# Databricks notebook source
# DBTITLE 1,Mirror Base Model Weights Into a Unity Catalog Volume
# MAGIC %md
# MAGIC # Load base model weights from Hugging Face into a Unity Catalog volume
# MAGIC
# MAGIC This setup notebook downloads the base checkpoint named in `setup.yaml` (`model_name`) from the Hugging Face Hub and mirrors it into the configured Unity Catalog volume (`model_volume`).
# MAGIC Training and registration can then load the weights from `/Volumes/...` instead of reaching out to the Hub from every GPU worker, which removes the Hub as a runtime dependency of the demo (no per-run download, no rate limits, no egress from the GPU nodes) and pins every run to one governed copy of the checkpoint.
# MAGIC
# MAGIC The transfer is deliberately **file-by-file**: each repo file is downloaded to local disk, copied to the volume, and deleted locally before the next one starts.
# MAGIC That bounds local disk use to the largest single safetensors shard, and it keeps volume writes sequential — large parallel writes to the volume FUSE mount fail with `OSError: [Errno 11] Resource temporarily unavailable`.
# MAGIC
# MAGIC Reruns are cheap: a file already present on the volume with the size the Hub reports is skipped unless `force_model_download` is set.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Unity Catalog volumes: https://docs.databricks.com/aws/en/volumes/
# MAGIC - `huggingface_hub` download API: https://huggingface.co/docs/huggingface_hub/en/guides/download
# MAGIC - Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute and dependency setup
# MAGIC
# MAGIC Run this notebook on Databricks serverless compute (no GPU needed — nothing is loaded into a model here, the bytes are only copied).
# MAGIC It has to run **in the workspace**: the destination is a volume FUSE path, which does not exist on a local Databricks Connect client.
# MAGIC
# MAGIC `hf_transfer` is installed for the Rust-based multipart downloader; it is enabled below only if the import succeeds.

# COMMAND ----------

# MAGIC %pip install -qqq "huggingface_hub>=0.34.0" "hf_transfer>=0.1.9" "pyyaml>=6.0.2" "pandas>=2.2.0"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load shared utilities and configuration

# COMMAND ----------

from fnmatch import fnmatch
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = notebook_context.notebookPath().get()
    script_dir = Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]

# training_utils is a plain Python module in train/ shared across the demo;
# the same import works for workspace-notebook and local-script runs. (It is
# not named `utils` because GPU base environments ship packages that register
# a top-level `utils` module, shadowing any local one.)
train_module_dir = str((script_dir.parent / "train").resolve())
if train_module_dir not in sys.path:
    sys.path.insert(0, train_module_dir)

from training_utils import (
    config_bool,
    config_str,
    config_value,
    full_name,
    get_spark_session,
    load_yaml_config,
)

config_path, config = load_yaml_config("setup.yaml", base_dir=script_dir)

catalog = config_str(config, "catalog")
schema = config_str(config, "schema")
model_volume = config_str(config, "model_volume")
model_name = config_str(config, "model_name")
model_revision = config_str(config, "model_revision")
force_model_download = config_bool(config, "force_model_download")

model_ignore_patterns = config_value(config, "model_ignore_patterns") or []
if not isinstance(model_ignore_patterns, list):
    raise ValueError("Config key model_ignore_patterns must be a list of fnmatch patterns")
model_ignore_patterns = [str(pattern) for pattern in model_ignore_patterns]

spark = get_spark_session()

schema_q = full_name(catalog, schema)
model_volume_q = full_name(catalog, schema, model_volume)

# One subdirectory per checkpoint, named after the repo's model id, so the
# volume can hold more than one base model.
model_volume_root = Path(f"/Volumes/{catalog}/{schema}/{model_volume}")
model_dir = model_volume_root / model_name.split("/")[-1]
metadata_path = model_volume_root / f"{model_dir.name}_download_metadata.json"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {model_volume_q}")

model_dir.mkdir(parents=True, exist_ok=True)

print(f"Config path: {config_path}")
print(f"Hugging Face repo: {model_name} (revision: {model_revision})")
print(f"Target volume: {model_volume_root}")
print(f"Target weights directory: {model_dir}")
print(f"Ignore patterns: {model_ignore_patterns}")
print(f"Force re-download: {force_model_download}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the repository file list
# MAGIC
# MAGIC The revision is resolved to a commit SHA up front and every file is fetched at that SHA, so the mirror is a single consistent snapshot even if the branch moves mid-run.
# MAGIC File sizes come from the same call and drive both the progress output and the resume check.
# MAGIC
# MAGIC Gated or private repositories need a Hub token: set `HF_TOKEN` in the notebook environment (for example from `dbutils.secrets.get(...)`) before running this cell.

# COMMAND ----------

# HF_HUB_ENABLE_HF_TRANSFER is read when huggingface_hub is imported, so it has
# to be set first — and only when hf_transfer is actually importable, because
# huggingface_hub raises instead of falling back if the package is missing.
try:
    import hf_transfer  # noqa: F401

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("hf_transfer is not installed; falling back to the default Python downloader.")

from huggingface_hub import HfApi, hf_hub_download

hf_token = os.environ.get("HF_TOKEN")
api = HfApi(token=hf_token)

repo_info = api.model_info(model_name, revision=model_revision, files_metadata=True)
resolved_revision = repo_info.sha
repo_file_sizes = {sibling.rfilename: sibling.size for sibling in repo_info.siblings}


def is_ignored(repo_path: str) -> bool:
    return any(fnmatch(repo_path, pattern) for pattern in model_ignore_patterns)


selected_files = sorted(path for path in repo_file_sizes if not is_ignored(path))
ignored_files = sorted(path for path in repo_file_sizes if is_ignored(path))

if not selected_files:
    raise ValueError(
        f"No files selected from {model_name}@{resolved_revision}; "
        "check model_ignore_patterns in setup.yaml."
    )

selected_bytes = sum(repo_file_sizes[path] or 0 for path in selected_files)

print(f"Resolved revision: {resolved_revision}")
print(f"Selected {len(selected_files)} files ({selected_bytes / 1024**3:.2f} GiB)")
if ignored_files:
    print(f"Ignored {len(ignored_files)} files: {ignored_files}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download to local disk and copy into the volume
# MAGIC
# MAGIC Each file lands in a local staging directory first, is streamed to the volume in 16 MiB chunks, and is then removed locally.
# MAGIC The copy writes to a `.partial` sibling and renames on completion, so an interrupted run never leaves a short file that the size check would mistake for a complete one.

# COMMAND ----------

COPY_BUFFER_BYTES = 16 * 1024 * 1024


def pick_staging_root() -> Path:
    """Return a writable local scratch directory.

    /local_disk0 is the large ephemeral disk on Databricks compute; fall back to
    the platform temp directory when it is absent (some serverless images).
    """
    for candidate in (Path("/local_disk0"), Path(tempfile.gettempdir())):
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    raise RuntimeError("No writable local staging directory found")


def copy_to_volume(source: Path, destination: Path) -> None:
    """Stream one file onto the volume, publishing it with an atomic rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_destination = destination.with_name(destination.name + ".partial")

    try:
        with source.open("rb") as source_file, partial_destination.open("wb") as output_file:
            shutil.copyfileobj(source_file, output_file, length=COPY_BUFFER_BYTES)
        partial_destination.replace(destination)
    except BaseException:
        if partial_destination.exists():
            partial_destination.unlink()
        raise


def is_already_mirrored(repo_path: str) -> bool:
    expected_size = repo_file_sizes[repo_path]
    destination = model_dir / repo_path
    if expected_size is None or not destination.exists():
        return False
    return destination.stat().st_size == expected_size


staging_dir = pick_staging_root() / "air-demo-hf-download" / model_dir.name
staging_dir.mkdir(parents=True, exist_ok=True)

copied_files: list[str] = []
reused_files: list[str] = []
copied_bytes = 0

try:
    for file_index, repo_path in enumerate(selected_files, start=1):
        expected_size = repo_file_sizes[repo_path] or 0
        destination = model_dir / repo_path

        if not force_model_download and is_already_mirrored(repo_path):
            reused_files.append(repo_path)
            print(f"[{file_index}/{len(selected_files)}] {repo_path}: already mirrored, skipping")
            continue

        print(
            f"[{file_index}/{len(selected_files)}] {repo_path}: "
            f"downloading {expected_size / 1024**2:.1f} MiB"
        )
        local_path = Path(
            hf_hub_download(
                repo_id=model_name,
                filename=repo_path,
                revision=resolved_revision,
                local_dir=staging_dir,
                token=hf_token,
            )
        )

        try:
            copy_to_volume(local_path, destination)
        finally:
            local_path.unlink(missing_ok=True)

        copied_files.append(repo_path)
        copied_bytes += destination.stat().st_size
finally:
    shutil.rmtree(staging_dir, ignore_errors=True)

print(
    f"Copied {len(copied_files)} files ({copied_bytes / 1024**3:.2f} GiB) to {model_dir}; "
    f"reused {len(reused_files)} already-mirrored files"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record the mirror's provenance
# MAGIC
# MAGIC The metadata file is written **next to** the weights directory rather than inside it, so the mirrored directory stays a byte-for-byte copy of the Hub snapshot and can be handed straight to `from_pretrained()` or vLLM's `--model`.

# COMMAND ----------

mirrored_files = sorted(path for path in model_dir.rglob("*") if path.is_file())

download_metadata = {
    "repo_id": model_name,
    "requested_revision": model_revision,
    "resolved_revision": resolved_revision,
    "mirrored_at_utc": datetime.now(timezone.utc).isoformat(),
    "weights_dir": str(model_dir),
    "file_count": len(mirrored_files),
    "total_bytes": sum(path.stat().st_size for path in mirrored_files),
    "ignore_patterns": model_ignore_patterns,
    "ignored_files": ignored_files,
}

metadata_path.write_text(json.dumps(download_metadata, indent=2), encoding="utf-8")

print(f"Wrote {metadata_path}")
print(json.dumps(download_metadata, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the mirrored snapshot
# MAGIC
# MAGIC The checks below are the ones that matter for loading the checkpoint: a config, a tokenizer, at least one safetensors file, and — for sharded checkpoints — every shard named in `model.safetensors.index.json`.
# MAGIC A missing shard is the failure mode worth catching here rather than at the start of a GPU run.

# COMMAND ----------

import pandas as pd

relative_paths = {path.relative_to(model_dir).as_posix() for path in mirrored_files}

if "config.json" not in relative_paths:
    raise FileNotFoundError(f"config.json is missing from {model_dir}")

if not {"tokenizer.json", "tokenizer_config.json"} & relative_paths:
    raise FileNotFoundError(f"No tokenizer files found in {model_dir}")

safetensors_files = sorted(path for path in relative_paths if path.endswith(".safetensors"))
if not safetensors_files:
    raise FileNotFoundError(f"No .safetensors weight files found in {model_dir}")

index_path = model_dir / "model.safetensors.index.json"
if index_path.exists():
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing_shards = sorted(set(weight_map.values()) - relative_paths)
    if missing_shards:
        raise FileNotFoundError(
            f"{len(missing_shards)} shard(s) listed in model.safetensors.index.json are missing "
            f"from {model_dir}: {missing_shards}"
        )
    print(f"All {len(set(weight_map.values()))} safetensors shards are present")

partial_files = sorted(path for path in relative_paths if path.endswith(".partial"))
if partial_files:
    raise RuntimeError(f"Incomplete files left on the volume: {partial_files}")

file_summary = pd.DataFrame(
    [
        {
            "file": path.relative_to(model_dir).as_posix(),
            "size_mib": round(path.stat().st_size / 1024**2, 2),
        }
        for path in mirrored_files
    ]
).sort_values("size_mib", ascending=False, ignore_index=True)

print(f"{len(mirrored_files)} files, {download_metadata['total_bytes'] / 1024**3:.2f} GiB total")
display(file_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consuming the mirror
# MAGIC
# MAGIC The weights directory printed above is a standard Hugging Face snapshot, so anything that takes a model id takes this path instead:
# MAGIC
# MAGIC ```python
# MAGIC AutoModelForCausalLM.from_pretrained("/Volumes/<catalog>/<schema>/<model_volume>/<model>")
# MAGIC ```
# MAGIC
# MAGIC To point the training run at the mirror, set `train.yaml`'s `parameters.training_config.model_name` to that path; `train.py` passes the value straight to `from_pretrained()`.
# MAGIC Keep `setup.yaml`'s `model_name` as the Hub repo id — it is what this notebook downloads.
# MAGIC
# MAGIC Volume reads are fine from every GPU worker (they are read-only and cached by the FUSE mount). Volume **writes** of large safetensors are not: the training and merge steps keep writing to `/local_disk0` for that reason.

