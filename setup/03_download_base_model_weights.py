# Databricks notebook source
# DBTITLE 1,Download Hugging Face model weights to Unity Catalog volumes
# MAGIC %md
# MAGIC # Download Hugging Face model weights to Unity Catalog volumes
# MAGIC
# MAGIC This setup notebook snapshots open-source models from Hugging Face into Unity Catalog volumes so downstream jobs load governed, workspace-local weights instead of downloading from Hugging Face on every GPU worker.
# MAGIC
# MAGIC The download list is the `models:` section of `setup/setup.yaml` — each entry pairs a Hugging Face repo id (`huggingface_path`) with the volume directory that receives the weights (`volume_path`). Point `train/train.yaml`'s `model_volume_path` at the entry the fine-tune should load; a final cell cross-checks that coupling.
# MAGIC Rerunning is cheap: a download is skipped when its volume already holds a complete snapshot.
# MAGIC
# MAGIC Run this as a Databricks workspace notebook on serverless (CPU) compute. It writes through the `/Volumes` FUSE mount, which is not available to local Databricks Connect runs.

# COMMAND ----------

# MAGIC %pip install -qqq "huggingface_hub>=0.30.0" hf_transfer
# MAGIC %restart_python

# COMMAND ----------

import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

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

from training_utils import ensure_uc_object, full_name, get_spark_session, load_global_config

config_path = script_dir / "setup.yaml"
with config_path.open("r", encoding="utf-8") as config_file:
    setup_config = yaml.safe_load(config_file)

# Optional Databricks secret with a Hugging Face token for gated models
# (e.g. meta-llama); huggingface_hub reads HF_TOKEN from the environment.
hf_token_secret_scope = str(setup_config.get("hf_token_secret_scope") or "").strip()
hf_token_secret_key = str(setup_config.get("hf_token_secret_key") or "").strip()

model_entries = setup_config.get("models") or []
if not isinstance(model_entries, list) or not model_entries:
    raise ValueError(
        "setup.yaml must define a non-empty `models:` list; each entry needs "
        "`huggingface_path` (Hugging Face repo id) and `volume_path` "
        "(/Volumes/<catalog>/<schema>/<volume>/... destination directory)."
    )

models = []
seen_volume_paths = set()
for entry in model_entries:
    if not isinstance(entry, dict):
        raise ValueError(f"models entries must be mappings, got: {entry!r}")
    huggingface_path = str(entry.get("huggingface_path") or "").strip()
    volume_path = str(entry.get("volume_path") or "").strip().rstrip("/")
    if not huggingface_path or not volume_path:
        raise ValueError(
            f"Each models entry needs a non-empty huggingface_path and volume_path: {entry!r}"
        )
    # ('/', 'Volumes', catalog, schema, volume, ...)
    path_parts = Path(volume_path).parts
    if len(path_parts) < 5 or path_parts[:2] != ("/", "Volumes"):
        raise ValueError(
            "volume_path must look like /Volumes/<catalog>/<schema>/<volume>/...: "
            f"{volume_path}"
        )
    # Two snapshots copied into one directory would interleave their files.
    if volume_path in seen_volume_paths:
        raise ValueError(f"Duplicate volume_path in models: {volume_path}")
    seen_volume_paths.add(volume_path)
    models.append(
        {
            "huggingface_path": huggingface_path,
            "destination_dir": Path(volume_path),
            "volume": path_parts[2:5],
        }
    )

for model in models:
    print(f"{model['huggingface_path']}  ->  {model['destination_dir']}")

# COMMAND ----------

spark = get_spark_session()

for volume_catalog, volume_schema, volume_name in sorted({model["volume"] for model in models}):
    schema_q = full_name(volume_catalog, volume_schema)
    volume_q = full_name(volume_catalog, volume_schema, volume_name)
    ensure_uc_object(spark, f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
    ensure_uc_object(spark, f"CREATE VOLUME IF NOT EXISTS {volume_q}")

    volume_root = Path(f"/Volumes/{volume_catalog}/{volume_schema}/{volume_name}")
    if not volume_root.exists():
        raise RuntimeError(
            f"{volume_root} is not accessible from this process. The volume exists in "
            "Unity Catalog, but the /Volumes FUSE mount is only available inside a "
            "Databricks workspace notebook — run this notebook there."
        )
    print(f"Ready: {volume_q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Snapshot each Hugging Face repo, then copy it into its volume
# MAGIC
# MAGIC Each snapshot lands on local disk first and is then copied file by file into its volume.
# MAGIC Downloading straight into `/Volumes` is unreliable: volume FUSE mounts only support sequential writes, while `hf_transfer` (and resumed downloads) write at random offsets.
# MAGIC The copy is sequential, which the mount supports, and Hugging Face's `.cache` bookkeeping folder is excluded so the volume holds only the model files.
# MAGIC
# MAGIC Set `FORCE_REDOWNLOAD = True` to wipe every destination and download again (for example after repointing a `volume_path` at a different repo); otherwise a complete existing snapshot is reused.

# COMMAND ----------

FORCE_REDOWNLOAD = False

if hf_token_secret_scope and hf_token_secret_key:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(hf_token_secret_scope, hf_token_secret_key)
    print("Hugging Face token loaded from Databricks secret (required for gated models).")

from huggingface_hub import snapshot_download

try:
    import hf_transfer  # noqa: F401 — accelerates the local-disk downloads

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass


def snapshot_is_complete(model_dir: Path) -> bool:
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "tokenizer_config.json").exists()
        and any(model_dir.glob("*.safetensors"))
    )


def download_snapshot_to_volume(huggingface_path: str, destination_dir: Path) -> None:
    local_disk_tmp = Path("/local_disk0/tmp")
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="hf-model-",
            dir=local_disk_tmp if local_disk_tmp.exists() else None,
        )
    )
    try:
        print(f"Downloading {huggingface_path} to {staging_dir}")
        snapshot_dir = Path(snapshot_download(repo_id=huggingface_path, local_dir=staging_dir))

        print(f"Copying snapshot to {destination_dir}")
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            snapshot_dir,
            destination_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".cache"),
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


for model in models:
    destination_dir = model["destination_dir"]
    if FORCE_REDOWNLOAD and destination_dir.exists():
        print(f"FORCE_REDOWNLOAD: removing {destination_dir}")
        shutil.rmtree(destination_dir)

    if snapshot_is_complete(destination_dir):
        print(f"Volume already holds a complete snapshot: {destination_dir}")
    else:
        download_snapshot_to_volume(model["huggingface_path"], destination_dir)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the volume snapshots
# MAGIC
# MAGIC Model loading fails fast when a volume snapshot is missing or incomplete, so this cell confirms every copied snapshot has the config, tokenizer, and safetensors weights a loader needs before any GPU time is spent.

# COMMAND ----------

for model in models:
    destination_dir = model["destination_dir"]
    copied_files = sorted(path for path in destination_dir.rglob("*") if path.is_file())
    if not copied_files:
        raise FileNotFoundError(f"No files found in {destination_dir} after download.")

    total_bytes = 0
    for copied_file in copied_files:
        size_bytes = copied_file.stat().st_size
        total_bytes += size_bytes
        print(f"{size_bytes / 1024**2:10.1f} MB  {copied_file.relative_to(destination_dir)}")

    if not snapshot_is_complete(destination_dir):
        raise FileNotFoundError(
            f"{destination_dir} is missing config.json, tokenizer_config.json, or "
            "*.safetensors — the snapshot is incomplete. Rerun the download cell with "
            "FORCE_REDOWNLOAD = True."
        )

    print(
        f"\n{model['huggingface_path']}: {total_bytes / 1024**3:.2f} GB in "
        f"{len(copied_files)} files at {destination_dir}\n"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-check against the training config
# MAGIC
# MAGIC Training loads base weights from `model_volume_path` in `train/train.yaml`'s `training_config`. This advisory cell confirms that path is covered by one of the snapshots above so the fine-tune doesn't fail at model load (it does not stop the notebook — the models list may legitimately serve other workloads).

# COMMAND ----------

_, global_config = load_global_config()
train_model_volume_path = str(global_config.get("model_volume_path") or "").rstrip("/")

if not train_model_volume_path:
    print(
        "global.yaml has no model_volume_path set — training will download "
        f"{global_config.get('model_name')} from Hugging Face on each GPU worker. "
        "Point model_volume_path at one of the snapshots above to load "
        "workspace-local weights instead."
    )
elif train_model_volume_path in {str(model["destination_dir"]) for model in models}:
    print(
        "global.yaml's model_volume_path matches a downloaded snapshot: "
        f"{train_model_volume_path}"
    )
else:
    print(
        "WARNING: global.yaml's model_volume_path is not among the snapshots "
        f"this notebook downloads: {train_model_volume_path}. Training will fail at "
        "model load unless that directory is populated some other way — add a "
        "matching entry to setup.yaml's models list or update model_volume_path."
    )
