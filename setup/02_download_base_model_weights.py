# Databricks notebook source
# DBTITLE 1,Download base model weights to a Unity Catalog volume
# MAGIC %md
# MAGIC # Download the base model weights to a Unity Catalog volume
# MAGIC
# MAGIC This setup notebook snapshots the base model from Hugging Face into a Unity Catalog volume so training loads governed, workspace-local weights instead of downloading from Hugging Face on every GPU worker.
# MAGIC
# MAGIC The destination is `model_volume_path` from `train/train.yaml`'s `training_config` section — the same value training reads — so setup and training always agree on one path and no separate setup key needs to stay in sync.
# MAGIC Rerunning is cheap: the download is skipped when the volume already holds a complete snapshot.
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

from training_utils import full_name, get_spark_session, load_training_config

# The download destination is train.yaml's model_volume_path — the exact path
# the training code loads from — read through the same load_training_config()
# helper the runner notebook and train.py use.
training_context = load_training_config()
MODEL_NAME = training_context["MODEL_NAME"]
MODEL_VOLUME_PATH = training_context["MODEL_VOLUME_PATH"]

if not MODEL_VOLUME_PATH:
    raise ValueError(
        "model_volume_path is empty in train/train.yaml. Set it to the UC volume "
        "directory that should hold the base model weights, e.g. "
        "/Volumes/<catalog>/<schema>/base_models/Qwen3-4B-Instruct-2507."
    )

# ('/', 'Volumes', catalog, schema, volume, ...)
path_parts = Path(MODEL_VOLUME_PATH).parts
if len(path_parts) < 5 or path_parts[:2] != ("/", "Volumes"):
    raise ValueError(
        "model_volume_path must look like /Volumes/<catalog>/<schema>/<volume>/...: "
        f"{MODEL_VOLUME_PATH}"
    )
volume_catalog, volume_schema, volume_name = path_parts[2:5]

print(f"Base model: {MODEL_NAME}")
print(f"Destination: {MODEL_VOLUME_PATH}")

# COMMAND ----------

spark = get_spark_session()

schema_q = full_name(volume_catalog, volume_schema)
volume_q = full_name(volume_catalog, volume_schema, volume_name)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_q}")

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
# MAGIC ## Snapshot the Hugging Face repo, then copy it into the volume
# MAGIC
# MAGIC The snapshot lands on local disk first and is then copied file by file into the volume.
# MAGIC Downloading straight into `/Volumes` is unreliable: volume FUSE mounts only support sequential writes, while `hf_transfer` (and resumed downloads) write at random offsets.
# MAGIC The copy is sequential, which the mount supports, and Hugging Face's `.cache` bookkeeping folder is excluded so the volume holds only the model files.
# MAGIC
# MAGIC Set `FORCE_REDOWNLOAD = True` to wipe the destination and download again (for example after changing `model_name` in `train.yaml`); otherwise a complete existing snapshot is reused.

# COMMAND ----------

FORCE_REDOWNLOAD = False

destination_dir = Path(MODEL_VOLUME_PATH)


def snapshot_is_complete(model_dir: Path) -> bool:
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "tokenizer_config.json").exists()
        and any(model_dir.glob("*.safetensors"))
    )


if FORCE_REDOWNLOAD and destination_dir.exists():
    print(f"FORCE_REDOWNLOAD: removing {destination_dir}")
    shutil.rmtree(destination_dir)

if snapshot_is_complete(destination_dir):
    print(f"Volume already holds a complete snapshot: {destination_dir}")
else:
    from huggingface_hub import snapshot_download

    try:
        import hf_transfer  # noqa: F401 — accelerates the local-disk download

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    local_disk_tmp = Path("/local_disk0/tmp")
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="hf-base-model-",
            dir=local_disk_tmp if local_disk_tmp.exists() else None,
        )
    )
    try:
        print(f"Downloading {MODEL_NAME} to {staging_dir}")
        snapshot_dir = Path(snapshot_download(repo_id=MODEL_NAME, local_dir=staging_dir))

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the volume snapshot
# MAGIC
# MAGIC Training fails fast when `model_volume_path` is missing or incomplete, so this cell confirms the copied snapshot has the config, tokenizer, and safetensors weights the loader needs before any GPU time is spent.

# COMMAND ----------

copied_files = sorted(path for path in destination_dir.rglob("*") if path.is_file())
if not copied_files:
    raise FileNotFoundError(f"No files found in {destination_dir} after download.")

total_bytes = 0
for copied_file in copied_files:
    size_bytes = copied_file.stat().st_size
    total_bytes += size_bytes
    print(f"{size_bytes / 1024**2:10.1f} MB  {copied_file.relative_to(destination_dir)}")

print(f"\nTotal: {total_bytes / 1024**3:.2f} GB in {len(copied_files)} files")

if not snapshot_is_complete(destination_dir):
    raise FileNotFoundError(
        f"{destination_dir} is missing config.json, tokenizer_config.json, or "
        "*.safetensors — the snapshot is incomplete. Rerun the download cell with "
        "FORCE_REDOWNLOAD = True."
    )

print(f"\nBase weights ready: {destination_dir}")
print("train/train.yaml's model_volume_path points here, so training now loads from the volume.")
