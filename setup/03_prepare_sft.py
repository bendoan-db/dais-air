# Databricks notebook source
# DBTITLE 1,Stage SFT-format training records in a Unity Catalog volume
# MAGIC %md
# MAGIC # Stage SFT-format records for the training loop
# MAGIC
# MAGIC This notebook turns the raw-record train/eval export staged by `setup/02_stage_training_data.py` into supervised fine-tuning records and stages them in their own Unity Catalog volume:
# MAGIC
# MAGIC 1. Reads the raw parquet export (`split=train|eval / shard_id=N`) from the setup stage's volume.
# MAGIC 2. Renders each record's `prompt` and `assistant_response` with the shared template and labeling heuristic in `setup/utils.py` (`render_fraud_prompt` / `render_fraud_response` — the load-test and trainer copies are checked against these canonical functions, so every stage sees byte-identical prompts).
# MAGIC 3. Writes the SFT records to `sft_staging_volume` from `setup.yaml` with the same `split`/`shard_id` partitioning consumed by both standalone training projects.
# MAGIC
# MAGIC The staged records stay **model-agnostic**: each model's own chat template is applied inside its training loop. Rerun this notebook after setup/02 or after changing the prompt template or `suspicious_amount_threshold` in `setup.yaml`.
# MAGIC
# MAGIC **The SFT staging export is overwritten on every run.** Runs on serverless compute; running locally via Databricks Connect additionally requires the local Python minor version to match the serverless runtime's — the render step is a Python UDF, and Connect rejects UDFs across mismatched Python versions.

# COMMAND ----------

import sys
from pathlib import Path

import yaml

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = notebook_context.notebookPath().get()
    script_dir = Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]

setup_module_dir = script_dir.resolve()
if str(setup_module_dir) not in sys.path:
    sys.path.insert(0, str(setup_module_dir))

from utils import (
    config_float,
    config_str,
    ensure_uc_object,
    full_name,
    get_spark_session,
    load_global_config,
)

# Databricks notebooks inject display(); local Databricks Connect runs need
# a plain-text fallback.
try:
    display
except NameError:
    def display(df):
        print(df.toPandas().to_string(index=False) if hasattr(df, "toPandas") else df)

# COMMAND ----------

config_path = script_dir / "setup.yaml"
with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

_, global_config = load_global_config()
catalog = config_str(global_config, "catalog")
schema = config_str(global_config, "schema")
sft_table = config_str(config, "sft_table")
sft_volume = config_str(config, "sft_volume")
sft_staging_volume = config_str(config, "sft_staging_volume")
SUSPICIOUS_AMOUNT_THRESHOLD = config_float(config, "suspicious_amount_threshold")

RAW_SPLIT_FILES_DIR = f"/Volumes/{catalog}/{schema}/{sft_volume}/{sft_table}"
SFT_FILES_DIR = f"/Volumes/{catalog}/{schema}/{sft_staging_volume}/{sft_table}"
schema_q = full_name(catalog, schema)
sft_staging_volume_q = full_name(catalog, schema, sft_staging_volume)

print(f"Raw split export (input): {RAW_SPLIT_FILES_DIR}")
print(f"SFT staging export (output): {SFT_FILES_DIR}")
print(f"suspicious_amount_threshold: {SUSPICIOUS_AMOUNT_THRESHOLD}")

# COMMAND ----------

from pyspark.sql import functions as F

spark = get_spark_session()

ensure_uc_object(spark, f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
ensure_uc_object(spark, f"CREATE VOLUME IF NOT EXISTS {sft_staging_volume_q}")

# Existence checks go through Spark (not the local filesystem) so this
# notebook also runs via Databricks Connect, where /Volumes is not mounted.
try:
    raw_df = spark.read.parquet(RAW_SPLIT_FILES_DIR)
except Exception as exc:
    raise FileNotFoundError(
        f"Could not read the raw split export at {RAW_SPLIT_FILES_DIR}. Run "
        "setup/01_load_dataset.py and setup/02_stage_training_data.py first."
    ) from exc

if "split" not in raw_df.columns or "shard_id" not in raw_df.columns:
    raise ValueError(
        f"{RAW_SPLIT_FILES_DIR} is not partitioned by split/shard_id — rerun "
        "setup/02_stage_training_data.py to restage the records with "
        "train/eval splits."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Render the SFT records
# MAGIC
# MAGIC The prompt/response text comes from the canonical renderers in `setup/utils.py` — the single source of the prompt contract (the monitor's `prompt_fields` extraction and the load test's payloads depend on this exact shape).
# MAGIC They run row-wise inside `mapInPandas`, so the rendering parallelizes across the cluster; the module file is shipped to the executors as a session artifact because they do not share this interpreter's `sys.path`.

# COMMAND ----------

SETUP_UTILS_PATH = str(setup_module_dir / "utils.py")

try:
    spark.addArtifact(SETUP_UTILS_PATH, pyfile=True)
except Exception as artifact_exc:
    # Rerunning in a session that already holds a different version of the
    # artifact raises; classic (non-Connect) clusters take the addPyFile path.
    try:
        spark.sparkContext.addPyFile(SETUP_UTILS_PATH)
    except Exception:
        print(
            f"Note: could not distribute setup/utils.py ({artifact_exc}); "
            "relying on the runtime's workspace-file sync. If the render step "
            "fails with ModuleNotFoundError, restart the session and rerun."
        )

SFT_EXPORT_SCHEMA = (
    "training_id string, split string, shard_id int, "
    "prompt string, assistant_response string, is_fraud int"
)

suspicious_amount_threshold = float(SUSPICIOUS_AMOUNT_THRESHOLD)


def render_sft_batches(batches):
    import pandas as pd
    from utils import render_fraud_prompt, render_fraud_response

    for raw_pdf in batches:
        records = raw_pdf.to_dict("records")
        yield pd.DataFrame(
            {
                "training_id": raw_pdf["training_id"],
                "split": raw_pdf["split"],
                "shard_id": raw_pdf["shard_id"],
                "prompt": [render_fraud_prompt(record) for record in records],
                "assistant_response": [
                    render_fraud_response(record, suspicious_amount_threshold)
                    for record in records
                ],
                # Kept so the post-training fraud-classification evaluation
                # can stratify its eval-split sample.
                "is_fraud": raw_pdf["is_fraud"],
            }
        )


sft_df = raw_df.mapInPandas(render_sft_batches, schema=SFT_EXPORT_SCHEMA)

(
    sft_df.repartition("split", "shard_id")
    .write.mode("overwrite")
    .partitionBy("split", "shard_id")
    .parquet(SFT_FILES_DIR)
)

print(f"Staged SFT-format export at {SFT_FILES_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the SFT staging
# MAGIC
# MAGIC Training fails fast when the staging is missing or incomplete, so this cell confirms every raw row was rendered into every split/shard before any GPU time is spent — checked by reading the export back through Spark, so verification also works on local Connect runs.

# COMMAND ----------

raw_split_summary = {
    row["split"]: row
    for row in raw_df.groupBy("split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("shard_id").alias("shard_count"),
    )
    .collect()
}

sft_export_df = spark.read.parquet(SFT_FILES_DIR)
sft_split_summary = {
    row["split"]: row
    for row in sft_export_df.groupBy("split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("shard_id").alias("shard_count"),
        F.sum("is_fraud").alias("fraud_row_count"),
    )
    .collect()
}

for split_name in ("train", "eval"):
    raw_row = raw_split_summary.get(split_name)
    sft_row = sft_split_summary.get(split_name)
    if raw_row is None or sft_row is None:
        raise ValueError(
            f"split={split_name} is missing from the "
            f"{'raw export' if raw_row is None else 'SFT staging'} — rerun "
            "setup/02_stage_training_data.py, then this notebook."
        )
    print(
        f"split={split_name}: {sft_row['row_count']} SFT rows in "
        f"{sft_row['shard_count']} shards ({sft_row['fraud_row_count']} fraud)"
    )
    if sft_row["row_count"] != raw_row["row_count"]:
        raise ValueError(
            f"split={split_name}: SFT staging has {sft_row['row_count']} rows "
            f"but the raw export has {raw_row['row_count']} — the render is "
            "incomplete; rerun this notebook."
        )
    if sft_row["shard_count"] != raw_row["shard_count"]:
        raise ValueError(
            f"split={split_name}: SFT staging covers {sft_row['shard_count']} "
            f"shards but the raw export has {raw_row['shard_count']} — the "
            "render is incomplete; rerun this notebook."
        )

display(
    sft_export_df.select(
        "training_id", "split", "shard_id", "prompt", "assistant_response"
    ).limit(5)
)

print(
    "SFT records staged. Next: run a project notebook under train/, or run "
    "`air run --file train.yaml` from that project directory."
)
