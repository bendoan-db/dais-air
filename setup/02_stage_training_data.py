# Databricks notebook source
# DBTITLE 1,Stage the training data in a Unity Catalog volume
# MAGIC %md
# MAGIC # Stage the training data in a Unity Catalog volume
# MAGIC
# MAGIC This setup notebook stages the cleaned transaction records written by `01_load_dataset.py` for AI Runtime training:
# MAGIC
# MAGIC 1. Writes the training-data Delta table: the raw transaction fields plus stable `training_id`, `split` (train/eval, `eval_fraction` in setup.yaml), and `shard_id` columns, all generated with Spark expressions so the work runs in parallel instead of row by row in Python. **SFT prompt/response formatting is deliberately not applied here** — the training loop renders prompts and labels from the raw fields.
# MAGIC 2. Exports the records to a Unity Catalog volume as Parquet files partitioned by `split` and `shard_id`, per the AI Runtime data-loading guidance for large Delta tables:
# MAGIC    https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
# MAGIC
# MAGIC Reading the exported files directly during training avoids Spark overhead on the GPU workers: each worker claims the `split=train/shard_id=N` directories where `N % world_size == rank` and loads only its own files with Hugging Face `datasets`; the `split=eval` shards are the held-out evaluation set.
# MAGIC Parquet is used because the training code (Unsloth) consumes Hugging Face `datasets`, which loads Parquet natively as memory-mapped Arrow tables.
# MAGIC
# MAGIC **Both the staged table and the volume export are overwritten on every run.**

# COMMAND ----------

from pathlib import Path

import yaml
from pyspark.sql import functions as F

# COMMAND ----------

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = notebook_context.notebookPath().get()
    script_dir = Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]

# utils.py is a plain setup-stage module, not a notebook. Add this notebook's
# directory explicitly so the import works in the workspace and local scripts.
import sys

setup_module_dir = str(script_dir.resolve())
if setup_module_dir not in sys.path:
    sys.path.insert(0, setup_module_dir)

from utils import (
    config_float,
    config_int,
    config_str,
    ensure_uc_object,
    get_spark_session,
    load_global_config,
    quote_identifier,
)

# Databricks notebooks inject display(); local Databricks Connect runs need
# a plain-text fallback so the verification cells work in both environments.
try:
    display
except NameError:
    def display(df):
        print(df.toPandas().to_string(index=False) if hasattr(df, "toPandas") else df)


config_path = script_dir / "setup.yaml"

with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Stage keys come from setup.yaml; catalog/schema come from the repo-root
# global.yaml.
global_config_path, global_config = load_global_config()
catalog = config_str(global_config, "catalog")
schema = config_str(global_config, "schema")

table = config_str(config, "source_table")
sft_table = config_str(config, "sft_table")
sft_volume = config_str(config, "sft_volume")
sft_files_path = f"/Volumes/{catalog}/{schema}/{sft_volume}/{sft_table}"

eval_fraction = config_float(config, "eval_fraction")
if not 0.0 < eval_fraction < 1.0:
    raise ValueError(f"eval_fraction must be between 0 and 1 (exclusive), got {eval_fraction}")
sft_shards = config_int(config, "sft_shards")

shard_key_columns = config.get("sft_shard_key_columns")
if not isinstance(shard_key_columns, list) or not shard_key_columns:
    raise ValueError("sft_shard_key_columns must be a non-empty list in setup.yaml")
shard_key_columns = [str(column_name) for column_name in shard_key_columns]

spark = get_spark_session()

catalog_q = quote_identifier(catalog)
schema_q = quote_identifier(schema)
table_q = quote_identifier(table)
sft_table_q = quote_identifier(sft_table)
sft_volume_q = quote_identifier(sft_volume)

full_schema_name = f"{catalog_q}.{schema_q}"
full_table_name = f"{full_schema_name}.{table_q}"
full_sft_table_name = f"{full_schema_name}.{sft_table_q}"
full_sft_volume_name = f"{full_schema_name}.{sft_volume_q}"

# COMMAND ----------

ensure_uc_object(spark, f"CREATE SCHEMA IF NOT EXISTS {full_schema_name}")
ensure_uc_object(spark, f"CREATE VOLUME IF NOT EXISTS {full_sft_volume_name}")

try:
    df = spark.table(full_table_name)
    source_columns = df.columns
except Exception as exc:
    raise RuntimeError(
        f"Source table {full_table_name} is not readable. Run "
        "setup/01_load_dataset.py first to download TabFormer and "
        "write the cleaned transaction table."
    ) from exc

missing_shard_key_columns = [
    column_name for column_name in shard_key_columns if column_name not in source_columns
]
if missing_shard_key_columns:
    raise ValueError(
        f"sft_shard_key_columns not found in {full_table_name}: "
        f"{missing_shard_key_columns}. Update sft_shard_key_columns in "
        "setup/setup.yaml to columns that exist in the source table."
    )

print(f"Stage config: {config_path}")
print(f"Global config: {global_config_path}")
print(f"Source table: {full_table_name} ({len(source_columns)} columns)")
print(f"Target training-data table: {full_sft_table_name}")
print(f"Target parquet export: {sft_files_path}")
print(f"Eval fraction: {eval_fraction}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the training records
# MAGIC
# MAGIC The staged table carries the raw transaction fields; the training loop renders the SFT prompt/response text from them.
# MAGIC `training_id`, `split`, and `shard_id` are generated with Spark expressions so the work runs in parallel.
# MAGIC
# MAGIC `shard_id` lets distributed training assign data slices by `rt.get_global_rank()` and `rt.get_world_size()` without loading the full table into every GPU worker.
# MAGIC `split` holds the deterministic 90/10 train/eval assignment (`eval_fraction` in setup.yaml).

# COMMAND ----------

# Row identity and shard assignment both derive from the configured shard
# key columns (sft_shard_key_columns in setup.yaml), so the export stays
# deterministic for any dataset that supplies its own key columns.
training_id_col = F.sha2(
    F.concat_ws(
        "||",
        *[F.col(column_name).cast("string") for column_name in shard_key_columns],
    ),
    256,
)

shard_key_sql = ", ".join(quote_identifier(column_name) for column_name in shard_key_columns)
shard_expr = f"pmod(xxhash64({shard_key_sql}), {sft_shards})"

# The split hashes the same key columns but with a salt literal, so it is
# deterministic across reruns yet independent of shard_id — reusing the
# unsalted shard hash would correlate the split with shard parity (both
# derive from the same low bits).
split_denominator = 1_000_000
eval_threshold = int(round(eval_fraction * split_denominator))
split_expr = (
    f"CASE WHEN pmod(xxhash64('eval_split', {shard_key_sql}), {split_denominator}) "
    f"< {eval_threshold} THEN 'eval' ELSE 'train' END"
)

staged_df = (
    df.withColumn("training_id", training_id_col)
    .withColumn("split", F.expr(split_expr))
    .withColumn("shard_id", F.expr(shard_expr).cast("int"))
    .select(
        "training_id",
        "split",
        "shard_id",
        "fraud_label",
        "is_fraud",
        "amount_usd",
        "user_id_text",
        "card_id_text",
        "transaction_ts_text",
        "use_chip_text",
        "merchant_city_text",
        "merchant_state_text",
        "mcc_text",
        "errors_text",
        "has_error_signal",
    )
)

(
    staged_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(full_sft_table_name)
)

spark.sql(
    f"""
    COMMENT ON TABLE {full_sft_table_name}
    IS 'Raw transaction training records (train/eval split, hash-sharded) for the AIR fraud demo; SFT formatting happens in the training loop'
    """
)

print(f"Overwrote training-data Delta table {full_sft_table_name}")

# COMMAND ----------

print(spark.table(full_sft_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export the training records to a Unity Catalog volume as Parquet
# MAGIC
# MAGIC AI Runtime's data-loading guidance recommends exporting large Delta tables to a UC volume and reading the files directly during training, which avoids Spark overhead on the GPU workers:
# MAGIC https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
# MAGIC
# MAGIC The export is partitioned by `split` and then `shard_id` (`split=train/shard_id=N/`, `split=eval/shard_id=N/`), so the rank-sharding contract carries over per split: each GPU worker claims the `split=train/shard_id=N` directories where `N % world_size == rank`, and the eval shards are read the same way for held-out evaluation.

# COMMAND ----------

(
    spark.table(full_sft_table_name)
    .repartition(sft_shards, "shard_id")
    .write.mode("overwrite")
    .partitionBy("split", "shard_id")
    .parquet(sft_files_path)
)

print(f"Exported train/eval parquet shards to {sft_files_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the staged table and the volume export
# MAGIC
# MAGIC Training fails fast when the parquet export is missing or incomplete, so this cell confirms both splits hold every shard and every staged row before any GPU time is spent.
# MAGIC The export is checked by reading it back through Spark (rather than listing the FUSE mount), so verification also works on local Databricks Connect runs.

# COMMAND ----------

staged_loaded_df = spark.table(full_sft_table_name)

split_summary_expressions = [
    F.count("*").alias("row_count"),
    F.countDistinct("shard_id").alias("shard_count"),
    F.min("shard_id").alias("min_shard_id"),
    F.max("shard_id").alias("max_shard_id"),
]
if "is_fraud" in staged_loaded_df.columns:
    split_summary_expressions.extend(
        [
            F.sum("is_fraud").alias("fraud_row_count"),
            F.avg("is_fraud").alias("fraud_rate"),
        ]
    )

display(staged_loaded_df.groupBy("split").agg(*split_summary_expressions))
display(
    staged_loaded_df.select(
        "training_id", "split", "shard_id", "amount_usd", "fraud_label", "merchant_city_text"
    ).limit(10)
)

export_df = spark.read.parquet(sft_files_path)
export_row_count = export_df.count()
table_row_count = staged_loaded_df.count()

export_split_summary = {
    row["split"]: row
    for row in export_df.groupBy("split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("shard_id").alias("shard_count"),
    )
    .collect()
}

print(f"Parquet export: {export_row_count} rows at {sft_files_path}")
for split_name in ("train", "eval"):
    summary_row = export_split_summary.get(split_name)
    if summary_row is None:
        raise ValueError(
            f"The parquet export has no split={split_name} partition. "
            "Check eval_fraction in setup/setup.yaml and rerun this notebook."
        )
    print(
        f"  split={split_name}: {summary_row['row_count']} rows "
        f"({summary_row['row_count'] / export_row_count:.1%}) "
        f"in {summary_row['shard_count']} shards"
    )
    if summary_row["shard_count"] != sft_shards:
        raise ValueError(
            f"Expected {sft_shards} shard_id partitions in the split={split_name} export, "
            f"found {summary_row['shard_count']}. For small datasets, lower sft_shards "
            "(or raise eval_fraction) in setup/setup.yaml so every shard has rows in both splits."
        )
if export_row_count != table_row_count:
    raise ValueError(
        f"Parquet export has {export_row_count} rows but {full_sft_table_name} has "
        f"{table_row_count} — the export is incomplete; rerun this notebook."
    )

print("Raw training data staged. Next: setup/03_prepare_sft.py.")
