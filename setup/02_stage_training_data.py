# Databricks notebook source
# DBTITLE 1,Stage the supervised fine-tuning data in a Unity Catalog volume
# MAGIC %md
# MAGIC # Stage the supervised fine-tuning data in a Unity Catalog volume
# MAGIC
# MAGIC This setup notebook builds the supervised fine-tuning records from the cleaned transaction table written by `01_load_dataset.py` and stages them for AI Runtime training:
# MAGIC
# MAGIC 1. Writes the SFT Delta table with `prompt`, `assistant_response`, and stable `shard_id` columns. Prompt text, assistant JSON, and shard IDs are generated with Spark expressions so the work runs in parallel instead of row by row in Python.
# MAGIC 2. Exports the SFT records to a Unity Catalog volume as Parquet files partitioned by `shard_id`, per the AI Runtime data-loading guidance for large Delta tables:
# MAGIC    https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
# MAGIC
# MAGIC Reading the exported files directly during training avoids Spark overhead on the GPU workers: each worker claims the `shard_id=N` directories where `N % world_size == rank` and loads only its own files with Hugging Face `datasets`.
# MAGIC Parquet is used because the training code (Unsloth) consumes Hugging Face `datasets`, which loads Parquet natively as memory-mapped Arrow tables.
# MAGIC
# MAGIC **Both the SFT table and the volume export are overwritten on every run.**

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

# training_utils is a plain Python module in train/ shared across the demo;
# the same import works for workspace-notebook and local-script runs. (It is
# not named `utils` because GPU base environments ship packages that register
# a top-level `utils` module, shadowing any local one.)
import sys

train_module_dir = str((script_dir.parent / "train").resolve())
if train_module_dir not in sys.path:
    sys.path.insert(0, train_module_dir)

from training_utils import (
    config_float,
    config_int,
    config_str,
    ensure_uc_object,
    get_spark_session,
    load_global_config,
    quote_identifier,
)

config_path = script_dir / "setup.yaml"

with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Stage keys come from setup.yaml; the pipeline-wide identity comes from
# the repo-root global.yaml.
global_config_path, global_config = load_global_config()
catalog = config_str(global_config, "catalog")
schema = config_str(global_config, "schema")
table = config_str(global_config, "source_table")
sft_table = config_str(global_config, "sft_table")
sft_volume = config_str(global_config, "sft_volume")
sft_files_path = f"/Volumes/{catalog}/{schema}/{sft_volume}/{sft_table}"

suspicious_amount_threshold = config_float(config, "suspicious_amount_threshold")
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
print(f"Target SFT table: {full_sft_table_name}")
print(f"Target SFT parquet export: {sft_files_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create supervised fine-tuning records
# MAGIC
# MAGIC The training notebook reads this table directly instead of building prompts row by row in Python.
# MAGIC Prompt text, assistant JSON, and `shard_id` are generated with Spark expressions so the work runs in parallel.
# MAGIC
# MAGIC `shard_id` lets distributed training assign data slices by `rt.get_global_rank()` and `rt.get_world_size()` without loading the full table into every GPU worker.

# COMMAND ----------

prompt_col = F.concat(
    F.lit("You are a fraud decision model for a credit-card transaction stream. "),
    F.lit("Classify the transaction as legitimate, suspicious, or likely_fraud. "),
    F.lit("Return only compact JSON with keys risk, action, and reason.\n\n"),
    F.lit("Transaction:\n"),
    F.lit("- user_id: "),
    F.col("user_id_text"),
    F.lit("\n- card_id: "),
    F.col("card_id_text"),
    F.lit("\n- timestamp: "),
    F.col("transaction_ts_text"),
    F.lit("\n- amount_usd: "),
    F.format_string("%.2f", F.col("amount_usd")),
    F.lit("\n- use_chip: "),
    F.col("use_chip_text"),
    F.lit("\n- merchant_city: "),
    F.col("merchant_city_text"),
    F.lit("\n- merchant_state: "),
    F.col("merchant_state_text"),
    F.lit("\n- merchant_category_code: "),
    F.col("mcc_text"),
    F.lit("\n- errors: "),
    F.col("errors_text"),
)

risk_col = (
    F.when(F.col("is_fraud") == 1, F.lit("likely_fraud"))
    .when(
        F.col("has_error_signal") | (F.col("amount_usd") >= F.lit(suspicious_amount_threshold)),
        F.lit("suspicious"),
    )
    .otherwise(F.lit("legitimate"))
)
action_col = (
    F.when(F.col("is_fraud") == 1, F.lit("decline_and_escalate"))
    .when(
        F.col("has_error_signal") | (F.col("amount_usd") >= F.lit(suspicious_amount_threshold)),
        F.lit("step_up_authentication"),
    )
    .otherwise(F.lit("approve"))
)
reason_col = (
    F.when(
        F.col("is_fraud") == 1,
        F.lit("The historical label marks this transaction as fraud."),
    )
    .when(
        F.col("has_error_signal") | (F.col("amount_usd") >= F.lit(suspicious_amount_threshold)),
        F.lit("The transaction is not labeled fraud, but amount or error signals warrant review."),
    )
    .otherwise(F.lit("The historical label is non-fraud and no strong review signal is present."))
)

assistant_response_col = F.to_json(
    F.struct(
        risk_col.alias("risk"),
        action_col.alias("action"),
        reason_col.alias("reason"),
    )
)

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

sft_df = (
    df.withColumn("training_id", training_id_col)
    .withColumn("prompt", prompt_col)
    .withColumn("assistant_response", assistant_response_col)
    .withColumn("shard_id", F.expr(shard_expr).cast("int"))
    .withColumn(
        "messages_json",
        F.to_json(
            F.array(
                F.struct(F.lit("user").alias("role"), F.col("prompt").alias("content")),
                F.struct(F.lit("assistant").alias("role"), F.col("assistant_response").alias("content")),
            )
        ),
    )
    .select(
        "training_id",
        "shard_id",
        "prompt",
        "assistant_response",
        "messages_json",
        "fraud_label",
        "is_fraud",
        "amount_usd",
        "user_id_text",
        "card_id_text",
        "transaction_ts_text",
        "merchant_city_text",
        "merchant_state_text",
        "mcc_text",
        "errors_text",
        "has_error_signal",
    )
)

(
    sft_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(full_sft_table_name)
)

spark.sql(
    f"""
    COMMENT ON TABLE {full_sft_table_name}
    IS 'Prompt/response supervised fine-tuning records for the AIR fraud demo'
    """
)

print(f"Overwrote SFT Delta table {full_sft_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export SFT records to a Unity Catalog volume as Parquet
# MAGIC
# MAGIC AI Runtime's data-loading guidance recommends exporting large Delta tables to a UC volume and reading the files directly during training, which avoids Spark overhead on the GPU workers:
# MAGIC https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
# MAGIC
# MAGIC The export is partitioned by `shard_id`, so the existing rank-sharding contract carries over: each GPU worker claims the `shard_id=N` directories where `N % world_size == rank` and reads only its own files.

# COMMAND ----------

(
    spark.table(full_sft_table_name)
    .repartition(sft_shards, "shard_id")
    .write.mode("overwrite")
    .partitionBy("shard_id")
    .parquet(sft_files_path)
)

print(f"Exported SFT parquet shards to {sft_files_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the SFT table and the volume export
# MAGIC
# MAGIC Training fails fast when the parquet export is missing or incomplete, so this cell confirms the export holds every shard and every SFT row before any GPU time is spent.
# MAGIC The export is checked by reading it back through Spark (rather than listing the FUSE mount), so verification also works on local Databricks Connect runs.

# COMMAND ----------

sft_loaded_df = spark.table(full_sft_table_name)

sft_summary_expressions = [
    F.count("*").alias("row_count"),
    F.countDistinct("shard_id").alias("shard_count"),
    F.min("shard_id").alias("min_shard_id"),
    F.max("shard_id").alias("max_shard_id"),
]
if "is_fraud" in sft_loaded_df.columns:
    sft_summary_expressions.extend(
        [
            F.sum("is_fraud").alias("fraud_row_count"),
            F.avg("is_fraud").alias("fraud_rate"),
        ]
    )

display(sft_loaded_df.agg(*sft_summary_expressions))
display(sft_loaded_df.select("training_id", "shard_id", "prompt", "assistant_response").limit(10))

export_df = spark.read.parquet(sft_files_path)
export_row_count = export_df.count()
export_shard_count = export_df.select("shard_id").distinct().count()
table_row_count = sft_loaded_df.count()

print(f"SFT parquet export: {export_row_count} rows in {export_shard_count} shards at {sft_files_path}")

if export_shard_count != sft_shards:
    raise ValueError(
        f"Expected {sft_shards} shard_id partitions in the parquet export, found {export_shard_count}."
    )
if export_row_count != table_row_count:
    raise ValueError(
        f"Parquet export has {export_row_count} rows but {full_sft_table_name} has "
        f"{table_row_count} — the export is incomplete; rerun this notebook."
    )

print("Training data staged. Next: train/01_runner.py (or `air run --file train.yaml` from train/).")
