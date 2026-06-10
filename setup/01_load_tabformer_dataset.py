# Databricks notebook source
# DBTITLE 1,Load IBM TabFormer Credit Card Dataset
# MAGIC %md
# MAGIC # Load and prepare IBM TabFormer credit-card transactions
# MAGIC
# MAGIC This setup notebook downloads the IBM TabFormer credit-card dataset, stages it in a Unity Catalog volume, and overwrites the configured Delta tables.
# MAGIC
# MAGIC The table is prepared for the AIR fine-tuning notebook by standardizing column names, casting core fields, creating `transaction_ts`, and adding reusable prompt-ready fields such as `amount_usd`, `*_text`, `errors_text`, `has_error_signal`, and `fraud_label`.
# MAGIC The setup also creates a supervised fine-tuning table with prompt/response columns and stable shard IDs. Moving these steps into ingestion keeps training focused on data selection, GPU execution, and model fine-tuning.

# COMMAND ----------

from pathlib import Path
import csv
import os
import re
import shutil
import tarfile
from urllib.request import Request, urlopen

import yaml
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------

# MAGIC %run ../train/training_utils

# COMMAND ----------

# The %run above provides the shared helpers in workspace notebook runs; when
# this file executes as a local script the MAGIC line is a plain comment, so
# import the same helpers from train/training_utils instead. (The module is not
# named `utils` because GPU base environments ship packages that register a
# top-level `utils` module, shadowing any local one.)
if "quote_identifier" not in globals():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from training_utils import (
        config_bool,
        config_float,
        config_int,
        config_str,
        get_spark_session,
        quote_identifier,
    )

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = notebook_context.notebookPath().get()
    script_dir = Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]

config_path = script_dir / "setup.yaml"

with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

catalog = config_str(config, "catalog")
schema = config_str(config, "schema")
table = config_str(config, "table")
sft_table = config_str(config, "sft_table")
staging_volume = config_str(config, "staging_volume")
sft_volume = config_str(config, "sft_volume")
dataset_name = config_str(config, "dataset_name")
source_url = config_str(config, "source_url")
archive_filename = config_str(config, "archive_filename")
force_download = config_bool(config, "force_download")
suspicious_amount_threshold = config_float(config, "suspicious_amount_threshold")
sft_shards = config_int(config, "sft_shards")

spark = get_spark_session()

catalog_q = quote_identifier(catalog)
schema_q = quote_identifier(schema)
table_q = quote_identifier(table)
sft_table_q = quote_identifier(sft_table)
volume_q = quote_identifier(staging_volume)
sft_volume_q = quote_identifier(sft_volume)

full_schema_name = f"{catalog_q}.{schema_q}"
full_table_name = f"{full_schema_name}.{table_q}"
full_sft_table_name = f"{full_schema_name}.{sft_table_q}"
full_volume_name = f"{full_schema_name}.{volume_q}"
full_sft_volume_name = f"{full_schema_name}.{sft_volume_q}"
sft_files_path = f"/Volumes/{catalog}/{schema}/{sft_volume}/{sft_table}"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_schema_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {full_volume_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {full_sft_volume_name}")

volume_root = Path(f"/Volumes/{catalog}/{schema}/{staging_volume}")
dataset_root = volume_root / dataset_name
extract_root = dataset_root / "extracted"
archive_path = dataset_root / archive_filename

dataset_root.mkdir(parents=True, exist_ok=True)

print(f"Config path: {config_path}")
print(f"Target table: {full_table_name}")
print(f"Target SFT table: {full_sft_table_name}")
print(f"Target SFT parquet export: {sft_files_path}")
print(f"Staging path: {dataset_root}")

# COMMAND ----------


def download_file(url: str, destination: Path) -> None:
    tmp_destination = destination.with_suffix(destination.suffix + ".download")
    if tmp_destination.exists():
        tmp_destination.unlink()

    request = Request(
        url,
        headers={
            "User-Agent": "Databricks-TabFormer-Setup/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        with tmp_destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file, length=16 * 1024 * 1024)

    tmp_destination.replace(destination)


if force_download or not archive_path.exists():
    print(f"Downloading TabFormer transactions archive from {source_url}")
    download_file(source_url, archive_path)
else:
    print(f"Using existing archive: {archive_path}")

with archive_path.open("rb") as archive_file:
    gzip_magic = archive_file.read(2)

if gzip_magic != b"\x1f\x8b":
    raise ValueError(
        "The downloaded file is not a gzip archive. "
        "If the source URL returned an HTML page, set source_url to the GitHub raw "
        "transactions.tgz URL or upload transactions.tgz to the staging path."
    )

# COMMAND ----------


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination_resolved / member.name).resolve()
        if os.path.commonpath([str(destination_resolved), str(member_path)]) != str(
            destination_resolved
        ):
            raise ValueError(f"Unsafe path in archive: {member.name}")
    tar.extractall(destination)


csv_files = sorted(path for path in extract_root.rglob("*.csv") if path.is_file())

if force_download or not csv_files:
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path} to {extract_root}")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, extract_root)

    csv_files = sorted(path for path in extract_root.rglob("*.csv") if path.is_file())

if not csv_files:
    raise FileNotFoundError(f"No CSV files found after extracting {archive_path}")

transactions_csv = max(csv_files, key=lambda path: path.stat().st_size)
print(f"Reading transactions CSV: {transactions_csv}")

# COMMAND ----------

EXPECTED_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]

EXPECTED_SCHEMA = StructType(
    [
        StructField("User", IntegerType(), True),
        StructField("Card", IntegerType(), True),
        StructField("Year", IntegerType(), True),
        StructField("Month", IntegerType(), True),
        StructField("Day", IntegerType(), True),
        StructField("Time", StringType(), True),
        StructField("Amount", StringType(), True),
        StructField("Use Chip", StringType(), True),
        StructField("Merchant Name", LongType(), True),
        StructField("Merchant City", StringType(), True),
        StructField("Merchant State", StringType(), True),
        StructField("Zip", StringType(), True),
        StructField("MCC", IntegerType(), True),
        StructField("Errors?", StringType(), True),
        StructField("Is Fraud?", StringType(), True),
    ]
)

with transactions_csv.open("r", newline="") as csv_file:
    header = next(csv.reader(csv_file))

reader = spark.read.option("header", True)
if header == EXPECTED_COLUMNS:
    reader = reader.schema(EXPECTED_SCHEMA)
else:
    print("Unexpected header found; falling back to Spark schema inference.")
    reader = reader.option("inferSchema", True)

raw_df = reader.csv(transactions_csv.as_posix())

# COMMAND ----------

COLUMN_RENAMES = {
    "User": "user_id",
    "Card": "card_id",
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "Time": "time",
    "Amount": "amount",
    "Use Chip": "use_chip",
    "Merchant Name": "merchant_name",
    "Merchant City": "merchant_city",
    "Merchant State": "merchant_state",
    "Zip": "zip_code",
    "MCC": "mcc",
    "Errors?": "errors",
    "Is Fraud?": "is_fraud",
}


def clean_column_name(column_name: str) -> str:
    if column_name in COLUMN_RENAMES:
        return COLUMN_RENAMES[column_name]

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", column_name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "column"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def make_unique(column_names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_names = []

    for column_name in column_names:
        count = seen.get(column_name, 0)
        seen[column_name] = count + 1
        unique_names.append(column_name if count == 0 else f"{column_name}_{count + 1}")

    return unique_names


clean_names = make_unique([clean_column_name(column_name) for column_name in raw_df.columns])
df = raw_df.select(
    *[
        raw_df[original_name].alias(clean_name)
        for original_name, clean_name in zip(raw_df.columns, clean_names)
    ]
)

integer_columns = [
    "user_id",
    "card_id",
    "year",
    "month",
    "day",
    "mcc",
]
for column_name in integer_columns:
    if column_name in df.columns:
        df = df.withColumn(column_name, F.col(column_name).cast("int"))

if "merchant_name" in df.columns:
    df = df.withColumn("merchant_name", F.col("merchant_name").cast("long"))

if "amount" in df.columns:
    df = df.withColumn(
        "amount",
        F.regexp_replace(F.col("amount").cast("string"), r"[$,]", "").cast("double"),
    )

if "is_fraud" in df.columns:
    fraud_text = F.lower(F.trim(F.col("is_fraud").cast("string")))
    df = df.withColumn(
        "is_fraud",
        F.when(fraud_text.isin("yes", "true", "1"), F.lit(1))
        .when(fraud_text.isin("no", "false", "0"), F.lit(0))
        .otherwise(F.col("is_fraud").cast("int"))
        .cast("int"),
    )

if {"year", "month", "day", "time"}.issubset(set(df.columns)):
    date_text = F.format_string(
        "%04d-%02d-%02d",
        F.col("year").cast("int"),
        F.col("month").cast("int"),
        F.col("day").cast("int"),
    )
    timestamp_text = F.concat_ws(" ", date_text, F.col("time").cast("string"))
    df = df.withColumn(
        "transaction_ts",
        F.coalesce(
            F.to_timestamp(timestamp_text, "yyyy-MM-dd HH:mm"),
            F.to_timestamp(timestamp_text, "yyyy-MM-dd H:mm"),
            F.to_timestamp(timestamp_text, "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp(timestamp_text, "yyyy-MM-dd H:mm:ss"),
        ),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare training-ready transaction fields
# MAGIC
# MAGIC The training notebook consumes a stable set of cleaned fields from the Delta table instead of applying row-by-row Pandas cleanup.
# MAGIC These fields preserve the raw columns while providing defaults and consistent text formatting for prompt construction.

# COMMAND ----------


def text_value(column_name: str, default: str = "unknown"):
    value = F.trim(F.col(column_name).cast("string"))
    return (
        F.when(F.col(column_name).isNull(), F.lit(default))
        .when(F.length(value) == 0, F.lit(default))
        .when(F.lower(value).isin("nan", "none", "null"), F.lit(default))
        .otherwise(value)
    )


if "amount" in df.columns:
    df = df.withColumn("amount_usd", F.coalesce(F.col("amount").cast("double"), F.lit(0.0)))
else:
    df = df.withColumn("amount_usd", F.lit(0.0))

if "transaction_ts" in df.columns:
    df = df.withColumn(
        "transaction_ts_text",
        F.coalesce(
            F.date_format(F.col("transaction_ts"), "yyyy-MM-dd HH:mm:ss"),
            F.lit("unknown"),
        ),
    )
else:
    df = df.withColumn("transaction_ts_text", F.lit("unknown"))

text_columns = {
    "user_id": "unknown",
    "card_id": "unknown",
    "use_chip": "unknown",
    "merchant_city": "unknown",
    "merchant_state": "unknown",
    "mcc": "unknown",
}
for source_column, default_value in text_columns.items():
    if source_column in df.columns:
        df = df.withColumn(f"{source_column}_text", text_value(source_column, default_value))
    else:
        df = df.withColumn(f"{source_column}_text", F.lit(default_value))

if "errors" in df.columns:
    df = df.withColumn("errors_text", text_value("errors", "none"))
else:
    df = df.withColumn("errors_text", F.lit("none"))

df = df.withColumn(
    "has_error_signal",
    ~F.lower(F.col("errors_text")).isin("none", "unknown", "nan", ""),
)

if "is_fraud" in df.columns:
    df = df.withColumn("is_fraud", F.coalesce(F.col("is_fraud").cast("int"), F.lit(0)))
else:
    df = df.withColumn("is_fraud", F.lit(0))

df = df.withColumn(
    "fraud_label",
    F.when(F.col("is_fraud") == 1, F.lit("fraud")).otherwise(F.lit("non_fraud")),
)

preferred_order = [
    "user_id",
    "user_id_text",
    "card_id",
    "card_id_text",
    "year",
    "month",
    "day",
    "time",
    "transaction_ts",
    "transaction_ts_text",
    "amount",
    "amount_usd",
    "use_chip",
    "use_chip_text",
    "merchant_name",
    "merchant_city",
    "merchant_city_text",
    "merchant_state",
    "merchant_state_text",
    "zip_code",
    "mcc",
    "mcc_text",
    "errors",
    "errors_text",
    "has_error_signal",
    "is_fraud",
    "fraud_label",
]
ordered_columns = [column_name for column_name in preferred_order if column_name in df.columns]
ordered_columns.extend(column_name for column_name in df.columns if column_name not in ordered_columns)
df = df.select(*ordered_columns)

# COMMAND ----------

writer = (
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
)

if "year" in df.columns:
    writer = writer.partitionBy("year")

writer.saveAsTable(full_table_name)

spark.sql(
    f"""
    COMMENT ON TABLE {full_table_name}
    IS 'IBM TabFormer synthetic credit card transactions loaded for the AIR demo'
    """
)

print(f"Overwrote Delta table {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create supervised fine-tuning records
# MAGIC
# MAGIC The training notebook reads this table directly instead of building prompts row by row in Python.
# MAGIC Prompt text, assistant JSON, and `shard_id` are generated with Spark expressions so the work runs in parallel during ingestion.
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

training_id_col = F.sha2(
    F.concat_ws(
        "||",
        F.col("user_id_text"),
        F.col("card_id_text"),
        F.col("transaction_ts_text"),
        F.format_string("%.2f", F.col("amount_usd")),
        F.col("merchant_city_text"),
        F.col("merchant_state_text"),
        F.col("mcc_text"),
    ),
    256,
)

shard_expr = f"""
pmod(
  xxhash64(
    user_id_text,
    card_id_text,
    transaction_ts_text,
    amount_usd,
    merchant_city_text,
    merchant_state_text,
    mcc_text
  ),
  {sft_shards}
)
"""

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
# MAGIC Parquet is used because the training code (Unsloth) consumes Hugging Face `datasets`, which loads Parquet natively as memory-mapped Arrow tables.
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

loaded_df = spark.sql(f"SELECT * FROM {full_table_name}")
sft_loaded_df = spark.sql(f"SELECT * FROM {full_sft_table_name}")

summary_expressions = [F.count("*").alias("row_count")]
if "is_fraud" in loaded_df.columns:
    summary_expressions.extend(
        [
            F.sum("is_fraud").alias("fraud_row_count"),
            F.avg("is_fraud").alias("fraud_rate"),
        ]
    )
if "transaction_ts" in loaded_df.columns:
    summary_expressions.extend(
        [
            F.min("transaction_ts").alias("min_transaction_ts"),
            F.max("transaction_ts").alias("max_transaction_ts"),
        ]
    )

display(loaded_df.agg(*summary_expressions))
display(loaded_df.limit(10))

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

sft_shard_dirs = sorted(Path(sft_files_path).glob("shard_id=*"))
print(f"SFT parquet export: {len(sft_shard_dirs)} shard directories under {sft_files_path}")
if len(sft_shard_dirs) != sft_shards:
    raise ValueError(
        f"Expected {sft_shards} shard directories in the parquet export, found {len(sft_shard_dirs)}."
    )
