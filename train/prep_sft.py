# Databricks notebook source
# DBTITLE 1,Prepare SFT Parquet records in a Unity Catalog volume
# MAGIC %md
# MAGIC # Prepare SFT records for training
# MAGIC
# MAGIC This training-stage notebook reads the raw `split`/`shard_id` Parquet export written by `setup/02_stage_training_data.py`, renders model-agnostic `prompt` and `assistant_response` columns, and writes prepared SFT Parquet files to a separate Unity Catalog volume.
# MAGIC
# MAGIC The output preserves the input partition layout (`split=train|eval/shard_id=N`) expected by every standalone trainer when `convert_sft: false`. The destination is overwritten on every run.

# COMMAND ----------

from pathlib import Path

from pyspark.sql import functions as F

# COMMAND ----------

# Edit these variables for the raw setup export and prepared SFT destination.
INPUT_RAW_PARQUET_PATH = (
    "/Volumes/catalog_sandbox_5nwjwb/air_hackathon_v2/"
    "training_data/fraud_raw_dataset"
)
OUTPUT_SFT_PARQUET_PATH = (
    "/Volumes/catalog_sandbox_5nwjwb/air_hackathon_v2/"
    "sft_staging/fraud_sft_dataset"
)
SUSPICIOUS_AMOUNT_THRESHOLD = 500.0

# COMMAND ----------


def get_spark_session():
    try:
        from pyspark.sql import SparkSession

        active_session = SparkSession.getActiveSession()
        if active_session is not None:
            return active_session
    except Exception:
        pass

    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    if hasattr(builder, "serverless"):
        return builder.serverless().getOrCreate()
    return builder.getOrCreate()


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def parse_volume_path(volume_path: str) -> tuple[str, str, str]:
    parts = Path(volume_path).parts
    if len(parts) < 6 or parts[:2] != ("/", "Volumes"):
        raise ValueError(
            "Path must include a directory below a UC volume: "
            f"/Volumes/<catalog>/<schema>/<volume>/<path>, got {volume_path!r}"
        )
    return parts[2], parts[3], parts[4]


if INPUT_RAW_PARQUET_PATH.rstrip("/") == OUTPUT_SFT_PARQUET_PATH.rstrip("/"):
    raise ValueError("Input and output paths must be different")

output_catalog, output_schema, output_volume = parse_volume_path(
    OUTPUT_SFT_PARQUET_PATH
)
parse_volume_path(INPUT_RAW_PARQUET_PATH)

spark = get_spark_session()
schema_q = ".".join(
    quote_identifier(part) for part in (output_catalog, output_schema)
)
volume_q = ".".join(
    quote_identifier(part)
    for part in (output_catalog, output_schema, output_volume)
)

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_q}")
except Exception as exc:
    raise RuntimeError(
        f"Could not create or access destination volume {volume_q}. "
        "Verify Unity Catalog privileges."
    ) from exc

print(f"Raw input: {INPUT_RAW_PARQUET_PATH}")
print(f"Prepared SFT output: {OUTPUT_SFT_PARQUET_PATH}")
print(f"Suspicious amount threshold: {SUSPICIOUS_AMOUNT_THRESHOLD}")

# COMMAND ----------

try:
    raw_df = spark.read.parquet(INPUT_RAW_PARQUET_PATH)
except Exception as exc:
    raise FileNotFoundError(
        f"Could not read raw Parquet at {INPUT_RAW_PARQUET_PATH}. Run "
        "setup/01_load_dataset.py and setup/02_stage_training_data.py first."
    ) from exc

required_raw_columns = {
    "training_id",
    "split",
    "shard_id",
    "user_id_text",
    "card_id_text",
    "transaction_ts_text",
    "amount_usd",
    "use_chip_text",
    "merchant_city_text",
    "merchant_state_text",
    "mcc_text",
    "errors_text",
    "is_fraud",
    "has_error_signal",
}
missing_columns = sorted(required_raw_columns - set(raw_df.columns))
if missing_columns:
    raise ValueError(
        f"Raw input is missing required columns: {missing_columns}. "
        "Point INPUT_RAW_PARQUET_PATH at setup/02's raw export."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Render model-agnostic SFT fields
# MAGIC
# MAGIC Spark expressions keep preparation distributed and avoid a Python UDF. Model-specific chat templates remain inside each trainer.

# COMMAND ----------


def text_column(name: str):
    return F.coalesce(F.col(name).cast("string"), F.lit(""))


amount_text = F.format_string(
    "%.2f", F.coalesce(F.col("amount_usd").cast("double"), F.lit(0.0))
)
prompt = F.concat(
    F.lit(
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        "- user_id: "
    ),
    text_column("user_id_text"),
    F.lit("\n- card_id: "),
    text_column("card_id_text"),
    F.lit("\n- timestamp: "),
    text_column("transaction_ts_text"),
    F.lit("\n- amount_usd: "),
    amount_text,
    F.lit("\n- use_chip: "),
    text_column("use_chip_text"),
    F.lit("\n- merchant_city: "),
    text_column("merchant_city_text"),
    F.lit("\n- merchant_state: "),
    text_column("merchant_state_text"),
    F.lit("\n- merchant_category_code: "),
    text_column("mcc_text"),
    F.lit("\n- errors: "),
    text_column("errors_text"),
)

is_fraud = F.coalesce(F.col("is_fraud").cast("int"), F.lit(0)) == 1
needs_review = F.coalesce(
    F.col("has_error_signal").cast("boolean"), F.lit(False)
) | (
    F.coalesce(F.col("amount_usd").cast("double"), F.lit(0.0))
    >= F.lit(float(SUSPICIOUS_AMOUNT_THRESHOLD))
)

risk = (
    F.when(is_fraud, F.lit("likely_fraud"))
    .when(needs_review, F.lit("suspicious"))
    .otherwise(F.lit("legitimate"))
)
action = (
    F.when(is_fraud, F.lit("decline_and_escalate"))
    .when(needs_review, F.lit("step_up_authentication"))
    .otherwise(F.lit("approve"))
)
reason = (
    F.when(
        is_fraud,
        F.lit("The historical label marks this transaction as fraud."),
    )
    .when(
        needs_review,
        F.lit(
            "The transaction is not labeled fraud, but amount or error signals "
            "warrant review."
        ),
    )
    .otherwise(
        F.lit(
            "The historical label is non-fraud and no strong review signal is present."
        )
    )
)

sft_df = raw_df.select(
    "training_id",
    "split",
    "shard_id",
    prompt.alias("prompt"),
    F.to_json(
        F.struct(
            risk.alias("risk"),
            action.alias("action"),
            reason.alias("reason"),
        )
    ).alias("assistant_response"),
    F.col("is_fraud").cast("int").alias("is_fraud"),
)

(
    sft_df.repartition("split", "shard_id")
    .write.mode("overwrite")
    .partitionBy("split", "shard_id")
    .parquet(OUTPUT_SFT_PARQUET_PATH)
)

print(f"Prepared SFT Parquet written to {OUTPUT_SFT_PARQUET_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the prepared export

# COMMAND ----------

prepared_df = spark.read.parquet(OUTPUT_SFT_PARQUET_PATH)

raw_summary = {
    row["split"]: row
    for row in raw_df.groupBy("split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("shard_id").alias("shard_count"),
    )
    .collect()
}
prepared_summary = {
    row["split"]: row
    for row in prepared_df.groupBy("split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("shard_id").alias("shard_count"),
    )
    .collect()
}

for split_name in ("train", "eval"):
    raw_row = raw_summary.get(split_name)
    prepared_row = prepared_summary.get(split_name)
    if raw_row is None or prepared_row is None:
        raise ValueError(f"split={split_name} is missing from the raw or SFT export")
    if (
        prepared_row["row_count"] != raw_row["row_count"]
        or prepared_row["shard_count"] != raw_row["shard_count"]
    ):
        raise ValueError(
            f"split={split_name} mismatch: raw={raw_row.asDict()}, "
            f"prepared={prepared_row.asDict()}"
        )
    print(
        f"split={split_name}: {prepared_row['row_count']} rows in "
        f"{prepared_row['shard_count']} shards"
    )

null_sft_rows = prepared_df.where(
    F.col("prompt").isNull() | F.col("assistant_response").isNull()
).limit(1)
if null_sft_rows.count():
    raise ValueError("Prepared export contains null prompt or assistant_response values")

try:
    display
except NameError:
    def display(df):
        print(df.toPandas().to_string(index=False))


display(
    prepared_df.select(
        "training_id", "split", "shard_id", "prompt", "assistant_response"
    ).limit(5)
)

print(
    "SFT preparation complete. Point a training project's train_data_path and "
    "eval_data_path at this output with convert_sft: false."
)
