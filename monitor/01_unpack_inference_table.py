# Databricks notebook source
# DBTITLE 1,Unpack the serving endpoint's inference table
# MAGIC %md
# MAGIC # Unpack the serving endpoint's inference table
# MAGIC
# MAGIC This is the first module of the monitoring stage: it converts the raw payload table that the serving endpoint writes (AI Gateway inference logging, enabled by `train/02_register_and_deploy.py`) into an analysis-ready Delta table with one row per request — the table the data-quality monitor is created over in the next module.
# MAGIC
# MAGIC The payload table stores each request/response pair as raw JSON strings alongside delivery metadata. This notebook:
# MAGIC
# MAGIC 1. Reads the payload table **incrementally** with Structured Streaming (`trigger(availableNow=True)` plus a checkpoint in a UC volume), so each run processes only new rows and the notebook can be scheduled as a recurring job.
# MAGIC 2. Parses the OpenAI-compatible chat payloads: the prompt (last user message), the completion text, finish reason, and token usage.
# MAGIC 3. Optionally extracts fields from structured JSON completions into top-level columns (`response_json_fields` in `monitor.yaml`) so the monitor can profile them as categoricals — the fraud example extracts `risk` and `action`.
# MAGIC 4. Appends to the unpacked Delta table, created on first run with **change data feed** enabled (data profiling refreshes incrementally from CDF) and partitioned by `request_date`.
# MAGIC
# MAGIC Run on Databricks serverless (CPU) compute — on demand, or scheduled (Databricks recommends at least weekly; hourly/daily keeps monitor metrics fresh).
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - AI Gateway inference tables (schema and delivery guarantees): https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/inference-tables-serving-endpoints
# MAGIC - Data quality monitoring (formerly Lakehouse Monitoring): https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/data-quality-monitoring/

# COMMAND ----------

# training_utils is a plain Python module in train/ (not a notebook),
# shared across the pipeline; insert that directory into sys.path.
import sys
from pathlib import Path

TRAIN_MODULE_DIR = str((Path.cwd().parent / "train").resolve())
if TRAIN_MODULE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_MODULE_DIR)

from training_utils import (
    config_bool,
    config_str,
    ensure_uc_object,
    full_name,
    get_spark_session,
    load_global_config,
    load_yaml_config,
)

# Stage keys come from monitor.yaml; the pipeline-wide identity comes from
# the repo-root global.yaml. The payload table name is derived from the
# global inference_table_prefix — the endpoint always writes
# <prefix>_payload, so the two can never drift.
config_path, monitor_config = load_yaml_config("monitor.yaml", base_dir=Path.cwd())
_, global_config = load_global_config()

UC_CATALOG = config_str(global_config, "catalog")
UC_SCHEMA = config_str(global_config, "schema")
INFERENCE_TABLE_NAME = f"{config_str(global_config, 'inference_table_prefix')}_payload"
UNPACKED_TABLE_NAME = config_str(monitor_config, "unpacked_table")
CHECKPOINT_VOLUME = config_str(monitor_config, "checkpoint_volume")
INCLUDE_FAILED_REQUESTS = config_bool(monitor_config, "include_failed_requests")

RESPONSE_JSON_FIELDS = monitor_config.get("response_json_fields") or []
if not isinstance(RESPONSE_JSON_FIELDS, list):
    raise ValueError("response_json_fields must be a list (or empty) in monitor.yaml")
RESPONSE_JSON_FIELDS = [str(field) for field in RESPONSE_JSON_FIELDS]

payload_table_q = full_name(UC_CATALOG, UC_SCHEMA, INFERENCE_TABLE_NAME)
unpacked_table_q = full_name(UC_CATALOG, UC_SCHEMA, UNPACKED_TABLE_NAME)
schema_q = full_name(UC_CATALOG, UC_SCHEMA)
checkpoint_volume_q = full_name(UC_CATALOG, UC_SCHEMA, CHECKPOINT_VOLUME)
checkpoint_path = (
    f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{CHECKPOINT_VOLUME}/checkpoints/{UNPACKED_TABLE_NAME}"
)

print(f"Monitor config: {config_path}")
print(f"Payload (inference) table: {payload_table_q}")
print(f"Unpacked requests table: {unpacked_table_q}")
print(f"Streaming checkpoint: {checkpoint_path}")
print(f"Response JSON fields: {RESPONSE_JSON_FIELDS or '(none — free-text responses)'}")

# COMMAND ----------

spark = get_spark_session()

ensure_uc_object(spark, f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
ensure_uc_object(spark, f"CREATE VOLUME IF NOT EXISTS {checkpoint_volume_q}")

try:
    payload_columns = spark.table(payload_table_q).columns
except Exception as exc:
    raise RuntimeError(
        f"Payload table {payload_table_q} is not readable. Deploy the endpoint "
        "first (train/02_register_and_deploy.py enables inference logging "
        "automatically) and send it some traffic — logs are delivered within "
        "~1 hour of a request."
    ) from exc

print(f"Payload table columns: {payload_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unpack the JSON payloads
# MAGIC
# MAGIC The endpoint speaks the OpenAI chat contract, so every `request` is `{"messages": [...], ...}` and every `response` is `{"choices": [...], "usage": {...}, ...}` — both stored as raw JSON strings.
# MAGIC `from_json` parses them against explicit schemas; requests or responses that do not match simply produce nulls rather than failing the stream.
# MAGIC
# MAGIC The transformation also normalizes the payload schema: AI Gateway inference tables carry `request_time`/`execution_duration_ms`, while the retired legacy tables carried `timestamp_ms`/`execution_time_ms` — both shapes are accepted so pre-migration tables still unpack.

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T

CHAT_REQUEST_SCHEMA = T.StructType(
    [
        T.StructField(
            "messages",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("role", T.StringType()),
                        T.StructField("content", T.StringType()),
                    ]
                )
            ),
        ),
        T.StructField("max_tokens", T.IntegerType()),
        T.StructField("temperature", T.DoubleType()),
    ]
)

CHAT_RESPONSE_SCHEMA = T.StructType(
    [
        T.StructField("id", T.StringType()),
        T.StructField("model", T.StringType()),
        T.StructField(
            "choices",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("index", T.IntegerType()),
                        T.StructField("finish_reason", T.StringType()),
                        T.StructField(
                            "message",
                            T.StructType(
                                [
                                    T.StructField("role", T.StringType()),
                                    T.StructField("content", T.StringType()),
                                ]
                            ),
                        ),
                    ]
                )
            ),
        ),
        T.StructField(
            "usage",
            T.StructType(
                [
                    T.StructField("prompt_tokens", T.IntegerType()),
                    T.StructField("completion_tokens", T.IntegerType()),
                    T.StructField("total_tokens", T.IntegerType()),
                ]
            ),
        ),
    ]
)


def unpack_payloads(payloads: DataFrame) -> DataFrame:
    """Convert raw payload rows into one analysis-ready row per request."""
    columns = payloads.columns

    # Normalize the AI Gateway schema vs the retired legacy schema.
    if "request_time" in columns:
        normalized = payloads.withColumn("execution_ms", F.col("execution_duration_ms"))
    else:
        normalized = (
            payloads.withColumn(
                "request_time", F.to_timestamp(F.col("timestamp_ms") / F.lit(1000.0))
            )
            .withColumn("request_date", F.col("date"))
            .withColumn("execution_ms", F.col("execution_time_ms"))
        )
    if "served_entity_id" not in columns:
        normalized = normalized.withColumn("served_entity_id", F.lit(None).cast("string"))

    if not INCLUDE_FAILED_REQUESTS:
        normalized = normalized.filter(F.col("status_code") == 200)

    parsed = normalized.withColumn(
        "request_struct", F.from_json(F.col("request"), CHAT_REQUEST_SCHEMA)
    ).withColumn("response_struct", F.from_json(F.col("response"), CHAT_RESPONSE_SCHEMA))

    unpacked = parsed.select(
        "request_date",
        "request_time",
        "databricks_request_id",
        "client_request_id",
        "status_code",
        F.col("execution_ms").alias("execution_duration_ms"),
        "sampling_fraction",
        # The served entity identifies the model version behind the endpoint —
        # the monitor's model_id dimension for cross-version comparisons.
        "served_entity_id",
        F.expr(
            "element_at(filter(request_struct.messages, m -> m.role = 'user'), -1).content"
        ).alias("prompt"),
        F.size(F.col("request_struct.messages")).alias("num_input_messages"),
        F.col("request_struct.max_tokens").alias("max_tokens"),
        F.col("request_struct.temperature").alias("temperature"),
        F.expr("response_struct.choices[0].message.content").alias("response_text"),
        F.expr("response_struct.choices[0].finish_reason").alias("finish_reason"),
        F.col("response_struct.model").alias("response_model"),
        F.col("response_struct.usage.prompt_tokens").alias("prompt_tokens"),
        F.col("response_struct.usage.completion_tokens").alias("completion_tokens"),
        F.col("response_struct.usage.total_tokens").alias("total_tokens"),
        F.length(F.col("response_struct.choices")[0]["message"]["content"]).alias(
            "response_chars"
        ),
    )

    # Structured-output models: surface configured fields of the assistant's
    # JSON completion as top-level columns (null when the completion is not
    # valid JSON or the field is absent), so the monitor can profile them.
    if RESPONSE_JSON_FIELDS:
        response_fields_schema = T.StructType(
            [T.StructField(field, T.StringType()) for field in RESPONSE_JSON_FIELDS]
        )
        unpacked = unpacked.withColumn(
            "_response_fields", F.from_json(F.col("response_text"), response_fields_schema)
        )
        for field in RESPONSE_JSON_FIELDS:
            unpacked = unpacked.withColumn(
                f"response_{field}", F.col(f"_response_fields.{field}")
            )
        unpacked = unpacked.drop("_response_fields")

    return unpacked

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the unpacked table
# MAGIC
# MAGIC The table is created empty on the first run so its properties are right from the start: **change data feed** on (data profiling reads CDF for efficient incremental refreshes) and partitioning by `request_date` (the monitor windows and prunes by time).

# COMMAND ----------

unpacked_schema = unpack_payloads(spark.table(payload_table_q)).schema

table_exists = spark.sql(
    f"SHOW TABLES IN {schema_q} LIKE '{UNPACKED_TABLE_NAME}'"
).count() > 0

if not table_exists:
    (
        spark.createDataFrame([], unpacked_schema)
        .write.format("delta")
        .partitionBy("request_date")
        .saveAsTable(unpacked_table_q)
    )
    spark.sql(
        f"ALTER TABLE {unpacked_table_q} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    spark.sql(
        f"""
        COMMENT ON TABLE {unpacked_table_q}
        IS 'Unpacked model serving requests/responses for data quality monitoring'
        """
    )
    print(f"Created {unpacked_table_q} (CDF enabled, partitioned by request_date)")
else:
    cdf_enabled = (
        spark.sql(f"SHOW TBLPROPERTIES {unpacked_table_q}")
        .filter("key = 'delta.enableChangeDataFeed'")
        .filter("value = 'true'")
        .count()
        > 0
    )
    if not cdf_enabled:
        spark.sql(
            f"ALTER TABLE {unpacked_table_q} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        print(f"Enabled change data feed on existing {unpacked_table_q}")
    else:
        print(f"{unpacked_table_q} exists with change data feed enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incrementally process new payload rows
# MAGIC
# MAGIC The payload table is read as a stream and drained with `trigger(availableNow=True)`: the run picks up exactly the rows added since the checkpoint's last position, appends their unpacked form, and exits — safe to schedule as a recurring job.
# MAGIC
# MAGIC To reprocess from scratch (for example after changing `response_json_fields`), delete the checkpoint directory and the unpacked table, then rerun.

# COMMAND ----------

unpack_stream = (
    unpack_payloads(spark.readStream.table(payload_table_q))
    .writeStream.format("delta")
    .outputMode("append")
    .trigger(availableNow=True)
    .option("checkpointLocation", checkpoint_path)
    .partitionBy("request_date")
    .toTable(unpacked_table_q)
)
unpack_stream.awaitTermination()

print(f"Unpacked new payload rows into {unpacked_table_q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the unpacked table

# COMMAND ----------

summary_columns = [
    F.count("*").alias("row_count"),
    F.min("request_time").alias("first_request"),
    F.max("request_time").alias("last_request"),
    F.countDistinct("served_entity_id").alias("served_entities"),
    F.avg("execution_duration_ms").alias("avg_execution_ms"),
    F.avg("completion_tokens").alias("avg_completion_tokens"),
]
for field in RESPONSE_JSON_FIELDS:
    summary_columns.append(
        F.avg(F.col(f"response_{field}").isNull().cast("int")).alias(
            f"response_{field}_null_rate"
        )
    )

unpacked_df = spark.table(unpacked_table_q)
display(unpacked_df.agg(*summary_columns))
display(unpacked_df.orderBy(F.col("request_time").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC The unpacked table now holds one analysis-ready row per serving request. The next module in this stage creates a data-quality monitor (data profiling) over it — tracking request volume, latency, token usage, and drift in the extracted response fields — and refreshes its metrics on a schedule.
