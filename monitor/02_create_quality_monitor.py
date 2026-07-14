# Databricks notebook source
# DBTITLE 1,Create the data quality monitor
# MAGIC %md
# MAGIC # Create the data quality monitor
# MAGIC
# MAGIC Second module of the monitoring stage: builds the training-set **baseline table** and creates (or updates) a **data quality monitor** (formerly Lakehouse Monitoring) over the unpacked requests table written by `01_unpack_inference_table.py`.
# MAGIC
# MAGIC The model is generative but wears a classifier's contract, so the monitor uses the **InferenceLog** profile with `problem_type = classification`: the label-like field of the compact JSON completion (`prediction_field` in `monitor.yaml`, extracted by module 01 as `response_<field>`) is the prediction column, `request_time` is the timestamp, and `served_entity_id` is the model-version dimension — every redeploy gets its own metric series. Every column of the unpacked table is profiled and drift-tested automatically, which is what makes the `txn_*` transaction features first-class drift signals: feature drift **and** prediction drift means the world changed; prediction drift alone points at the model.
# MAGIC
# MAGIC What one run does:
# MAGIC
# MAGIC 1. Rebuilds the baseline table from the SFT dataset, applying the **same** prompt-field and response-JSON extraction module 01 applies to serving traffic (both import `monitoring_utils`) — so baseline drift compares serving traffic against exactly what the model was trained on: covariate shift on `txn_*`, label shift on `response_*`.
# MAGIC 2. Creates or updates the monitor: prediction/timestamp/model-id columns, granularities, slices, the baseline, and custom contract-integrity metrics (JSON output breakage, prompt-template breakage, out-of-vocabulary labels, truncation, per-class prediction rates).
# MAGIC 3. Triggers a metrics refresh (incremental via the table's change data feed) and prints where the metrics tables and auto-generated dashboard live.
# MAGIC
# MAGIC Run on Databricks serverless (CPU) compute **after** module 01 has created and filled the unpacked table. Rerun when a retrain changes the SFT table (to refresh the baseline) or after changing the monitor keys in `monitor.yaml`. Recurring refreshes belong in the job that runs module 01 (unpack task → `quality_monitors.run_refresh` task), so metrics never lag the data — the monitor itself is created without a schedule.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Data quality monitoring (formerly Lakehouse Monitoring): https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/data-quality-monitoring/
# MAGIC - Custom metrics: https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/data-quality-monitoring/data-profiling/custom-metrics

# COMMAND ----------

# training_utils (train/) and monitoring_utils (this folder) are plain
# Python modules, not notebooks; insert their directories into sys.path.
import sys
from pathlib import Path

TRAIN_MODULE_DIR = str((Path.cwd().parent / "train").resolve())
MONITOR_MODULE_DIR = str(Path.cwd().resolve())
for module_dir in (TRAIN_MODULE_DIR, MONITOR_MODULE_DIR):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from monitoring_utils import (
    PROMPT_COLUMN_PREFIX,
    parse_quality_monitor_config,
    with_prompt_fields,
    with_response_fields,
)
from training_utils import (
    config_str,
    full_name,
    get_spark_session,
    load_global_config,
    load_yaml_config,
)

# Stage keys come from monitor.yaml (validated/normalized by
# parse_quality_monitor_config — the same parser scripts/validate_config.py
# exercises in CI); catalog/schema come from the repo-root global.yaml.
config_path, monitor_config = load_yaml_config("monitor.yaml", base_dir=Path.cwd())
_, global_config = load_global_config()

UC_CATALOG = config_str(global_config, "catalog")
UC_SCHEMA = config_str(global_config, "schema")
SFT_TABLE_NAME = config_str(monitor_config, "sft_table")
UNPACKED_TABLE_NAME = config_str(monitor_config, "unpacked_table")

monitor_settings = parse_quality_monitor_config(monitor_config)
RESPONSE_JSON_FIELDS = monitor_settings["response_json_fields"]
PROMPT_FIELDS = monitor_settings["prompt_fields"]
PREDICTION_FIELD = monitor_settings["prediction_field"]
PREDICTION_COL = monitor_settings["prediction_col"]
EXPECTED_PREDICTION_VALUES = monitor_settings["expected_prediction_values"]
GRANULARITIES = monitor_settings["granularities"]
SLICING_EXPRS = monitor_settings["slicing_exprs"]
BASELINE_TABLE_NAME = monitor_settings["baseline_table"]
BASELINE_SAMPLE_FRACTION = monitor_settings["baseline_sample_fraction"]
LABEL_FIELD = monitor_settings["label_field"]

# The quality-monitor API takes unquoted three-part names.
MONITORED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{UNPACKED_TABLE_NAME}"
BASELINE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{BASELINE_TABLE_NAME}"
OUTPUT_SCHEMA_NAME = monitor_settings["monitor_output_schema"] or f"{UC_CATALOG}.{UC_SCHEMA}"

sft_table_q = full_name(UC_CATALOG, UC_SCHEMA, SFT_TABLE_NAME)
unpacked_table_q = full_name(UC_CATALOG, UC_SCHEMA, UNPACKED_TABLE_NAME)
baseline_table_q = full_name(UC_CATALOG, UC_SCHEMA, BASELINE_TABLE_NAME)

print(f"Monitor config: {config_path}")
print(f"Monitored (unpacked) table: {MONITORED_TABLE}")
print(f"Baseline table: {BASELINE_TABLE} (from {SFT_TABLE_NAME}, fraction {BASELINE_SAMPLE_FRACTION})")
print(f"Prediction column: {PREDICTION_COL} in {EXPECTED_PREDICTION_VALUES}")
print(f"Granularities: {GRANULARITIES}; slices: {SLICING_EXPRS or '(none)'}")
print(f"Label column: {LABEL_FIELD or '(none — drift-only monitoring)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the training-set baseline table
# MAGIC
# MAGIC Window-over-window drift misses slow drift and cannot answer "is serving traffic still what we trained on?" — the baseline anchors both. Because the SFT prompts use the same transaction template as serving traffic, the baseline carries the training-time **feature** distribution (`txn_*`), not just the label distribution (`response_*`).
# MAGIC
# MAGIC The baseline also carries `prompt`, `response_text` (the training target), a synthetic `finish_reason = 'stop'`, and the monitor's `model_id_col` (`served_entity_id = 'training_baseline'`) so every column the custom metrics reference exists on both sides. Columns with no training-time analogue (latency, token usage, ...) are simply absent — the monitor skips them for baseline drift. The table is overwritten on every run, so rerunning after a retrain refreshes the baseline to match.

# COMMAND ----------

from pyspark.sql import functions as F

spark = get_spark_session()

try:
    unpacked_columns = set(spark.table(unpacked_table_q).columns)
except Exception as exc:
    raise RuntimeError(
        f"Unpacked table {unpacked_table_q} is not readable — run "
        "01_unpack_inference_table.py first (it creates and fills the table "
        "this monitor profiles)."
    ) from exc

if LABEL_FIELD and LABEL_FIELD not in unpacked_columns:
    raise RuntimeError(
        f"label_field {LABEL_FIELD!r} is not a column of {unpacked_table_q}. "
        "Join the ground-truth labels into the unpacked table first, or set "
        "label_field to '' in monitor.yaml."
    )

TRAINING_BASELINE_MODEL_ID = "training_baseline"

baseline_df = spark.table(sft_table_q).select("prompt", "assistant_response")
if BASELINE_SAMPLE_FRACTION < 1.0:
    baseline_df = baseline_df.sample(fraction=BASELINE_SAMPLE_FRACTION, seed=42)

baseline_df = (
    with_prompt_fields(
        with_response_fields(baseline_df, RESPONSE_JSON_FIELDS, "assistant_response"),
        PROMPT_FIELDS,
    )
    .withColumnRenamed("assistant_response", "response_text")
    .withColumn("finish_reason", F.lit("stop"))
    .withColumn("served_entity_id", F.lit(TRAINING_BASELINE_MODEL_ID))
)

# Baseline drift is computed only for columns present in both tables; a
# baseline column missing from the unpacked table means module 01 predates
# the current field configuration and its output must be rebuilt.
missing_columns = sorted(set(baseline_df.columns) - unpacked_columns)
if missing_columns:
    raise RuntimeError(
        f"{unpacked_table_q} lacks columns the baseline derives: "
        f"{missing_columns}. The unpacked table predates the current "
        "response_json_fields/prompt_fields — delete its streaming checkpoint "
        "and the table, then rerun 01_unpack_inference_table.py."
    )

(
    baseline_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(baseline_table_q)
)
spark.sql(
    f"""
    COMMENT ON TABLE {baseline_table_q}
    IS 'Training-set baseline (parsed SFT records) for the data quality monitor'
    """
)
baseline_count = spark.table(baseline_table_q).count()
print(f"Baseline table {baseline_table_q}: {baseline_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity-check the baseline extraction
# MAGIC
# MAGIC The SFT targets are valid JSON and the SFT prompts follow the transaction template, so the extracted columns should be near-zero-null here. A fully null prediction column or fully null `txn_*` columns means the extraction does not match this dataset — fail now rather than shipping a baseline that would make every drift comparison meaningless.

# COMMAND ----------

extracted_columns = [PREDICTION_COL] + [
    f"{PROMPT_COLUMN_PREFIX}{name}" for name, _ in PROMPT_FIELDS
]
null_rates = (
    spark.table(baseline_table_q)
    .agg(
        *[
            F.avg(F.col(column).isNull().cast("int")).alias(column)
            for column in extracted_columns
        ]
    )
    .collect()[0]
    .asDict()
)
for column, null_rate in null_rates.items():
    print(f"{column}: {null_rate:.1%} null")

if null_rates[PREDICTION_COL] >= 1.0:
    raise RuntimeError(
        f"{PREDICTION_COL} is 100% null in the baseline — the SFT table's "
        "assistant_response does not parse as JSON with a "
        f"{PREDICTION_FIELD!r} key. Check response_json_fields against the "
        "prompt contract in setup/02_stage_training_data.py."
    )
if PROMPT_FIELDS and all(
    null_rates[f"{PROMPT_COLUMN_PREFIX}{name}"] >= 1.0 for name, _ in PROMPT_FIELDS
):
    raise RuntimeError(
        "Every txn_* column is 100% null in the baseline — the SFT prompts do "
        "not contain the '- <field>: <value>' lines prompt_fields expects. "
        "Check prompt_fields against the prompt template in "
        "setup/02_stage_training_data.py."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the custom contract-integrity metrics
# MAGIC
# MAGIC Built-in profiling covers distributions (per-column stats, `frequent_items`, null rates) and drift tests (chi-square / TV / L∞ / JS for categoricals, KS / Wasserstein for numerics). The SLM-specific breakage signals are added as **aggregate custom metrics** — table-scoped, computed per window / model version / slice, and therefore alertable like any other metric:
# MAGIC
# MAGIC | Metric | Catches |
# MAGIC |---|---|
# MAGIC | `json_contract_failure_rate` | completions that are not valid JSON or lost the prediction key — the model breaking its output format |
# MAGIC | `prompt_parse_failure_rate` | prompts where every `txn_*` field failed to extract — the prompt template changed and feature monitoring has gone blind |
# MAGIC | `invalid_<prediction>_rate` | out-of-vocabulary labels (values outside `expected_prediction_values`) |
# MAGIC | `truncation_rate` | completions cut off at `max_tokens` (`finish_reason = 'length'`) — the reasoning-leakage / verbosity failure mode |
# MAGIC | `<prediction>_<value>_rate` | one scalar time series per expected class — for the fraud example, `risk_likely_fraud_rate` is the business-level block rate an on-call person checks first |

# COMMAND ----------

import re

from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
from pyspark.sql import types as T


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def aggregate_metric(name: str, definition: str) -> MonitorMetric:
    return MonitorMetric(
        name=name,
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        input_columns=[":table"],
        definition=definition,
        output_data_type=T.StructField(name, T.DoubleType()).json(),
    )


expected_values_sql = ", ".join(
    sql_string_literal(value) for value in EXPECTED_PREDICTION_VALUES
)

custom_metrics = [
    aggregate_metric(
        "json_contract_failure_rate",
        "avg(CASE WHEN response_text IS NOT NULL "
        f"AND {PREDICTION_COL} IS NULL THEN 1.0 ELSE 0.0 END)",
    ),
    aggregate_metric(
        f"invalid_{PREDICTION_FIELD}_rate",
        f"avg(CASE WHEN {PREDICTION_COL} IS NOT NULL "
        f"AND {PREDICTION_COL} NOT IN ({expected_values_sql}) THEN 1.0 ELSE 0.0 END)",
    ),
    aggregate_metric(
        "truncation_rate",
        "avg(CASE WHEN finish_reason = 'length' THEN 1.0 ELSE 0.0 END)",
    ),
]

if PROMPT_FIELDS:
    all_fields_null = " AND ".join(
        f"{PROMPT_COLUMN_PREFIX}{name} IS NULL" for name, _ in PROMPT_FIELDS
    )
    custom_metrics.append(
        aggregate_metric(
            "prompt_parse_failure_rate",
            f"avg(CASE WHEN prompt IS NOT NULL AND {all_fields_null} "
            "THEN 1.0 ELSE 0.0 END)",
        )
    )

for value in EXPECTED_PREDICTION_VALUES:
    metric_name = f"{PREDICTION_FIELD}_{re.sub(r'[^A-Za-z0-9_]', '_', value)}_rate"
    custom_metrics.append(
        aggregate_metric(
            metric_name,
            f"avg(CASE WHEN {PREDICTION_COL} = {sql_string_literal(value)} "
            "THEN 1.0 ELSE 0.0 END)",
        )
    )

for metric in custom_metrics:
    print(f"{metric.name}: {metric.definition}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the monitor
# MAGIC
# MAGIC Unity Catalog allows exactly one monitor per table, so this cell is idempotent: it updates the existing monitor (keeping its dashboard) or creates a new one. The assets directory is derived from the current user — nothing user-specific is hardcoded. Creation is asynchronous; the cell waits until the monitor reports `ACTIVE` before refreshing.

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorInfoStatus,
)

w = WorkspaceClient()

inference_log = MonitorInferenceLog(
    granularities=GRANULARITIES,
    timestamp_col="request_time",
    model_id_col="served_entity_id",
    prediction_col=PREDICTION_COL,
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    label_col=LABEL_FIELD or None,
)

try:
    existing_monitor = w.quality_monitors.get(table_name=MONITORED_TABLE)
except NotFound:
    existing_monitor = None

if existing_monitor is None:
    current_user = w.current_user.me().user_name
    assets_dir = f"/Workspace/Users/{current_user}/quality_monitoring/{UNPACKED_TABLE_NAME}"
    w.quality_monitors.create(
        table_name=MONITORED_TABLE,
        assets_dir=assets_dir,
        output_schema_name=OUTPUT_SCHEMA_NAME,
        inference_log=inference_log,
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS or None,
        custom_metrics=custom_metrics,
    )
    print(f"Created monitor over {MONITORED_TABLE} (assets: {assets_dir})")
else:
    w.quality_monitors.update(
        table_name=MONITORED_TABLE,
        output_schema_name=OUTPUT_SCHEMA_NAME,
        inference_log=inference_log,
        baseline_table_name=BASELINE_TABLE,
        slicing_exprs=SLICING_EXPRS or None,
        custom_metrics=custom_metrics,
        dashboard_id=existing_monitor.dashboard_id,
    )
    print(f"Updated monitor over {MONITORED_TABLE}")

deadline = time.monotonic() + 600
while True:
    monitor_info = w.quality_monitors.get(table_name=MONITORED_TABLE)
    if monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE:
        break
    if monitor_info.status in (
        MonitorInfoStatus.MONITOR_STATUS_ERROR,
        MonitorInfoStatus.MONITOR_STATUS_FAILED,
    ):
        raise RuntimeError(
            f"Monitor over {MONITORED_TABLE} entered status "
            f"{monitor_info.status.value} — inspect it on the table's Quality "
            "tab in Catalog Explorer, or delete it and rerun this notebook."
        )
    if time.monotonic() > deadline:
        raise TimeoutError(
            f"Monitor over {MONITORED_TABLE} did not become ACTIVE within 10 "
            f"minutes (last status: {monitor_info.status})."
        )
    print(f"Monitor status: {monitor_info.status.value} — waiting...")
    time.sleep(15)

print("Monitor is ACTIVE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trigger a metrics refresh
# MAGIC
# MAGIC The refresh computes profile and drift metrics incrementally from the unpacked table's change data feed. It runs on Databricks-managed serverless compute and can take several minutes; this cell starts it without blocking — check progress with `w.quality_monitors.get_refresh(...)`, on the table's Quality tab, or just query the metrics tables once it finishes.

# COMMAND ----------

refresh_info = w.quality_monitors.run_refresh(table_name=MONITORED_TABLE)
refresh_state = refresh_info.state.value if refresh_info.state else "UNKNOWN"
print(f"Refresh {refresh_info.refresh_id} started (state: {refresh_state})")
print(f"Profile metrics table: {monitor_info.profile_metrics_table_name}")
print(f"Drift metrics table:   {monitor_info.drift_metrics_table_name}")
print(
    f"Dashboard id {monitor_info.dashboard_id} — open it from the table's "
    "Quality tab in Catalog Explorer:"
)
print(f"{w.config.host}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{UNPACKED_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - **Schedule**: one job with two sequential tasks — `01_unpack_inference_table.py`, then `quality_monitors.run_refresh` — daily by default (matching the `1 day` granularity). Rerun **this** notebook only after a retrain (baseline rebuild) or a `monitor.yaml` change.
# MAGIC - **Alerts**: SQL alerts over the drift and profile metrics tables — prediction/feature drift vs. the `training_baseline` model id (`drift_type = 'BASELINE'`), the custom contract metrics (`column_name = ':table'`), and volume collapse. `monitor/monitoring.md` §11 has the suggested queries; exclude the current in-progress window (inference logs arrive with up to ~1 hour of lag).
# MAGIC - **Phase 2 (ground truth)**: when late-arriving fraud labels exist, MERGE them into the unpacked table keyed by `client_request_id` (callers must send the header) and set `label_field` in `monitor.yaml` — the monitor then computes precision/recall/confusion per window and slice.
