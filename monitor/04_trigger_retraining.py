# Databricks notebook source
# DBTITLE 1,Trigger retraining when baseline drift breaches thresholds
# MAGIC %md
# MAGIC # Trigger retraining from baseline drift
# MAGIC
# MAGIC Schedule this notebook after the data quality monitor refresh. It checks the generated drift metrics table with the same predicate used by `03_create_drift_sql_alert.py`. When drift is present, it starts the Lakeflow Job configured by `retraining_job_id` in `monitor.yaml`.
# MAGIC
# MAGIC Duplicate protection has three layers: an active target-job run blocks a launch, any run within `retraining_cooldown_hours` blocks a launch, and the Jobs API receives an idempotency token derived from the latest breached drift window.

# COMMAND ----------

import hashlib
import sys
import time
from pathlib import Path

MONITOR_MODULE_DIR = str(Path.cwd().resolve())
if MONITOR_MODULE_DIR not in sys.path:
    sys.path.insert(0, MONITOR_MODULE_DIR)

from monitoring_utils import (
    build_drift_breach_details_query,
    build_drift_breach_summary_query,
    parse_drift_detection_config,
)
from utils import config_str, get_spark_session, load_yaml_config

# COMMAND ----------

config_path, monitor_config = load_yaml_config("monitor.yaml", base_dir=Path.cwd())

UC_CATALOG = config_str(monitor_config, "catalog")
UC_SCHEMA = config_str(monitor_config, "schema")
UNPACKED_TABLE_NAME = config_str(monitor_config, "unpacked_table")
DRIFT_SETTINGS = parse_drift_detection_config(monitor_config)
RETRAINING_COOLDOWN_HOURS = int(monitor_config.get("retraining_cooldown_hours", 24))

if RETRAINING_COOLDOWN_HOURS < 0:
    raise ValueError("retraining_cooldown_hours cannot be negative")

raw_job_parameters = monitor_config.get("retraining_job_parameters") or {}
if not isinstance(raw_job_parameters, dict):
    raise ValueError("retraining_job_parameters must be a mapping")
RETRAINING_JOB_PARAMETERS = {
    str(key): str(value) for key, value in raw_job_parameters.items()
}

MONITORED_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{UNPACKED_TABLE_NAME}"

print(f"Monitor config: {config_path}")
print(f"Monitored table: {MONITORED_TABLE}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
spark = get_spark_session()

monitor_info = w.quality_monitors.get(table_name=MONITORED_TABLE)
drift_table = monitor_info.drift_metrics_table_name
if not drift_table:
    raise RuntimeError(
        f"The monitor over {MONITORED_TABLE} has no drift metrics table. Run "
        "02_create_quality_monitor.py and wait for its refresh to finish."
    )

summary_query = build_drift_breach_summary_query(drift_table, DRIFT_SETTINGS)
summary = spark.sql(summary_query).first()
breach_count = int(summary["breach_count"] or 0)
latest_window_end = summary["latest_window_end"]

print(f"Drift metrics table: {drift_table}")
print(f"Breached metric rows: {breach_count}")
print(f"Latest breached window end: {latest_window_end or '(none)'}")

if breach_count:
    breach_details = spark.sql(
        build_drift_breach_details_query(drift_table, DRIFT_SETTINGS)
    )
    try:
        display(breach_details)
    except NameError:
        breach_details.show(truncate=False)

# COMMAND ----------


def trigger_retraining_if_needed() -> None:
    if breach_count == 0:
        print("No drift threshold breach; retraining was not requested.")
        return

    raw_job_id = str(monitor_config.get("retraining_job_id") or "").strip()
    if not raw_job_id:
        raise ValueError(
            "Drift breached a configured threshold, but retraining_job_id is "
            "empty in monitor.yaml. Set it to the target Lakeflow Job ID."
        )
    try:
        job_id = int(raw_job_id)
    except ValueError as exc:
        raise ValueError("retraining_job_id must be an integer Lakeflow Job ID") from exc

    job = w.jobs.get(job_id=job_id)
    job_name = (job.settings.name if job.settings else None) or str(job_id)

    active_run = next(
        w.jobs.list_runs(job_id=job_id, active_only=True, limit=1),
        None,
    )
    if active_run is not None:
        print(
            f"Retraining job {job_name!r} already has active run "
            f"{active_run.run_id}; no new run was started."
        )
        return

    if RETRAINING_COOLDOWN_HOURS:
        cooldown_start_ms = int(
            (time.time() - RETRAINING_COOLDOWN_HOURS * 60 * 60) * 1000
        )
        recent_run = next(
            w.jobs.list_runs(
                job_id=job_id,
                start_time_from=cooldown_start_ms,
                limit=1,
            ),
            None,
        )
        if recent_run is not None:
            print(
                f"Retraining job {job_name!r} ran within the "
                f"{RETRAINING_COOLDOWN_HOURS}-hour cooldown (run "
                f"{recent_run.run_id}); no new run was started."
            )
            return

    window_key = (
        latest_window_end.isoformat()
        if hasattr(latest_window_end, "isoformat")
        else str(latest_window_end)
    )
    token_digest = hashlib.sha256(
        f"drift-retrain:{job_id}:{window_key}".encode("utf-8")
    ).hexdigest()
    idempotency_token = f"drift-{token_digest}"[:64]

    run_waiter = w.jobs.run_now(
        job_id=job_id,
        idempotency_token=idempotency_token,
        job_parameters=RETRAINING_JOB_PARAMETERS or None,
    )
    run_id = run_waiter.run_id
    print(f"Started retraining job {job_name!r}: run {run_id}")
    print(f"{w.config.host}/#job/{job_id}/run/{run_id}")


trigger_retraining_if_needed()
