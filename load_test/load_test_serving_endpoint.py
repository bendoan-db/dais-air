# Databricks notebook source
# DBTITLE 1,AIR fraud model serving load test
# MAGIC %md
# MAGIC # Load test the deployed fraud model serving endpoint
# MAGIC
# MAGIC This notebook exercises the deployed fraud decision model with high-volume chat requests.
# MAGIC It follows the Databricks custom LLM serving query contract: send a request to the serving endpoint with a `messages` array, where the user message contains the transaction prompt.
# MAGIC
# MAGIC The goal of this notebook is to simulate a **10,000 query-per-second** production traffic pattern and record the achieved throughput, response status distribution, and latency percentiles.
# MAGIC The load generator runs across Spark tasks so the notebook driver does not become the only request source.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Query custom LLM serving endpoint: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms#step-7-query-your-endpoint
# MAGIC - Custom LLM serving monitoring: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms#monitor-your-endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute and dependency setup
# MAGIC
# MAGIC Attach this notebook to Databricks serverless compute with enough load-generator capacity for the configured request rate.
# MAGIC The serving endpoint must already exist and be in a ready state before the load test starts.
# MAGIC
# MAGIC The notebook uses `aiohttp` for asynchronous HTTP requests. The Databricks SDK is not used for the hot path because the load test needs connection pooling, high concurrency, and precise request pacing.

# COMMAND ----------

# MAGIC %pip install -qqq "aiohttp>=3.9.0" "pyyaml>=6.0.2" "pandas>=2.2.0" "requests>=2.31.0"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load shared utilities and configuration

# COMMAND ----------

# training_utils is a plain Python module in train/ (not a notebook), so it is
# imported rather than %run. The notebook's working directory is the notebook's
# folder on serverless, so ../train resolves to the module's directory.
import sys
from pathlib import Path

TRAIN_MODULE_DIR = str((Path.cwd().parent / "train").resolve())
if TRAIN_MODULE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_MODULE_DIR)

from training_utils import (
    config_float,
    config_int,
    config_str,
    full_name,
    get_spark_session,
    load_yaml_config,
)

# COMMAND ----------

import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from pyspark.sql import functions as F

# COMMAND ----------

config_path, load_test_config = load_yaml_config("serving_load_test.yaml", base_dir=Path.cwd())

UC_CATALOG = config_str(load_test_config, "catalog")
UC_SCHEMA = config_str(load_test_config, "schema")
SFT_TABLE_NAME = config_str(load_test_config, "sft_table")
ENDPOINT_NAME = config_str(load_test_config, "endpoint_name")
RESULTS_TABLE_NAME = config_str(load_test_config, "results_table")

TARGET_QPS = config_int(load_test_config, "target_qps")
DURATION_SECONDS = config_int(load_test_config, "duration_seconds")
LOAD_GENERATOR_WORKERS = config_int(load_test_config, "load_generator_workers")
WORKER_CONCURRENCY = config_int(load_test_config, "worker_concurrency")
REQUEST_TIMEOUT_SECONDS = config_int(load_test_config, "request_timeout_seconds")

PROMPT_SAMPLE_FRACTION = config_float(load_test_config, "prompt_sample_fraction")
PROMPT_SAMPLE_SIZE = config_int(load_test_config, "prompt_sample_size")
SMOKE_TEST_REQUESTS = config_int(load_test_config, "smoke_test_requests")
MAX_LATENCY_SAMPLES_PER_WORKER = config_int(load_test_config, "max_latency_samples_per_worker")

MAX_TOKENS = config_int(load_test_config, "max_tokens")
TEMPERATURE = config_float(load_test_config, "temperature")

if TARGET_QPS <= 0:
    raise ValueError("target_qps must be greater than zero.")
if DURATION_SECONDS <= 0:
    raise ValueError("duration_seconds must be greater than zero.")
if LOAD_GENERATOR_WORKERS <= 0:
    raise ValueError("load_generator_workers must be greater than zero.")
if WORKER_CONCURRENCY <= 0:
    raise ValueError("worker_concurrency must be greater than zero.")

spark = get_spark_session()

schema_q = full_name(UC_CATALOG, UC_SCHEMA)
sft_table_q = full_name(UC_CATALOG, UC_SCHEMA, SFT_TABLE_NAME)
results_table_q = full_name(UC_CATALOG, UC_SCHEMA, RESULTS_TABLE_NAME)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")

config_summary = {
    "config_path": str(config_path),
    "sft_table": f"{UC_CATALOG}.{UC_SCHEMA}.{SFT_TABLE_NAME}",
    "endpoint_name": ENDPOINT_NAME,
    "results_table": f"{UC_CATALOG}.{UC_SCHEMA}.{RESULTS_TABLE_NAME}",
    "target_qps": TARGET_QPS,
    "duration_seconds": DURATION_SECONDS,
    "load_generator_workers": LOAD_GENERATOR_WORKERS,
    "worker_concurrency": WORKER_CONCURRENCY,
    "planned_requests": TARGET_QPS * DURATION_SECONDS,
}

display(pd.DataFrame([config_summary]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve endpoint URL and authentication
# MAGIC
# MAGIC Databricks custom LLM serving endpoints accept chat requests at `/serving-endpoints/<endpoint-name>/invocations`.
# MAGIC This notebook uses the notebook context token when running inside Databricks and falls back to `DATABRICKS_HOST` / `DATABRICKS_TOKEN` for job or local execution.
# MAGIC
# MAGIC The token is never displayed. It is only used to build the authorization header for the endpoint request.

# COMMAND ----------

def get_databricks_host_and_token() -> tuple[str, str]:
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        try:
            context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            if not host:
                host = context.apiUrl().get()
            if not token:
                token = context.apiToken().get()
        except Exception as exc:
            raise RuntimeError(
                "Set DATABRICKS_HOST and DATABRICKS_TOKEN, or run this notebook inside Databricks."
            ) from exc

    if not host:
        raise RuntimeError("DATABRICKS_HOST could not be resolved.")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN could not be resolved.")
    if not host.startswith("http"):
        host = f"https://{host}"

    return host.rstrip("/"), token


DATABRICKS_HOST, DATABRICKS_TOKEN = get_databricks_host_and_token()
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
AUTH_HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

print(f"Endpoint URL: {ENDPOINT_URL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify endpoint readiness
# MAGIC
# MAGIC Before generating load, check that the endpoint exists and is ready.
# MAGIC If the endpoint has scale-to-zero enabled and is currently cold, this check may wake it up; wait for readiness before running the full load test.

# COMMAND ----------

endpoint_status_response = requests.get(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers=AUTH_HEADERS,
    timeout=REQUEST_TIMEOUT_SECONDS,
)
endpoint_status_response.raise_for_status()
endpoint_info = endpoint_status_response.json()

endpoint_state = endpoint_info.get("state", {})
ready_state = str(endpoint_state.get("ready", endpoint_state.get("ready_state", "")))
config_update_state = str(endpoint_state.get("config_update", ""))

display(
    pd.DataFrame(
        [
            {
                "endpoint_name": endpoint_info.get("name", ENDPOINT_NAME),
                "ready_state": ready_state,
                "config_update_state": config_update_state,
                "creator": endpoint_info.get("creator"),
            }
        ]
    )
)

if "READY" not in ready_state.upper():
    raise RuntimeError(f"Endpoint {ENDPOINT_NAME} is not ready. Endpoint state: {endpoint_state}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build request payloads from SFT prompts
# MAGIC
# MAGIC The load test uses prompts from the prepared SFT Delta table so the request shape matches the prompts used during fine-tuning.
# MAGIC Each request sends a chat payload with one user message, plus deterministic generation settings.

# COMMAND ----------

prompt_pdf = (
    spark.table(sft_table_q)
    .select("prompt")
    .sample(withReplacement=False, fraction=PROMPT_SAMPLE_FRACTION, seed=3407)
    .limit(PROMPT_SAMPLE_SIZE)
    .toPandas()
)

if prompt_pdf.empty:
    raise ValueError(f"No prompt records were loaded from {UC_CATALOG}.{UC_SCHEMA}.{SFT_TABLE_NAME}.")

payload_templates = [
    {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    for prompt in prompt_pdf["prompt"].tolist()
]

display(
    pd.DataFrame(
        [
            {
                "prompt_count": len(payload_templates),
                "first_prompt_preview": payload_templates[0]["messages"][0]["content"][:700],
            }
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test the deployed endpoint
# MAGIC
# MAGIC Send a small number of synchronous requests first.
# MAGIC The full load test is skipped if the endpoint cannot successfully handle the basic chat payload.

# COMMAND ----------

def summarize_response(response: requests.Response) -> dict[str, Any]:
    elapsed_ms = response.elapsed.total_seconds() * 1000
    try:
        body = response.json()
        body_preview = json.dumps(body)[:1000]
    except Exception:
        body_preview = response.text[:1000]

    return {
        "status_code": response.status_code,
        "elapsed_ms": elapsed_ms,
        "body_preview": body_preview,
    }


smoke_test_results = []
for request_index in range(SMOKE_TEST_REQUESTS):
    response = requests.post(
        ENDPOINT_URL,
        headers=AUTH_HEADERS,
        json=payload_templates[request_index % len(payload_templates)],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    smoke_test_results.append(
        {
            "request_index": request_index,
            **summarize_response(response),
        }
    )

smoke_test_pdf = pd.DataFrame(smoke_test_results)
display(smoke_test_pdf)

if (smoke_test_pdf["status_code"] >= 400).any():
    raise RuntimeError("Smoke test failed. Fix endpoint or payload errors before running the load test.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run distributed 10,000 QPS load test
# MAGIC
# MAGIC The load test divides `target_qps` across `load_generator_workers` Spark tasks.
# MAGIC Each task uses asynchronous HTTP requests with bounded concurrency and attempts to pace requests at its assigned share of the target rate.
# MAGIC
# MAGIC The achieved QPS may be lower than the target if the load-generator compute, network path, endpoint queue, or provisioned endpoint replicas become saturated.
# MAGIC The summary metrics below make that gap visible.

# COMMAND ----------

load_test_id = str(uuid.uuid4())


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, percentile_value))


worker_result_schema = """
worker_id LONG,
target_qps DOUBLE,
request_count LONG,
success_count LONG,
failure_count LONG,
elapsed_seconds DOUBLE,
achieved_qps DOUBLE,
status_counts_json STRING,
failure_examples_json STRING,
latency_samples_json STRING
"""


def run_worker_batches(batch_iterator):
    import asyncio
    import json
    import random
    import time
    from collections import Counter

    import aiohttp
    import pandas as pd

    async def run_single_worker(worker_id: int) -> dict[str, Any]:
        payloads = payload_templates
        worker_target_qps = TARGET_QPS / LOAD_GENERATOR_WORKERS
        worker_total_requests = int(worker_target_qps * DURATION_SECONDS)
        request_interval_seconds = 1.0 / worker_target_qps

        status_counts = Counter()
        failure_examples = []
        latency_samples = []
        observed_latency_count = 0

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        connector = aiohttp.TCPConnector(
            limit=WORKER_CONCURRENCY,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        def record_latency(latency_ms: float) -> None:
            nonlocal observed_latency_count
            observed_latency_count += 1
            if len(latency_samples) < MAX_LATENCY_SAMPLES_PER_WORKER:
                latency_samples.append(latency_ms)
                return

            replacement_index = random.randint(0, observed_latency_count - 1)
            if replacement_index < MAX_LATENCY_SAMPLES_PER_WORKER:
                latency_samples[replacement_index] = latency_ms

        async def send_request(session: aiohttp.ClientSession, request_index: int):
            payload = payloads[(worker_id + request_index) % len(payloads)]
            started_request = time.perf_counter()
            try:
                async with session.post(ENDPOINT_URL, json=payload) as response:
                    response_text = await response.text()
                    latency_ms = (time.perf_counter() - started_request) * 1000
                    return response.status, latency_ms, None if response.status < 400 else response_text[:500]
            except Exception as exc:
                latency_ms = (time.perf_counter() - started_request) * 1000
                return "exception", latency_ms, repr(exc)[:500]

        async def drain_completed(pending_tasks):
            done, remaining = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                status, latency_ms, error_preview = await task
                status_counts[str(status)] += 1
                record_latency(latency_ms)
                if error_preview and len(failure_examples) < 5:
                    failure_examples.append({"status": str(status), "preview": error_preview})
            return remaining

        async with aiohttp.ClientSession(headers=AUTH_HEADERS, timeout=timeout, connector=connector) as session:
            pending = set()
            started_at = time.perf_counter()

            for request_index in range(worker_total_requests):
                scheduled_at = started_at + request_index * request_interval_seconds
                now = time.perf_counter()
                if scheduled_at > now:
                    await asyncio.sleep(scheduled_at - now)

                pending.add(asyncio.create_task(send_request(session, request_index)))

                if len(pending) >= WORKER_CONCURRENCY:
                    pending = await drain_completed(pending)

            while pending:
                pending = await drain_completed(pending)

        elapsed_seconds = time.perf_counter() - started_at
        success_count = sum(count for status, count in status_counts.items() if status.isdigit() and 200 <= int(status) < 300)
        failure_count = sum(status_counts.values()) - success_count

        return {
            "worker_id": worker_id,
            "target_qps": worker_target_qps,
            "request_count": sum(status_counts.values()),
            "success_count": success_count,
            "failure_count": failure_count,
            "elapsed_seconds": elapsed_seconds,
            "achieved_qps": sum(status_counts.values()) / elapsed_seconds if elapsed_seconds else None,
            "status_counts_json": json.dumps(dict(status_counts), sort_keys=True),
            "failure_examples_json": json.dumps(failure_examples),
            "latency_samples_json": json.dumps(latency_samples),
        }

    async def run_worker_group(worker_ids: list[int]) -> list[dict[str, Any]]:
        return await asyncio.gather(*(run_single_worker(worker_id) for worker_id in worker_ids))

    for batch_pdf in batch_iterator:
        worker_ids = [int(worker_id) for worker_id in batch_pdf["worker_id"].tolist()]
        worker_rows = asyncio.run(run_worker_group(worker_ids))
        yield pd.DataFrame(worker_rows)


load_test_started_at = datetime.now(timezone.utc)
load_test_started_perf = time.perf_counter()

worker_results_pdf = (
    spark.range(LOAD_GENERATOR_WORKERS)
    .select(F.col("id").cast("long").alias("worker_id"))
    .repartition(LOAD_GENERATOR_WORKERS)
    .mapInPandas(run_worker_batches, schema=worker_result_schema)
    .toPandas()
)

load_test_elapsed_seconds = time.perf_counter() - load_test_started_perf
load_test_ended_at = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summarize load test results
# MAGIC
# MAGIC Review the achieved throughput, success rate, status-code mix, and latency percentiles.
# MAGIC The result is also appended to the configured Delta results table for comparison across endpoint sizes and model versions.

# COMMAND ----------

status_counts = Counter()
all_latency_samples_ms = []

for _, worker_result in worker_results_pdf.iterrows():
    status_counts.update(json.loads(worker_result["status_counts_json"] or "{}"))
    all_latency_samples_ms.extend(json.loads(worker_result["latency_samples_json"] or "[]"))

total_requests = int(worker_results_pdf["request_count"].sum())
success_count = int(worker_results_pdf["success_count"].sum())
failure_count = int(worker_results_pdf["failure_count"].sum())
achieved_qps = total_requests / load_test_elapsed_seconds if load_test_elapsed_seconds else None
success_rate = success_count / total_requests if total_requests else None

summary_row = {
    "load_test_id": load_test_id,
    "endpoint_name": ENDPOINT_NAME,
    "started_at_utc": load_test_started_at.isoformat(),
    "ended_at_utc": load_test_ended_at.isoformat(),
    "target_qps": TARGET_QPS,
    "duration_seconds": DURATION_SECONDS,
    "planned_requests": TARGET_QPS * DURATION_SECONDS,
    "actual_elapsed_seconds": load_test_elapsed_seconds,
    "request_count": total_requests,
    "success_count": success_count,
    "failure_count": failure_count,
    "success_rate": success_rate,
    "achieved_qps": achieved_qps,
    "load_generator_workers": LOAD_GENERATOR_WORKERS,
    "worker_concurrency": WORKER_CONCURRENCY,
    "prompt_sample_size": len(payload_templates),
    "latency_sample_count": len(all_latency_samples_ms),
    "latency_p50_ms": percentile(all_latency_samples_ms, 50),
    "latency_p90_ms": percentile(all_latency_samples_ms, 90),
    "latency_p95_ms": percentile(all_latency_samples_ms, 95),
    "latency_p99_ms": percentile(all_latency_samples_ms, 99),
    "status_counts_json": json.dumps(dict(status_counts), sort_keys=True),
}

summary_pdf = pd.DataFrame([summary_row])
display(summary_pdf)

status_counts_pdf = pd.DataFrame(
    [{"status": status, "count": count} for status, count in sorted(status_counts.items())]
)
display(status_counts_pdf)

display(worker_results_pdf.drop(columns=["latency_samples_json"]).sort_values("worker_id"))

# COMMAND ----------

(
    spark.createDataFrame(summary_pdf)
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(results_table_q)
)

print(f"Saved load test summary to {results_table_q}")

# COMMAND ----------

display(spark.table(results_table_q))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpret the result
# MAGIC
# MAGIC Compare `achieved_qps` with `target_qps`.
# MAGIC If the endpoint does not sustain 10,000 QPS, use the result tables to separate load-generator limits from serving limits:
# MAGIC
# MAGIC - High client-side failures or exceptions usually indicate the load generator needs more workers, more concurrency, or a longer timeout.
# MAGIC - HTTP `429`, `503`, or long tail latency usually indicates endpoint queueing or insufficient provisioned serving capacity.
# MAGIC - Low achieved QPS with low endpoint latency usually indicates the notebook load generator is saturated before the endpoint.
# MAGIC
# MAGIC For production capacity testing, run this notebook against each endpoint size you plan to evaluate and compare the persisted summaries in the results table.
