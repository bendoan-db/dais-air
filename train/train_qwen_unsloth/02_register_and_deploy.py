# Databricks notebook source
# DBTITLE 1,Register the fine-tuned model and deploy it to Model Serving
# MAGIC %md
# MAGIC # Register the fine-tuned model and deploy it to Model Serving
# MAGIC
# MAGIC This project-local deployment step takes a training run produced by `01_runner.py` (or the AIR CLI), merges the run's LoRA adapter into the base model, registers the merged model to Unity Catalog as a custom LLM, and creates or updates a Mosaic AI Model Serving endpoint for it.
# MAGIC
# MAGIC Run selection is driven by this directory's `train.yaml` `deploy_config` section:
# MAGIC
# MAGIC - `run_id` set — register exactly that MLflow run's adapter.
# MAGIC - `run_id` empty — search this project's `experiment_name` for FINISHED runs and pick the best one by `best_run_metric` / `best_run_metric_goal`.
# MAGIC
# MAGIC Either way, the adapter location is read from the run's `adapter_output_dir` parameter (logged by training), so this notebook needs no knowledge of checkpoint-volume layout.
# MAGIC
# MAGIC **Compute**: attach to **Serverless GPU** with the **AI v5** base environment and enough memory/local disk to load, merge, and save this project's model.
# MAGIC
# MAGIC Reference: [Serve custom LLMs with Custom Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms).

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# project_config.py and train.py are plain modules in this directory.
import json
import sys
from pathlib import Path

import pandas as pd

NOTEBOOK_DIR = str(Path.cwd())
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

from project_config import load_deploy_config

# Registration/serving settings come from this project's deploy_config.
# Keep endpoint_name aligned with the load test and inference_table_prefix
# aligned with the monitor.
deploy_context = load_deploy_config()
globals().update(deploy_context)

print(f"Deploy config: {DEPLOY_CONFIG_PATH} (parameters.deploy_config)")
print(f"Registered model target: {FULL_MODEL_NAME}")
print(f"Serving endpoint: {ENDPOINT_NAME}")
print(
    "Inference payload table: "
    f"{UC_CATALOG}.{UC_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload"
)
print(f"Run selection: {'run_id=' + RUN_ID if RUN_ID else f'best {BEST_RUN_METRIC} ({BEST_RUN_METRIC_GOAL}) in {EXPERIMENT_NAME}'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the training run to deploy
# MAGIC
# MAGIC The adapter location comes from the selected run's `adapter_output_dir` parameter, which training logs on the rank-0 run after saving artifacts.
# MAGIC When `run_id` is empty, only FINISHED runs that logged both the ranking metric and an adapter are candidates — an incomplete or non-rank-0 run can never be selected.

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")

# A bare experiment name resolves under the current user's folder — the same
# path the training notebook and AIR CLI use; absolute paths pass through.
if EXPERIMENT_NAME.startswith("/"):
    experiment_path = EXPERIMENT_NAME
else:
    current_user = WorkspaceClient().current_user.me().user_name
    experiment_path = f"/Users/{current_user}/{EXPERIMENT_NAME}"

experiment = mlflow.get_experiment_by_name(experiment_path)
if experiment is None:
    raise ValueError(
        f"MLflow experiment not found: {experiment_path}. Run training "
        "(01_runner.py or the AIR CLI) first, or fix experiment_name in train.yaml."
    )

if RUN_ID:
    source_run = mlflow.get_run(RUN_ID)
    selection_reason = "run_id from train.yaml's deploy_config"
    selection_metric_value = source_run.data.metrics.get(BEST_RUN_METRIC)
else:
    metric_order = "ASC" if BEST_RUN_METRIC_GOAL == "minimize" else "DESC"
    runs_pdf = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.`{BEST_RUN_METRIC}` {metric_order}"],
    )
    metric_column = f"metrics.{BEST_RUN_METRIC}"
    adapter_column = "params.adapter_output_dir"
    if runs_pdf.empty or metric_column not in runs_pdf.columns or adapter_column not in runs_pdf.columns:
        raise ValueError(
            f"No finished runs in {experiment_path} logged both "
            f"{BEST_RUN_METRIC!r} and adapter_output_dir — nothing to deploy. "
            "Complete a training run first or set run_id in train.yaml's deploy_config."
        )
    candidate_runs = runs_pdf[
        runs_pdf[metric_column].notna() & runs_pdf[adapter_column].notna()
    ]
    if candidate_runs.empty:
        raise ValueError(
            f"No finished run in {experiment_path} has both {BEST_RUN_METRIC!r} "
            "and adapter_output_dir. Complete a training run first or set "
            "run_id in train.yaml's deploy_config."
        )
    best_row = candidate_runs.iloc[0]
    source_run = mlflow.get_run(best_row["run_id"])
    selection_reason = (
        f"best {BEST_RUN_METRIC} ({BEST_RUN_METRIC_GOAL}) of "
        f"{len(candidate_runs)} candidate run(s)"
    )
    selection_metric_value = best_row[metric_column]

SOURCE_RUN_ID = source_run.info.run_id
ADAPTER_OUTPUT_DIR = source_run.data.params.get("adapter_output_dir")
if not ADAPTER_OUTPUT_DIR:
    raise ValueError(
        f"Run {SOURCE_RUN_ID} has no adapter_output_dir parameter — it did not "
        "save adapter artifacts (only completed rank-0 training runs do). "
        "Pick a different run."
    )
BASE_MODEL = source_run.data.params.get("base_model", "unknown")

display(
    pd.DataFrame(
        [
            {
                "source_run_id": SOURCE_RUN_ID,
                "run_name": source_run.info.run_name,
                "selection": selection_reason,
                BEST_RUN_METRIC: selection_metric_value,
                "adapter_output_dir": ADAPTER_OUTPUT_DIR,
                "base_model": BASE_MODEL,
                "experiment": experiment_path,
            }
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the trained model for custom LLM serving
# MAGIC
# MAGIC This cell loads the selected run's adapter artifacts, merges them with the base model, packages the merged Hugging Face weights into an MLflow model artifact, and registers that model to Unity Catalog.
# MAGIC
# MAGIC Databricks custom LLM serving runs a vLLM OpenAI-compatible server from a custom MLflow entrypoint. The important serving choices are visible below:
# MAGIC
# MAGIC - `task` is `llm/v1/chat`, matching the chat request contract used by the serving endpoint.
# MAGIC - The vLLM process listens on port `8080`, which is the port Model Serving expects.
# MAGIC - The entrypoint launches from the MLflow model's `artifacts/` folder, so the `--model` path is the bare artifact name relative to that folder.
# MAGIC - Registration uses `env_pack="databricks_model_serving"` so Databricks can build the express serving environment.
# MAGIC - The serving container installs packages from this project's `requirements.txt` (referenced by `deploy_config`'s `serving_requirements_file`). Its vLLM, transformers, MLflow, SDK, and OpenCV pins follow the current Databricks custom LLM starter environment; the base model's architecture must be in that vLLM version's supported model list.
# MAGIC
# MAGIC Registration is separate from training so a failed registration or deployment can be rerun without re-training.

# COMMAND ----------

from train import load_model_for_merge

CUSTOM_LLM_TASK = "llm/v1/chat"
# Bare directory name for the merged weights inside the MLflow model's
# artifacts/ folder (the vLLM entrypoint's --model path).
CUSTOM_LLM_MODEL_ARTIFACT_NAME = UC_MODEL_NAME


def local_model_work_dir() -> Path:
    import tempfile

    local_disk_tmp = Path("/local_disk0/tmp")
    if local_disk_tmp.exists():
        return Path(tempfile.mkdtemp(prefix="air-custom-llm-", dir=local_disk_tmp))
    return Path(tempfile.mkdtemp(prefix="air-custom-llm-"))


def register_custom_llm_model(adapter_output_dir: str, run_name: str):
    import shutil

    import mlflow
    from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

    mlflow.set_registry_uri("databricks-uc")

    # Defined inline (not in a project module) on purpose: cloudpickle
    # serializes notebook-local classes BY VALUE, so the serving container can
    # unpickle the model without any repo code and no code_paths are needed in
    # log_model. If this class ever moves into a module or imports repo helpers,
    # registration must add the relevant project files as code_paths.
    class CustomLlmEntrypointPlaceholder(ChatModel):
        def predict(self, context, messages, params):
            return ChatCompletionResponse.from_dict({"choices": []})

    metadata = {
        "task": CUSTOM_LLM_TASK,
        "entrypoint": (
            "python -u -m vllm.entrypoints.openai.api_server "
            f"--model {CUSTOM_LLM_MODEL_ARTIFACT_NAME} "
            f"--served-model-name {SERVED_MODEL_NAME} "
            "--host 0.0.0.0 --port 8080 "
            # SFT prompts typically share the same instruction header, so
            # prefix caching skips most prefill work (explicit for visibility;
            # the vLLM v1 engine defaults it on).
            "--enable-prefix-caching "
            f"--dtype {VLLM_DTYPE} "
            f"--max-model-len {VLLM_MAX_MODEL_LEN} "
            f"--gpu-memory-utilization {VLLM_GPU_MEMORY_UTILIZATION}"
        ),
    }

    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "Classify this card transaction and return compact JSON.",
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    temp_dir = local_model_work_dir()
    try:
        merged_model_dir = temp_dir / CUSTOM_LLM_MODEL_ARTIFACT_NAME

        model, tokenizer = load_model_for_merge(adapter_output_dir)

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_dir)

        # Preflight: serving only works for architectures supported by the
        # vLLM version pinned in serving_pip_requirements. Surface the
        # architecture here so an unsupported base model is caught before a
        # failed endpoint rollout.
        model_config = json.loads((merged_model_dir / "config.json").read_text())
        architectures = model_config.get("architectures", [])
        print(
            f"Base model architecture(s): {architectures} — verify these appear in "
            "the supported-models list of the vLLM pinned in "
            "requirements.txt before deploying."
        )

        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            mlflow.log_params(
                {
                    "base_model": BASE_MODEL,
                    "adapter_output_dir": adapter_output_dir,
                    "registered_model_name": FULL_MODEL_NAME,
                    "source_training_run_id": SOURCE_RUN_ID,
                    "custom_llm_task": CUSTOM_LLM_TASK,
                    "custom_llm_model_artifact": CUSTOM_LLM_MODEL_ARTIFACT_NAME,
                    "served_model_name": SERVED_MODEL_NAME,
                    "vllm_dtype": VLLM_DTYPE,
                    "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
                    "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
                }
            )
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=CustomLlmEntrypointPlaceholder(),
                artifacts={CUSTOM_LLM_MODEL_ARTIFACT_NAME: str(merged_model_dir)},
                input_example=input_example,
                # The serving container's packages come from the
                # Project-local requirements, including the FIPS-safe vLLM
                # pins configured for the serving container.
                pip_requirements=SERVING_PIP_REQUIREMENTS,
                metadata=metadata,
            )
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=FULL_MODEL_NAME,
                await_registration_for=3600,
                env_pack="databricks_model_serving",
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "registration_run_id": run.info.run_id,
        "registered_model_name": FULL_MODEL_NAME,
        "model_version": model_version.version,
        "model_uri": model_info.model_uri,
        "source_training_run_id": SOURCE_RUN_ID,
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = register_custom_llm_model(
    adapter_output_dir=ADAPTER_OUTPUT_DIR,
    run_name=f"{UC_MODEL_NAME}-registration",
)
REGISTERED_MODEL_VERSION = str(registration_result["model_version"])
display(pd.DataFrame([registration_result]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the custom LLM endpoint
# MAGIC
# MAGIC This cell creates or updates a Mosaic AI Model Serving endpoint for the registered custom LLM, routing 100% of traffic to the version registered above.
# MAGIC
# MAGIC The endpoint configuration is controlled by this project's `train.yaml` `deploy_config`:
# MAGIC
# MAGIC - `endpoint_name` is the serving endpoint name used by the load-test notebook.
# MAGIC - `serving_workload_type` selects a documented custom LLM GPU class: `GPU_SMALL` (T4), `GPU_MEDIUM` (A10), or `GPU_XLARGE` (H100).
# MAGIC - `serving_workload_size` (`Small`, `Medium`, or `Large`) controls the fixed replica capacity; custom LLM serving does not autoscale between non-zero replica counts during beta.
# MAGIC - `serving_scale_to_zero` is useful for development, but should be disabled for latency-sensitive production traffic.
# MAGIC
# MAGIC The served entity also sets `VLLM_USE_FLASHINFER_SAMPLER=0`: the serving container cannot JIT-compile FlashInfer kernels (no `ninja`/`nvcc`), so vLLM must use its native PyTorch sampler.
# MAGIC
# MAGIC **Inference logging is always enabled** as part of the deployment: the endpoint's AI Gateway configuration logs every request/response to `<catalog>.<schema>.<inference_table_prefix>_payload` — the raw table the monitoring stage (`monitor/`) unpacks. AI Gateway inference tables are the recommended capture mechanism for custom model endpoints (the legacy `auto_capture_config` path is retired); logs are delivered within about an hour of traffic.

# COMMAND ----------

def served_entity_name_for_version(model_name: str, version: str) -> str:
    clean_name = model_name.rsplit(".", 1)[-1].replace("_", "-").replace(".", "-")
    return f"{clean_name}-{version}"[:64]


def create_or_update_custom_llm_endpoint(model_version: str) -> dict:
    if SERVING_WORKLOAD_TYPE == "GPU_XLARGE" and SERVING_SCALE_TO_ZERO:
        raise ValueError(
            "Custom LLM serving beta does not support scale-to-zero for GPU_XLARGE. "
            "Set serving_scale_to_zero: false in train.yaml's deploy_config."
        )

    from datetime import timedelta

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, ResourceDoesNotExist
    from databricks.sdk.service.serving import (
        AiGatewayInferenceTableConfig,
        AiGatewayUsageTrackingConfig,
        EndpointCoreConfigInput,
        Route,
        ServedEntityInput,
        ServingModelWorkloadType,
        TrafficConfig,
    )

    w = WorkspaceClient()
    workload_type = ServingModelWorkloadType(SERVING_WORKLOAD_TYPE)
    served_entity_name = served_entity_name_for_version(FULL_MODEL_NAME, model_version)
    served_entity_kwargs = dict(
        name=served_entity_name,
        entity_name=FULL_MODEL_NAME,
        entity_version=str(model_version),
        workload_type=workload_type,
        workload_size=SERVING_WORKLOAD_SIZE,
        scale_to_zero_enabled=SERVING_SCALE_TO_ZERO,
        environment_vars={
            # The serving container has no ninja/nvcc, so FlashInfer (shipped in
            # the Databricks AI base env) cannot JIT-compile its sampling kernels
            # at startup; fall back to vLLM's native PyTorch sampler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
    )
    served_entity = ServedEntityInput(**served_entity_kwargs)
    traffic_config = TrafficConfig(
        routes=[
            Route(
                served_entity_name=served_entity_name,
                traffic_percentage=100,
            )
        ]
    )

    try:
        w.serving_endpoints.get(ENDPOINT_NAME)
        endpoint = w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
            traffic_config=traffic_config,
            timeout=timedelta(minutes=60),
        )
        deployment_action = "updated"
    except (NotFound, ResourceDoesNotExist):
        endpoint = w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[served_entity],
                traffic_config=traffic_config,
            ),
            description=ENDPOINT_DESCRIPTION,
            timeout=timedelta(minutes=60),
        )
        deployment_action = "created"

    # AI Gateway is configured separately from the endpoint model config, so
    # apply it after both create and update rollouts. PUT replaces the entire
    # gateway configuration; preserve unrelated settings already on the
    # endpoint while enabling inference tables and usage tracking.
    endpoint_details = w.serving_endpoints.get(ENDPOINT_NAME)
    current_gateway = getattr(endpoint_details, "ai_gateway", None)
    requested_inference_table = AiGatewayInferenceTableConfig(
        catalog_name=UC_CATALOG,
        schema_name=UC_SCHEMA,
        table_name_prefix=INFERENCE_TABLE_PREFIX,
        enabled=True,
    )
    gateway_response = w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        fallback_config=getattr(current_gateway, "fallback_config", None),
        guardrails=getattr(current_gateway, "guardrails", None),
        inference_table_config=requested_inference_table,
        rate_limits=getattr(current_gateway, "rate_limits", None),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    )

    configured_inference_table = getattr(
        gateway_response, "inference_table_config", None
    )
    if configured_inference_table is None:
        refreshed_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
        refreshed_gateway = getattr(refreshed_endpoint, "ai_gateway", None)
        configured_inference_table = getattr(
            refreshed_gateway, "inference_table_config", None
        )
    expected_config = requested_inference_table.as_dict()
    actual_config = (
        configured_inference_table.as_dict()
        if configured_inference_table is not None
        else None
    )
    if actual_config != expected_config:
        raise RuntimeError(
            f"Inference table configuration failed for {ENDPOINT_NAME}: "
            f"expected {expected_config}, got {actual_config}"
        )

    inference_payload_table = f"{UC_CATALOG}.{UC_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload"

    endpoint_state = getattr(endpoint, "state", None)
    workspace_url = (w.config.host or "").rstrip("/")
    endpoint_url = (
        f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}"
        if workspace_url
        else f"/serving-endpoints/{ENDPOINT_NAME}"
    )

    return {
        "deployment_action": deployment_action,
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_url": endpoint_url,
        "registered_model_name": FULL_MODEL_NAME,
        "model_version": str(model_version),
        "served_entity_name": served_entity_name,
        "workload_type": SERVING_WORKLOAD_TYPE,
        "workload_size": SERVING_WORKLOAD_SIZE,
        "scale_to_zero_enabled": SERVING_SCALE_TO_ZERO,
        "inference_table_enabled": configured_inference_table.enabled,
        "inference_payload_table": inference_payload_table,
        "endpoint_ready": str(getattr(endpoint_state, "ready", None)),
        "config_update": str(getattr(endpoint_state, "config_update", None)),
    }


deployment_result = create_or_update_custom_llm_endpoint(REGISTERED_MODEL_VERSION)
display(pd.DataFrame([deployment_result]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC The endpoint serves the fine-tuned model behind the OpenAI-compatible chat contract (`/serving-endpoints/<endpoint_name>/invocations`).
# MAGIC
# MAGIC - Load test it with `load_test/load_test_serving_endpoint.py`, which renders prompts from the raw eval Parquet and records throughput/latency to a results table.
# MAGIC - Rerunning this notebook after a new training run re-selects the best run (or honors `run_id`), registers a new model version, and rolls the endpoint to it.
