# Databricks notebook source
# DBTITLE 1,Register the fine-tuned Qwen3.5 model and deploy it to Model Serving
# MAGIC %md
# MAGIC # Register the fine-tuned Qwen3.5 model and deploy it to Model Serving
# MAGIC
# MAGIC This project-local deployment step takes a training run produced by `01_runner.py` (or the AIR CLI), merges the run's LoRA adapter into the base model, registers the merged model to Unity Catalog as a custom LLM, and creates or updates a Mosaic AI Model Serving endpoint for it.
# MAGIC
# MAGIC Run selection is driven by this directory's `train.yaml` `deploy_config` section:
# MAGIC
# MAGIC - `run_id` set — register exactly that MLflow run's adapter.
# MAGIC - `run_id` empty — search this project's configured `experiment_path` for FINISHED runs and pick the best one by `best_run_metric` / `best_run_metric_goal`.
# MAGIC
# MAGIC Either way, the adapter location is read from the run's `adapter_output_dir` parameter (logged by training), so this notebook needs no knowledge of checkpoint-volume layout.
# MAGIC
# MAGIC Qwen3.5 needs a newer serving stack than the repository's vLLM 0.11 pins, and the training and serving environments cannot share one Python session, so registration crosses a `%restart_python` boundary:
# MAGIC
# MAGIC 1. **Merge** in the training environment (`requirements.txt`) and write the serving checkpoint to node-local disk.
# MAGIC 2. **Install the serving stack** in two `pip` passes: `serving_requirements.txt`, then the FIPS-safe OpenCV pin.
# MAGIC 3. **Register** from that serving environment, which `env_pack` captures into the model version.
# MAGIC
# MAGIC State crosses the restart only through `merge_metadata.json` beside the merged weights, never through Python globals.
# MAGIC
# MAGIC **Compute**: attach to **Serverless GPU** with the **AI v6** base environment and enough memory/local disk to load, merge, and save this project's model.
# MAGIC
# MAGIC Reference: [Serve custom LLMs with Custom Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms).

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# project_config.py and train.py are plain modules in this directory.
import json
import sys
import tempfile
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

# Node-local merge workspace, resolved identically on both sides of the
# %restart_python below. Not /Volumes: large safetensors writes there fail
# with EAGAIN.
LOCAL_DISK_TMP = Path("/local_disk0/tmp")
MERGE_WORK_DIR = (
    LOCAL_DISK_TMP if LOCAL_DISK_TMP.exists() else Path(tempfile.gettempdir())
) / "air-custom-llm" / UC_MODEL_NAME
MERGE_METADATA_PATH = MERGE_WORK_DIR / "merge_metadata.json"

print(f"Deploy config: {DEPLOY_CONFIG_PATH} (parameters.deploy_config)")
print(f"Registered model target: {FULL_MODEL_NAME}")
print(f"Serving endpoint: {ENDPOINT_NAME}")
print(
    "Inference payload table: "
    f"{UC_CATALOG}.{UC_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload"
)
print(f"Run selection: {'run_id=' + RUN_ID if RUN_ID else f'best {BEST_RUN_METRIC} ({BEST_RUN_METRIC_GOAL}) in {EXPERIMENT_PATH}'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the training run to deploy
# MAGIC
# MAGIC The adapter location comes from the selected run's `adapter_output_dir` parameter, which training logs on the rank-0 run after saving artifacts.
# MAGIC When `run_id` is empty, only FINISHED runs that logged both the ranking metric and an adapter are candidates — an incomplete or non-rank-0 run can never be selected.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
if experiment is None:
    raise ValueError(
        f"MLflow experiment not found: {EXPERIMENT_PATH}. Run training "
        "(01_runner.py or the AIR CLI) first, or fix experiment_path in train.yaml."
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
            f"No finished runs in {EXPERIMENT_PATH} logged both "
            f"{BEST_RUN_METRIC!r} and adapter_output_dir — nothing to deploy. "
            "Complete a training run first or set run_id in train.yaml's deploy_config."
        )
    candidate_runs = runs_pdf[
        runs_pdf[metric_column].notna() & runs_pdf[adapter_column].notna()
    ]
    if candidate_runs.empty:
        raise ValueError(
            f"No finished run in {EXPERIMENT_PATH} has both {BEST_RUN_METRIC!r} "
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
                "experiment": EXPERIMENT_PATH,
            }
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge the adapter into a servable checkpoint
# MAGIC
# MAGIC This cell merges the selected adapter into the base weights and writes plain Hugging Face weights to node-local disk, which survives the `%restart_python` below.
# MAGIC
# MAGIC The merged weights are saved in the base model's **composite (vision + text) shape**, not the text-only shape training used. vLLM 0.24 implements `Qwen3_5ForCausalLM` but never registers it, so a text-only checkpoint silently builds `Qwen3_5ForConditionalGeneration` and crashes with `'Qwen3_5TextConfig' object has no attribute 'vision_config'`. Saving the composite shape makes the architecture match; `--language-model-only` then keeps the vision tower idle, but its weights must still ship because vLLM's loader raises on any parameter missing from the checkpoint.
# MAGIC
# MAGIC The merge also rewrites the saved `chat_template.jinja` so thinking is **off by default**: Qwen3.5 has no non-thinking variant, and a client that omits `chat_template_kwargs` would otherwise get a reasoning preamble instead of JSON.

# COMMAND ----------

import shutil

from train import merge_adapter_to_serving_checkpoint

# Bare directory name for the merged weights inside the MLflow model's
# artifacts/ folder (the vLLM entrypoint's --model path).
CUSTOM_LLM_MODEL_ARTIFACT_NAME = UC_MODEL_NAME
MERGED_MODEL_DIR = MERGE_WORK_DIR / CUSTOM_LLM_MODEL_ARTIFACT_NAME

shutil.rmtree(MERGE_WORK_DIR, ignore_errors=True)
MERGED_MODEL_DIR.mkdir(parents=True)

merge_summary = merge_adapter_to_serving_checkpoint(ADAPTER_OUTPUT_DIR, str(MERGED_MODEL_DIR))

# Preflight: serving only works for architectures registered by the vLLM in
# serving_requirements.txt. Surface it before a failed endpoint rollout.
architectures = json.loads((MERGED_MODEL_DIR / "config.json").read_text()).get("architectures", [])
print(f"Merged checkpoint architecture(s): {architectures}")

# %restart_python clears the session, so hand the registration cell what it
# needs through a file rather than Python state.
MERGE_METADATA_PATH.write_text(
    json.dumps(
        {
            "source_run_id": SOURCE_RUN_ID,
            "adapter_output_dir": ADAPTER_OUTPUT_DIR,
            "base_model": BASE_MODEL,
            "merged_model_dir": str(MERGED_MODEL_DIR),
            **merge_summary,
        },
        indent=2,
    )
)
display(pd.DataFrame([{"merged_model_dir": str(MERGED_MODEL_DIR), **merge_summary}]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the serving stack (two pip passes)
# MAGIC
# MAGIC The order matters:
# MAGIC
# MAGIC 1. **vLLM first** from `serving_requirements.txt`. `vllm==0.24.0` registers Qwen3.5's architecture; `mlflow==3.12` is uninstallable beside it (starlette conflict), hence `mlflow==3.14.0`.
# MAGIC 2. **OpenCV second**, downgrading what vLLM just pulled in. `pip` prints a dependency-conflict warning; that is expected. Anything `>=4.13` bundles an OpenSSL that fails Model Serving's FIPS self-test and aborts vLLM at startup.
# MAGIC
# MAGIC `env_pack="databricks_model_serving"` packs this environment into the registered model version, which is why the serving stack is installed here instead of declared as `pip_requirements`: one requirements list is resolved in a single pass and cannot express the conflicting OpenCV pin.
# MAGIC
# MAGIC **Security note:** `opencv-python-headless<4.13` carries a known CVE. Call it out alongside this recipe when reusing it.

# COMMAND ----------

# MAGIC %pip install -qqq -r serving_requirements.txt
# MAGIC %pip install -qqq opencv-python-headless==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the merged model for custom LLM serving
# MAGIC
# MAGIC `%restart_python` cleared the session, so this cell reloads `deploy_config` and reads the merge hand-off file. The serving choices worth seeing:
# MAGIC
# MAGIC - `task` is `llm/v1/chat`, matching the chat request contract used by the serving endpoint.
# MAGIC - The vLLM process listens on port `8080`, which is the port Model Serving expects.
# MAGIC - The entrypoint launches from the MLflow model's `artifacts/` folder, so the `--model` path is the bare artifact name relative to that folder.
# MAGIC - `--language-model-only` serves Qwen3.5's text backbone and keeps the vision tower out of the request path.
# MAGIC - Registration uses `env_pack="databricks_model_serving"` so Databricks can build the express serving environment from the packages installed above.
# MAGIC
# MAGIC Registration is separate from training so a failed registration or deployment can be rerun without re-training.

# COMMAND ----------

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

NOTEBOOK_DIR = str(Path.cwd())
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

from project_config import load_deploy_config

deploy_context = load_deploy_config()
globals().update(deploy_context)

LOCAL_DISK_TMP = Path("/local_disk0/tmp")
MERGE_WORK_DIR = (
    LOCAL_DISK_TMP if LOCAL_DISK_TMP.exists() else Path(tempfile.gettempdir())
) / "air-custom-llm" / UC_MODEL_NAME
MERGE_METADATA_PATH = MERGE_WORK_DIR / "merge_metadata.json"
if not MERGE_METADATA_PATH.exists():
    raise ValueError("Run the merge cell before registering the model.")
merge_metadata = json.loads(MERGE_METADATA_PATH.read_text())

CUSTOM_LLM_TASK = "llm/v1/chat"
CUSTOM_LLM_MODEL_ARTIFACT_NAME = Path(merge_metadata["merged_model_dir"]).name
# Only MLflow is pinned on the logged model; env_pack captures the rest of the
# two-pass environment installed above.
MLFLOW_PIP_REQUIREMENTS = [
    requirement for requirement in SERVING_PIP_REQUIREMENTS if requirement.startswith("mlflow")
]


def register_custom_llm_model(merge_metadata: dict, run_name: str):
    mlflow.set_registry_uri("databricks-uc")

    # Defined inline (not in a project module) on purpose: cloudpickle
    # serializes notebook-local classes BY VALUE, so the serving container can
    # unpickle the model without any repo code and no code_paths are needed in
    # log_model. Serving runs the vLLM entrypoint, never this predict method.
    class CustomLlmEntrypointPlaceholder(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Inference is handled by the custom vLLM entrypoint.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

    metadata = {
        "task": CUSTOM_LLM_TASK,
        "entrypoint": (
            "python -u -m vllm.entrypoints.openai.api_server "
            f"--model {CUSTOM_LLM_MODEL_ARTIFACT_NAME} "
            f"--served-model-name {SERVED_MODEL_NAME} "
            "--host 0.0.0.0 --port 8080 "
            # The checkpoint carries Qwen3.5's vision tower and this task is
            # text-only. The flag zeroes the per-prompt modality limits; it
            # does NOT skip loading the tower, hence the composite merge.
            "--language-model-only "
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
        # The merged chat template already defaults thinking off; this keeps
        # the example explicit. It must sit at the top level of the request
        # body — vLLM ignores it under extra_body.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.log_params(
            {
                "base_model": merge_metadata["base_model"],
                "adapter_output_dir": merge_metadata["adapter_output_dir"],
                "registered_model_name": FULL_MODEL_NAME,
                "source_training_run_id": merge_metadata["source_run_id"],
                "custom_llm_task": CUSTOM_LLM_TASK,
                "custom_llm_model_artifact": CUSTOM_LLM_MODEL_ARTIFACT_NAME,
                "served_model_name": SERVED_MODEL_NAME,
                "serving_architecture": merge_metadata["architecture"],
                "vllm_dtype": VLLM_DTYPE,
                "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
                "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CustomLlmEntrypointPlaceholder(),
            artifacts={CUSTOM_LLM_MODEL_ARTIFACT_NAME: merge_metadata["merged_model_dir"]},
            input_example=input_example,
            # Deliberately NOT pip_requirements: env_pack captures the
            # vLLM 0.24 + OpenCV 4.12 environment, which no single
            # requirements list can express.
            extra_pip_requirements=MLFLOW_PIP_REQUIREMENTS,
            metadata=metadata,
        )
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=FULL_MODEL_NAME,
            await_registration_for=3600,
            env_pack="databricks_model_serving",
        )

    return {
        "registration_run_id": run.info.run_id,
        "registered_model_name": FULL_MODEL_NAME,
        "model_version": model_version.version,
        "model_uri": model_info.model_uri,
        "source_training_run_id": merge_metadata["source_run_id"],
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = register_custom_llm_model(
    merge_metadata,
    run_name=f"{UC_MODEL_NAME}-registration",
)
REGISTERED_MODEL_VERSION = str(registration_result["model_version"])
# The merged weights are now in the model version; free the node-local copy.
shutil.rmtree(MERGE_WORK_DIR, ignore_errors=True)
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
# MAGIC - `serving_workload_type` selects a documented custom LLM GPU class such as `GPU_SMALL`, `GPU_MEDIUM`, `GPU_LARGE`, or `GPU_XLARGE`.
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
# MAGIC The endpoint serves the fine-tuned model behind the OpenAI-compatible chat contract (`/serving-endpoints/<endpoint_name>/invocations`). Thinking is off by default in the served chat template; requests may still send `chat_template_kwargs: {"enable_thinking": false}` at the **top level** of the body (nesting it under `extra_body` is silently ignored by vLLM).
# MAGIC
# MAGIC - Load test it with `load_test/load_test_serving_endpoint.py` after pointing `serving_load_test.yaml`'s `endpoint_name` at this endpoint; its default `disable_thinking: true` matches this contract.
# MAGIC - Rerunning this notebook after a new training run re-selects the best run (or honors `run_id`), registers a new model version, and rolls the endpoint to it.
