# Databricks notebook source
# DBTITLE 1,Register and deploy the full-weight Qwen3.6 model
# MAGIC %md
# MAGIC # Register and deploy the full-weight Qwen3.6 model
# MAGIC
# MAGIC This project-local notebook selects a completed full-weight training
# MAGIC run, packages its `model_output_dir` checkpoint directly as a custom
# MAGIC LLM, registers it in Unity Catalog, and creates or updates a Mosaic AI
# MAGIC Model Serving endpoint with AI Gateway inference tables enabled.
# MAGIC
# MAGIC Run selection uses this directory's `train.yaml` `deploy_config`:
# MAGIC
# MAGIC - Set `run_id` to deploy one exact training run.
# MAGIC - Leave it empty to select the best finished run by
# MAGIC   `best_run_metric` and `best_run_metric_goal`.
# MAGIC
# MAGIC Qwen3.6 requires a newer engine than the repository's vLLM 0.11 stack.
# MAGIC This module therefore uses the Transformers 5 OpenAI-compatible server,
# MAGIC pinned in its own `requirements.txt`, and forces non-thinking mode for
# MAGIC the fraud-classification contract.
# MAGIC
# MAGIC Reference: [Serve custom LLMs with Custom Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms).

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import sys
from pathlib import Path

import pandas as pd

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_config import load_deploy_config

deploy_context = load_deploy_config()
globals().update(deploy_context)

print(f"Deploy config: {DEPLOY_CONFIG_PATH} (parameters.deploy_config)")
print(f"Registered model target: {FULL_MODEL_NAME}")
print(f"Serving endpoint: {ENDPOINT_NAME}")
print(
    "Inference payload table: "
    f"{UC_CATALOG}.{UC_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload"
)
print(
    "Run selection: "
    + (
        f"run_id={RUN_ID}"
        if RUN_ID
        else f"best {BEST_RUN_METRIC} ({BEST_RUN_METRIC_GOAL}) in {EXPERIMENT_PATH}"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select a complete training checkpoint
# MAGIC
# MAGIC Training logs `model_output_dir` only after rank zero has copied the
# MAGIC assembled FSDP checkpoint to the UC volume. Runs without that parameter
# MAGIC are not deployable and are excluded from automatic selection.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
if experiment is None:
    raise ValueError(
        f"MLflow experiment not found: {EXPERIMENT_PATH}. Run training first "
        "or correct experiment_path in train.yaml."
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
    output_column = "params.model_output_dir"
    if (
        runs_pdf.empty
        or metric_column not in runs_pdf.columns
        or output_column not in runs_pdf.columns
    ):
        raise ValueError(
            f"No finished runs in {EXPERIMENT_PATH} logged both "
            f"{BEST_RUN_METRIC!r} and model_output_dir."
        )
    candidates = runs_pdf[
        runs_pdf[metric_column].notna() & runs_pdf[output_column].notna()
    ]
    if candidates.empty:
        raise ValueError(
            f"No finished run in {EXPERIMENT_PATH} has a complete checkpoint."
        )
    best_row = candidates.iloc[0]
    source_run = mlflow.get_run(best_row["run_id"])
    selection_reason = (
        f"best {BEST_RUN_METRIC} ({BEST_RUN_METRIC_GOAL}) of "
        f"{len(candidates)} candidate run(s)"
    )
    selection_metric_value = best_row[metric_column]

SOURCE_RUN_ID = source_run.info.run_id
MODEL_OUTPUT_DIR = source_run.data.params.get("model_output_dir")
if not MODEL_OUTPUT_DIR:
    raise ValueError(
        f"Run {SOURCE_RUN_ID} has no model_output_dir parameter. Pick a "
        "completed rank-zero full-weight training run."
    )

model_output_path = Path(MODEL_OUTPUT_DIR)
if not model_output_path.exists():
    raise FileNotFoundError(f"Training checkpoint does not exist: {MODEL_OUTPUT_DIR}")
required_files = [model_output_path / "config.json", model_output_path / "tokenizer_config.json"]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files or not list(model_output_path.glob("*.safetensors")):
    raise ValueError(
        f"Checkpoint at {MODEL_OUTPUT_DIR} is incomplete; missing {missing_files} "
        "or no safetensors weights were found."
    )

display(
    pd.DataFrame(
        [
            {
                "source_run_id": SOURCE_RUN_ID,
                "run_name": source_run.info.run_name,
                "selection": selection_reason,
                BEST_RUN_METRIC: selection_metric_value,
                "model_output_dir": MODEL_OUTPUT_DIR,
                "training_scope": source_run.data.params.get("training_scope"),
                "experiment": EXPERIMENT_PATH,
            }
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the custom LLM
# MAGIC
# MAGIC The complete checkpoint is already merged because this is full-weight
# MAGIC training. The MLflow model packages that directory as one artifact and
# MAGIC starts `transformers serve` from the artifact root on port 8080.
# MAGIC Registration uses `env_pack="databricks_model_serving"` for express
# MAGIC deployment. The entrypoint implements `/v1/chat/completions`, matching
# MAGIC the `llm/v1/chat` task contract.

# COMMAND ----------

CUSTOM_LLM_TASK = "llm/v1/chat"
CUSTOM_LLM_MODEL_ARTIFACT_NAME = UC_MODEL_NAME


def transformers_entrypoint() -> str:
    command = [
        "transformers serve",
        CUSTOM_LLM_MODEL_ARTIFACT_NAME,
        "--host 0.0.0.0",
        "--port 8080",
        f"--dtype {SERVING_DTYPE}",
        f"--reasoning {SERVING_REASONING}",
    ]
    if SERVING_CONTINUOUS_BATCHING:
        command.append("--continuous-batching")
    return " ".join(command)


def register_custom_llm_model(model_output_dir: str, run_name: str) -> dict:
    import mlflow
    from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

    mlflow.set_registry_uri("databricks-uc")

    # Keep this placeholder inline so cloudpickle serializes it by value. The
    # entrypoint server, not predict(), handles serving requests.
    class CustomLlmEntrypointPlaceholder(ChatModel):
        def predict(self, context, messages, params):
            return ChatCompletionResponse.from_dict({"choices": []})

    metadata = {
        "task": CUSTOM_LLM_TASK,
        "entrypoint": transformers_entrypoint(),
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

    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.log_params(
            {
                "source_training_run_id": SOURCE_RUN_ID,
                "model_output_dir": model_output_dir,
                "registered_model_name": FULL_MODEL_NAME,
                "custom_llm_task": CUSTOM_LLM_TASK,
                "custom_llm_model_artifact": CUSTOM_LLM_MODEL_ARTIFACT_NAME,
                "serving_engine": "transformers",
                "serving_dtype": SERVING_DTYPE,
                "serving_reasoning": SERVING_REASONING,
                "serving_continuous_batching": SERVING_CONTINUOUS_BATCHING,
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CustomLlmEntrypointPlaceholder(),
            artifacts={CUSTOM_LLM_MODEL_ARTIFACT_NAME: model_output_dir},
            input_example=input_example,
            pip_requirements=SERVING_PIP_REQUIREMENTS,
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
        "source_training_run_id": SOURCE_RUN_ID,
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = register_custom_llm_model(
    model_output_dir=MODEL_OUTPUT_DIR,
    run_name=f"{UC_MODEL_NAME}-registration",
)
REGISTERED_MODEL_VERSION = str(registration_result["model_version"])
display(pd.DataFrame([registration_result]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy with inference tables
# MAGIC
# MAGIC The endpoint uses the `GPU_LARGE` workload class configured in
# MAGIC `train.yaml`. After each create or update, AI Gateway inference logging
# MAGIC is enabled at
# MAGIC `<catalog>.<schema>.<inference_table_prefix>_payload` for the monitoring
# MAGIC pipeline.

# COMMAND ----------


def served_entity_name_for_version(model_name: str, version: str) -> str:
    clean_name = model_name.rsplit(".", 1)[-1].replace("_", "-").replace(".", "-")
    return f"{clean_name}-{version}"[:64]


def create_or_update_custom_llm_endpoint(model_version: str) -> dict:
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
    served_entity = ServedEntityInput(
        name=served_entity_name,
        entity_name=FULL_MODEL_NAME,
        entity_version=str(model_version),
        workload_type=workload_type,
        workload_size=SERVING_WORKLOAD_SIZE,
        scale_to_zero_enabled=SERVING_SCALE_TO_ZERO,
        environment_vars={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
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
        refreshed_gateway = getattr(
            w.serving_endpoints.get(ENDPOINT_NAME), "ai_gateway", None
        )
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

    workspace_url = (w.config.host or "").rstrip("/")
    endpoint_state = getattr(endpoint, "state", None)
    return {
        "deployment_action": deployment_action,
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_url": (
            f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}"
            if workspace_url
            else f"/serving-endpoints/{ENDPOINT_NAME}"
        ),
        "registered_model_name": FULL_MODEL_NAME,
        "model_version": str(model_version),
        "served_entity_name": served_entity_name,
        "workload_type": SERVING_WORKLOAD_TYPE,
        "workload_size": SERVING_WORKLOAD_SIZE,
        "scale_to_zero_enabled": SERVING_SCALE_TO_ZERO,
        "inference_table_enabled": configured_inference_table.enabled,
        "inference_payload_table": (
            f"{UC_CATALOG}.{UC_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload"
        ),
        "endpoint_ready": str(getattr(endpoint_state, "ready", None)),
        "config_update": str(getattr(endpoint_state, "config_update", None)),
    }


deployment_result = create_or_update_custom_llm_endpoint(REGISTERED_MODEL_VERSION)
display(pd.DataFrame([deployment_result]))

# COMMAND ----------

# MAGIC %md
# MAGIC The endpoint now serves the full-weight checkpoint through the
# MAGIC OpenAI-compatible chat contract. Rerunning this notebook after another
# MAGIC training run registers a new version and rolls the endpoint to it.
