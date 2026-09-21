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

# Bootstrap only: a notebook's own folder is not reliably on sys.path, so this
# has to run before any project module can be imported.
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_config import DeployConfig, load_deploy_config

DEPLOY = load_deploy_config()

print(f"Deploy config: {DEPLOY.config_path} (parameters.deploy_config)")
print(f"Registered model target: {DEPLOY.full_model_name}")
print(f"Serving endpoint: {DEPLOY.endpoint_name}")
print(f"Inference payload table: {DEPLOY.inference_payload_table}")
print(
    "Run selection: "
    + (
        f"run_id={DEPLOY.run_id}"
        if DEPLOY.run_id
        else f"best {DEPLOY.best_run_metric} ({DEPLOY.best_run_metric_goal}) "
        f"in {DEPLOY.experiment_path}"
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


def select_source_run(deploy: DeployConfig):
    """Return (run, selection_reason, metric_value) for the run to deploy."""
    experiment = mlflow.get_experiment_by_name(deploy.experiment_path)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {deploy.experiment_path}. Run "
            "training first or correct experiment_path in train.yaml."
        )

    if deploy.run_id:
        source_run = mlflow.get_run(deploy.run_id)
        return (
            source_run,
            "run_id from train.yaml's deploy_config",
            source_run.data.metrics.get(deploy.best_run_metric),
        )

    metric_order = "ASC" if deploy.best_run_metric_goal == "minimize" else "DESC"
    runs_pdf = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.`{deploy.best_run_metric}` {metric_order}"],
    )
    metric_column = f"metrics.{deploy.best_run_metric}"
    output_column = "params.model_output_dir"
    if (
        runs_pdf.empty
        or metric_column not in runs_pdf.columns
        or output_column not in runs_pdf.columns
    ):
        raise ValueError(
            f"No finished runs in {deploy.experiment_path} logged both "
            f"{deploy.best_run_metric!r} and model_output_dir."
        )
    candidates = runs_pdf[
        runs_pdf[metric_column].notna() & runs_pdf[output_column].notna()
    ]
    if candidates.empty:
        raise ValueError(
            f"No finished run in {deploy.experiment_path} has a complete checkpoint."
        )
    best_row = candidates.iloc[0]
    return (
        mlflow.get_run(best_row["run_id"]),
        f"best {deploy.best_run_metric} ({deploy.best_run_metric_goal}) of "
        f"{len(candidates)} candidate run(s)",
        best_row[metric_column],
    )


def validate_checkpoint(model_output_dir: str) -> Path:
    """Confirm the volume holds a reloadable Hugging Face checkpoint."""
    model_output_path = Path(model_output_dir)
    if not model_output_path.exists():
        raise FileNotFoundError(f"Training checkpoint does not exist: {model_output_dir}")
    required_files = [
        model_output_path / "config.json",
        model_output_path / "tokenizer_config.json",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files or not list(model_output_path.glob("*.safetensors")):
        raise ValueError(
            f"Checkpoint at {model_output_dir} is incomplete; missing "
            f"{missing_files} or no safetensors weights were found."
        )
    return model_output_path


source_run, selection_reason, selection_metric_value = select_source_run(DEPLOY)
SOURCE_RUN_ID = source_run.info.run_id
MODEL_OUTPUT_DIR = source_run.data.params.get("model_output_dir")
if not MODEL_OUTPUT_DIR:
    raise ValueError(
        f"Run {SOURCE_RUN_ID} has no model_output_dir parameter. Pick a "
        "completed rank-zero full-weight training run."
    )
validate_checkpoint(MODEL_OUTPUT_DIR)

display(
    pd.DataFrame(
        [
            {
                "source_run_id": SOURCE_RUN_ID,
                "run_name": source_run.info.run_name,
                "selection": selection_reason,
                DEPLOY.best_run_metric: selection_metric_value,
                "model_output_dir": MODEL_OUTPUT_DIR,
                "training_scope": source_run.data.params.get("training_scope"),
                "experiment": DEPLOY.experiment_path,
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


def transformers_entrypoint(deploy: DeployConfig, model_artifact_name: str) -> str:
    """The serving command. Receives the bare MLflow artifact name."""
    command = [
        "transformers serve",
        model_artifact_name,
        "--host 0.0.0.0",
        "--port 8080",
        f"--dtype {deploy.serving_dtype}",
        f"--reasoning {deploy.serving_reasoning}",
    ]
    if deploy.serving_continuous_batching:
        command.append("--continuous-batching")
    return " ".join(command)


def register_custom_llm_model(
    deploy: DeployConfig, model_output_dir: str, source_run_id: str, run_name: str
) -> dict:
    from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

    mlflow.set_registry_uri("databricks-uc")
    model_artifact_name = deploy.uc_model_name

    # Keep this placeholder inline so cloudpickle serializes it by value. The
    # entrypoint server, not predict(), handles serving requests.
    class CustomLlmEntrypointPlaceholder(ChatModel):
        def predict(self, context, messages, params):
            return ChatCompletionResponse.from_dict({"choices": []})

    metadata = {
        "task": CUSTOM_LLM_TASK,
        "entrypoint": transformers_entrypoint(deploy, model_artifact_name),
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
                "source_training_run_id": source_run_id,
                "model_output_dir": model_output_dir,
                "registered_model_name": deploy.full_model_name,
                "custom_llm_task": CUSTOM_LLM_TASK,
                "custom_llm_model_artifact": model_artifact_name,
                "serving_engine": "transformers",
                "serving_dtype": deploy.serving_dtype,
                "serving_reasoning": deploy.serving_reasoning,
                "serving_continuous_batching": deploy.serving_continuous_batching,
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CustomLlmEntrypointPlaceholder(),
            artifacts={model_artifact_name: model_output_dir},
            input_example=input_example,
            pip_requirements=deploy.serving_pip_requirements,
            metadata=metadata,
        )
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=deploy.full_model_name,
            await_registration_for=3600,
            env_pack="databricks_model_serving",
        )

    return {
        "registration_run_id": run.info.run_id,
        "registered_model_name": deploy.full_model_name,
        "model_version": model_version.version,
        "model_uri": model_info.model_uri,
        "source_training_run_id": source_run_id,
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = register_custom_llm_model(
    DEPLOY,
    model_output_dir=MODEL_OUTPUT_DIR,
    source_run_id=SOURCE_RUN_ID,
    run_name=f"{DEPLOY.uc_model_name}-registration",
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


def _enable_inference_table(w, deploy: DeployConfig):
    """Turn on AI Gateway payload logging, preserving other gateway settings."""
    from databricks.sdk.service.serving import (
        AiGatewayInferenceTableConfig,
        AiGatewayUsageTrackingConfig,
    )

    current_gateway = getattr(w.serving_endpoints.get(deploy.endpoint_name), "ai_gateway", None)
    requested_inference_table = AiGatewayInferenceTableConfig(
        catalog_name=deploy.uc_catalog,
        schema_name=deploy.uc_schema,
        table_name_prefix=deploy.inference_table_prefix,
        enabled=True,
    )
    gateway_response = w.serving_endpoints.put_ai_gateway(
        name=deploy.endpoint_name,
        fallback_config=getattr(current_gateway, "fallback_config", None),
        guardrails=getattr(current_gateway, "guardrails", None),
        inference_table_config=requested_inference_table,
        rate_limits=getattr(current_gateway, "rate_limits", None),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    )

    configured_inference_table = getattr(gateway_response, "inference_table_config", None)
    if configured_inference_table is None:
        refreshed_gateway = getattr(
            w.serving_endpoints.get(deploy.endpoint_name), "ai_gateway", None
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
            f"Inference table configuration failed for {deploy.endpoint_name}: "
            f"expected {expected_config}, got {actual_config}"
        )
    return configured_inference_table


def create_or_update_custom_llm_endpoint(deploy: DeployConfig, model_version: str) -> dict:
    from datetime import timedelta

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, ResourceDoesNotExist
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        Route,
        ServedEntityInput,
        ServingModelWorkloadType,
        TrafficConfig,
    )

    w = WorkspaceClient()
    served_entity_name = served_entity_name_for_version(
        deploy.full_model_name, model_version
    )
    served_entity = ServedEntityInput(
        name=served_entity_name,
        entity_name=deploy.full_model_name,
        entity_version=str(model_version),
        workload_type=ServingModelWorkloadType(deploy.serving_workload_type),
        workload_size=deploy.serving_workload_size,
        scale_to_zero_enabled=deploy.serving_scale_to_zero,
        environment_vars={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    traffic_config = TrafficConfig(
        routes=[Route(served_entity_name=served_entity_name, traffic_percentage=100)]
    )

    try:
        w.serving_endpoints.get(deploy.endpoint_name)
        endpoint = w.serving_endpoints.update_config_and_wait(
            name=deploy.endpoint_name,
            served_entities=[served_entity],
            traffic_config=traffic_config,
            timeout=timedelta(minutes=60),
        )
        deployment_action = "updated"
    except (NotFound, ResourceDoesNotExist):
        endpoint = w.serving_endpoints.create_and_wait(
            name=deploy.endpoint_name,
            config=EndpointCoreConfigInput(
                name=deploy.endpoint_name,
                served_entities=[served_entity],
                traffic_config=traffic_config,
            ),
            description=deploy.endpoint_description,
            timeout=timedelta(minutes=60),
        )
        deployment_action = "created"

    configured_inference_table = _enable_inference_table(w, deploy)

    workspace_url = (w.config.host or "").rstrip("/")
    endpoint_state = getattr(endpoint, "state", None)
    return {
        "deployment_action": deployment_action,
        "endpoint_name": deploy.endpoint_name,
        "endpoint_url": (
            f"{workspace_url}/serving-endpoints/{deploy.endpoint_name}"
            if workspace_url
            else f"/serving-endpoints/{deploy.endpoint_name}"
        ),
        "registered_model_name": deploy.full_model_name,
        "model_version": str(model_version),
        "served_entity_name": served_entity_name,
        "workload_type": deploy.serving_workload_type,
        "workload_size": deploy.serving_workload_size,
        "scale_to_zero_enabled": deploy.serving_scale_to_zero,
        "inference_table_enabled": configured_inference_table.enabled,
        "inference_payload_table": deploy.inference_payload_table,
        "endpoint_ready": str(getattr(endpoint_state, "ready", None)),
        "config_update": str(getattr(endpoint_state, "config_update", None)),
    }


deployment_result = create_or_update_custom_llm_endpoint(DEPLOY, REGISTERED_MODEL_VERSION)
display(pd.DataFrame([deployment_result]))

# COMMAND ----------

# MAGIC %md
# MAGIC The endpoint now serves the full-weight checkpoint through the
# MAGIC OpenAI-compatible chat contract. Rerunning this notebook after another
# MAGIC training run registers a new version and rolls the endpoint to it.
