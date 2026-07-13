# Databricks notebook source
# MAGIC %md
# MAGIC ## Install notebook requirements
# MAGIC
# MAGIC Install the Python packages required by the notebook.
# MAGIC AI Runtime already includes many common AI and ML libraries; this cell makes the notebook reproducible when package versions need to be pinned for the project.

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# training_utils is a plain Python module (not a notebook) so the same file
# can be imported here, by train.py, and under the AI Runtime CLI. Put this
# notebook's directory on sys.path first; NOTEBOOK_DIR is reused inside the
# @distributed cell so GPU workers can import train.py the same way.
import sys
from pathlib import Path

NOTEBOOK_DIR = str(Path.cwd())
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

from training_utils import init_training_workspace, load_training_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training configuration
# MAGIC
# MAGIC Training, registration, and serving settings are loaded from the `training_config` section of `train/train.yaml` — the same file that defines the AI Runtime CLI workload, so the notebook and CLI launch paths share one configuration.
# MAGIC This keeps the notebook body stable while making the experiment easy to tune:
# MAGIC
# MAGIC - `catalog`, `schema`, and `source_table` point to the governed transaction Delta table.
# MAGIC - `sft_table` points to the prepared prompt/response Delta table.
# MAGIC - `checkpoint_volume` controls where adapters and model artifacts are written.
# MAGIC - `model_volume_path` (optional) points at a Unity Catalog volume snapshot of the base model weights, populated by `setup/03_download_base_model_weights.py`; when set, the GPU workers load the model from the volume instead of downloading it from Hugging Face. Volume-hosted weights are first staged to node-local disk (once per node) because safetensors mmap reads through the volume FUSE mount are slow.
# MAGIC - `max_steps`, batch size, learning rate, and the LoRA settings control the training cost and quality.
# MAGIC - `training_sample_fraction` controls how much of the staged SFT data is used; the training cell reads it from the config and can override it inline for quick smoke runs.
# MAGIC - `notebook_gpus` / `notebook_gpu_type` size the `@distributed` training cell. Start with 1 GPU to validate the workflow, then raise `notebook_gpus` to scale out — the training code is unchanged.
# MAGIC
# MAGIC For a quick validation run, keep `max_steps` low. For a real fine-tune, increase `max_steps` (or train on the full data), and compare runs in MLflow.

# COMMAND ----------

import json
import sys
from pathlib import Path

import pandas as pd

# COMMAND ----------

# load_training_config (defined in training_utils) parses train.yaml's training_config
# section, derives the UC names/paths, and returns one flat dict; binding it
# into globals gives every later cell the same constants train.py uses.
training_context = load_training_config()
globals().update(training_context)

print(f"Training config: {CONFIG_PATH}")
print(f"Source table: {SOURCE_TABLE}")
print(f"SFT table: {SFT_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Base weights source: {MODEL_LOAD_PATH}")
print(f"Training output dir: {TRAINING_OUTPUT_DIR}")
print(f"Register model: {REGISTER_MODEL}")
print(f"Deploy endpoint: {DEPLOY_ENDPOINT}")
print(f"Serving endpoint: {ENDPOINT_NAME}")
print(f"Serving workload: {SERVING_WORKLOAD_TYPE} @ provisioned concurrency {SERVING_PROVISIONED_CONCURRENCY}")

# COMMAND ----------

spark = init_training_workspace(training_context)

print(f"Ready: {schema_q}")
print(f"Ready: {volume_q}")
print(f"SFT table: {sft_table_q}")

# COMMAND ----------

# DBTITLE 1,AI Runtime fraud fine-tuning with Qwen3 4B and Unsloth
# MAGIC %md
# MAGIC # Fine-tune Qwen3 4B for fraud decisions with AI Runtime
# MAGIC
# MAGIC ![](./images/air-finetuning-workflow.png)
# MAGIC
# MAGIC This notebook shows how to fine-tune a small language model for real-time credit-card fraud decisions on Databricks AI Runtime. 
# MAGIC
# MAGIC The workflow uses the IBM TabFormer credit-card dataset prepared by the setup notebooks: `setup/01_load_tabformer_dataset.py` writes the cleaned transaction table, and `setup/02_stage_training_data.py` builds the supervised fine-tuning table with prompt/response records and stages it in a Unity Catalog volume. This notebook samples or shards those SFT rows, fine-tunes with Unsloth LoRA, logs with MLflow, and optionally registers the model to Unity Catalog for serving.
# MAGIC
# MAGIC **Features demonstrated in this notebook**
# MAGIC
# MAGIC - **On-demand GPU access:** run deep learning workloads on serverless GPU compute without provisioning or maintaining GPU clusters.
# MAGIC - **Managed AI environment:** use the AI Runtime base environment with common model-training libraries already available.
# MAGIC - **Unified data and governance:** read source transactions from Unity Catalog Delta tables and write checkpoints, adapters, and models to governed Unity Catalog assets.
# MAGIC - **Simple scaling path:** start with `@distributed(gpus=1)`, then change that single decorator parameter to use multiple GPUs while the training code stays the same.
# MAGIC - **Operational handoff:** use MLflow and Unity Catalog to move from experimentation toward managed custom LLM serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business scenario and model contract
# MAGIC Fraud detection is a high-volume, low-latency decision problem. A production payment system needs a clear response for each transaction: approve it, ask for additional authentication, or decline and escalate it. We will finetune Qwen3-4B-Instruct-2507` to emit a structured fraud decision with additional triage steps.
# MAGIC
# MAGIC ![](./images/fraud-decision-contract.png)
# MAGIC
# MAGIC The output contract is a compact JSON object with:
# MAGIC
# MAGIC - `risk`: `legitimate`, `suspicious`, or `likely_fraud`
# MAGIC - `action`: downstream routing guidance
# MAGIC - `reason`: a short analyst-facing explanation
# MAGIC
# MAGIC Keeping the response schema explicit makes the model easier to evaluate, serve, and integrate into downstream applications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute: attach to AI Runtime serverless GPU
# MAGIC
# MAGIC Attach this notebook to **Serverless GPU** from the notebook compute picker and choose the **AI v5** environment.
# MAGIC AI Runtime is designed for deep learning workloads on Databricks serverless GPU compute, so the notebook can focus on model development instead of cluster provisioning, driver setup, or GPU library management.
# MAGIC
# MAGIC Recommended compute:
# MAGIC
# MAGIC - Accelerator: `1xH100` or `1xA10` for the validation path, or `8xH100` to demonstrate multi-GPU scaling.
# MAGIC - Base environment: `AI v5`.
# MAGIC
# MAGIC If `1xH100` is not available in the workspace, `1xA10` is enough for this 4B bf16 LoRA workflow.
# MAGIC The model is intentionally small so the notebook highlights the platform workflow: governed data, GPU-backed training, experiment tracking, and production handoff.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and summarize the fraud data
# MAGIC
# MAGIC Start by summarizing the transaction table and the prepared SFT table.
# MAGIC Fraud datasets are typically highly imbalanced, so the row count, fraud count, fraud rate, time range, and SFT shard coverage provide useful context before any modeling work starts.
# MAGIC
# MAGIC This step also verifies that the ingestion notebook has successfully loaded the data before GPU time is used for training.

# COMMAND ----------

display(spark.table(sft_table_q).select('fraud_label', 'is_fraud', 'amount_usd', 'user_id_text', 'card_id_text', 'transaction_ts_text', 'merchant_city_text', 'merchant_state_text', 'mcc_text', 'errors_text', 'has_error_signal').limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="./images/unsloth_green_sticker_cME6ryC59BlZg-VtqGN4p.avif" alt="drawing" width="200"/>
# MAGIC
# MAGIC ## Fine-tune Qwen3 4B Instruct with Unsloth
# MAGIC
# MAGIC This section fine-tunes `unsloth/Qwen3-4B-Instruct-2507` with LoRA adapters.
# MAGIC The Instruct-2507 variant is non-thinking: it answers directly instead of emitting reasoning tokens first, which keeps served responses inside the compact JSON contract and the per-request generation budget. (Base Qwen3 is a hybrid reasoning model whose serving-time chat template defaults to thinking mode.)
# MAGIC It uses bf16/16-bit LoRA for accuracy; the 4B model fits comfortably in GPU memory without quantization.
# MAGIC
# MAGIC The implementation highlights the production workflow around training:
# MAGIC
# MAGIC - MLflow records parameters, metrics, and run metadata.
# MAGIC - Checkpoints and adapters are saved to a Unity Catalog volume.
# MAGIC - Model registration is handled in a separate section after training completes.
# MAGIC - GPU memory metrics are logged when CUDA is available, which helps compare the `gpus=1` and `gpus>1` runs.
# MAGIC
# MAGIC The training implementation lives in `train/train.py`, a plain Python module shared by two launchers: this notebook's `@distributed` cell and the AI Runtime CLI (`air run --file train.yaml`), which runs the same file standalone on serverless GPUs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale training by changing one config value
# MAGIC
# MAGIC This is the only training cell in the pipeline: a thin wrapper that imports `train.py` on each GPU worker and runs one rank of training.
# MAGIC Run it first with `notebook_gpus: 1` to validate the workflow, then raise `notebook_gpus` in `train.yaml` (for example to `8`) and rerun the same cell to distribute training across multiple GPUs.
# MAGIC `TRAINING_SAMPLE_FRACTION` comes from `train.yaml`'s `training_sample_fraction`; override it inline below for a quick smoke run without editing the config.
# MAGIC
# MAGIC Each worker reads its rank-assigned `shard_id=N` parquet directories from the UC volume inside `run_rank_training`, so nothing large ships from the notebook driver to the GPU workers.
# MAGIC The same function runs without a notebook through the AI Runtime CLI: `air run --file train.yaml` executes `python train.py` on serverless GPUs.

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

# The experiment lands under the current user's workspace folder, named by
# train.yaml's experiment_name — the same experiment AIR CLI runs resolve to.
current_user = WorkspaceClient().current_user.me().user_name
mlflow.set_experiment(f"/Users/{current_user}/{EXPERIMENT_NAME}")

# COMMAND ----------

from serverless_gpu import distributed

# From train.yaml's training_sample_fraction; override inline (e.g. 0.01) for
# a quick smoke run without editing the config.
TRAINING_SAMPLE_FRACTION = training_context["TRAINING_SAMPLE_FRACTION"]

@distributed(gpus=NOTEBOOK_GPUS, gpu_type=NOTEBOOK_GPU_TYPE)
def run_training_job():
    import sys

    if NOTEBOOK_DIR not in sys.path:
        sys.path.insert(0, NOTEBOOK_DIR)

    from train import run_rank_training

    return run_rank_training(sample_fraction=TRAINING_SAMPLE_FRACTION)

distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_ADAPTER_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/{TRAINING_WORLD_SIZE}gpu"

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained adapter output dir: {TRAINED_ADAPTER_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the trained model for custom LLM serving
# MAGIC
# MAGIC Registration is separate from training so the distributed training cell stays focused on GPU optimization and adapter checkpointing.
# MAGIC This cell loads the rank-0 adapter artifacts saved by training, merges them with the base model, packages the merged Hugging Face weights into an MLflow model artifact, and registers that model to Unity Catalog.
# MAGIC
# MAGIC Databricks custom LLM serving runs a vLLM OpenAI-compatible server from a custom MLflow entrypoint. The important serving choices are visible below:
# MAGIC
# MAGIC - `task` is `llm/v1/chat`, matching the chat request contract used by the serving endpoint.
# MAGIC - The vLLM process listens on port `8080`, which is the port Model Serving expects.
# MAGIC - The entrypoint launches from the MLflow model's `artifacts/` folder, so the `--model` path is the bare artifact name relative to that folder.
# MAGIC - Registration uses `env_pack="databricks_model_serving"` so Databricks can build the express serving environment.
# MAGIC - The serving container installs its packages from `train.yaml`'s `serving_pip_requirements`, not from `requirements.txt`. The pinned `vllm==0.11.0` + `transformers<5` + `opencv-python-headless==4.12.0.88` combination is what runs on Model Serving's FIPS-enabled pods, and the base model's architecture must be in that vLLM's supported model list (a preflight check below prints the architecture to verify).
# MAGIC
# MAGIC Keeping registration as a separate step also makes reruns cheaper: if training succeeds but registration or deployment fails, rerun only this section.

# COMMAND ----------

from train import load_unsloth_model

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

    mlflow.set_registry_uri("databricks-uc")

    # Defined inline (not in train.py/training_utils.py) on purpose: cloudpickle
    # serializes notebook-local classes BY VALUE, so the serving container can
    # unpickle the model without any repo code and no code_paths are needed in
    # log_model. If this class ever moves into a module or imports repo helpers,
    # registration must add code_paths=["train.py", "training_utils.py"].
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
            # All fraud prompts share the same instruction header, so prefix
            # caching skips most prefill work (explicit for visibility; the
            # vLLM v1 engine defaults it on).
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

        model, tokenizer = load_unsloth_model(adapter_output_dir)

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
            "the supported-models list of the vLLM pinned in train.yaml's "
            "serving_pip_requirements before deploying."
        )

        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            mlflow.log_params(
                {
                    "base_model": MODEL_NAME,
                    "adapter_output_dir": adapter_output_dir,
                    "registered_model_name": FULL_MODEL_NAME,
                    "source_training_run_id": TRAINING_RUN_ID,
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
                # The serving container's packages come from train.yaml's
                # serving_pip_requirements — the FIPS-safe vLLM combination;
                # see the comments there before changing pins or base model.
                pip_requirements=SERVING_PIP_REQUIREMENTS,
                metadata=metadata,
            )
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=FULL_MODEL_NAME,
                await_registration_for=3600,
                env_pack="databricks_model_serving"

            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "registration_run_id": run.info.run_id,
        "registered_model_name": FULL_MODEL_NAME,
        "model_version": model_version.version,
        "model_uri": model_info.model_uri,
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = None
REGISTERED_MODEL_VERSION = None

if REGISTER_MODEL:
    if "TRAINED_ADAPTER_OUTPUT_DIR" not in globals() or not TRAINED_ADAPTER_OUTPUT_DIR:
        raise ValueError("Run the training cell before registering the model.")

    registration_result = register_custom_llm_model(
        adapter_output_dir=TRAINED_ADAPTER_OUTPUT_DIR,
        run_name=f"{TRAINING_RUN_NAME}-registration",
    )
    REGISTERED_MODEL_VERSION = str(registration_result["model_version"])
    display(pd.DataFrame([registration_result]))
else:
    print("Model registration skipped because register_model is false in train.yaml.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the custom LLM endpoint
# MAGIC
# MAGIC This cell creates or updates a Mosaic AI Model Serving endpoint for the registered custom LLM.
# MAGIC The deployment uses the Databricks SDK so the demo can be run end to end from the notebook instead of switching to the UI.
# MAGIC
# MAGIC The endpoint configuration is controlled by the `training_config` section of `train.yaml`:
# MAGIC
# MAGIC - `endpoint_name` is the serving endpoint name used by the load-test notebook.
# MAGIC - `serving_workload_type` selects the GPU class, such as `GPU_MEDIUM` for A10 or `GPU_XLARGE` for H100.
# MAGIC - `serving_provisioned_concurrency` sets the fixed provisioned capacity behind the endpoint (custom LLM serving does not autoscale during beta — size for peak traffic).
# MAGIC - `serving_scale_to_zero` is useful for development, but should be disabled for latency-sensitive production traffic.
# MAGIC
# MAGIC The served entity also sets `VLLM_USE_FLASHINFER_SAMPLER=0`: the serving container cannot JIT-compile FlashInfer kernels (no `ninja`/`nvcc`), so vLLM must use its native PyTorch sampler.
# MAGIC
# MAGIC Custom LLM serving is currently a fixed-capacity serving path during beta. Size the workload for the traffic target before running a high-QPS load test.

# COMMAND ----------

def served_entity_name_for_version(model_name: str, version: str) -> str:
    clean_name = model_name.rsplit(".", 1)[-1].replace("_", "-").replace(".", "-")
    return f"{clean_name}-{version}"[:64]


def create_or_update_custom_llm_endpoint(model_version: str) -> dict:
    if SERVING_WORKLOAD_TYPE == "GPU_XLARGE" and SERVING_SCALE_TO_ZERO:
        raise ValueError(
            "Custom LLM serving beta does not support scale-to-zero for GPU_XLARGE. "
            "Set serving_scale_to_zero: false in train.yaml."
        )

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
    workload_type = ServingModelWorkloadType(SERVING_WORKLOAD_TYPE)
    served_entity_name = served_entity_name_for_version(FULL_MODEL_NAME, model_version)
    served_entity_kwargs = dict(
        name=served_entity_name,
        entity_name=FULL_MODEL_NAME,
        entity_version=str(model_version),
        workload_type=workload_type,
        # Fixed capacity — custom LLM serving does not autoscale during beta.
        min_provisioned_concurrency=SERVING_PROVISIONED_CONCURRENCY,
        max_provisioned_concurrency=SERVING_PROVISIONED_CONCURRENCY,
        environment_vars={
            # The serving container has no ninja/nvcc, so FlashInfer (shipped in
            # the Databricks AI base env) cannot JIT-compile its sampling kernels
            # at startup; fall back to vLLM's native PyTorch sampler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
    )
    if SERVING_WORKLOAD_SIZE:
        served_entity_kwargs["workload_size"] = SERVING_WORKLOAD_SIZE
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
        "workload_size": SERVING_WORKLOAD_SIZE or None,
        "provisioned_concurrency": SERVING_PROVISIONED_CONCURRENCY,
        "endpoint_ready": str(getattr(endpoint_state, "ready", None)),
        "config_update": str(getattr(endpoint_state, "config_update", None)),
    }


deployment_result = None

if DEPLOY_ENDPOINT:
    if not REGISTERED_MODEL_VERSION:
        raise ValueError("Deployment requires register_model: true so a model version is available.")

    deployment_result = create_or_update_custom_llm_endpoint(REGISTERED_MODEL_VERSION)
    display(pd.DataFrame([deployment_result]))
else:
    print("Endpoint deployment skipped because deploy_endpoint is false in train.yaml.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query payload for validation and load testing
# MAGIC
# MAGIC The request payload keeps the same prompt contract used during fine-tuning: ask for compact JSON with `risk`, `action`, and `reason`.
# MAGIC This keeps training, serving validation, and the load-test notebook aligned around the same interface.

# COMMAND ----------

# Sample a real prompt from the SFT table so the payload always matches the
# prompt contract the model was fine-tuned on — no hardcoded copy to drift.
sample_prompt_rows = spark.table(sft_table_q).select("prompt").limit(1).collect()
if not sample_prompt_rows:
    raise ValueError(f"No rows found in {SFT_TABLE}. Run the setup notebooks first.")
sample_transaction_prompt = sample_prompt_rows[0]["prompt"]

serving_payload = {
    "messages": [
        {
            "role": "user",
            "content": sample_transaction_prompt,
        }
    ],
    "max_tokens": 64,
    "temperature": 0.0,
}

print(f"Registered model name: {FULL_MODEL_NAME}")
print(f"Serving endpoint name: {ENDPOINT_NAME}")
print(json.dumps(serving_payload, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrates an end-to-end AI Runtime fine-tuning workflow for fraud decisions:
# MAGIC
# MAGIC - Ingested transactions are governed in Unity Catalog.
# MAGIC - Supervised chat records are generated from real table rows during ingestion and stored in the prepared SFT Delta table.
# MAGIC - AI Runtime provides managed serverless GPU compute for model training.
# MAGIC - The same training cell supports `gpus=1` validation and a scaled multi-GPU path.
# MAGIC - MLflow captures the experiment record, Unity Catalog stores the registered model version, and the deployment cell creates or updates a custom LLM serving endpoint.
# MAGIC
# MAGIC The main platform outcome is speed with control: teams can move from governed data to GPU fine-tuning to registered model artifacts without leaving Databricks or stitching together separate infrastructure.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
# MAGIC - Serverless GPU H100 starter: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-api-h100-starter
# MAGIC - Databricks Unsloth example: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth
# MAGIC - Custom LLM serving with vLLM: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms
# MAGIC - Unsloth Qwen3: https://unsloth.ai/docs/models/qwen3
# MAGIC - Unsloth Qwen3 fine-tuning: https://unsloth.ai/docs/models/qwen3/fine-tune
