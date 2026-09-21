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

from training_utils import (
    init_training_workspace,
    load_training_config,
    resolve_experiment_path,
)

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
# MAGIC - The workload's top-level `experiment_name` names the MLflow experiment, so this notebook and AI Runtime CLI runs log to the same place.
# MAGIC - `max_steps`, batch size, and learning rate control the training cost and runtime.
# MAGIC - `training_sample_fraction` controls how much of each rank's shard slice is trained on (`1.0` uses every row); this notebook and AI Runtime CLI runs both read it from `train.yaml`.
# MAGIC
# MAGIC The demo uses one training cell. Run it first with `@distributed(gpus=1, gpu_type="h100")`, then change only `gpus` to a larger value such as `8` to distribute the same training workflow.
# MAGIC For a short walkthrough, keep `max_steps` low. For a real experiment, increase `max_steps`, broaden the sampled dataset, and compare runs in MLflow.

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
print(f"MLflow experiment name: {EXPERIMENT_NAME}")
print(f"Training sample fraction: {TRAINING_SAMPLE_FRACTION}")
print(f"Source table: {SOURCE_TABLE}")
print(f"SFT table: {SFT_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Training output dir: {TRAINING_OUTPUT_DIR}")
print(f"Register model: {REGISTER_MODEL}")
print(f"Deploy endpoint: {DEPLOY_ENDPOINT}")
print(f"Serving endpoint: {ENDPOINT_NAME}")
print(f"Serving workload: {SERVING_WORKLOAD_TYPE} / {SERVING_WORKLOAD_SIZE}")

# COMMAND ----------

spark = init_training_workspace(training_context)

print(f"Ready: {schema_q}")
print(f"Ready: {volume_q}")
print(f"SFT table: {sft_table_q}")

# COMMAND ----------

# DBTITLE 1,AI Runtime fraud fine-tuning with Qwen3.5 4B and Hugging Face TRL
# MAGIC %md
# MAGIC # Fine-tune Qwen3.5 4B for fraud decisions with AI Runtime
# MAGIC
# MAGIC ![](/Workspace/Users/ben.doan@databricks.com/dais-air/train/images/Screenshot 2026-06-11 at 12.04.39 PM.png)
# MAGIC
# MAGIC This notebook shows how to fine-tune a small language model for real-time credit-card fraud decisions on Databricks AI Runtime. 
# MAGIC
# MAGIC The workflow uses the IBM TabFormer credit-card dataset loaded and prepared by `setup/01_load_tabformer_dataset.py`.
# MAGIC The setup notebook creates both a cleaned transaction table and a supervised fine-tuning table with prompt/response records. This notebook samples or shards those SFT rows, fine-tunes with Hugging Face TRL supervised fine-tuning and PEFT LoRA, logs with MLflow, and optionally registers the model to Unity Catalog for serving.
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
# MAGIC
# MAGIC Fraud detection is a high-volume, low-latency decision problem. A production payment system needs a clear response for each transaction: approve it, ask for additional authentication, or decline and escalate it. We will finetune `Qwen/Qwen3.5-4B` to emit a structured fraud decision with additional triage steps.
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
# MAGIC Attach this notebook to **Serverless GPU** from the notebook compute picker and choose the **AI v6** environment.
# MAGIC AI Runtime is designed for deep learning workloads on Databricks serverless GPU compute, so the notebook can focus on model development instead of cluster provisioning, driver setup, or GPU library management.
# MAGIC
# MAGIC Recommended compute:
# MAGIC
# MAGIC - Accelerator: `1xH100` or `1xA10` for the validation path, or `8xH100` to demonstrate multi-GPU scaling.
# MAGIC - Base environment: `AI v6` (Public Preview).
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

display(spark.table(sft_table_q).select('fraud_label', 'is_fraud', 'amount_usd', 'user_id_text', 'card_id_text', 'transaction_ts_text', 'merchant_city_text', 'merchant_state_text', 'mcc_text', 'errors_text', 'has_error_signal'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune Qwen3.5 4B with Hugging Face TRL
# MAGIC
# MAGIC This section fine-tunes `Qwen/Qwen3.5-4B` with PEFT LoRA adapters through TRL's `SFTTrainer`.
# MAGIC Qwen3.5 runs in thinking mode by default and ships no non-thinking variant, so every render passes `enable_thinking=False` and the endpoint sends the matching `chat_template_kwargs`. Without that the model emits a reasoning preamble and the compact JSON gets truncated at `max_tokens`.
# MAGIC Only the text backbone is loaded (`Qwen3_5ForCausalLM`); the checkpoint's vision tower is irrelevant to this task, and `--language-model-only` keeps it out of the request path at serving time (it does not skip loading it — see the merge cell).
# MAGIC It uses bf16/16-bit LoRA for accuracy; the 4B model fits comfortably in GPU memory without quantization. Qwen3.5's 3:1 hybrid stack means the adapter lands on the Gated Attention layers' `q/k/v/o_proj` plus every layer's MLP projections — the Gated DeltaNet layers' `linear_attn.*` are left alone.
# MAGIC Loss is computed on the assistant response only: each SFT row becomes a `prompt`/`completion` pair rendered through the chat template, and `completion_only_loss` masks the prompt.
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
# MAGIC ## Scale training by changing one decorator parameter
# MAGIC
# MAGIC This is the only training cell in the demo: a thin wrapper that imports `train.py` on each GPU worker and runs one rank of training.
# MAGIC Run it first with `gpus=1` to validate the workflow, then change the decorator to `gpus=8` and rerun the same cell to distribute training across multiple GPUs.
# MAGIC `train.yaml`'s `training_sample_fraction` controls how much of each rank's shard slice is used — raise it there to broaden the dataset between runs (or pass `--override parameters.training_config.training_sample_fraction=...` to `air run`).
# MAGIC
# MAGIC Each worker reads its rank-assigned `shard_id=N` parquet directories from the UC volume inside `run_rank_training`, so nothing large ships from the notebook driver to the GPU workers.
# MAGIC The same function runs without a notebook through the AI Runtime CLI: `air run --file train.yaml` executes `python train.py` on serverless GPUs.

# COMMAND ----------

# train.yaml's top-level `experiment_name` is the only place the experiment is
# named: the AI Runtime CLI resolves it to /Users/<user>/<experiment_name>, and
# resolve_experiment_path derives the same path here so notebook runs and CLI
# runs share one experiment.
import mlflow

MLFLOW_EXPERIMENT_PATH = resolve_experiment_path(EXPERIMENT_NAME)
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

print(f"MLflow experiment: {MLFLOW_EXPERIMENT_PATH}")

# COMMAND ----------

from serverless_gpu import distributed

# TRAINING_SAMPLE_FRACTION comes from train.yaml's `training_sample_fraction`
# (bound above by load_training_config), so notebook and AI Runtime CLI runs
# train on the same slice of data; set 1.0 there to use every row. Uncomment the
# line below only for a one-off experiment that should not change the config.
# TRAINING_SAMPLE_FRACTION = 0.01

@distributed(gpus=1, gpu_type="h100")
def run_training_job():
    import sys

    if NOTEBOOK_DIR not in sys.path:
        sys.path.insert(0, NOTEBOOK_DIR)

    from train import run_rank_training

    return run_rank_training(
        sample_fraction=TRAINING_SAMPLE_FRACTION,
        experiment_path=MLFLOW_EXPERIMENT_PATH,
    )

distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_ADAPTER_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/{TRAINING_WORLD_SIZE}gpu"

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained adapter output dir: {TRAINED_ADAPTER_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge the trained adapter
# MAGIC
# MAGIC Registration is split into three cells — merge, install the serving stack, register — because the training and serving environments cannot share one Python session.
# MAGIC vLLM requires `opencv-python-headless>=4.13`, whose bundled OpenSSL aborts with `FATAL FIPS SELFTEST FAILURE` on Model Serving's FIPS pods, so opencv must be forced back to `4.12.0.88` in a second `pip` pass. A single `pip_requirements` list is resolved in one pass and cannot express that conflict, so the environment is built here in the notebook and captured by `env_pack` instead.
# MAGIC
# MAGIC This cell merges the rank-0 LoRA adapter into the base weights and writes plain Hugging Face weights to `/local_disk0`, which survives the `%restart_python` below and avoids the EAGAIN failures that large safetensors writes hit on `/Volumes`.
# MAGIC
# MAGIC The merged weights are saved in the base model's **composite (vision + text) shape**, not the text-only shape training used. vLLM 0.24.0 implements `Qwen3_5ForCausalLM` but never registers it, and its architecture normalisation rewrites the `ForCausalLM` suffix until a registered name matches — so a text-only checkpoint silently builds `Qwen3_5ForConditionalGeneration` and crashes with `'Qwen3_5TextConfig' object has no attribute 'vision_config'`. Saving the composite shape makes the architecture match; `--language-model-only` then keeps the vision tower idle, but its weights must still ship because vLLM's loader raises on any parameter missing from the checkpoint.
# MAGIC
# MAGIC Splitting registration from training also makes reruns cheap: if training succeeds but registration or deployment fails, rerun only these cells.

# COMMAND ----------

from train import merge_adapter_to_serving_checkpoint
from training_utils import local_staging_dir

CUSTOM_LLM_MODEL_ARTIFACT_NAME = "qwen35_fraud_model"
# Node-local staging, resolved the same way on both sides of the %restart_python
# below. Not /Volumes: large safetensors writes there have failed with EAGAIN.
MERGE_WORK_ROOT = local_staging_dir("air-demo-merged")
MERGED_MODEL_DIR = MERGE_WORK_ROOT / CUSTOM_LLM_MODEL_ARTIFACT_NAME
MERGE_METADATA_PATH = MERGE_WORK_ROOT / "merge_metadata.json"


def merge_adapter_to_local_disk(adapter_output_dir: str) -> Path:
    import shutil

    if MERGED_MODEL_DIR.exists():
        shutil.rmtree(MERGED_MODEL_DIR)
    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary = merge_adapter_to_serving_checkpoint(adapter_output_dir, str(MERGED_MODEL_DIR))
    print(
        f"Merged checkpoint: {summary['architecture']} — "
        f"{summary['grafted_tensors']} fine-tuned tensors, "
        f"{summary['vision_tensors_from_base']} carried from the base vision tower"
    )
    return MERGED_MODEL_DIR


if REGISTER_MODEL:
    if "TRAINED_ADAPTER_OUTPUT_DIR" not in globals() or not TRAINED_ADAPTER_OUTPUT_DIR:
        raise ValueError("Run the training cell before merging the adapter.")

    merge_adapter_to_local_disk(TRAINED_ADAPTER_OUTPUT_DIR)
    # %restart_python clears the session, so hand the registration cell what it
    # needs through a file rather than Python state.
    MERGE_METADATA_PATH.write_text(
        json.dumps(
            {
                "adapter_output_dir": TRAINED_ADAPTER_OUTPUT_DIR,
                "training_run_id": TRAINING_RUN_ID,
                "merged_model_dir": str(MERGED_MODEL_DIR),
            }
        )
    )
    print(f"Merged weights: {MERGED_MODEL_DIR}")
else:
    print("Merge skipped because register_model is false in train.yaml.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the serving stack (two pip passes)
# MAGIC
# MAGIC These pins mirror a Custom LLM Serving deployment validated on a live workspace, and the order matters:
# MAGIC
# MAGIC 1. **vLLM first.** `vllm==0.24.0` is the version proven on Custom LLM Serving and new enough to register Qwen3.5's architecture. `mlflow==3.12` is uninstallable beside it (starlette conflict), hence `mlflow==3.14.0`.
# MAGIC 2. **opencv second**, downgrading what vLLM just pulled in. `pip` prints a dependency-conflict warning and that is expected. Anything `>=4.13` bundles an OpenSSL that fails the FIPS self-test and aborts vLLM at startup.
# MAGIC 3. `flashinfer-cubin` is **not** pinned here: vLLM depends on an exact version (`vllm==0.24.0` requires `flashinfer-cubin==0.6.12`), so pinning one yourself is an instant `ResolutionImpossible`. Databricks' starter notebook pins `0.5.2` because it pairs with `vllm==0.11.2`. Either way the precompiled cubins arrive, which is what stops the sampler JIT-compiling in a container with no `ninja`/`nvcc`.
# MAGIC
# MAGIC `env_pack="databricks_model_serving"` packs this environment into the registered model version, which is why the serving stack is installed here instead of declared as `pip_requirements`.
# MAGIC
# MAGIC **Security note for anyone reusing this pattern:** `opencv-python-headless<4.13` carries a known RCE CVE. That is precisely why the managed Foundation Model path will not ship this combination centrally, and it should be called out to customers alongside the recipe.

# COMMAND ----------

# MAGIC %pip install vllm==0.24.0 transformers==5.13.0 mlflow==3.14.0 openai==2.17.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0
# MAGIC %pip install opencv-python-headless==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the merged model for custom LLM serving
# MAGIC
# MAGIC `%restart_python` cleared the session, so this cell re-derives its configuration from `train.yaml` and reads the merge hand-off file. The serving choices worth seeing:
# MAGIC
# MAGIC - `task` is `llm/v1/chat`, matching the request contract the endpoint and the load test use.
# MAGIC - The vLLM process listens on port `8080`, the port Model Serving expects.
# MAGIC - The entrypoint launches from the model's `artifacts/` folder, so `--model` is the bare artifact name. An `artifacts/` prefix makes vLLM treat it as a Hugging Face repo id and fail with a 401.
# MAGIC - `--language-model-only` serves Qwen3.5's text backbone and skips the vision tower this task never uses.
# MAGIC - Registration requires `env_pack="databricks_model_serving"`: custom LLM serving runs on [Serverless Optimized Deployments](https://docs.databricks.com/aws/en/machine-learning/model-serving/serverless-optimized-deployments).

# COMMAND ----------

import json
import sys
from pathlib import Path

NOTEBOOK_DIR = str(Path.cwd())
if NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOK_DIR)

import mlflow
import pandas as pd

from training_utils import (
    fraud_response_format,
    load_training_config,
    local_staging_dir,
    resolve_experiment_path,
)

# The restart wiped the bindings from the configuration cell; reload them so the
# registration and deployment cells see the same constants as training did.
globals().update(load_training_config())
mlflow.set_experiment(resolve_experiment_path(EXPERIMENT_NAME))

CUSTOM_LLM_TASK = "llm/v1/chat"
CUSTOM_LLM_MODEL_ARTIFACT_NAME = "qwen35_fraud_model"
MERGE_METADATA_PATH = local_staging_dir("air-demo-merged") / "merge_metadata.json"


def register_custom_llm_model(merge_metadata: dict):
    mlflow.set_registry_uri("databricks-uc")

    # Defined inline (not in train.py/training_utils.py) on purpose: cloudpickle
    # serializes notebook-local classes BY VALUE, so the serving container can
    # unpickle the model without any repo code and no code_paths are needed.
    # Serving runs the vLLM entrypoint, never this predict method.
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
            # Qwen3.5 checkpoints carry a vision tower and this fraud task is
            # text-only. The flag zeroes the per-prompt modality limits (and
            # enables a fused qwen3-next kernel) — it does NOT skip the tower,
            # which is why the merged checkpoint still ships its weights.
            "--language-model-only "
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
        # Belt: the merged checkpoint's chat template already suppresses thinking
        # by default, and this keeps working for clients that send the kwarg.
        "chat_template_kwargs": {"enable_thinking": False},
        # Braces: grammar-constrained decoding admits only tokens that fit the
        # risk/action/reason schema, so no reasoning preamble is representable.
        "response_format": fraud_response_format(),
    }

    with mlflow.start_run(
        run_name=f"{TRAINING_RUN_NAME}-registration", log_system_metrics=True
    ) as run:
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "adapter_output_dir": merge_metadata["adapter_output_dir"],
                "registered_model_name": FULL_MODEL_NAME,
                "source_training_run_id": merge_metadata["training_run_id"],
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
            artifacts={CUSTOM_LLM_MODEL_ARTIFACT_NAME: merge_metadata["merged_model_dir"]},
            input_example=input_example,
            # Deliberately NOT pip_requirements: the environment installed above
            # (vLLM 0.24 with opencv held at 4.12.0.88) is what env_pack captures,
            # and a single requirements list cannot express that conflicting pair.
            extra_pip_requirements=["mlflow==3.14.0"],
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
        "custom_llm_task": CUSTOM_LLM_TASK,
        "entrypoint": metadata["entrypoint"],
    }


registration_result = None
REGISTERED_MODEL_VERSION = None

if REGISTER_MODEL:
    if not MERGE_METADATA_PATH.exists():
        raise ValueError("Run the merge cell before registering the model.")

    registration_result = register_custom_llm_model(
        json.loads(MERGE_METADATA_PATH.read_text())
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
# MAGIC - `serving_workload_size` controls provisioned capacity behind the endpoint.
# MAGIC - `serving_scale_to_zero` is useful for demos and development, but should be disabled for latency-sensitive production traffic.
# MAGIC
# MAGIC The served entity also sets `VLLM_USE_FLASHINFER_SAMPLER=0`: the serving container cannot JIT-compile FlashInfer kernels (no `ninja`/`nvcc`), so vLLM falls back to its native PyTorch sampler. Now that `flashinfer-cubin` is installed with the serving stack, the precompiled kernels are present and this variable can be dropped to get the faster sampler back — worth testing once the endpoint is otherwise healthy.
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
    served_entity = ServedEntityInput(
        name=served_entity_name,
        entity_name=FULL_MODEL_NAME,
        entity_version=str(model_version),
        workload_type=workload_type,
        #workload_size=SERVING_WORKLOAD_SIZE,
        min_provisioned_concurrency=4,
        max_provisioned_concurrency=4,
        environment_vars={
            # The serving container has no ninja/nvcc, so FlashInfer (shipped in
            # the Databricks AI base env) cannot JIT-compile its sampling kernels
            # at startup; fall back to vLLM's native PyTorch sampler.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
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
            description="AIR demo custom LLM endpoint for TabFormer fraud decisions.",
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
        "workload_size": SERVING_WORKLOAD_SIZE,
        #"min_provisioned_concurrency": 256,
        #"max_provisioned_concurrency": 256,
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

sample_transaction_prompt = (
    "You are a fraud decision model for a credit-card transaction stream. "
    "Classify the transaction as legitimate, suspicious, or likely_fraud. "
    "Return only compact JSON with keys risk, action, and reason.\n\n"
    "Transaction:\n"
    "- user_id: 492\n"
    "- card_id: 3\n"
    "- timestamp: 2026-06-08 13:45:00\n"
    "- amount_usd: 2499.99\n"
    "- use_chip: Online Transaction\n"
    "- merchant_city: Miami\n"
    "- merchant_state: FL\n"
    "- merchant_category_code: 5732\n"
    "- errors: Bad PIN"
)

serving_payload = {
    "messages": [
        {
            "role": "user",
            "content": sample_transaction_prompt,
        }
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    # The served checkpoint's chat template defaults to thinking off, so this is
    # belt-and-braces rather than load-bearing -- it keeps the payload explicit
    # about the contract, and it is what the load test sends too. Note that
    # chat_template_kwargs must sit at the TOP level of the request body: nesting
    # it under extra_body (a client-side-only concept in the OpenAI SDK) is
    # silently ignored by vLLM and the response fills with reasoning.
    "chat_template_kwargs": {"enable_thinking": False},
    # Grammar-constrained decoding pins the JSON contract regardless.
    "response_format": fraud_response_format(),
}

print(f"Registered model name: {FULL_MODEL_NAME}")
print(f"Serving endpoint name: {ENDPOINT_NAME}")
print(json.dumps(serving_payload, indent=2))

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

response = w.serving_endpoints.query(
    name="qwen35_4b_finetuned_lora",
    messages=[
        ChatMessage(
            role=ChatMessageRole.USER,
            content=(
                "You are a fraud decision model for a credit-card transaction stream. "
                "Classify the transaction as legitimate, suspicious, or likely_fraud. "
                "Return only compact JSON with keys risk, action, and reason.\n\n"
                "Transaction:\n"
                "- user_id: 492\n"
                "- card_id: 3\n"
                "- timestamp: 2026-06-08 13:45:00\n"
                "- amount_usd: 2499.99\n"
                "- use_chip: Online Transaction\n"
                "- merchant_city: Miami\n"
                "- merchant_state: FL\n"
                "- merchant_category_code: 5732\n"
                "- errors: Bad PIN"
            ),
        )
    ],
    max_tokens=64,
    temperature=0.0,
    extra_params={"chat_template_kwargs": {"enable_thinking": False}},
)

# COMMAND ----------

response.as_dict()['choices'][0]['message']['content']

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
# MAGIC - Custom LLM serving with vLLM: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms
# MAGIC - TRL supervised fine-tuning: https://huggingface.co/docs/trl/en/sft_trainer
# MAGIC - PEFT LoRA: https://huggingface.co/docs/peft/en/developer_guides/lora
# MAGIC - Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B