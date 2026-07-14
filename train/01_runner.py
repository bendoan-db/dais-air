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
# MAGIC Training settings are loaded from the `training_config` section of `train/train.yaml` — the same file that defines the AI Runtime CLI workload, so the notebook and CLI launch paths share one configuration. (Registration and serving settings live in the `deploy_config` section of the same file, read by `02_register_and_deploy.py`.)
# MAGIC This keeps the notebook body stable while making the experiment easy to tune:
# MAGIC
# MAGIC - `catalog`, `schema`, and `source_table` point to the governed transaction Delta table.
# MAGIC - `sft_table` points to the prepared prompt/response Delta table.
# MAGIC - `checkpoint_volume` controls where adapters and model artifacts are written.
# MAGIC - `model_volume_path` (optional) points at a Unity Catalog volume snapshot of the base model weights, populated by `setup/03_download_base_model_weights.py`; when set, the GPU workers load the model from the volume instead of downloading it from Hugging Face. Volume-hosted weights are first staged to node-local disk (once per node) because safetensors mmap reads through the volume FUSE mount are slow.
# MAGIC - `max_steps`, batch size, learning rate, and the LoRA settings control the training cost and quality.
# MAGIC - `training_sample_fraction` controls how much of the staged SFT data is used; the training cell reads it from the config and can override it inline for quick smoke runs.
# MAGIC - The workload-level `compute` block sizes the `@distributed` training cell and AIR CLI runs alike (`num_accelerators` → `gpus`, `accelerator_type`'s chip suffix → `gpu_type`). Start with 1 GPU to validate the workflow, then raise `num_accelerators` to scale out — the training code is unchanged.
# MAGIC
# MAGIC For a quick validation run, keep `max_steps` low. For a real fine-tune, increase `max_steps` (or train on the full data), and compare runs in MLflow.

# COMMAND ----------

# Select the training implementation and its sibling config file:
#   "train"      — Unsloth LoRA + DDP (default; the 4B worked example)
#   "train_fsdp" — TRL + FSDP2 for models too large for one GPU
#                  (gpt-oss-120b; see train_fsdp.yaml)
TRAINING_MODULE = "train"

# load_training_config parses the selected workload file's training_config
# section, derives the UC names/paths, and returns one flat dict; binding it
# into globals gives every later cell the same constants the module uses.
training_context = load_training_config(f"{TRAINING_MODULE}.yaml")
globals().update(training_context)

print(f"Training config: {CONFIG_PATH}")
print(f"Source table: {SOURCE_TABLE}")
print(f"SFT table: {SFT_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Base weights source: {MODEL_LOAD_PATH}")
print(f"Training output dir: {TRAINING_OUTPUT_DIR}")

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
# MAGIC The workflow uses the IBM TabFormer credit-card dataset prepared by the setup notebooks: `setup/01_load_dataset.py` writes the cleaned transaction table, and `setup/02_stage_training_data.py` builds the supervised fine-tuning table with prompt/response records and stages it in a Unity Catalog volume. This notebook samples or shards those SFT rows, fine-tunes with Unsloth LoRA, and logs with MLflow; registration and serving are handled by the deployment notebook (`02_register_and_deploy.py`, next in this directory).
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
# MAGIC - MLflow records parameters, metrics, and run metadata — including held-out fraud-classification quality (`eval_fraud_accuracy`/`_precision`/`_recall`/`_f1`, scored after training on a stratified holdout the training data excludes; `eval_sample_size` in `train.yaml`).
# MAGIC - Checkpoints and adapters are saved to a Unity Catalog volume.
# MAGIC - Model registration and deployment are handled by `02_register_and_deploy.py` (next in this directory) after training completes.
# MAGIC - GPU memory metrics are logged when CUDA is available, which helps compare the `gpus=1` and `gpus>1` runs.
# MAGIC
# MAGIC The training implementation lives in `train/train.py`, a plain Python module shared by two launchers: this notebook's `@distributed` cell and the AI Runtime CLI (`air run --file train.yaml`), which runs the same file standalone on serverless GPUs. A sibling FSDP implementation (`train/train_fsdp.py` + `train_fsdp.yaml`, following the Databricks gpt-oss-120b tutorial) shards models too large for one GPU across the node — select it with `TRAINING_MODULE = "train_fsdp"` in the configuration cell below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale training by changing one config value
# MAGIC
# MAGIC This is the only training cell in the pipeline: a thin wrapper that imports `train.py` on each GPU worker and runs one rank of training.
# MAGIC Run it first with `compute.num_accelerators: 1` to validate the workflow, then raise it in the workload YAML (for example to `8`, with a matching `accelerator_type`) and rerun the same cell to distribute training across multiple GPUs.
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

    import importlib

    # TRAINING_MODULE selects train (Unsloth DDP) or train_fsdp (TRL+FSDP2);
    # both expose the same run_rank_training contract.
    run_rank_training = importlib.import_module(TRAINING_MODULE).run_rank_training

    return run_rank_training(sample_fraction=TRAINING_SAMPLE_FRACTION)

distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_ADAPTER_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/{TRAINING_WORLD_SIZE}gpu"

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained adapter output dir: {TRAINED_ADAPTER_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: register and deploy (02_register_and_deploy.py)
# MAGIC
# MAGIC Training is done. The run ID printed above identifies this run in MLflow, and the adapter's volume location is logged on the run as the `adapter_output_dir` parameter.
# MAGIC
# MAGIC `02_register_and_deploy.py` (next in this directory) merges the adapter into the base model, registers it to Unity Catalog as a custom LLM, and creates or updates the serving endpoint. Point `run_id` in `train.yaml`'s `deploy_config` section at the run ID above — or leave it empty to auto-select the best finished run in this experiment by the configured metric (`best_run_metric` / `best_run_metric_goal`).

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
# MAGIC - MLflow captures the experiment record; the deployment notebook (`02_register_and_deploy.py`) registers the model to Unity Catalog and creates or updates the custom LLM serving endpoint.
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
