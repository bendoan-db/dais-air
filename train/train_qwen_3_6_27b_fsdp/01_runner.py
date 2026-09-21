# Databricks notebook source
# MAGIC %md
# MAGIC # Full-weight fine-tune Qwen3.6 27B with FSDP2
# MAGIC
# MAGIC This runner and the AI Runtime CLI execute this directory's `train.py`
# MAGIC with the settings in `train.yaml`. The default workload uses eight H100
# MAGIC GPUs and saves a complete Hugging Face checkpoint, not a LoRA adapter.

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import sys
from pathlib import Path

# Bootstrap only: a notebook's own folder is not reliably on sys.path, so this
# has to run before any project module can be imported.
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_config import load_notebook_compute, load_project_config

CONFIG = load_project_config()
NOTEBOOK_GPUS, NOTEBOOK_GPU_TYPE = load_notebook_compute()
SAMPLE_FRACTION = CONFIG.training_sample_fraction

print(f"Config: {CONFIG.config_path}")
print(f"Model weights: {CONFIG.model_weights_path}")
print(f"Training data: {CONFIG.train_data_path}")
print(f"Evaluation data: {CONFIG.eval_data_path}")
print(
    "SFT conversion: "
    f"{'inside trainer' if CONFIG.convert_sft else 'pre-converted input'}"
)
print(
    "Partition loading: "
    f"{'all files per rank' if CONFIG.ignore_partitions else 'rank-assigned shards'}"
)
print(f"Full-model output: {CONFIG.output_dir}")
print(f"MLflow experiment: {CONFIG.experiment_path}")
print(
    f"MLflow cadence: train every {CONFIG.logging_steps} step(s), "
    f"eval every {CONFIG.eval_steps} step(s)"
)
print(f"Compute: {NOTEBOOK_GPUS} x {NOTEBOOK_GPU_TYPE}")

# COMMAND ----------

import mlflow

mlflow.set_experiment(CONFIG.experiment_path)

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=NOTEBOOK_GPUS, gpu_type=NOTEBOOK_GPU_TYPE)
def run_training_job():
    # This function is serialized to every rank, so it closes over plain
    # builtins only (a Path and a float). Each rank loads its own config from
    # train.yaml rather than receiving a TrainingConfig across the boundary,
    # which would require project_config to be importable before unpickling.
    import sys

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from train import run_rank_training

    return run_rank_training(sample_fraction=SAMPLE_FRACTION)


distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_MODEL_OUTPUT_DIR = CONFIG.output_dir_for(TRAINING_WORLD_SIZE)

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained full-model output dir: {TRAINED_MODEL_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC The rank-zero MLflow run logs `model_output_dir`. Use that run with
# MAGIC this project's `02_register_and_deploy.py` to register the complete
# MAGIC fine-tuned checkpoint and deploy it with inference tables enabled.
