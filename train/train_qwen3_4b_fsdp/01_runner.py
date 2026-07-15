# Databricks notebook source
# MAGIC %md
# MAGIC # Full-weight fine-tune Qwen3 4B with FSDP2
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

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_config import load_project_config

training_context = load_project_config()
globals().update(training_context)

print(f"Config: {CONFIG_PATH}")
print(f"Model weights: {MODEL_WEIGHTS_PATH}")
print(f"Training data: {TRAIN_DATA_PATH}")
print(f"Evaluation data: {EVAL_DATA_PATH}")
print(f"SFT conversion: {'inside trainer' if CONVERT_SFT else 'pre-converted input'}")
print(
    "Partition loading: "
    f"{'all files per rank' if IGNORE_PARTITIONS else 'rank-assigned shards'}"
)
print(f"Full-model output: {TRAINING_OUTPUT_DIR}")
print(f"MLflow experiment: {EXPERIMENT_PATH}")
print(f"MLflow cadence: train every {LOGGING_STEPS} step(s), eval every {EVAL_STEPS} step(s)")

# COMMAND ----------

import mlflow

mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=NOTEBOOK_GPUS, gpu_type=NOTEBOOK_GPU_TYPE)
def run_training_job():
    import sys

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from train import run_rank_training

    return run_rank_training(sample_fraction=TRAINING_SAMPLE_FRACTION)


distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_MODEL_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/{TRAINING_WORLD_SIZE}gpu"

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained full-model output dir: {TRAINED_MODEL_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC The rank-zero MLflow run logs `model_output_dir`. Use that run with
# MAGIC this project's `02_register_and_deploy.py` to register the complete
# MAGIC fine-tuned checkpoint and deploy it with inference tables enabled.
