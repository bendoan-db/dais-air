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

# DBTITLE 1,AI Runtime fraud fine-tuning with Qwen3.5 2B and Unsloth
# MAGIC %md
# MAGIC # Fine-tune Qwen3.5 2B for fraud decisions with AI Runtime
# MAGIC
# MAGIC This notebook shows how to fine-tune a small language model for real-time credit-card fraud decisions on Databricks AI Runtime.
# MAGIC It is designed to stand alone for an external technical audience: each section explains what is happening, why it matters, and how the step contributes to a production AI workflow.
# MAGIC
# MAGIC The workflow uses the IBM TabFormer credit-card dataset loaded and prepared by `setup/01_load_tabformer_dataset.py`.
# MAGIC The setup notebook creates both a cleaned transaction table and a supervised fine-tuning table with prompt/response records. This notebook samples or shards those SFT rows, fine-tunes with Unsloth LoRA, logs with MLflow, and optionally registers the model to Unity Catalog for serving.
# MAGIC
# MAGIC **AI Runtime value drivers demonstrated in this notebook**
# MAGIC
# MAGIC - **On-demand GPU access:** run deep learning workloads on serverless GPU compute without provisioning or maintaining GPU clusters.
# MAGIC - **Managed AI environment:** use the AI Runtime base environment with common model-training libraries already available.
# MAGIC - **Unified data and governance:** read source transactions from Unity Catalog Delta tables and write checkpoints, adapters, and models to governed Unity Catalog assets.
# MAGIC - **Simple scaling path:** start with `@distributed(gpus=1)`, then change that single decorator parameter to use multiple GPUs while the training code stays the same.
# MAGIC - **Operational handoff:** use MLflow and Unity Catalog to move from experimentation toward managed serving and autoscaling.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
# MAGIC - Serverless GPU H100 starter: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-api-h100-starter
# MAGIC - Databricks Unsloth example: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth
# MAGIC - Unsloth Qwen3.5: https://unsloth.ai/docs/models/qwen3.5
# MAGIC - Unsloth Qwen3.5 fine-tuning: https://unsloth.ai/docs/models/qwen3.5/fine-tune

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business scenario and model contract
# MAGIC
# MAGIC Fraud detection is a high-volume, low-latency decision problem. A production payment system needs a clear response for each transaction: approve it, ask for additional authentication, or decline and escalate it.
# MAGIC
# MAGIC The setup notebook loads IBM TabFormer credit-card transactions into Unity Catalog Delta tables and builds prompt/response records for supervised fine-tuning.
# MAGIC This training notebook reads those prepared SFT records and fine-tunes `unsloth/Qwen3.5-2B` to emit a structured fraud decision.
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
# MAGIC If `1xH100` is not available in the workspace, `1xA10` is enough for this 2B bf16 LoRA workflow.
# MAGIC The model is intentionally small so the notebook highlights the platform workflow: governed data, GPU-backed training, experiment tracking, and production handoff.

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training configuration
# MAGIC
# MAGIC Training settings are loaded from `air/training.yaml`.
# MAGIC This keeps the notebook body stable while making the experiment easy to tune:
# MAGIC
# MAGIC - `catalog`, `schema`, and `source_table` point to the governed transaction Delta table.
# MAGIC - `sft_table` points to the prepared prompt/response Delta table.
# MAGIC - `checkpoint_volume` controls where adapters and model artifacts are written.
# MAGIC - `max_steps`, batch size, and learning rate control the training cost and runtime.
# MAGIC
# MAGIC The demo uses one training cell. Run it first with `@distributed(gpus=1, gpu_type="h100")`, then change only `gpus` to a larger value such as `8` to distribute the same training workflow.
# MAGIC For a short walkthrough, keep `max_steps` low. For a real experiment, increase `max_steps`, broaden the sampled dataset, and compare runs in MLflow.

# COMMAND ----------

import json
import os
from pathlib import Path

import pandas as pd

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print("Environment flags configured before importing Unsloth.")

# COMMAND ----------

config_path, training_config = load_yaml_config("training.yaml")

UC_CATALOG = config_str(training_config, "catalog")
UC_SCHEMA = config_str(training_config, "schema")
SOURCE_TABLE_NAME = config_str(training_config, "source_table")
SFT_TABLE_NAME = config_str(training_config, "sft_table")
UC_VOLUME = config_str(training_config, "checkpoint_volume")
UC_MODEL_NAME = config_str(training_config, "uc_model_name")

MODEL_NAME = config_str(training_config, "model_name")
MAX_SEQ_LENGTH = config_int(training_config, "max_seq_length")
MAX_STEPS = config_int(training_config, "max_steps")
PER_DEVICE_TRAIN_BATCH_SIZE = config_int(training_config, "per_device_train_batch_size")
GRADIENT_ACCUMULATION_STEPS = config_int(training_config, "gradient_accumulation_steps")
LEARNING_RATE = config_float(training_config, "learning_rate")
REGISTER_MODEL = config_bool(training_config, "register_model")
SEED = config_int(training_config, "seed")

SOURCE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SOURCE_TABLE_NAME}"
SFT_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SFT_TABLE_NAME}"
FULL_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
OUTPUT_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{UC_MODEL_NAME}"
TRAINING_OUTPUT_DIR = f"{OUTPUT_ROOT}/training_demo"
TRAINING_RUN_NAME = f"air-demo-{UC_MODEL_NAME}-training-steps{MAX_STEPS}"

print(f"Training config: {config_path}")
print(f"Source table: {SOURCE_TABLE}")
print(f"SFT table: {SFT_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Training output dir: {TRAINING_OUTPUT_DIR}")
print(f"Register model: {REGISTER_MODEL}")

# COMMAND ----------

spark = get_spark_session()

schema_q = full_name(UC_CATALOG, UC_SCHEMA)
volume_q = full_name(UC_CATALOG, UC_SCHEMA, UC_VOLUME)
source_table_q = full_name(UC_CATALOG, UC_SCHEMA, SOURCE_TABLE_NAME)
sft_table_q = full_name(UC_CATALOG, UC_SCHEMA, SFT_TABLE_NAME)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_q}")

Path(TRAINING_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Ready: {schema_q}")
print(f"Ready: {volume_q}")
print(f"SFT table: {sft_table_q}")

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

summary_sql = f"""
SELECT
  COUNT(*) AS row_count,
  SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_row_count,
  AVG(CAST(is_fraud AS DOUBLE)) AS fraud_rate,
  MIN(transaction_ts) AS min_transaction_ts,
  MAX(transaction_ts) AS max_transaction_ts
FROM {source_table_q}
"""

summary_pdf = spark.sql(summary_sql).toPandas()
display(summary_pdf)

sft_summary_sql = f"""
SELECT
  COUNT(*) AS row_count,
  COUNT(DISTINCT shard_id) AS shard_count,
  MIN(shard_id) AS min_shard_id,
  MAX(shard_id) AS max_shard_id,
  SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_row_count,
  AVG(CAST(is_fraud AS DOUBLE)) AS fraud_rate
FROM {sft_table_q}
"""

sft_summary_pdf = spark.sql(sft_summary_sql).toPandas()
display(sft_summary_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale plan
# MAGIC
# MAGIC The notebook loads training records directly from the prepared SFT Delta table inside the `@distributed` training function.
# MAGIC Prompt and response generation already happened in ingestion, so the training path avoids row-by-row prompt construction on the driver:
# MAGIC
# MAGIC - With `gpus=1`, one worker reads the sample and validates the end-to-end flow.
# MAGIC - With `gpus>1`, each worker reads a different set of SFT shards by rank.
# MAGIC
# MAGIC The model, prompt contract, LoRA setup, MLflow logging, and Unity Catalog artifact layout stay fixed.
# MAGIC The main change during the demo is the `gpus` value in the decorator.

# COMMAND ----------

scale_config = {
    "initial_strategy": "@distributed(gpus=1, gpu_type=\"h100\")",
    "scale_up_strategy": "change only the decorator to @distributed(gpus=8, gpu_type=\"h100\")",
    "delta_load": "rank-sharded SFT table inside the distributed function",
    "intermediate_dataset_files": "none",
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_micro_batch_per_step": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
}

display(pd.DataFrame([scale_config]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune Qwen3.5 2B with Unsloth
# MAGIC
# MAGIC This section fine-tunes `unsloth/Qwen3.5-2B` with LoRA adapters.
# MAGIC It uses bf16/16-bit LoRA instead of QLoRA because Unsloth's Qwen3.5 guidance does not recommend QLoRA for this model family.
# MAGIC
# MAGIC The implementation highlights the production workflow around training:
# MAGIC
# MAGIC - MLflow records parameters, metrics, and run metadata.
# MAGIC - Checkpoints and adapters are saved to a Unity Catalog volume.
# MAGIC - Model registration is handled in a separate section after training completes.
# MAGIC - GPU memory metrics are logged when CUDA is available, which helps compare the `gpus=1` and `gpus>1` runs.
# MAGIC
# MAGIC The training implementation is kept inline below so readers can inspect the Unsloth, TRL, MLflow, and distributed execution code directly.

# COMMAND ----------

# DBTITLE 1,Cell 18
from contextlib import contextmanager, nullcontext

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig


@contextmanager
def start_mlflow_run(mlflow_module, run_name: str):
    try:
        with mlflow_module.start_run(run_name=run_name, log_system_metrics=True) as run:
            yield run
    except TypeError:
        with mlflow_module.start_run(run_name=run_name) as run:
            yield run


def render_chat_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def train_qwen35_unsloth(
    *,
    examples_pdf: pd.DataFrame,
    output_dir: str,
    run_name: str,
    training_mode: str,
    num_gpus: int,
    device_map=None,
    save_artifacts: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> str | None:
    import mlflow
    import torch
    from datasets import Dataset

    mlflow.set_registry_uri("databricks-uc")
    is_main_process = rank == 0
    save_artifacts = save_artifacts and is_main_process

    dataset = Dataset.from_pandas(
        examples_pdf[["prompt", "assistant_response"]],
        preserve_index=False,
    )

    load_kwargs = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": None,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("load_in_16bit", None)
        load_kwargs.pop("full_finetuning", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    def formatting_prompts_func(examples):
        texts = [
            render_chat_messages(
                tokenizer,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
            )
            for prompt, assistant_response in zip(
                examples["prompt"],
                examples["assistant_response"],
            )
        ]
        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,
    )

    peft_kwargs = {
        "r": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": SEED,
        "use_rslora": False,
        "loftq_config": None,
        "max_seq_length": MAX_SEQ_LENGTH,
    }

    try:
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    except TypeError:
        peft_kwargs.pop("max_seq_length", None)
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=output_dir,
        report_to="none",
        run_name=run_name,
        save_strategy="steps",
        save_steps=max(5, MAX_STEPS // 2),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=training_args,
    )

    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            num_proc=1,
        )
    except Exception as exc:
        print(f"Response-only masking was skipped: {exc}")

    run_context = start_mlflow_run(mlflow, run_name) if is_main_process else nullcontext()

    with run_context as run:
        if is_main_process:
            mlflow.log_params(
                {
                    "base_model": MODEL_NAME,
                    "training_mode": training_mode,
                    "num_gpus": num_gpus,
                    "rank": rank,
                    "world_size": world_size,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "max_steps": MAX_STEPS,
                    "rank_0_training_record_count": len(examples_pdf),
                    "source_table": SOURCE_TABLE,
                    "sft_table": SFT_TABLE,
                    "lora_r": 16,
                    "lora_alpha": 16,
                }
            )

        train_output = trainer.train()
        metrics = getattr(train_output, "metrics", {}) or {}

        if not is_main_process:
            return None

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"trainer_{metric_name}", float(metric_value))

        if save_artifacts:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.log_param("adapter_output_dir", output_dir)

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale training by changing one decorator parameter
# MAGIC
# MAGIC This is the only training cell in the demo.
# MAGIC Run it first with `gpus=1` to validate the workflow, then change the decorator to `gpus=8` and rerun the same cell to distribute training across multiple GPUs.
# MAGIC
# MAGIC Each worker reads a rank-assigned slice of the prepared SFT Delta table. Keeping the Delta read inside the decorated function avoids shipping a large in-memory dataset from the notebook driver to the GPU workers.

# COMMAND ----------

import mlflow
mlflow.set_experiment("/Users/ben.doan@databricks.com/unsloth_qwen_2b_training")

# COMMAND ----------

from serverless_gpu import distributed

TRAINING_SAMPLE_FRACTION = 0.001

@distributed(gpus=8, gpu_type="h100")
def run_training_job():
    import os
    import torch
    from serverless_gpu import runtime as rt

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = rt.get_global_rank()
    world_size = rt.get_world_size()
    torch.cuda.set_device(local_rank)

    distributed_sql = f"""
    SELECT
      training_id,
      shard_id,
      prompt,
      assistant_response,
      fraud_label,
      is_fraud
    FROM {sft_table_q}
    WHERE pmod(shard_id, {world_size}) = {rank}
    """

    dataset = get_spark_session().sql(distributed_sql).sample(TRAINING_SAMPLE_FRACTION).toPandas()
    run_output_dir = f"{TRAINING_OUTPUT_DIR}/{world_size}gpu"
    run_name = f"{TRAINING_RUN_NAME}-{world_size}gpu"
    training_mode = f"{world_size}_gpu_rank_sharded_sample"

    try:
        return train_qwen35_unsloth(
            examples_pdf=dataset,
            output_dir=run_output_dir,
            run_name=run_name,
            training_mode=training_mode,
            num_gpus=world_size,
            device_map={"": local_rank},
            save_artifacts=rank == 0,
            rank=rank,
            world_size=world_size,
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

distributed_run_ids = run_training_job.distributed()
TRAINING_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)
TRAINING_WORLD_SIZE = len(distributed_run_ids)
TRAINED_ADAPTER_OUTPUT_DIR = f"{TRAINING_OUTPUT_DIR}/{TRAINING_WORLD_SIZE}gpu"

print(f"Training MLflow run ID: {TRAINING_RUN_ID}")
print(f"Trained adapter output dir: {TRAINED_ADAPTER_OUTPUT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance optimization loop
# MAGIC
# MAGIC After a successful baseline run, change `gpus` in the decorator, rerun the training cell, and compare runtime, loss, and GPU utilization in MLflow and the notebook resource panels.
# MAGIC For multi-GPU training, a common first bottleneck is an effective batch size that is too small to keep the GPUs busy.
# MAGIC
# MAGIC Use the suggested configuration below as a starting point for the next experiment:
# MAGIC
# MAGIC - Increase `per_device_train_batch_size` if memory allows.
# MAGIC - Otherwise increase `gradient_accumulation_steps`.
# MAGIC - Keep `learning_rate`, `max_steps`, and dataset size controlled while comparing runs.
# MAGIC - Use MLflow metrics and system telemetry to confirm that throughput improves without destabilizing training.
# MAGIC
# MAGIC AI Runtime keeps this loop inside the same platform experience: data, GPU compute, metrics, artifacts, and registered models remain connected.

# COMMAND ----------

genie_suggested_config = {
    "current_per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "current_gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "suggested_per_device_train_batch_size": max(PER_DEVICE_TRAIN_BATCH_SIZE, 4),
    "suggested_gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "rerun_instruction": "Update training.yaml, rerun from Training configuration, and compare MLflow loss/runtime/GPU metrics.",
}

display(pd.DataFrame([genie_suggested_config]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the trained model
# MAGIC
# MAGIC Registration is separate from training so the distributed training cell stays focused on GPU optimization and adapter checkpointing.
# MAGIC This cell loads the rank-0 adapter artifacts saved by training, merges them with the base model, and registers the merged model to Unity Catalog for serving.
# MAGIC
# MAGIC Keeping registration as a separate step also makes reruns cheaper: if training succeeds but registration fails, rerun only this section.

# COMMAND ----------

def register_trained_qwen35_model(adapter_output_dir: str, run_name: str):
    import mlflow

    mlflow.set_registry_uri("databricks-uc")

    load_kwargs = {
        "model_name": adapter_output_dir,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": None,
        "load_in_4bit": False,
        "load_in_16bit": True,
    }
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("load_in_16bit", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    merged_model = model.merge_and_unload()

    with start_mlflow_run(mlflow, run_name) as run:
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "adapter_output_dir": adapter_output_dir,
                "registered_model_name": FULL_MODEL_NAME,
                "source_training_run_id": TRAINING_RUN_ID,
            }
        )
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": merged_model, "tokenizer": tokenizer},
            name="model",
            task="llm/v1/chat",
            registered_model_name=FULL_MODEL_NAME,
            await_registration_for=3600,
            metadata={
                "task": "llm/v1/chat",
                "pretrained_model_name": MODEL_NAME,
                "databricks_model_family": "Qwen3.5",
                "demo": "AIR DAIS fraud detection",
            },
        )

    return {
        "registration_run_id": run.info.run_id,
        "registered_model_name": FULL_MODEL_NAME,
        "model_uri": model_info.model_uri,
    }


if REGISTER_MODEL:
    if "TRAINED_ADAPTER_OUTPUT_DIR" not in globals() or not TRAINED_ADAPTER_OUTPUT_DIR:
        raise ValueError("Run the training cell before registering the model.")

    registration_result = register_trained_qwen35_model(
        adapter_output_dir=TRAINED_ADAPTER_OUTPUT_DIR,
        run_name=f"{TRAINING_RUN_NAME}-registration",
    )
    display(pd.DataFrame([registration_result]))
else:
    print("Model registration skipped because register_model is false in training.yaml.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production handoff: model serving and autoscaling
# MAGIC
# MAGIC The previous section registers the fine-tuned model to Unity Catalog when `register_model = true`.
# MAGIC
# MAGIC From there, the production path is:
# MAGIC
# MAGIC 1. Open the registered model in Unity Catalog.
# MAGIC 2. Click **Serve this model**.
# MAGIC 3. Create a Mosaic AI Model Serving endpoint with autoscaling enabled.
# MAGIC 4. Send a sample transaction request.
# MAGIC 5. Increase request traffic and observe endpoint health and autoscaling behavior.
# MAGIC
# MAGIC The request payload should keep the same prompt contract used during fine-tuning: ask for compact JSON with `risk`, `action`, and `reason`.
# MAGIC This keeps development and serving aligned around the same interface.

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
    "max_tokens": 128,
    "temperature": 0.0,
}

print(f"Registered model name: {FULL_MODEL_NAME}")
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
# MAGIC - MLflow captures the experiment record, and Unity Catalog provides the handoff point for serving.
# MAGIC
# MAGIC The main platform outcome is speed with control: teams can move from governed data to GPU fine-tuning to registered model artifacts without leaving Databricks or stitching together separate infrastructure.
