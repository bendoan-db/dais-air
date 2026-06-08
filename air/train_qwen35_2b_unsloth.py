# Databricks notebook source
# DBTITLE 1,AI Runtime fraud fine-tuning with Qwen3.5 2B and Unsloth
# MAGIC %md
# MAGIC # Fine-tune Qwen3.5 2B for fraud decisions with AI Runtime
# MAGIC
# MAGIC This notebook shows how to fine-tune a small language model for real-time credit-card fraud decisions on Databricks AI Runtime.
# MAGIC It is designed to stand alone for an external technical audience: each section explains what is happening, why it matters, and how the step contributes to a production AI workflow.
# MAGIC
# MAGIC The workflow uses the IBM TabFormer credit-card dataset loaded and prepared by `setup/01_load_tabformer_dataset.py`.
# MAGIC The setup table provides cleaned prompt-ready transaction fields. This notebook samples those rows, converts them into supervised chat examples, fine-tunes with Unsloth LoRA, logs with MLflow, and optionally registers the model to Unity Catalog for serving.
# MAGIC
# MAGIC **AI Runtime value drivers demonstrated in this notebook**
# MAGIC
# MAGIC - **On-demand GPU access:** run deep learning workloads on serverless GPU compute without provisioning or maintaining GPU clusters.
# MAGIC - **Managed AI environment:** use the AI Runtime base environment with common model-training libraries already available.
# MAGIC - **Unified data and governance:** read source transactions from Unity Catalog Delta tables and write checkpoints, adapters, and models to governed Unity Catalog assets.
# MAGIC - **Simple scaling path:** start with a single-node run on 10% of the data, then use the same training function on the full training dataset with a distributed multi-GPU strategy.
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
# MAGIC The setup notebook loads IBM TabFormer credit-card transactions into a Unity Catalog Delta table and adds cleaned fields for prompt construction.
# MAGIC This training notebook turns those prepared rows into supervised chat examples and fine-tunes `unsloth/Qwen3.5-2B` to emit a structured fraud decision.
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
# MAGIC - Accelerator: `1xH100` or `1xA10` for the single-node 10% validation path, or `8xH100` for the distributed full-dataset path.
# MAGIC - Base environment: `AI v5`.
# MAGIC
# MAGIC If `1xH100` is not available in the workspace, `1xA10` is enough for this 2B bf16 LoRA workflow.
# MAGIC The model is intentionally small so the notebook highlights the platform workflow: governed data, GPU-backed training, experiment tracking, and production handoff.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install notebook requirements
# MAGIC
# MAGIC Install the Python packages required by the notebook.
# MAGIC AI Runtime already includes many common AI and ML libraries; this cell makes the notebook reproducible when package versions need to be pinned for the project.

# COMMAND ----------

# MAGIC %pip install -qqq -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load shared utilities

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training configuration
# MAGIC
# MAGIC Training settings are loaded from `air/training.yaml`.
# MAGIC This keeps the notebook body stable while making the experiment easy to tune:
# MAGIC
# MAGIC - `catalog`, `schema`, and `source_table` point to the governed Delta table.
# MAGIC - `checkpoint_volume` controls where adapters and model artifacts are written.
# MAGIC - `distributed_gpus` and `distributed_gpu_type` define the scaled training strategy.
# MAGIC - `max_steps`, batch size, and learning rate control the training cost and runtime.
# MAGIC
# MAGIC The notebook intentionally runs two training phases:
# MAGIC
# MAGIC 1. **Part 1:** train on one node with a 10% random Delta sample to validate the workflow cheaply.
# MAGIC 2. **Part 2:** train on the full prepared Delta table with distributed GPUs to show the scale-up path.
# MAGIC
# MAGIC For a short walkthrough, keep `max_steps` low. For a real experiment, increase `max_steps`, broaden the sampled dataset, and compare both phases in MLflow.

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
UC_VOLUME = config_str(training_config, "checkpoint_volume")
UC_MODEL_NAME = config_str(training_config, "uc_model_name")

MODEL_NAME = config_str(training_config, "model_name")
DISTRIBUTED_GPUS = config_int(training_config, "distributed_gpus")
DISTRIBUTED_GPU_TYPE = config_str(training_config, "distributed_gpu_type")
MAX_SEQ_LENGTH = config_int(training_config, "max_seq_length")
MAX_STEPS = config_int(training_config, "max_steps")
PER_DEVICE_TRAIN_BATCH_SIZE = config_int(training_config, "per_device_train_batch_size")
GRADIENT_ACCUMULATION_STEPS = config_int(training_config, "gradient_accumulation_steps")
LEARNING_RATE = config_float(training_config, "learning_rate")
SUSPICIOUS_AMOUNT_THRESHOLD = config_float(training_config, "suspicious_amount_threshold")
REGISTER_MODEL = config_bool(training_config, "register_model")
SEED = config_int(training_config, "seed")

SOURCE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SOURCE_TABLE_NAME}"
FULL_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
OUTPUT_ROOT = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{UC_MODEL_NAME}"
SINGLE_NODE_OUTPUT_DIR = f"{OUTPUT_ROOT}/single_node_10pct"
DISTRIBUTED_OUTPUT_DIR = f"{OUTPUT_ROOT}/distributed_full"
SINGLE_NODE_RUN_NAME = f"air-demo-{UC_MODEL_NAME}-single-node-10pct-steps{MAX_STEPS}"
DISTRIBUTED_RUN_NAME = f"air-demo-{UC_MODEL_NAME}-distributed-full-steps{MAX_STEPS}"

print(f"Training config: {config_path}")
print(f"Source table: {SOURCE_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Distributed strategy: {DISTRIBUTED_GPUS}x{DISTRIBUTED_GPU_TYPE}")
print(f"Single-node output dir: {SINGLE_NODE_OUTPUT_DIR}")
print(f"Distributed output dir: {DISTRIBUTED_OUTPUT_DIR}")
print(f"Register model: {REGISTER_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog and Spark Connect setup
# MAGIC
# MAGIC This cell creates the target schema and checkpoint volume if needed, then opens a Spark session for reading the transaction table.
# MAGIC
# MAGIC Unity Catalog is central to the workflow:
# MAGIC
# MAGIC - The source dataset is a governed Delta table.
# MAGIC - The generated supervised fine-tuning records are built from governed table rows.
# MAGIC - Training checkpoints and model artifacts are stored in the same catalog and schema.
# MAGIC - The final registered model can inherit the same governance boundary as the data used to train it.

# COMMAND ----------

spark = get_spark_session()

schema_q = full_name(UC_CATALOG, UC_SCHEMA)
volume_q = full_name(UC_CATALOG, UC_SCHEMA, UC_VOLUME)
source_table_q = full_name(UC_CATALOG, UC_SCHEMA, SOURCE_TABLE_NAME)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_q}")

Path(SINGLE_NODE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(DISTRIBUTED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Ready: {schema_q}")
print(f"Ready: {volume_q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and summarize the fraud data
# MAGIC
# MAGIC Start by summarizing the transaction table.
# MAGIC Fraud datasets are typically highly imbalanced, so the row count, fraud count, fraud rate, and time range provide useful context before any modeling work starts.
# MAGIC
# MAGIC This step also verifies that the ingestion notebook has successfully loaded the data before GPU time is used for training.

# COMMAND ----------

source_table_q = full_name(UC_CATALOG, UC_SCHEMA, SOURCE_TABLE_NAME)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale plan
# MAGIC
# MAGIC The notebook loads training records directly from the prepared Delta table inside each training section.
# MAGIC This creates a little duplicated code, but keeps the demo readable:
# MAGIC
# MAGIC - **Part 1:** load a 10% random sample from Delta and train on one node.
# MAGIC - **Part 2:** load the full prepared Delta table and train with the distributed strategy.
# MAGIC
# MAGIC Both parts use the same model, prompt contract, LoRA setup, MLflow logging, and Unity Catalog artifact layout.
# MAGIC The main change is the record count and compute strategy, which makes the scale-up path explicit and easy to compare.

# COMMAND ----------

scale_config = {
    "part_1_strategy": "single node",
    "part_1_delta_load": "10% random sample",
    "part_2_strategy": f"distributed {DISTRIBUTED_GPUS}x{DISTRIBUTED_GPU_TYPE}",
    "part_2_delta_load": "full prepared Delta table",
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
# MAGIC - If enabled, the merged model is registered to Unity Catalog for downstream serving.
# MAGIC - GPU memory metrics are logged when CUDA is available, which helps compare the single-node and distributed runs.
# MAGIC
# MAGIC The training implementation is kept inline below so readers can inspect the Unsloth, TRL, MLflow, and distributed execution code directly.

# COMMAND ----------

# DBTITLE 1,Cell 18
from contextlib import contextmanager

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
    records: list[dict[str, object]],
    output_dir: str,
    run_name: str,
    training_mode: str,
    num_gpus: int,
    register_model: bool,
    device_map=None,
    save_artifacts: bool = True,
) -> str:
    import mlflow
    import torch
    from datasets import Dataset

    mlflow.set_registry_uri("databricks-uc")

    dataset = Dataset.from_list(records)

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
        texts = [render_chat_messages(tokenizer, messages) for messages in examples["messages"]]
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
        report_to="mlflow",
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

    with start_mlflow_run(mlflow, run_name) as run:
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "training_mode": training_mode,
                "num_gpus": num_gpus,
                "max_seq_length": MAX_SEQ_LENGTH,
                "max_steps": MAX_STEPS,
                "training_record_count": len(records),
                "source_table": SOURCE_TABLE,
                "lora_r": 16,
                "lora_alpha": 16,
            }
        )

        train_output = trainer.train()
        metrics = getattr(train_output, "metrics", {}) or {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"trainer_{metric_name}", float(metric_value))

        if save_artifacts:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.log_param("adapter_output_dir", output_dir)

            if register_model:
                try:
                    merged_model = model.merge_and_unload()
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
                    print(f"Registered model: {FULL_MODEL_NAME}")
                    print(f"MLflow model URI: {model_info.model_uri}")
                except Exception as exc:
                    print(f"Model registration skipped or failed: {exc}")
                    print(f"LoRA adapters remain saved at: {output_dir}")

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: single-node training on 10% of the data
# MAGIC
# MAGIC First, validate the full training workflow on one node with a 10% random sample loaded from Delta.
# MAGIC This run catches data, dependency, prompt-format, and model-loading issues before using larger GPU capacity.
# MAGIC
# MAGIC This is the low-cost development pass:
# MAGIC
# MAGIC - Dataset: 10% random Delta sample converted to chat records in notebook memory
# MAGIC - Compute: attached single-node AI Runtime GPU compute
# MAGIC - Model registration: skipped, because this is a validation run
# MAGIC - Output: LoRA adapters and checkpoints under the single-node output directory

# COMMAND ----------

source_table_q

# COMMAND ----------


single_node_pdf = spark.table(source_table_q).sample(0.0001).toPandas()
single_node_records = [
    make_chat_record(row, SUSPICIOUS_AMOUNT_THRESHOLD)
    for _, row in single_node_pdf.iterrows()
]


single_node_preview_records = [
    {
        "prompt": record["messages"][0]["content"][:700],
        "assistant": record["messages"][1]["content"],
    }
    for record in single_node_records[:3]
]

# COMMAND ----------

display(pd.DataFrame(single_node_preview_records))

# COMMAND ----------

SINGLE_NODE_RUN_ID = train_qwen35_unsloth(
    records=single_node_records,
    output_dir=SINGLE_NODE_OUTPUT_DIR,
    run_name=SINGLE_NODE_RUN_NAME,
    training_mode="single_node_10pct",
    num_gpus=1,
    register_model=False,
)

print(f"Single-node MLflow run ID: {SINGLE_NODE_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: distributed training on the full dataset
# MAGIC
# MAGIC After the single-node pass succeeds, load the full prepared Delta table and run the same training function with the distributed AI Runtime strategy.
# MAGIC This section demonstrates the scale-up path: the data contract, LoRA configuration, MLflow logging, and Unity Catalog artifact locations stay the same, while the compute strategy changes.
# MAGIC
# MAGIC This is the production-oriented training pass:
# MAGIC
# MAGIC - Dataset: full prepared Delta table converted to chat records in notebook memory
# MAGIC - Compute: distributed `serverless_gpu` execution using the configured GPU count and type
# MAGIC - Model registration: controlled by `register_model` in `training.yaml`
# MAGIC - Output: full-run adapters, checkpoints, and optional Unity Catalog registered model
# MAGIC
# MAGIC The full-data load happens inside the decorated function. This follows the `serverless_gpu` best practice: keep large datasets out of the function closure so the launcher does not need to pickle and ship them to workers.

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=DISTRIBUTED_GPUS, gpu_type=DISTRIBUTED_GPU_TYPE)
def run_distributed_train():
    import os
    import torch
    from serverless_gpu import runtime as rt

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    distributed_sql = f"""
    SELECT
      user_id_text,
      card_id_text,
      transaction_ts_text,
      amount_usd,
      use_chip_text,
      merchant_city_text,
      merchant_state_text,
      mcc_text,
      errors_text,
      has_error_signal,
      fraud_label,
      is_fraud
    FROM {source_table_q}
    """

    distributed_pdf = get_spark_session().sql(distributed_sql).toPandas()
    if distributed_pdf.empty:
        raise ValueError(f"No distributed training rows were loaded from {SOURCE_TABLE}. Run setup/01_load_tabformer_dataset.py first.")

    distributed_records = [
        make_chat_record(row, SUSPICIOUS_AMOUNT_THRESHOLD)
        for _, row in distributed_pdf.iterrows()
    ]

    if rt.get_global_rank() == 0:
        print(f"Prepared {len(distributed_records)} distributed examples from the full Delta table")
        print(
            distributed_pdf.groupby("is_fraud")
            .size()
            .reset_index(name="row_count")
            .assign(dataset="distributed_full")
            .to_string(index=False)
        )

    return train_qwen35_unsloth(
        records=distributed_records,
        output_dir=DISTRIBUTED_OUTPUT_DIR,
        run_name=DISTRIBUTED_RUN_NAME,
        training_mode=f"distributed_{DISTRIBUTED_GPUS}x{DISTRIBUTED_GPU_TYPE}_full",
        num_gpus=DISTRIBUTED_GPUS,
        register_model=REGISTER_MODEL,
        device_map={"": local_rank},
        save_artifacts=rt.get_global_rank() == 0,
    )


distributed_run_ids = run_distributed_train.distributed()
DISTRIBUTED_RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)

print(f"Distributed MLflow run ID: {DISTRIBUTED_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance optimization loop
# MAGIC
# MAGIC After a successful baseline run, compare runtime, loss, and GPU utilization in MLflow and the notebook resource panels.
# MAGIC For distributed training, a common first bottleneck is an effective batch size that is too small to keep the GPUs busy.
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
# MAGIC ## Production handoff: model serving and autoscaling
# MAGIC
# MAGIC The fine-tuned model is registered to the Unity Catalog name printed in the next cell when `register_model = true`.
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
# MAGIC - Supervised chat records are generated from real table rows and passed directly into the training loops.
# MAGIC - AI Runtime provides managed serverless GPU compute for model training.
# MAGIC - The same training logic supports single-node validation and a scaled multi-GPU path.
# MAGIC - MLflow captures the experiment record, and Unity Catalog provides the handoff point for serving.
# MAGIC
# MAGIC The main platform outcome is speed with control: teams can move from governed data to GPU fine-tuning to registered model artifacts without leaving Databricks or stitching together separate infrastructure.
