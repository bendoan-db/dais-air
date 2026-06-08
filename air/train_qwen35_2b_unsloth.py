# Databricks notebook source
# DBTITLE 1,AIR DAIS demo - Qwen3.5 2B fraud fine-tuning with Unsloth
# MAGIC %md
# MAGIC # AIR DAIS demo: Qwen3.5 2B fraud fine-tuning with Unsloth
# MAGIC
# MAGIC This notebook follows the demo story in `demo_script/AIR DAIS Demo Script (go_air-dais-demo).pdf`:
# MAGIC
# MAGIC 1. Start with the fraud-detection business problem.
# MAGIC 2. Attach to AI Runtime serverless GPU compute.
# MAGIC 3. Validate the workload on a single GPU.
# MAGIC 4. Scale the training path by changing one demo integer.
# MAGIC 5. Use Genie Code guidance to tune throughput.
# MAGIC 6. Register the trained model for production serving and autoscaling.
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
# MAGIC - Databricks Unsloth example: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-llama-unsloth
# MAGIC - Unsloth Qwen3.5: https://unsloth.ai/docs/models/qwen3.5
# MAGIC - Unsloth Qwen3.5 fine-tuning: https://unsloth.ai/docs/models/qwen3.5/fine-tune

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0:00-0:45 - Scenario
# MAGIC
# MAGIC We are training a fraud decision model for a high-volume credit-card transaction business.
# MAGIC The setup notebook in `setup/01_load_tabformer_dataset.py` loads IBM TabFormer credit-card transactions into a Unity Catalog Delta table.
# MAGIC This notebook turns those transactions into supervised chat examples and fine-tunes `unsloth/Qwen3.5-2B` to emit structured fraud decisions.
# MAGIC
# MAGIC The output contract is a compact JSON object with:
# MAGIC
# MAGIC - `risk`: `legitimate`, `suspicious`, or `likely_fraud`
# MAGIC - `action`: downstream routing guidance
# MAGIC - `reason`: a short analyst-facing explanation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0:45-1:45 - WOW Moment 1: instant access to H100 compute
# MAGIC
# MAGIC Attach this notebook to **Serverless GPU** from the notebook compute picker.
# MAGIC
# MAGIC Recommended demo compute:
# MAGIC
# MAGIC - Accelerator: `1xH100` for the single-GPU validation path, or `8xH100` for the distributed path.
# MAGIC - Base environment: `AI v5`.
# MAGIC
# MAGIC If `1xH100` is not available in the workspace preview, `1xA10` is enough for this 2B bf16 LoRA demo.
# MAGIC Qwen3.5 2B LoRA is intentionally small so the workflow focuses on platform experience rather than model size.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install notebook requirements
# MAGIC
# MAGIC The path below is relative to this `air/` notebook source file and installs the top-level project requirements for both ingestion and training.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

print("Environment flags configured before importing Unsloth.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependency note
# MAGIC
# MAGIC AI Runtime `AI v5` includes Unsloth and common ML dependencies. If the version check below shows an older stack or Qwen3.5 load errors, run this once at the top of the notebook and restart Python:
# MAGIC
# MAGIC ```python
# MAGIC %pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
# MAGIC %restart_python
# MAGIC ```
# MAGIC
# MAGIC Unsloth recommends current Unsloth packages and Transformers v5 support for Qwen3.5. The demo keeps `%pip` out of the default execution path so an already-compatible AI v5 image is not unnecessarily modified.

# COMMAND ----------

import importlib.metadata as metadata

for package_name in [
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "mlflow",
]:
    try:
        print(f"{package_name}: {metadata.version(package_name)}")
    except metadata.PackageNotFoundError:
        print(f"{package_name}: not installed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo configuration
# MAGIC
# MAGIC Training settings are loaded from `air/training.yaml`, matching the setup notebook's YAML-driven ingestion pattern.
# MAGIC The PDF's second wow moment is the `num_nodes = 1 -> num_nodes = 4` change. Current AI Runtime training is selected from serverless GPU accelerators; the distributed notebook path maps the scaled demo mode to an `8xH100` multi-GPU run through the `serverless_gpu` API.
# MAGIC
# MAGIC For a live five-minute DAIS demo, keep `max_steps` low. Increase it for a real experiment.

# COMMAND ----------

from pathlib import Path

import yaml

try:
    notebook_dir = Path(__file__).resolve().parent
except NameError:
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = notebook_context.notebookPath().get()
    notebook_dir = Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]

config_path = notebook_dir / "training.yaml"

with config_path.open("r", encoding="utf-8") as config_file:
    training_config = yaml.safe_load(config_file)

if not isinstance(training_config, dict):
    raise ValueError(f"Expected YAML mapping in {config_path}, got {type(training_config).__name__}")


def config_value(key: str):
    if key not in training_config:
        raise KeyError(f"Missing required training config key: {key}")
    return training_config[key]


def config_str(key: str) -> str:
    value = str(config_value(key)).strip()
    if not value:
        raise ValueError(f"Training config key cannot be empty: {key}")
    return value


def config_int(key: str) -> int:
    return int(config_value(key))


def config_float(key: str) -> float:
    return float(config_value(key))


def config_bool(key: str) -> bool:
    value = config_value(key)
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Training config key must be boolean-like: {key}")


UC_CATALOG = config_str("catalog")
UC_SCHEMA = config_str("schema")
SOURCE_TABLE_NAME = config_str("source_table")
UC_VOLUME = config_str("checkpoint_volume")
UC_MODEL_NAME = config_str("uc_model_name")

MODEL_NAME = config_str("model_name")
NUM_NODES = config_int("num_nodes")
FRAUD_EXAMPLES = config_int("fraud_examples")
LEGIT_EXAMPLES = config_int("legit_examples")
MAX_SEQ_LENGTH = config_int("max_seq_length")
MAX_STEPS = config_int("max_steps")
PER_DEVICE_TRAIN_BATCH_SIZE = config_int("per_device_train_batch_size")
GRADIENT_ACCUMULATION_STEPS = config_int("gradient_accumulation_steps")
LEARNING_RATE = config_float("learning_rate")
SUSPICIOUS_AMOUNT_THRESHOLD = config_float("suspicious_amount_threshold")
REGISTER_MODEL = config_bool("register_model")
SEED = config_int("seed")
TRAINING_MODE = "single_gpu" if NUM_NODES == 1 else "distributed_8xh100"

SOURCE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SOURCE_TABLE_NAME}"
FULL_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
OUTPUT_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{UC_MODEL_NAME}"
DATASET_DIR = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/datasets/{UC_MODEL_NAME}"
DATASET_JSONL = f"{DATASET_DIR}/train.jsonl"
RUN_NAME = f"air-demo-{UC_MODEL_NAME}-{TRAINING_MODE}-steps{MAX_STEPS}"

print(f"Training config: {config_path}")
print(f"Source table: {SOURCE_TABLE}")
print(f"Base model: {MODEL_NAME}")
print(f"Demo num_nodes: {NUM_NODES}")
print(f"Training mode: {TRAINING_MODE}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"SFT dataset: {DATASET_JSONL}")
print(f"Register model: {REGISTER_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog and Spark Connect setup
# MAGIC
# MAGIC AI Runtime uses Unity Catalog for data and volume access. This cell creates the checkpoint volume if needed and opens a Spark session for reading the transaction Delta table.

# COMMAND ----------

from pathlib import Path


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def full_name(*parts: str) -> str:
    return ".".join(quote_identifier(part) for part in parts)


def get_spark_session():
    if "spark" in globals():
        return globals()["spark"]

    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.serverless().getOrCreate()


spark = get_spark_session()

schema_q = full_name(UC_CATALOG, UC_SCHEMA)
volume_q = full_name(UC_CATALOG, UC_SCHEMA, UC_VOLUME)
source_table_q = full_name(UC_CATALOG, UC_SCHEMA, SOURCE_TABLE_NAME)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_q}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_q}")

Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Ready: {schema_q}")
print(f"Ready: {volume_q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read and summarize the fraud data
# MAGIC
# MAGIC The live demo should show the table summary before fine-tuning to anchor the business problem: many credit-card transactions, rare fraud labels, and a need for a real-time decision.

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
# MAGIC ## Build a supervised fine-tuning dataset
# MAGIC
# MAGIC This cell samples fraudulent and legitimate TabFormer transactions, converts each row into a fraud analyst instruction, and writes JSONL to a Unity Catalog volume. The training loop reads from this local volume path with Hugging Face Datasets.

# COMMAND ----------

import json
from typing import Any

import pandas as pd


def clean_value(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def clean_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def transaction_prompt(row: pd.Series) -> str:
    amount = clean_float(row.get("amount"))
    return (
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        f"- user_id: {clean_value(row.get('user_id'))}\n"
        f"- card_id: {clean_value(row.get('card_id'))}\n"
        f"- timestamp: {clean_value(row.get('transaction_ts'))}\n"
        f"- amount_usd: {amount:.2f}\n"
        f"- use_chip: {clean_value(row.get('use_chip'))}\n"
        f"- merchant_city: {clean_value(row.get('merchant_city'))}\n"
        f"- merchant_state: {clean_value(row.get('merchant_state'))}\n"
        f"- merchant_category_code: {clean_value(row.get('mcc'))}\n"
        f"- errors: {clean_value(row.get('errors'), default='none')}"
    )


def transaction_answer(row: pd.Series) -> str:
    is_fraud = int(clean_float(row.get("is_fraud")))
    amount = clean_float(row.get("amount"))
    errors = clean_value(row.get("errors"), default="none")
    has_error_signal = errors.lower() not in {"none", "nan", "unknown", ""}

    if is_fraud == 1:
        payload = {
            "risk": "likely_fraud",
            "action": "decline_and_escalate",
            "reason": "The historical label marks this transaction as fraud.",
        }
    elif has_error_signal or amount >= SUSPICIOUS_AMOUNT_THRESHOLD:
        payload = {
            "risk": "suspicious",
            "action": "step_up_authentication",
            "reason": "The transaction is not labeled fraud, but amount or error signals warrant review.",
        }
    else:
        payload = {
            "risk": "legitimate",
            "action": "approve",
            "reason": "The historical label is non-fraud and no strong review signal is present.",
        }

    return json.dumps(payload, separators=(",", ": "))


def make_chat_record(row: pd.Series) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": transaction_prompt(row)},
            {"role": "assistant", "content": transaction_answer(row)},
        ],
        "label": "fraud" if int(clean_float(row.get("is_fraud"))) == 1 else "non_fraud",
        "transaction": {
            "user_id": clean_value(row.get("user_id")),
            "card_id": clean_value(row.get("card_id")),
            "amount": clean_float(row.get("amount")),
            "merchant_city": clean_value(row.get("merchant_city")),
            "merchant_state": clean_value(row.get("merchant_state")),
            "mcc": clean_value(row.get("mcc")),
            "is_fraud": int(clean_float(row.get("is_fraud"))),
        },
    }


dataset_sql = f"""
WITH fraud AS (
  SELECT
    user_id,
    card_id,
    CAST(transaction_ts AS STRING) AS transaction_ts,
    amount,
    use_chip,
    merchant_city,
    merchant_state,
    mcc,
    errors,
    is_fraud
  FROM {source_table_q}
  WHERE is_fraud = 1
  ORDER BY rand({SEED})
  LIMIT {FRAUD_EXAMPLES}
),
legit AS (
  SELECT
    user_id,
    card_id,
    CAST(transaction_ts AS STRING) AS transaction_ts,
    amount,
    use_chip,
    merchant_city,
    merchant_state,
    mcc,
    errors,
    is_fraud
  FROM {source_table_q}
  WHERE is_fraud = 0
  ORDER BY rand({SEED})
  LIMIT {LEGIT_EXAMPLES}
)
SELECT * FROM fraud
UNION ALL
SELECT * FROM legit
"""

training_pdf = spark.sql(dataset_sql).toPandas()
if training_pdf.empty:
    raise ValueError(f"No training rows were sampled from {SOURCE_TABLE}. Run setup/01_load_tabformer_dataset.py first.")

training_pdf = training_pdf.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
records = [make_chat_record(row) for _, row in training_pdf.iterrows()]

with open(DATASET_JSONL, "w", encoding="utf-8") as dataset_file:
    for record in records:
        dataset_file.write(json.dumps(record, ensure_ascii=True) + "\n")

print(f"Wrote {len(records)} supervised chat examples to {DATASET_JSONL}")
display(training_pdf.groupby("is_fraud").size().reset_index(name="row_count"))

# COMMAND ----------

# DBTITLE 1,Preview training records
preview_records = [
    {
        "prompt": record["messages"][0]["content"][:700],
        "assistant": record["messages"][1]["content"],
    }
    for record in records[:3]
]

display(pd.DataFrame(preview_records))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1:45-2:45 - WOW Moment 2: scale by changing one integer
# MAGIC
# MAGIC The `num_nodes` value in `air/training.yaml` is the demo control:
# MAGIC
# MAGIC - `num_nodes = 1`: train on the attached single GPU.
# MAGIC - `num_nodes = 4`: run the distributed demo path using AI Runtime `8xH100` and the `serverless_gpu` API.
# MAGIC
# MAGIC For the live demo, show this cell, change `num_nodes` in `training.yaml` from `1` to `4`, and rerun from the demo configuration cell on an `8xH100` AI Runtime notebook.

# COMMAND ----------

scale_config = {
    "num_nodes": NUM_NODES,
    "training_mode": TRAINING_MODE,
    "gpu_accelerator_to_select": "1xH100 or 1xA10" if TRAINING_MODE == "single_gpu" else "8xH100",
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_micro_batch_per_step": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
}

display(pd.DataFrame([scale_config]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune Qwen3.5 2B with Unsloth
# MAGIC
# MAGIC This uses bf16/16-bit LoRA instead of QLoRA. Unsloth's Qwen3.5 guide calls out that QLoRA is not recommended for this model family because quantization differences are higher than normal.
# MAGIC
# MAGIC The trainer logs to MLflow, saves checkpoints to the Unity Catalog volume, and optionally registers the merged model in Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Cell 21
from contextlib import contextmanager


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


def train_qwen35_unsloth(device_map=None, save_artifacts: bool = True) -> str:
    import mlflow
    import torch
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    mlflow.set_registry_uri("databricks-uc")

    dataset = load_dataset("json", data_files={"train": DATASET_JSONL}, split="train")

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
        output_dir=OUTPUT_DIR,
        report_to="mlflow",
        run_name=RUN_NAME,
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

    with start_mlflow_run(mlflow, RUN_NAME) as run:
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "training_mode": TRAINING_MODE,
                "num_nodes": NUM_NODES,
                "max_seq_length": MAX_SEQ_LENGTH,
                "max_steps": MAX_STEPS,
                "dataset_jsonl": DATASET_JSONL,
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
            trainer.save_model(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            mlflow.log_param("adapter_output_dir", OUTPUT_DIR)

            if REGISTER_MODEL:
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
                    print(f"LoRA adapters remain saved at: {OUTPUT_DIR}")

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run training
# MAGIC
# MAGIC Keep `num_nodes = 1` in `training.yaml` for the initial validation pass. Switch to `num_nodes = 4` and attach `8xH100` compute to show the scaled path.

# COMMAND ----------

if TRAINING_MODE == "single_gpu":
    RUN_ID = train_qwen35_unsloth()
else:
    from serverless_gpu import distributed

    @distributed(gpus=8, gpu_type="h100")
    def run_distributed_train():
        import os
        import torch
        from serverless_gpu import runtime as rt

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return train_qwen35_unsloth(
            device_map={"": local_rank},
            save_artifacts=rt.get_global_rank() == 0,
        )

    distributed_run_ids = run_distributed_train.distributed()
    RUN_ID = next((run_id for run_id in distributed_run_ids if run_id), None)

print(f"MLflow run ID: {RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2:45-4:15 - WOW Moment 3: agent-first performance optimization
# MAGIC
# MAGIC At this point in the live demo, open Genie Code and ask:
# MAGIC
# MAGIC > Why is my distributed training job not scaling as expected?
# MAGIC
# MAGIC If Genie Code identifies small batch size as the throughput bottleneck, update the YAML values and rerun training:
# MAGIC
# MAGIC - Increase `per_device_train_batch_size` if memory allows.
# MAGIC - Otherwise increase `gradient_accumulation_steps`.
# MAGIC - Watch the GPU resources pane and MLflow metrics for higher utilization and stable loss.

# COMMAND ----------

genie_suggested_config = {
    "current_per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "current_gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "suggested_per_device_train_batch_size": max(PER_DEVICE_TRAIN_BATCH_SIZE, 4),
    "suggested_gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "rerun_instruction": "Update training.yaml, rerun from Demo configuration, and compare MLflow loss/runtime/GPU metrics.",
}

display(pd.DataFrame([genie_suggested_config]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production handoff: model serving and autoscaling
# MAGIC
# MAGIC The fine-tuned model is registered to the Unity Catalog name printed in the next cell when `register_model = true`.
# MAGIC
# MAGIC For the final demo beat:
# MAGIC
# MAGIC 1. Open the registered model in Unity Catalog.
# MAGIC 2. Click **Serve this model**.
# MAGIC 3. Create a Mosaic AI Model Serving endpoint with autoscaling enabled.
# MAGIC 4. Send a sample transaction request.
# MAGIC 5. Start the load generator and show replicas scaling while the endpoint remains healthy.
# MAGIC
# MAGIC The request payload should keep the same prompt contract used during fine-tuning: ask for compact JSON with `risk`, `action`, and `reason`.

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
# MAGIC ## 4:15-5:00 - Conclusion
# MAGIC
# MAGIC In one notebook:
# MAGIC
# MAGIC - The fraud use case is anchored in a Unity Catalog Delta table.
# MAGIC - AI Runtime supplies serverless GPU access.
# MAGIC - The same notebook validates single-GPU training and exposes the scaled training path.
# MAGIC - Genie Code provides a concrete performance tuning moment.
# MAGIC - MLflow and Unity Catalog prepare the fine-tuned Qwen3.5 2B model for autoscaled serving.
