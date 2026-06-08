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
# MAGIC - **Unified data and governance:** read source transactions from Unity Catalog Delta tables and write checkpoints, datasets, and models to governed Unity Catalog assets.
# MAGIC - **Simple scaling path:** start with a single-GPU validation run, then use the same notebook and configuration to move into a multi-GPU training path.
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
# MAGIC - Accelerator: `1xH100` for the single-GPU validation path, or `8xH100` for the distributed path.
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
# MAGIC Unsloth recommends current Unsloth packages and Transformers v5 support for Qwen3.5.
# MAGIC The notebook keeps unnecessary package mutation out of the default execution path when the selected AI Runtime image is already compatible.

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
# MAGIC ## Training configuration
# MAGIC
# MAGIC Training settings are loaded from `air/training.yaml`.
# MAGIC This keeps the notebook body stable while making the experiment easy to tune:
# MAGIC
# MAGIC - `catalog`, `schema`, and `source_table` point to the governed Delta table.
# MAGIC - `checkpoint_volume` controls where datasets, adapters, and model artifacts are written.
# MAGIC - `num_nodes` selects the validation path (`1`) or the scaled training path (`4`).
# MAGIC - `max_steps`, batch size, and learning rate control the training cost and runtime.
# MAGIC
# MAGIC For a short walkthrough, keep `max_steps` low. For a real experiment, increase `max_steps`, broaden the sampled dataset, and compare runs in MLflow.

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
print(f"Configured num_nodes: {NUM_NODES}")
print(f"Training mode: {TRAINING_MODE}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"SFT dataset: {DATASET_JSONL}")
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
# MAGIC - The generated supervised fine-tuning dataset is written to a governed volume.
# MAGIC - Training checkpoints and model artifacts are stored in the same catalog and schema.
# MAGIC - The final registered model can inherit the same governance boundary as the data used to train it.

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
# MAGIC ## Build a supervised fine-tuning dataset
# MAGIC
# MAGIC This cell samples fraudulent and legitimate transactions from the prepared Delta table, converts each row into a fraud analyst instruction, and writes JSONL to a Unity Catalog volume.
# MAGIC The training loop reads the JSONL file with Hugging Face Datasets.
# MAGIC
# MAGIC Data cleaning has already happened in the ingestion notebook. This cell only assembles prompts from stable fields such as `amount_usd`, `transaction_ts_text`, `errors_text`, and `fraud_label`.
# MAGIC Prompt construction does two important things:
# MAGIC
# MAGIC - It presents structured transaction attributes in a stable format.
# MAGIC - It teaches the assistant response to follow the JSON contract used later for serving.
# MAGIC
# MAGIC Writing the generated dataset to a volume makes the training data inspectable and reusable across reruns.

# COMMAND ----------

import json

import pandas as pd


def transaction_prompt(row: pd.Series) -> str:
    amount = float(row["amount_usd"])
    return (
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        f"- user_id: {row['user_id_text']}\n"
        f"- card_id: {row['card_id_text']}\n"
        f"- timestamp: {row['transaction_ts_text']}\n"
        f"- amount_usd: {amount:.2f}\n"
        f"- use_chip: {row['use_chip_text']}\n"
        f"- merchant_city: {row['merchant_city_text']}\n"
        f"- merchant_state: {row['merchant_state_text']}\n"
        f"- merchant_category_code: {row['mcc_text']}\n"
        f"- errors: {row['errors_text']}"
    )


def transaction_answer(row: pd.Series) -> str:
    is_fraud = int(row["is_fraud"])
    amount = float(row["amount_usd"])
    has_error_signal = bool(row["has_error_signal"])

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


def make_chat_record(row: pd.Series) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": transaction_prompt(row)},
            {"role": "assistant", "content": transaction_answer(row)},
        ],
        "label": row["fraud_label"],
        "transaction": {
            "user_id": row["user_id_text"],
            "card_id": row["card_id_text"],
            "amount": float(row["amount_usd"]),
            "merchant_city": row["merchant_city_text"],
            "merchant_state": row["merchant_state_text"],
            "mcc": row["mcc_text"],
            "is_fraud": int(row["is_fraud"]),
        },
    }


dataset_sql = f"""
WITH fraud AS (
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
  WHERE is_fraud = 1
  ORDER BY rand({SEED})
  LIMIT {FRAUD_EXAMPLES}
),
legit AS (
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
# MAGIC ## Scale the training path with configuration
# MAGIC
# MAGIC Start with `num_nodes: 1` to validate the data pipeline, prompt formatting, model loading, and LoRA training loop on a single GPU.
# MAGIC After the workflow is correct, change `num_nodes` in `air/training.yaml` to use the scaled path:
# MAGIC
# MAGIC - `num_nodes: 1`: train on the attached single GPU.
# MAGIC - `num_nodes: 4`: run the distributed path using AI Runtime `8xH100` and the `serverless_gpu` API.
# MAGIC
# MAGIC The value driver is operational simplicity: the notebook keeps the same data access pattern, training function, MLflow logging, and artifact locations while the compute shape changes.
# MAGIC This lets teams validate cheaply, then scale when the workload is ready.

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
# MAGIC This section fine-tunes `unsloth/Qwen3.5-2B` with LoRA adapters.
# MAGIC It uses bf16/16-bit LoRA instead of QLoRA because Unsloth's Qwen3.5 guidance does not recommend QLoRA for this model family.
# MAGIC
# MAGIC The implementation highlights the production workflow around training:
# MAGIC
# MAGIC - MLflow records parameters, metrics, and run metadata.
# MAGIC - Checkpoints and adapters are saved to a Unity Catalog volume.
# MAGIC - If enabled, the merged model is registered to Unity Catalog for downstream serving.
# MAGIC - GPU memory metrics are logged when CUDA is available, which helps compare single-GPU and scaled runs.

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
# MAGIC Execute the training function selected by `training.yaml`.
# MAGIC A single-GPU run calls the trainer directly. A scaled run uses the `serverless_gpu` distributed API to launch the same trainer across H100 GPUs.
# MAGIC
# MAGIC The recommended operating pattern is:
# MAGIC
# MAGIC 1. Run `num_nodes: 1` first to catch data, dependency, and prompt-format issues quickly.
# MAGIC 2. Review MLflow metrics and sample outputs.
# MAGIC 3. Move to the scaled path when the training loop is stable and the workload needs more throughput.

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
# MAGIC - Supervised chat records are generated from real table rows and stored in a Unity Catalog volume.
# MAGIC - AI Runtime provides managed serverless GPU compute for model training.
# MAGIC - The same training logic supports single-GPU validation and a scaled multi-GPU path.
# MAGIC - MLflow captures the experiment record, and Unity Catalog provides the handoff point for serving.
# MAGIC
# MAGIC The main platform outcome is speed with control: teams can move from governed data to GPU fine-tuning to registered model artifacts without leaving Databricks or stitching together separate infrastructure.
