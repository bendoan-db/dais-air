"""Standalone training entrypoint for the AIR fraud fine-tuning demo.

This module owns the Unsloth LoRA training implementation and runs two ways:

- Imported by the ``train_qwen3_4b_unsloth`` notebook, whose ``@distributed``
  cell calls :func:`run_rank_training` on each GPU worker.
- Executed directly as ``python train.py`` by the AI Runtime CLI
  (``air run --file train.yaml``), where each GPU worker runs this file.

Configuration comes from the ``training_config`` section of ``train.yaml``
(the same file that defines the AI Runtime CLI workload) and shared helpers
from ``utils.py``, both in the same directory as this file.
"""

import os

# These must be set before unsloth is imported.
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import nullcontext

import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

from utils import (
    config_float,
    config_int,
    config_str,
    config_value,
    full_name,
    get_spark_session,
    load_yaml_config,
)

CONFIG_PATH, WORKLOAD_CONFIG = load_yaml_config("train.yaml")
TRAINING_CONFIG = config_value(WORKLOAD_CONFIG, "training_config")

UC_CATALOG = config_str(TRAINING_CONFIG, "catalog")
UC_SCHEMA = config_str(TRAINING_CONFIG, "schema")
SOURCE_TABLE_NAME = config_str(TRAINING_CONFIG, "source_table")
SFT_TABLE_NAME = config_str(TRAINING_CONFIG, "sft_table")
UC_VOLUME = config_str(TRAINING_CONFIG, "checkpoint_volume")
UC_MODEL_NAME = config_str(TRAINING_CONFIG, "uc_model_name")

MODEL_NAME = config_str(TRAINING_CONFIG, "model_name")
MAX_SEQ_LENGTH = config_int(TRAINING_CONFIG, "max_seq_length")
MAX_STEPS = config_int(TRAINING_CONFIG, "max_steps")
PER_DEVICE_TRAIN_BATCH_SIZE = config_int(TRAINING_CONFIG, "per_device_train_batch_size")
GRADIENT_ACCUMULATION_STEPS = config_int(TRAINING_CONFIG, "gradient_accumulation_steps")
LEARNING_RATE = config_float(TRAINING_CONFIG, "learning_rate")
TRAINING_SAMPLE_FRACTION = config_float(TRAINING_CONFIG, "training_sample_fraction")
SEED = config_int(TRAINING_CONFIG, "seed")

SOURCE_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SOURCE_TABLE_NAME}"
SFT_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{SFT_TABLE_NAME}"
SFT_TABLE_QUALIFIED = full_name(UC_CATALOG, UC_SCHEMA, SFT_TABLE_NAME)
TRAINING_OUTPUT_DIR = (
    f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{UC_MODEL_NAME}/training_demo"
)
TRAINING_RUN_NAME = f"air-demo-{UC_MODEL_NAME}-training-steps{MAX_STEPS}"


def load_unsloth_model(model_name: str, device_map=None):
    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": None,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    return FastLanguageModel.from_pretrained(**load_kwargs)


def render_chat_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def train_qwen3_unsloth(
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

    model, tokenizer = load_unsloth_model(MODEL_NAME, device_map=device_map)

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
        # Unsloth's custom checkpointing is reentrant and fires DDP gradient hooks
        # twice per LoRA param under multi-GPU runs; non-reentrant HF checkpointing
        # is enabled through SFTConfig below instead.
        "use_gradient_checkpointing": False,
        "random_state": SEED,
        "use_rslora": False,
        "loftq_config": None,
        "max_seq_length": MAX_SEQ_LENGTH,
    }

    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    run_context = (
        mlflow.start_run(run_name=run_name, log_system_metrics=True)
        if is_main_process
        else nullcontext()
    )

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


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) under any launcher.

    Prefers the serverless_gpu runtime (notebook @distributed and AI Runtime
    workloads); falls back to torchrun-style environment variables, and to a
    single-process default when neither is present.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        from serverless_gpu import runtime as rt

        return rt.get_global_rank(), rt.get_world_size(), local_rank
    except Exception:
        return (
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
            local_rank,
        )


def run_rank_training() -> str | None:
    """Train this rank's shard slice of the SFT table.

    Returns the MLflow run id on rank 0 and ``None`` on other ranks.
    """
    import torch

    rank, world_size, local_rank = get_distributed_context()
    torch.cuda.set_device(local_rank)

    distributed_sql = f"""
    SELECT
      training_id,
      shard_id,
      prompt,
      assistant_response,
      fraud_label,
      is_fraud
    FROM {SFT_TABLE_QUALIFIED}
    WHERE pmod(shard_id, {world_size}) = {rank}
    """

    examples_pdf = (
        get_spark_session().sql(distributed_sql).sample(TRAINING_SAMPLE_FRACTION).toPandas()
    )

    try:
        return train_qwen3_unsloth(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-{world_size}gpu",
            training_mode=f"{world_size}_gpu_rank_sharded_sample",
            num_gpus=world_size,
            device_map={"": local_rank},
            save_artifacts=rank == 0,
            rank=rank,
            world_size=world_size,
        )
    finally:
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    rank, world_size, _ = get_distributed_context()
    run_id = run_rank_training()
    if rank == 0:
        print(f"Training MLflow run ID: {run_id}")
        print(f"Trained adapter output dir: {TRAINING_OUTPUT_DIR}/{world_size}gpu")


if __name__ == "__main__":
    main()
