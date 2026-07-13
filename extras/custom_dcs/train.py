"""Standalone training entrypoint for the AIR fraud fine-tuning demo.

This module owns the Unsloth LoRA training implementation and runs two ways:

- Imported by the ``runner`` notebook, whose ``@distributed`` cell calls
  :func:`run_rank_training` on each GPU worker.
- Executed directly as ``python train.py`` by the AI Runtime CLI
  (``air run --file train.yaml``), where each GPU worker runs this file.

Configuration comes from the ``parameters.training_config`` section of
``train.yaml`` (the same file that defines the AI Runtime CLI workload;
under an AIR run it arrives via ``$HYPERPARAMETERS_PATH``) and shared
helpers from ``training_utils.py``, both in the same directory as this file.
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

from pathlib import Path

# The helpers module is deliberately NOT named `utils`: the Databricks AI base
# environment's nvidia_cutlass_dsl package registers its own top-level `utils`
# module once the torch/CUDA stack loads, which shadows any local utils.py.
from training_utils import load_training_config

# Bind the shared configuration into module globals: typed training_config
# values (MODEL_NAME, MAX_SEQ_LENGTH, hyperparameters, SEED, ...), derived
# names (SOURCE_TABLE, SFT_TABLE, TRAINING_OUTPUT_DIR, TRAINING_RUN_NAME), and
# quoted SQL identifiers (sft_table_q) — the same names the runner notebook
# binds into its session.
globals().update(load_training_config())

# The AI Runtime CLI launch wrapper exports these before running the script;
# neither is present under the notebook's @distributed path. Used to label
# MLflow runs with their launcher so CLI and notebook runs are
# distinguishable in the experiment.
LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


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
            mlflow.set_tags(
                {
                    "submitted_via": LAUNCHER,
                    # AIR pre-creates the workload's MLflow run and start_run
                    # resumes it, ignoring run_name — set the name explicitly
                    # so the launcher-suffixed name sticks on both paths.
                    "mlflow.runName": run_name,
                }
            )
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


def run_rank_training(sample_fraction: float | None = None) -> str | None:
    """Train this rank's shard slice of the exported SFT parquet files.

    ``sample_fraction`` overrides the ``training_sample_fraction`` value from
    the config — the runner notebook passes it from its training cell so the
    fraction can be changed live during the demo; the AIR CLI path leaves it
    ``None`` and uses the config value.

    Returns the MLflow run id on rank 0 and ``None`` on other ranks.
    """
    import torch
    from datasets import load_dataset

    if sample_fraction is None:
        sample_fraction = TRAINING_SAMPLE_FRACTION

    rank, world_size, local_rank = get_distributed_context()
    torch.cuda.set_device(local_rank)

    import math
    import random

    # Read the SFT records as parquet shard files from the UC volume instead
    # of querying Delta through Spark on the GPU workers, per the AIR
    # data-loading guidance for large Delta tables:
    # https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
    # Ingestion writes one shard_id=N directory per stable hash shard; each
    # rank claims the shards where N % world_size == rank, preserving the
    # original Delta rank-sharding contract.
    shard_dirs = sorted(Path(SFT_FILES_DIR).glob("shard_id=*"))
    if not shard_dirs:
        raise FileNotFoundError(
            f"No SFT parquet shards found under {SFT_FILES_DIR}. "
            "Run setup/01_load_tabformer_dataset.py first."
        )

    rank_shard_dirs = [
        shard_dir
        for shard_dir in shard_dirs
        if int(shard_dir.name.split("=", 1)[1]) % world_size == rank
    ]

    # Two-level sampling. shard_id is a uniform hash, so loading a subset of
    # shard directories is statistically equivalent to row sampling — and it
    # keeps the HF datasets Arrow conversion ("Generating train split")
    # proportional to sample_fraction instead of always materializing the
    # rank's full slice. Row-level sampling within the loaded shards then
    # lands on the exact requested fraction.
    within_shard_fraction = 1.0
    if sample_fraction < 1.0 and rank_shard_dirs:
        total_rank_dirs = len(rank_shard_dirs)
        dirs_to_load = max(1, math.ceil(total_rank_dirs * sample_fraction))
        rank_shard_dirs = sorted(random.Random(SEED).sample(rank_shard_dirs, dirs_to_load))
        within_shard_fraction = min(1.0, sample_fraction * total_rank_dirs / dirs_to_load)

    rank_files = [
        str(parquet_file)
        for shard_dir in rank_shard_dirs
        for parquet_file in sorted(shard_dir.glob("*.parquet"))
    ]

    dataset = load_dataset("parquet", data_files=rank_files, split="train")
    examples_pdf = dataset.to_pandas()
    if within_shard_fraction < 1.0:
        examples_pdf = examples_pdf.sample(frac=within_shard_fraction, random_state=SEED)

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""

    try:
        return train_qwen3_unsloth(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-{world_size}gpu{run_suffix}",
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
