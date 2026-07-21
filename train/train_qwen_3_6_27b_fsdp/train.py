"""Full-weight Qwen3.6 27B supervised fine-tuning with PyTorch FSDP2.

The project runs either from ``01_runner.py`` on Databricks Serverless GPU or
directly through ``air run --file train.yaml``. It owns its configuration,
dependencies, data conversion, and output paths; no other training project is
imported.

All model parameters are trainable. FSDP2 full-shards parameters, gradients,
and optimizer states across the configured GPUs and activation checkpointing
reduces activation memory. Stepwise evaluation reports loss, token accuracy,
and teacher-forced risk-classification metrics; autoregressive generation is
not safe while the model remains sharded. Every rank participates in the final
``save_model`` collective; rank zero then copies the complete Hugging Face
checkpoint to the configured Unity Catalog volume and logs ``model_output_dir``
on the MLflow run.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")
# Accelerate defaults FSDP2 to SHARDED_STATE_DICT, which Trainer.save_model
# does not export as a reloadable Hugging Face checkpoint. Final save must
# gather one full state dict on rank zero before save_pretrained shards it.
os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"

from contextlib import nullcontext
from pathlib import Path

import pandas as pd

try:
    from .project_config import (
        VOLUME_PATH_PREFIX,
        claim_rank_shard_files,
        load_project_config,
        sample_eval_records,
        stage_model_locally,
    )
    from .sft_conversion import prepare_sft_records
    from .training_metrics import (
        build_compute_metrics,
        build_mlflow_metrics_callback,
        preprocess_logits_for_metrics,
    )
except ImportError:
    from project_config import (
        VOLUME_PATH_PREFIX,
        claim_rank_shard_files,
        load_project_config,
        sample_eval_records,
        stage_model_locally,
    )
    from sft_conversion import prepare_sft_records
    from training_metrics import (
        build_compute_metrics,
        build_mlflow_metrics_callback,
        preprocess_logits_for_metrics,
    )


globals().update(load_project_config())

LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


def load_fsdp_model_and_tokenizer():
    """Load text-only Qwen3.6 weights in bf16 without placing the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = MODEL_LOAD_PATH
    if model_source.startswith("/") and not Path(model_source).exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_source}. Populate "
            "model_weights_path with setup/04_download_base_model_weights.py "
            f"before training {MODEL_NAME}."
        )
    if model_source.startswith(VOLUME_PATH_PREFIX):
        model_source = stage_model_locally(model_source)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    tokenizer.truncation_side = "right"

    # AutoModelForCausalLM selects Qwen3_5ForCausalLM for the qwen3_5 config,
    # excluding the vision tower and multimodal projection from text SFT.
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if model.__class__.__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(
            "Expected AutoModelForCausalLM to load Qwen3_5ForCausalLM, got "
            f"{model.__class__.__name__}. Check the Transformers version and "
            "model snapshot."
        )
    return model, tokenizer


def render_training_text(records_pdf: pd.DataFrame, tokenizer) -> list[str]:
    """Render complete user/assistant conversations with thinking disabled."""
    rendered = []
    for prompt, assistant_response in zip(
        records_pdf["prompt"], records_pdf["assistant_response"]
    ):
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(assistant_response)},
        ]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
    return rendered


def infer_transformer_blocks_for_fsdp(model) -> list[str]:
    """Detect decoder block classes for FSDP auto-wrapping."""
    common_block_classes = {
        "Qwen3_5DecoderLayer",
        "Qwen3_5MoeDecoderLayer",
        "Qwen3DecoderLayer",
        "Qwen2DecoderLayer",
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "MixtralDecoderLayer",
        "Gemma2DecoderLayer",
        "Phi3DecoderLayer",
    }
    hits = {
        module.__class__.__name__
        for _, module in model.named_modules()
        if module.__class__.__name__ in common_block_classes
    }
    if not hits:
        raise RuntimeError(
            "Could not find a supported decoder layer for FSDP auto-wrapping. "
            "Inspect model.named_modules() and add the block class explicitly."
        )
    return sorted(hits)


def _dataset_from_records(records_pdf: pd.DataFrame, tokenizer):
    from datasets import Dataset

    normalized_pdf = prepare_sft_records(
        records_pdf,
        convert_sft=CONVERT_SFT,
        suspicious_amount_threshold=SUSPICIOUS_AMOUNT_THRESHOLD,
    )
    return normalized_pdf, Dataset.from_dict(
        {"text": render_training_text(normalized_pdf, tokenizer)}
    )


def train_full_weight_fsdp(
    *,
    examples_pdf: pd.DataFrame,
    output_dir: str,
    run_name: str,
    training_mode: str,
    num_gpus: int,
    rank: int = 0,
    world_size: int = 1,
) -> str | None:
    """Train all model weights and save one complete Hugging Face checkpoint."""
    import shutil
    import tempfile

    import mlflow
    import mlflow.transformers
    import torch
    from trl import SFTConfig, SFTTrainer

    mlflow.set_registry_uri("databricks-uc")
    is_main_process = rank == 0
    if is_main_process:
        mlflow.set_experiment(EXPERIMENT_PATH)
        mlflow.transformers.autolog(
            log_models=False,
            log_datasets=False,
            exclusive=False,
        )

    model, tokenizer = load_fsdp_model_and_tokenizer()
    examples_pdf, train_dataset = _dataset_from_records(examples_pdf, tokenizer)

    eval_dataset = None
    if EVAL_SAMPLE_SIZE > 0:
        eval_pdf = sample_eval_records(
            EVAL_DATA_PATH,
            EVAL_SAMPLE_SIZE,
            SEED,
            stratify_column="is_fraud",
            ignore_partitions=IGNORE_PARTITIONS,
        )
        _, eval_dataset = _dataset_from_records(eval_pdf, tokenizer)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full-weight training requires every parameter to be trainable: "
            f"{trainable_parameters:,} of {total_parameters:,} are trainable."
        )

    fsdp_wrap_classes = infer_transformer_blocks_for_fsdp(model)
    if is_main_process:
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"FSDP auto-wrap classes: {fsdp_wrap_classes}")

    local_disk_tmp = Path("/local_disk0/tmp")
    staging_base = (
        local_disk_tmp if local_disk_tmp.exists() else Path(tempfile.gettempdir())
    )
    local_output_dir = staging_base / "air-training-output" / run_name
    if is_main_process:
        shutil.rmtree(local_output_dir, ignore_errors=True)

    training_args = SFTConfig(
        output_dir=str(local_output_dir),
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        # TRL's chunked_nll path omits logits and returns scalar counters,
        # which makes decoded classification metrics impossible.
        loss_type="nll",
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        do_eval=eval_dataset is not None,
        prediction_loss_only=False,
        eval_do_concat_batches=True,
        save_strategy="no",
        report_to=["mlflow"],
        run_name=run_name,
        seed=SEED,
        data_seed=SEED,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "version": 2,
            "state_dict_type": "FULL_STATE_DICT",
            "transformer_layer_cls_to_wrap": fsdp_wrap_classes,
            "reshard_after_forward": True,
            "activation_checkpointing": True,
            "xla": False,
            "limit_all_gathers": True,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[build_mlflow_metrics_callback(eval_dataset is not None)],
    )

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
                    "mlflow.runName": run_name,
                    "training_scope": "full_weight",
                }
            )
            mlflow.log_params(
                {
                    "base_model": MODEL_NAME,
                    "base_model_load_path": MODEL_LOAD_PATH,
                    "training_scope": "full_weight",
                    "training_mode": training_mode,
                    "num_gpus": num_gpus,
                    "world_size": world_size,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "max_steps": MAX_STEPS,
                    "logging_steps": LOGGING_STEPS,
                    "eval_steps": EVAL_STEPS,
                    "configured_eval_sample_size": EVAL_SAMPLE_SIZE,
                    "rank_0_training_record_count": len(examples_pdf),
                    "train_data_path": TRAIN_DATA_PATH,
                    "eval_data_path": EVAL_DATA_PATH,
                    "convert_sft": CONVERT_SFT,
                    "ignore_partitions": IGNORE_PARTITIONS,
                    "suspicious_amount_threshold": SUSPICIOUS_AMOUNT_THRESHOLD,
                    "trainable_parameters": trainable_parameters,
                    "total_parameters": total_parameters,
                    "fsdp_wrap_classes": ",".join(fsdp_wrap_classes),
                }
            )

        train_output = trainer.train()
        metrics = getattr(train_output, "metrics", {}) or {}

        # FSDP state gathering is collective. All ranks must enter save_model;
        # only the main process writes the assembled checkpoint files.
        trainer.save_model(str(local_output_dir))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not is_main_process:
            return None

        tokenizer.save_pretrained(local_output_dir)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"trainer_{metric_name}", float(metric_value))

        volume_dir = Path(output_dir)
        volume_dir.mkdir(parents=True, exist_ok=True)
        for existing_artifact in volume_dir.iterdir():
            if existing_artifact.is_dir():
                shutil.rmtree(existing_artifact)
            else:
                existing_artifact.unlink()
        for artifact_file in sorted(local_output_dir.iterdir()):
            if artifact_file.is_file():
                shutil.copy2(artifact_file, volume_dir / artifact_file.name)

        checkpoint_files = list(volume_dir.glob("*.safetensors"))
        if not checkpoint_files:
            raise RuntimeError(
                f"No safetensors checkpoint files were copied to {volume_dir}"
            )
        mlflow.log_param("model_output_dir", output_dir)

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id


def get_distributed_context() -> tuple[int, int, int]:
    """Return rank, world size, and local rank under either launcher."""
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
    """Load this rank's records and run collective full-weight FSDP training."""
    import torch

    if sample_fraction is None:
        sample_fraction = TRAINING_SAMPLE_FRACTION

    rank, world_size, local_rank = get_distributed_context()
    torch.cuda.set_device(local_rank)

    rank_files, within_shard_fraction = claim_rank_shard_files(
        TRAIN_DATA_PATH,
        rank,
        world_size,
        sample_fraction,
        SEED,
        ignore_partitions=IGNORE_PARTITIONS,
    )

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=rank_files, split="train")
    examples_pdf = dataset.to_pandas()
    if within_shard_fraction < 1.0:
        examples_pdf = examples_pdf.sample(
            frac=within_shard_fraction, random_state=SEED
        )
    if examples_pdf.empty:
        raise ValueError(
            "Training sampling produced no rows. Increase training_sample_fraction."
        )

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""
    data_mode = "full_dataset" if IGNORE_PARTITIONS else "rank_sharded"
    output_dir = f"{TRAINING_OUTPUT_DIR}/{world_size}gpu"

    try:
        return train_full_weight_fsdp(
            examples_pdf=examples_pdf,
            output_dir=output_dir,
            run_name=(
                f"{TRAINING_RUN_NAME}-full-fsdp-{world_size}gpu{run_suffix}"
            ),
            training_mode=f"full_weight_fsdp_{world_size}_gpu_{data_mode}",
            num_gpus=world_size,
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
        print(f"Trained full-model output dir: {TRAINING_OUTPUT_DIR}/{world_size}gpu")


if __name__ == "__main__":
    main()
