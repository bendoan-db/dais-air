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

Model identity lives in ``train.yaml`` (``model_name``, ``model_weights_path``,
``expected_model_classes``, ``fsdp_wrap_classes``), so retargeting this project
at another causal LM is a configuration change, not a code change.
"""

import os

# These must be set before torch/transformers/trl initialize, which is why the
# heavy imports below are deferred into function bodies rather than hoisted.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")
# Accelerate defaults FSDP2 to SHARDED_STATE_DICT, which Trainer.save_model
# does not export as a reloadable Hugging Face checkpoint. Final save must
# gather one full state dict on rank zero before save_pretrained shards it.
os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"

import shutil
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from project_config import (
    VOLUME_PATH_PREFIX,
    TrainingConfig,
    claim_rank_shard_files,
    load_project_config,
    local_staging_root,
    sample_eval_records,
    stage_model_locally,
)
from sft_conversion import prepare_sft_records
from training_metrics import (
    build_compute_metrics,
    build_mlflow_metrics_callback,
    preprocess_logits_for_metrics,
)

CONFIG = load_project_config()

LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


# --------------------------------------------------------------------------
# Model and data
# --------------------------------------------------------------------------
def load_fsdp_model_and_tokenizer(config: TrainingConfig):
    """Load the configured weights in bf16 without placing the model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = config.model_weights_path
    if model_source.startswith("/") and not Path(model_source).exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_source}. Populate "
            "model_weights_path with setup/04_download_base_model_weights.py "
            f"before training {config.model_name}."
        )
    if model_source.startswith(VOLUME_PATH_PREFIX):
        model_source = stage_model_locally(model_source)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_seq_length
    tokenizer.truncation_side = "right"

    # A text-only causal class is required: the vision tower and multimodal
    # projection of a multimodal checkpoint must stay out of text SFT.
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    loaded_class = model.__class__.__name__
    if loaded_class not in config.expected_model_classes:
        raise TypeError(
            f"AutoModelForCausalLM loaded {loaded_class}, which is not one of "
            f"the configured expected_model_classes "
            f"{list(config.expected_model_classes)}. Check the Transformers "
            "version and the model snapshot, then update train.yaml."
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


def infer_transformer_blocks_for_fsdp(
    model, candidate_classes: tuple[str, ...]
) -> list[str]:
    """Detect which configured decoder block classes this model actually has."""
    hits = {
        module.__class__.__name__
        for _, module in model.named_modules()
        if module.__class__.__name__ in candidate_classes
    }
    if not hits:
        raise RuntimeError(
            "No configured decoder layer was found for FSDP auto-wrapping. "
            f"fsdp_wrap_classes={list(candidate_classes)}. Inspect "
            "model.named_modules() and update fsdp_wrap_classes in train.yaml."
        )
    return sorted(hits)


def _dataset_from_records(
    records_pdf: pd.DataFrame, tokenizer, config: TrainingConfig
):
    from datasets import Dataset

    normalized_pdf = prepare_sft_records(
        records_pdf,
        convert_sft=config.convert_sft,
        suspicious_amount_threshold=config.suspicious_amount_threshold,
    )
    return normalized_pdf, Dataset.from_dict(
        {"text": render_training_text(normalized_pdf, tokenizer)}
    )


def _build_eval_dataset(config: TrainingConfig, tokenizer):
    """Build the held-out dataset, or None when evaluation is disabled."""
    if config.eval_sample_size <= 0:
        return None
    eval_pdf = sample_eval_records(
        config.eval_data_path,
        config.eval_sample_size,
        config.seed,
        stratify_column="is_fraud",
        ignore_partitions=config.ignore_partitions,
    )
    _, eval_dataset = _dataset_from_records(eval_pdf, tokenizer, config)
    return eval_dataset


# --------------------------------------------------------------------------
# Trainer construction
# --------------------------------------------------------------------------
def _assert_all_parameters_trainable(model) -> tuple[int, int]:
    """Return (trainable, total) after confirming this is a full-weight run."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full-weight training requires every parameter to be trainable: "
            f"{trainable_parameters:,} of {total_parameters:,} are trainable."
        )
    return trainable_parameters, total_parameters


def _build_sft_config(
    config: TrainingConfig,
    fsdp_wrap_classes: list[str],
    local_output_dir: Path,
    run_name: str,
    has_eval_dataset: bool,
):
    """Assemble the TRL/Transformers training arguments.

    This is the block to tune first when adapting the project: batch shape,
    optimizer, precision, evaluation cadence, and the FSDP2 sharding plan.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(local_output_dir),
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        # TRL's chunked_nll path omits logits and returns scalar counters,
        # which makes decoded classification metrics impossible.
        loss_type="nll",
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        eval_strategy="steps" if has_eval_dataset else "no",
        eval_steps=config.eval_steps,
        do_eval=has_eval_dataset,
        prediction_loss_only=False,
        eval_do_concat_batches=True,
        save_strategy="no",
        report_to=["mlflow"],
        run_name=run_name,
        seed=config.seed,
        data_seed=config.seed,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        # Activation checkpointing is requested through fsdp_config below;
        # enabling it here as well would wrap the modules twice.
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


def _log_training_params(
    config: TrainingConfig,
    *,
    training_mode: str,
    num_gpus: int,
    world_size: int,
    record_count: int,
    trainable_parameters: int,
    total_parameters: int,
    fsdp_wrap_classes: list[str],
) -> None:
    import mlflow

    mlflow.log_params(
        {
            "base_model": config.model_name,
            "base_model_load_path": config.model_weights_path,
            "training_scope": "full_weight",
            "training_mode": training_mode,
            "num_gpus": num_gpus,
            "world_size": world_size,
            "max_seq_length": config.max_seq_length,
            "max_steps": config.max_steps,
            "logging_steps": config.logging_steps,
            "eval_steps": config.eval_steps,
            "configured_eval_sample_size": config.eval_sample_size,
            "rank_0_training_record_count": record_count,
            "train_data_path": config.train_data_path,
            "eval_data_path": config.eval_data_path,
            "convert_sft": config.convert_sft,
            "ignore_partitions": config.ignore_partitions,
            "suspicious_amount_threshold": config.suspicious_amount_threshold,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "fsdp_wrap_classes": ",".join(fsdp_wrap_classes),
        }
    )


def _publish_checkpoint(local_output_dir: Path, output_dir: str) -> None:
    """Replace the UC volume checkpoint with the assembled local one.

    Files are copied one at a time: concurrent safetensors writes through the
    volume FUSE mount fail with EAGAIN.
    """
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

    if not list(volume_dir.glob("*.safetensors")):
        raise RuntimeError(
            f"No safetensors checkpoint files were copied to {volume_dir}"
        )


# --------------------------------------------------------------------------
# Training entrypoints
# --------------------------------------------------------------------------
def train_full_weight_fsdp(
    *,
    config: TrainingConfig,
    examples_pdf: pd.DataFrame,
    output_dir: str,
    run_name: str,
    training_mode: str,
    num_gpus: int,
    rank: int = 0,
    world_size: int = 1,
) -> str | None:
    """Train all model weights and save one complete Hugging Face checkpoint."""
    import mlflow
    import mlflow.transformers
    import torch
    from trl import SFTTrainer

    mlflow.set_registry_uri("databricks-uc")
    is_main_process = rank == 0
    if is_main_process:
        mlflow.set_experiment(config.experiment_path)
        mlflow.transformers.autolog(
            log_models=False,
            log_datasets=False,
            exclusive=False,
        )

    model, tokenizer = load_fsdp_model_and_tokenizer(config)
    examples_pdf, train_dataset = _dataset_from_records(examples_pdf, tokenizer, config)
    eval_dataset = _build_eval_dataset(config, tokenizer)
    has_eval_dataset = eval_dataset is not None

    trainable_parameters, total_parameters = _assert_all_parameters_trainable(model)
    fsdp_wrap_classes = infer_transformer_blocks_for_fsdp(
        model, config.fsdp_wrap_classes
    )
    if is_main_process:
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"FSDP auto-wrap classes: {fsdp_wrap_classes}")

    local_output_dir = local_staging_root() / "air-training-output" / run_name
    if is_main_process:
        shutil.rmtree(local_output_dir, ignore_errors=True)

    trainer = SFTTrainer(
        model=model,
        args=_build_sft_config(
            config, fsdp_wrap_classes, local_output_dir, run_name, has_eval_dataset
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[build_mlflow_metrics_callback(has_eval_dataset)],
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
            _log_training_params(
                config,
                training_mode=training_mode,
                num_gpus=num_gpus,
                world_size=world_size,
                record_count=len(examples_pdf),
                trainable_parameters=trainable_parameters,
                total_parameters=total_parameters,
                fsdp_wrap_classes=fsdp_wrap_classes,
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

        _publish_checkpoint(local_output_dir, output_dir)
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


def run_rank_training(
    sample_fraction: float | None = None,
    config: TrainingConfig | None = None,
) -> str | None:
    """Load this rank's records and run collective full-weight FSDP training."""
    import torch

    config = config or CONFIG
    if sample_fraction is None:
        sample_fraction = config.training_sample_fraction

    rank, world_size, local_rank = get_distributed_context()
    torch.cuda.set_device(local_rank)

    rank_files, within_shard_fraction = claim_rank_shard_files(
        config.train_data_path,
        rank,
        world_size,
        sample_fraction,
        config.seed,
        ignore_partitions=config.ignore_partitions,
    )

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=rank_files, split="train")
    examples_pdf = dataset.to_pandas()
    if within_shard_fraction < 1.0:
        examples_pdf = examples_pdf.sample(
            frac=within_shard_fraction, random_state=config.seed
        )
    if examples_pdf.empty:
        raise ValueError(
            "Training sampling produced no rows. Increase training_sample_fraction."
        )

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""
    data_mode = "full_dataset" if config.ignore_partitions else "rank_sharded"

    try:
        return train_full_weight_fsdp(
            config=config,
            examples_pdf=examples_pdf,
            output_dir=config.output_dir_for(world_size),
            run_name=f"{config.run_name}-full-fsdp-{world_size}gpu{run_suffix}",
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
        print(f"Trained full-model output dir: {CONFIG.output_dir_for(world_size)}")


if __name__ == "__main__":
    main()
