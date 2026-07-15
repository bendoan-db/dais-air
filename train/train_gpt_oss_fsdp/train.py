"""FSDP training entrypoint for models too large for single-GPU LoRA.

A sibling implementation of ``train.py`` for large models (the worked example
targets ``openai/gpt-oss-120b``), following the Databricks AI Runtime
gpt-oss-120b DDP/FSDP tutorial:
https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-gpt-oss-120b-ddp-fsdp

Differences from ``train.py``:

- Plain TRL ``SFTTrainer`` + PyTorch **FSDP2** (``full_shard auto_wrap`` with
  activation checkpointing) instead of Unsloth — model parameters, gradients,
  and optimizer states shard across GPUs, so the model does not need to fit
  on one device. No Unsloth import anywhere in this module.
- LoRA via plain PEFT, targeting ``all-linear`` (plus reduced-rank MoE expert
  projections for gpt-oss models).
- Full-sequence loss (response-only masking is an Unsloth feature of the DDP
  path).
- Stepwise evaluation reports held-out loss, token accuracy, and teacher-forced
  risk-classification metrics. Autoregressive fraud evaluation remains
  unavailable because the model is FSDP-sharded during training and batched
  generation would require gathering it.

The project owns its runner, workload YAML, dependencies, and config helpers;
it does not load the Qwen project's configuration.

Runs two ways, like ``train.py``:

- Imported by this project's runner notebook's ``@distributed`` cell.
- Executed directly by the AI Runtime CLI: ``air run --file train.yaml``.

Configuration comes from ``parameters.training_config`` of ``train.yaml``
(via ``$HYPERPARAMETERS_PATH`` under an AIR run).
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_DEBUG", "WARN")
# Surface NCCL failures as exceptions instead of hangs.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")

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
except ImportError:
    from project_config import (
        VOLUME_PATH_PREFIX,
        claim_rank_shard_files,
        load_project_config,
        sample_eval_records,
        stage_model_locally,
    )

try:
    from .sft_conversion import prepare_sft_records
    from .training_metrics import build_compute_metrics, preprocess_logits_for_metrics
except ImportError:
    from sft_conversion import prepare_sft_records
    from training_metrics import build_compute_metrics, preprocess_logits_for_metrics


def conversational_records(records_pdf: "pd.DataFrame") -> list[dict]:
    """Convert normalized SFT rows to TRL's conversational format."""
    return [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        for prompt, assistant_response in zip(
            records_pdf["prompt"], records_pdf["assistant_response"]
        )
    ]

globals().update(load_project_config())

LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


def _restore_saved_adapter_base_path(output_dir: str) -> None:
    """Point the saved adapter config back at the configured base source.

    Same fixup as train.py: training loads volume-hosted weights from an
    ephemeral node-local staged copy, so PEFT records that local path as
    ``base_model_name_or_path`` — rewrite it to the durable MODEL_LOAD_PATH
    so registration on another machine can resolve the base model.
    """
    import json

    adapter_config_path = Path(output_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        return
    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("base_model_name_or_path") != MODEL_LOAD_PATH:
        adapter_config["base_model_name_or_path"] = MODEL_LOAD_PATH
        adapter_config_path.write_text(json.dumps(adapter_config, indent=2))


def load_fsdp_model_and_tokenizer():
    """Load the base model for FSDP training (no Unsloth, no device_map).

    Placement is deliberately left to Trainer/Accelerate+FSDP — no
    ``device_map`` and no ``.to(device)``. gpt-oss checkpoints ship
    MXFP4-quantized; ``Mxfp4Config(dequantize=True)`` unpacks them to bf16 at
    load time so FSDP shards plain bf16 parameters.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = MODEL_LOAD_PATH
    if model_source.startswith("/") and not Path(model_source).exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_source}. If this is "
            "model_weights_path from train.yaml; populate the volume "
            "first via setup/04_download_base_model_weights.py (add a "
            f"models entry for {MODEL_NAME})."
        )
    if model_source.startswith(VOLUME_PATH_PREFIX):
        # Same staging rationale as train.py: mmap reads through the volume
        # FUSE mount are latency-bound; stage to node-local disk first.
        model_source = stage_model_locally(model_source)

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    tokenizer.truncation_side = "right"

    load_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "use_cache": False,  # required with activation checkpointing
        "low_cpu_mem_usage": True,
    }
    if "gpt-oss" in MODEL_NAME.lower():
        from transformers import Mxfp4Config

        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    return model, tokenizer


def load_model_for_merge(adapter_output_dir: str):
    """Load the PEFT adapter and base model for deployment-time merging."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_source = adapter_output_dir
    if adapter_source.startswith("/") and not Path(adapter_source).exists():
        raise FileNotFoundError(f"Adapter output path does not exist: {adapter_source}")
    if adapter_source.startswith(VOLUME_PATH_PREFIX):
        adapter_source = stage_model_locally(adapter_source)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_source)
    return model, tokenizer


def infer_transformer_blocks_for_fsdp(model) -> list[str]:
    """Detect the transformer block classes FSDP should auto-wrap."""
    common_block_classes = {
        "LlamaDecoderLayer", "MistralDecoderLayer", "MixtralDecoderLayer",
        "Qwen2DecoderLayer", "Gemma2DecoderLayer", "Phi3DecoderLayer",
        "GPTNeoXLayer", "MPTBlock", "BloomBlock", "FalconDecoderLayer",
        "DecoderLayer", "GPTJBlock", "OPTDecoderLayer",
    }
    hits = set()
    for _, module in model.named_modules():
        name = module.__class__.__name__
        if name in common_block_classes:
            hits.add(name)
    if not hits:
        # Fallback: anything that looks like a decoder block.
        for _, module in model.named_modules():
            name = module.__class__.__name__
            if any(part in name for part in ("Block", "DecoderLayer", "Layer")) and (
                "Embedding" not in name
            ):
                hits.add(name)
    if not hits:
        raise RuntimeError(
            "Could not infer transformer block classes for FSDP wrapping; "
            "print(model) and add the block class explicitly."
        )
    return sorted(hits)


def build_peft_config():
    """LoRA over all linear layers; reduced-rank MoE expert targeting for gpt-oss."""
    from peft import LoraConfig

    # The loader validates lora_target_modules as a list; a single
    # "all-linear" entry means PEFT's all-linear shorthand.
    if LORA_TARGET_MODULES == ["all-linear"]:
        target_modules = "all-linear"
    else:
        target_modules = LORA_TARGET_MODULES

    peft_kwargs = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if "gpt-oss" in MODEL_NAME.lower():
        # Adapt the MoE expert projections too, at a reduced rank — the
        # expert tensors are huge, and rank 8 keeps the adapter tractable
        # (pattern from the Databricks gpt-oss-120b FSDP tutorial).
        peft_kwargs["rank_pattern"] = {
            "mlp.experts.gate_up_proj": 8,
            "mlp.experts.down_proj": 8,
        }
        peft_kwargs["target_parameters"] = [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]
    return LoraConfig(**peft_kwargs)


def train_fsdp(
    *,
    examples_pdf: pd.DataFrame,
    output_dir: str,
    run_name: str,
    training_mode: str,
    num_gpus: int,
    save_artifacts: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> str | None:
    import shutil
    import tempfile

    import mlflow
    import mlflow.transformers
    import torch
    from datasets import Dataset
    from peft import get_peft_model
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

    examples_pdf = prepare_sft_records(
        examples_pdf,
        convert_sft=CONVERT_SFT,
        suspicious_amount_threshold=SUSPICIOUS_AMOUNT_THRESHOLD,
    )
    # SFTTrainer applies the model's chat template to the normalized records.
    # This FSDP path trains with full-sequence loss.
    dataset = Dataset.from_list(conversational_records(examples_pdf))

    # Stepwise loss and teacher-forced classification metrics use the SFT eval
    # split. Forward passes on the sharded model are collective-safe, unlike
    # generation. Every rank draws the identical seeded sample; the Trainer's
    # distributed sampler then splits it across ranks.
    eval_dataset = None
    if EVAL_SAMPLE_SIZE > 0:
        eval_pdf = sample_eval_records(
            EVAL_DATA_PATH,
            EVAL_SAMPLE_SIZE,
            SEED,
            stratify_column="is_fraud",
            ignore_partitions=IGNORE_PARTITIONS,
        )
        eval_pdf = prepare_sft_records(
            eval_pdf,
            convert_sft=CONVERT_SFT,
            suspicious_amount_threshold=SUSPICIOUS_AMOUNT_THRESHOLD,
        )
        eval_dataset = Dataset.from_list(conversational_records(eval_pdf))

    model, tokenizer = load_fsdp_model_and_tokenizer()
    model = get_peft_model(model, build_peft_config())
    # Cast everything to bf16 so FSDP sees a uniform dtype (LoRA adapters
    # initialize in fp32).
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)

    fsdp_wrap_classes = infer_transformer_blocks_for_fsdp(model)
    if is_main_process:
        print(f"FSDP auto-wrap classes: {fsdp_wrap_classes}")

    # Same /Volumes-FUSE constraint as train.py: safetensors serialization to
    # the volume mount fails with EAGAIN, so train and save on node-local
    # disk, then copy the finished files to the volume sequentially.
    local_disk_tmp = Path("/local_disk0/tmp")
    staging_base = local_disk_tmp if local_disk_tmp.exists() else Path(tempfile.gettempdir())
    local_output_dir = str(staging_base / "air-training-output" / run_name)
    if is_main_process:
        shutil.rmtree(local_output_dir, ignore_errors=True)

    training_args = SFTConfig(
        output_dir=local_output_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        report_to=["mlflow"],
        run_name=run_name,
        seed=SEED,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        max_length=MAX_SEQ_LENGTH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        # Activation checkpointing below replaces gradient checkpointing.
        gradient_checkpointing=False,
        # ---- FSDP2 ----
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "version": 2,
            "fsdp_transformer_layer_cls_to_wrap": fsdp_wrap_classes,
            "reshard_after_forward": True,
            "activation_checkpointing": True,
            "xla": False,
            "limit_all_gathers": True,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    run_context = (
        mlflow.start_run(run_name=run_name, log_system_metrics=True)
        if is_main_process
        else nullcontext()
    )

    with run_context as run:
        if is_main_process:
            mlflow.set_tags({"submitted_via": LAUNCHER, "mlflow.runName": run_name})
            mlflow.log_params(
                {
                    "base_model": MODEL_NAME,
                    "base_model_load_path": MODEL_LOAD_PATH,
                    "training_mode": training_mode,
                    "num_gpus": num_gpus,
                    "rank": rank,
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
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                    "fsdp_wrap_classes": ",".join(fsdp_wrap_classes),
                }
            )

        train_output = trainer.train()
        metrics = getattr(train_output, "metrics", {}) or {}

        # Gathering the sharded parameters for the final adapter is a
        # collective operation — EVERY rank must call save_model (the
        # Trainer writes files only on the main process).
        if save_artifacts:
            trainer.save_model(local_output_dir)

        if not is_main_process:
            return None

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"trainer_{metric_name}", float(metric_value))

        if save_artifacts:
            tokenizer.save_pretrained(local_output_dir)
            _restore_saved_adapter_base_path(local_output_dir)
            # Top-level files only; sequential whole-file copies are the
            # write pattern the volume mount supports.
            volume_dir = Path(output_dir)
            volume_dir.mkdir(parents=True, exist_ok=True)
            for artifact_file in sorted(Path(local_output_dir).iterdir()):
                if artifact_file.is_file():
                    shutil.copy2(artifact_file, volume_dir / artifact_file.name)
            mlflow.log_param("adapter_output_dir", output_dir)

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) under any launcher."""
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
    """Train this rank's shard slice with FSDP — same contract as train.py.

    Note that under FSDP every rank trains the SAME sharded model; the
    shard-claiming here still governs which data each rank feeds it (FSDP is
    still data-parallel across ranks).
    """
    import torch

    if sample_fraction is None:
        sample_fraction = TRAINING_SAMPLE_FRACTION

    rank, world_size, local_rank = get_distributed_context()
    torch.cuda.set_device(local_rank)

    # Claim raw or pre-converted split shards where N % world_size == rank;
    # conversion, when enabled, happens after loading and sampling.
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
        examples_pdf = examples_pdf.sample(frac=within_shard_fraction, random_state=SEED)

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""
    data_mode = "full_dataset" if IGNORE_PARTITIONS else "rank_sharded"

    try:
        return train_fsdp(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-fsdp-{world_size}gpu{run_suffix}",
            training_mode=f"fsdp_{world_size}_gpu_{data_mode}_sample",
            num_gpus=world_size,
            save_artifacts=True,
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
