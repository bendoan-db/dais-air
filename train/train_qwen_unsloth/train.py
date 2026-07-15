"""Standalone training entrypoint for the AIR fine-tuning pipeline.

This module owns the Unsloth LoRA training implementation and runs two ways:

- Imported by the ``runner`` notebook, whose ``@distributed`` cell calls
  :func:`run_rank_training` on each GPU worker.
- Executed directly as ``python train.py`` by the AI Runtime CLI
  (``air run --file train.yaml``), where each GPU worker runs this file.

Configuration and helper code live beside this file. Under an AIR run the
submitted configuration arrives through ``$HYPERPARAMETERS_PATH``.
"""

import os

# These must be set before unsloth is imported.
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")

from contextlib import nullcontext

import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

from pathlib import Path

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
    from .training_metrics import (
        build_compute_metrics,
        build_mlflow_metrics_callback,
        preprocess_logits_for_metrics,
    )
except ImportError:
    from sft_conversion import prepare_sft_records
    from training_metrics import (
        build_compute_metrics,
        build_mlflow_metrics_callback,
        preprocess_logits_for_metrics,
    )

globals().update(load_project_config())
if RESPONSE_INSTRUCTION_PART is None or RESPONSE_PART is None:
    raise ValueError(
        "train.yaml must define response_instruction_part and response_part "
        "for response-only loss masking"
    )

# The AI Runtime CLI launch wrapper exports these before running the script;
# neither is present under the notebook's @distributed path. Used to label
# MLflow runs with their launcher so CLI and notebook runs are
# distinguishable in the experiment.
LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


def load_unsloth_model(model_name: str, device_map=None):
    # model_name is either a Hugging Face repo id or a local directory — the
    # UC volume snapshot configured by model_weights_path in train.yaml.
    # or a trained adapter dir. Fail fast on a missing local path so the error
    # isn't unsloth/transformers treating it as a repo id and dying with an
    # HF 401/404.
    if model_name.startswith("/") and not Path(model_name).exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_name}. If this is "
            "model_weights_path from train.yaml; populate the volume first by "
            "running setup/04_download_base_model_weights.py (or `hf download "
            f"{MODEL_NAME} --local-dir {model_name}`)."
        )
    if model_name.startswith(VOLUME_PATH_PREFIX):
        # safetensors mmap reads through the volume FUSE mount are
        # latency-bound and take minutes for multi-GB weights; stage the
        # directory to node-local disk and load from there instead.
        model_name = stage_model_locally(model_name)
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


def load_model_for_merge(adapter_output_dir: str):
    """Load this project's adapter for the deployment merge notebook."""
    return load_unsloth_model(adapter_output_dir)


def _restore_saved_adapter_base_path(output_dir: str) -> None:
    """Point the saved adapter config back at the configured base source.

    When the base weights come from a volume, training loads them from a
    node-local staged copy, so PEFT records that ephemeral local path as
    ``base_model_name_or_path``. Registration runs on a different machine and
    resolves the base model through this field — rewrite it to the durable
    MODEL_LOAD_PATH (volume path or HF repo id) after the adapter is saved.
    """
    import json

    adapter_config_path = Path(output_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        return
    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("base_model_name_or_path") != MODEL_LOAD_PATH:
        adapter_config["base_model_name_or_path"] = MODEL_LOAD_PATH
        adapter_config_path.write_text(json.dumps(adapter_config, indent=2))


def render_chat_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def parse_risk_prediction(completion: str) -> str | None:
    """Extract the ``risk`` value from a generated JSON completion.

    Falls back to a regex so a completion truncated by the generation budget
    still yields a prediction (``risk`` is the first key in the response
    contract, so it survives truncation).
    """
    import json
    import re

    try:
        risk = json.loads(completion.strip()).get("risk")
        return str(risk) if risk is not None else None
    except Exception:
        match = re.search(r'"risk"\s*:\s*"([^"]+)"', completion)
        return match.group(1) if match else None


def binary_classification_metrics(
    ground_truth: list[bool], predictions: list[bool]
) -> dict[str, float]:
    """Accuracy/precision/recall/F1 with fraud as the positive class."""
    true_positives = sum(1 for truth, pred in zip(ground_truth, predictions) if truth and pred)
    false_positives = sum(1 for truth, pred in zip(ground_truth, predictions) if not truth and pred)
    false_negatives = sum(1 for truth, pred in zip(ground_truth, predictions) if truth and not pred)
    true_negatives = sum(
        1 for truth, pred in zip(ground_truth, predictions) if not truth and not pred
    )
    total = len(ground_truth)

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "eval_fraud_accuracy": (true_positives + true_negatives) / total if total else 0.0,
        "eval_fraud_precision": precision,
        "eval_fraud_recall": recall,
        "eval_fraud_f1": f1,
        "eval_fraud_true_positives": float(true_positives),
        "eval_fraud_false_positives": float(false_positives),
        "eval_fraud_false_negatives": float(false_negatives),
        "eval_fraud_true_negatives": float(true_negatives),
    }


def evaluate_fraud_classification(model, tokenizer, eval_pdf, batch_size: int = 16) -> dict:
    """Score the fine-tuned model as a binary fraud classifier on eval-split rows.

    Generates a completion for each held-out prompt, parses the ``risk``
    field, and treats ``likely_fraud`` as the positive prediction against the
    records' ``is_fraud`` label. Returns the metrics dict (plus the rate of
    completions with no parseable ``risk``, which count as non-fraud
    predictions).
    """
    import torch

    FastLanguageModel.for_inference(model)
    # Decoder-only batched generation needs left padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = eval_pdf["prompt"].tolist()
    ground_truth = [bool(int(value) == 1) for value in eval_pdf["is_fraud"].tolist()]

    predictions: list[bool] = []
    unparseable_count = 0
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in batch_prompts
        ]
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                # The compact JSON answer fits in ~50-60 tokens (the same
                # budget the serving payloads use); risk is the first key, so
                # even a truncated reason leaves the prediction parseable.
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for completion in completions:
            risk = parse_risk_prediction(completion)
            if risk is None:
                unparseable_count += 1
            predictions.append(risk == "likely_fraud")

    metrics = binary_classification_metrics(ground_truth, predictions)
    metrics["eval_unparseable_rate"] = unparseable_count / len(prompts) if prompts else 0.0
    return metrics


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
    import mlflow.transformers
    import torch
    from datasets import Dataset

    mlflow.set_registry_uri("databricks-uc")
    is_main_process = rank == 0
    save_artifacts = save_artifacts and is_main_process
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
    dataset = Dataset.from_pandas(
        examples_pdf[["prompt", "assistant_response"]],
        preserve_index=False,
    )

    eval_pdf = None
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
        eval_dataset = Dataset.from_pandas(
            eval_pdf[["prompt", "assistant_response"]],
            preserve_index=False,
        )

    model, tokenizer = load_unsloth_model(MODEL_LOAD_PATH, device_map=device_map)

    # Raw records are converted above when configured; pre-converted records
    # pass through unchanged. The loaded model's own chat template is always
    # applied here so model-specific formatting remains local to the trainer.
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
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    peft_kwargs = {
        "r": LORA_R,
        "target_modules": LORA_TARGET_MODULES,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
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

    # The /Volumes FUSE mount rejects safetensors' serialization write
    # pattern with EAGAIN (os error 11) — the same limitation that forces
    # setup/02 to stage its download on local disk — so the trainer must
    # write checkpoints and the final adapter to node-local disk; the
    # finished adapter files are copied to the volume sequentially after
    # training.
    import shutil
    import tempfile

    local_disk_tmp = Path("/local_disk0/tmp")
    staging_base = local_disk_tmp if local_disk_tmp.exists() else Path(tempfile.gettempdir())
    local_output_dir = str(staging_base / "air-training-output" / run_name)
    if is_main_process:
        # Only the world-zero process writes checkpoints (save_on_each_node
        # is False), so clearing a stale directory from a previous run can't
        # race the other ranks on this node.
        shutil.rmtree(local_output_dir, ignore_errors=True)

    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        # TRL's chunked_nll path omits logits and returns scalar counters,
        # which makes decoded classification metrics impossible.
        loss_type="nll",
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        do_eval=eval_dataset is not None,
        prediction_loss_only=False,
        eval_do_concat_batches=True,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=local_output_dir,
        report_to=["mlflow"],
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
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=training_args,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[build_mlflow_metrics_callback(eval_dataset is not None)],
    )

    try:
        # Marker strings come from train.yaml (response_instruction_part /
        # response_part) and must match the base model's chat template.
        trainer = train_on_responses_only(
            trainer,
            instruction_part=RESPONSE_INSTRUCTION_PART,
            response_part=RESPONSE_PART,
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
            trainer.save_model(local_output_dir)
            tokenizer.save_pretrained(local_output_dir)
            _restore_saved_adapter_base_path(local_output_dir)
            # Top-level files only: skips the throwaway checkpoint-*/ dirs,
            # and whole-file sequential copies are the write pattern the
            # volume mount supports.
            volume_dir = Path(output_dir)
            volume_dir.mkdir(parents=True, exist_ok=True)
            for artifact_file in sorted(Path(local_output_dir).iterdir()):
                if artifact_file.is_file():
                    shutil.copy2(artifact_file, volume_dir / artifact_file.name)
            mlflow.log_param("adapter_output_dir", output_dir)

        # Fraud-classification quality of the fine-tuned model on the staged
        # eval split (split=eval, never trained on by any rank). Stratified
        # half fraud / half non-fraud: at the natural ~1% fraud rate a small
        # random sample would carry almost no positives, leaving
        # recall/precision meaningless. A failed evaluation logs eval_error
        # but keeps the run FINISHED — a completed training (and its adapter)
        # should stay deployable.
        if eval_pdf is not None:
            try:
                fraud_metrics = evaluate_fraud_classification(model, tokenizer, eval_pdf)
                for metric_name, metric_value in fraud_metrics.items():
                    mlflow.log_metric(metric_name, float(metric_value))
                mlflow.log_param("eval_sample_size", len(eval_pdf))
                print(
                    "Held-out fraud classification — "
                    f"accuracy: {fraud_metrics['eval_fraud_accuracy']:.3f}, "
                    f"precision: {fraud_metrics['eval_fraud_precision']:.3f}, "
                    f"recall: {fraud_metrics['eval_fraud_recall']:.3f}, "
                    f"f1: {fraud_metrics['eval_fraud_f1']:.3f} "
                    f"(n={len(eval_pdf)}, unparseable rate "
                    f"{fraud_metrics['eval_unparseable_rate']:.3f})"
                )
            except Exception as exc:
                mlflow.log_param("eval_error", str(exc)[:250])
                print(f"Fraud-classification evaluation failed (run continues): {exc}")

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
    """Train this rank's shard slice of the staged train split.

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

    # Read raw or pre-converted records as parquet shard files from the UC
    # volume instead of querying Delta
    # through Spark on the GPU workers, per the AIR data-loading guidance:
    # https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes
    # Training reads only the train split; the eval split feeds the
    # post-training fraud-classification evaluation on rank 0.
    rank_files, within_shard_fraction = claim_rank_shard_files(
        TRAIN_DATA_PATH,
        rank,
        world_size,
        sample_fraction,
        SEED,
        ignore_partitions=IGNORE_PARTITIONS,
    )

    dataset = load_dataset("parquet", data_files=rank_files, split="train")
    examples_pdf = dataset.to_pandas()
    if within_shard_fraction < 1.0:
        examples_pdf = examples_pdf.sample(frac=within_shard_fraction, random_state=SEED)

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""
    data_mode = "full_dataset" if IGNORE_PARTITIONS else "rank_sharded"

    try:
        return train_qwen3_unsloth(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-{world_size}gpu{run_suffix}",
            training_mode=f"{world_size}_gpu_{data_mode}_sample",
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
