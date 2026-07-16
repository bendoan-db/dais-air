"""Standalone Phi-4 QLoRA (HuggingFace PEFT) training entrypoint.

This module owns a plain TRL + PEFT + bitsandbytes 4-bit (QLoRA) LoRA training
implementation and runs two ways, like the sibling projects:

- Imported by the ``runner`` notebook, whose ``@distributed`` cell calls
  :func:`run_rank_training` on each GPU worker.
- Executed directly as ``python train.py`` by the AI Runtime CLI
  (``air run --file train.yaml``), where each GPU worker runs this file.

Differences from ``train_phi_4_unsloth/train.py`` (the project this is modeled
on):

- No Unsloth. The base model loads through ``transformers``
  ``AutoModelForCausalLM`` / ``AutoTokenizer`` and the LoRA adapter is applied
  with plain ``peft`` (``LoraConfig`` + ``get_peft_model``).
- QLoRA: the base weights are quantized to 4-bit with a
  ``BitsAndBytesConfig`` (NF4, double quant, bf16 compute dtype), and
  ``prepare_model_for_kbit_training`` runs before the adapter is attached.
- Distributed strategy stays **DDP** (each rank holds the full 4-bit model on
  its own GPU via ``device_map={"": local_rank}``); this is intentionally NOT
  the FSDP path used by ``train_gpt_oss_fsdp``.
- Response-only loss masking uses ``trl.DataCollatorForCompletionOnlyLM`` with
  the ``response_part`` marker from ``train.yaml`` (full-sequence loss when the
  markers are absent; missing markers never hard-fail).

Configuration and helper code live beside this file. Under an AIR run the
submitted configuration arrives through ``$HYPERPARAMETERS_PATH``.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "TRUE")

from contextlib import nullcontext
from pathlib import Path

import pandas as pd

try:
    from .project_config import (
        VOLUME_PATH_PREFIX,
        _load_workload,
        claim_rank_shard_files,
        load_project_config,
        sample_eval_records,
        stage_model_locally,
    )
except ImportError:
    from project_config import (
        VOLUME_PATH_PREFIX,
        _load_workload,
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

# The AI Runtime CLI launch wrapper exports these before running the script;
# neither is present under the notebook's @distributed path. Used to label
# MLflow runs with their launcher so CLI and notebook runs are
# distinguishable in the experiment.
LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"


def _load_quantization_settings() -> dict:
    """Read this project's QLoRA 4-bit keys from the workload config.

    The shared ``project_config.load_project_config`` loader is intentionally
    byte-identical across the training projects and does not parse
    quantization keys, so this QLoRA-only project reads them here from the same
    workload config (``train.yaml`` locally, ``$HYPERPARAMETERS_PATH`` under an
    AIR run) with 4-bit NF4 defaults.
    """
    _, _, raw_training_config = _load_workload("train.yaml")
    return {
        "load_in_4bit": bool(raw_training_config.get("load_in_4bit", True)),
        "bnb_4bit_quant_type": str(raw_training_config.get("bnb_4bit_quant_type", "nf4")),
        "bnb_4bit_use_double_quant": bool(
            raw_training_config.get("bnb_4bit_use_double_quant", True)
        ),
        "bnb_4bit_compute_dtype": str(
            raw_training_config.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    }


QUANTIZATION_SETTINGS = _load_quantization_settings()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(dtype_name).lower(), torch.bfloat16)


def load_base_model_and_tokenizer(model_name: str, device_map=None):
    """Load the 4-bit QLoRA base model and tokenizer (no Unsloth).

    ``model_name`` is either a Hugging Face repo id or a local directory — the
    UC volume snapshot configured by ``model_weights_path`` in ``train.yaml``.
    Fail fast on a missing local path so the error isn't transformers treating
    it as a repo id and dying with an HF 401/404.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    # Right padding for training; the generation eval flips to left padding.
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # NET-NEW for this repo: QLoRA 4-bit base quantization. The base weights
    # stay quantized (do NOT blanket-cast them to bf16); only the LoRA adapter
    # and the layers upcast by prepare_model_for_kbit_training train in higher
    # precision.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_SETTINGS["load_in_4bit"],
        bnb_4bit_quant_type=QUANTIZATION_SETTINGS["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=QUANTIZATION_SETTINGS["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=_resolve_torch_dtype(
            QUANTIZATION_SETTINGS["bnb_4bit_compute_dtype"]
        ),
    )
    load_kwargs = {
        "quantization_config": quantization_config,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "use_cache": False,  # required with gradient checkpointing
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        # DDP: pin the whole 4-bit model to this rank's GPU.
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer


def load_model_for_merge(adapter_output_dir: str):
    """Load the PEFT adapter and base model for deployment-time merging.

    Uses the plain PEFT merge path (``AutoPeftModelForCausalLM``) rather than
    an Unsloth loader — the deploy notebook calls ``merge_and_unload`` on the
    returned model.
    """
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_source = adapter_output_dir
    if adapter_source.startswith("/") and not Path(adapter_source).exists():
        raise FileNotFoundError(f"Adapter output path does not exist: {adapter_source}")
    if adapter_source.startswith(VOLUME_PATH_PREFIX):
        adapter_source = stage_model_locally(adapter_source)

    # Merge in bf16 (not 4-bit): merge_and_unload on a quantized base produces
    # degraded weights, and the merged checkpoint is served by vLLM in bf16.
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_source)
    return model, tokenizer


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


def conversational_records(records_pdf: "pd.DataFrame") -> list[dict]:
    """Convert normalized SFT rows to TRL's conversational format.

    SFTTrainer applies the model's own chat template to these records, so
    model-specific formatting stays local to the trainer.
    """
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


def build_peft_config():
    """LoRA config for the QLoRA adapter (plain PEFT)."""
    from peft import LoraConfig

    # The loader validates lora_target_modules as a non-empty list; a single
    # "all-linear" entry means PEFT's all-linear shorthand.
    if LORA_TARGET_MODULES == ["all-linear"]:
        target_modules = "all-linear"
    else:
        target_modules = LORA_TARGET_MODULES

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
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

    Under DDP each rank holds the full (unsharded) model, so batched
    generation works directly — unlike the FSDP project, which cannot run this
    without gathering the sharded parameters.
    """
    import torch

    model.eval()
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


def build_response_only_collator(tokenizer):
    """Build a response-only loss-masking collator, or None for full-sequence.

    Uses ``trl.DataCollatorForCompletionOnlyLM`` with the ``response_part``
    marker from ``train.yaml`` (and the ``response_instruction_part`` marker
    when present, for multi-turn masking). Missing markers fall back to
    full-sequence causal-LM loss and never hard-fail.
    """
    if RESPONSE_PART is None:
        print(
            "response_part is not set in train.yaml; training with "
            "full-sequence loss (no response-only masking)."
        )
        return None
    from trl import DataCollatorForCompletionOnlyLM

    collator_kwargs = {"response_template": RESPONSE_PART, "tokenizer": tokenizer}
    if RESPONSE_INSTRUCTION_PART is not None:
        collator_kwargs["instruction_template"] = RESPONSE_INSTRUCTION_PART
    return DataCollatorForCompletionOnlyLM(**collator_kwargs)


def train_phi4_qlora(
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
    import shutil
    import tempfile

    import mlflow
    import mlflow.transformers
    import torch
    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

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
    # SFTTrainer applies the model's chat template to the normalized records.
    dataset = Dataset.from_list(conversational_records(examples_pdf))

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
        eval_dataset = Dataset.from_list(conversational_records(eval_pdf))

    model, tokenizer = load_base_model_and_tokenizer(MODEL_LOAD_PATH, device_map=device_map)

    # QLoRA: prepare the 4-bit base for k-bit training BEFORE attaching the
    # adapter. This upcasts layernorms to fp32, enables input-gradient hooks,
    # and turns on gradient checkpointing. HF non-reentrant checkpointing is
    # required under DDP here (Unsloth's reentrant variant double-fires the DDP
    # gradient hooks); keep use_reentrant=False in sync with SFTConfig below.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, build_peft_config())
    if is_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # The /Volumes FUSE mount rejects safetensors' serialization write
    # pattern with EAGAIN (os error 11) — the same limitation that forces
    # setup/02 to stage its download on local disk — so the trainer must
    # write checkpoints and the final adapter to node-local disk; the
    # finished adapter files are copied to the volume sequentially after
    # training.
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
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
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
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        # DDP (not FSDP): no fsdp / fsdp_config keys.
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=build_response_only_collator(tokenizer),
        args=training_args,
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
                    "quant_load_in_4bit": QUANTIZATION_SETTINGS["load_in_4bit"],
                    "quant_bnb_4bit_quant_type": QUANTIZATION_SETTINGS["bnb_4bit_quant_type"],
                    "quant_bnb_4bit_use_double_quant": QUANTIZATION_SETTINGS[
                        "bnb_4bit_use_double_quant"
                    ],
                    "quant_bnb_4bit_compute_dtype": QUANTIZATION_SETTINGS[
                        "bnb_4bit_compute_dtype"
                    ],
                    "response_only_masking": RESPONSE_PART is not None,
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
        return train_phi4_qlora(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-qlora-{world_size}gpu{run_suffix}",
            training_mode=f"qlora_{world_size}_gpu_{data_mode}_sample",
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
