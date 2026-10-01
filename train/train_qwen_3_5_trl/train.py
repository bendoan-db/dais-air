"""Standalone Qwen3.5 TRL + PEFT LoRA training entrypoint.

This module owns a plain TRL ``SFTTrainer`` + PEFT LoRA implementation and runs
two ways, like the sibling projects:

- Imported by the ``runner`` notebook, whose ``@distributed`` cell calls
  :func:`run_rank_training` on each GPU worker.
- Executed by the AI Runtime CLI (``air run --file train.yaml``), whose
  ``torchrun`` command starts one copy of this file per GPU.

Differences from ``train_qwen_unsloth/train.py``:

- No Unsloth (serverless GPU environment v6 dropped it). ``AutoModelForCausalLM``
  loads the bf16 text backbone (``Qwen3_5ForCausalLM``) of the natively
  multimodal checkpoint, and ``SFTTrainer`` applies the ``LoraConfig`` itself.
- Qwen3.5 has no non-thinking variant, so every chat-template render passes
  ``enable_thinking=False``. The template is applied here rather than by TRL
  because that is the only way the flag reaches the render.
- Assistant-only loss comes from TRL's prompt-completion format with
  ``completion_only_loss=True`` instead of response-marker masking.
- Registration re-wraps the merged backbone in the base checkpoint's composite
  (vision + text) shape; see :func:`merge_adapter_to_serving_checkpoint`.

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

# The AI Runtime CLI launch wrapper exports these before running the script;
# neither is present under the notebook's @distributed path. Used to label
# MLflow runs with their launcher so CLI and notebook runs are
# distinguishable in the experiment.
LAUNCHED_VIA_AIR_CLI = bool(
    os.environ.get("HYPERPARAMETERS_PATH") or os.environ.get("CODE_SOURCE_PATH")
)
LAUNCHER = "air-cli" if LAUNCHED_VIA_AIR_CLI else "notebook"

# Qwen3.5 runs in thinking mode by default. Training renders, the generation
# eval, and the served chat template (see default_thinking_off) all disable it
# so the fine-tune's direct-JSON behaviour matches what the endpoint renders.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# vLLM builds Qwen3.5's multimodal processor even with --language-model-only,
# so these base-snapshot configs must ship with the merged weights.
PROCESSOR_CONFIG_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)


def preferred_dtype():
    """bfloat16 where the GPU supports it, else float16 (matches SFTConfig flags)."""
    import torch

    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def load_base_model_and_tokenizer(model_name: str, device_map=None):
    """Load the Qwen3.5 text backbone and tokenizer for LoRA fine-tuning.

    ``model_name`` is either a Hugging Face repo id or a local directory — the
    UC volume snapshot configured by ``model_weights_path`` in ``train.yaml``.
    ``AutoModelForCausalLM`` resolves the composite checkpoint to
    ``Qwen3_5ForCausalLM``, so the vision tower never loads. The model is
    returned unwrapped: ``SFTTrainer`` applies the LoRA config itself.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    load_kwargs = {"dtype": preferred_dtype(), "low_cpu_mem_usage": True}
    if device_map is not None:
        # DDP: pin the whole model to this rank's GPU.
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer


def load_model_for_merge(adapter_output_dir: str):
    """Load the trained adapter on its text backbone for deployment-time merging."""
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    adapter_source = adapter_output_dir
    if adapter_source.startswith("/") and not Path(adapter_source).exists():
        raise FileNotFoundError(f"Adapter output path does not exist: {adapter_source}")
    if adapter_source.startswith(VOLUME_PATH_PREFIX):
        # Also stages the volume base model the adapter config points at.
        adapter_source = stage_model_locally(adapter_source)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_source,
        dtype=preferred_dtype(),
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


def prompt_completion_records(records_pdf: pd.DataFrame, tokenizer) -> list[dict]:
    """Render normalized SFT rows into TRL's prompt-completion format.

    With ``completion_only_loss`` TRL puts the loss on the completion alone.
    Splitting one full render at the generation prompt keeps the completion
    byte-identical to what the template emits, including its end-of-turn
    token, and the prompt identical to what the endpoint renders.
    """
    records = []
    fallback_count = 0
    for prompt, assistant_response in zip(
        records_pdf["prompt"], records_pdf["assistant_response"]
    ):
        user_turn = [{"role": "user", "content": str(prompt)}]
        prompt_text = tokenizer.apply_chat_template(
            user_turn,
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        full_text = tokenizer.apply_chat_template(
            user_turn + [{"role": "assistant", "content": str(assistant_response)}],
            tokenize=False,
            add_generation_prompt=False,
            **CHAT_TEMPLATE_KWARGS,
        )
        if full_text.startswith(prompt_text):
            completion_text = full_text[len(prompt_text) :]
        else:
            # Some templates place generation-prompt tokens differently in the
            # full render; fall back to the bare response.
            completion_text = str(assistant_response) + (tokenizer.eos_token or "")
            fallback_count += 1
        records.append({"prompt": prompt_text, "completion": completion_text})

    if fallback_count:
        print(
            f"Chat template's generation prompt is not a prefix of the full render "
            f"for {fallback_count} of {len(records)} rows; used bare response + "
            "eos_token as their completion."
        )
    return records


def build_peft_config():
    """LoRA config for the 16-bit base (plain PEFT, applied by SFTTrainer)."""
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
    generation works directly.
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
                **CHAT_TEMPLATE_KWARGS,
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


# The upstream Qwen3.5 chat template opens a `<think>` block unless the caller
# passes enable_thinking=False, so a client that omits the kwarg gets a prompt
# state the fine-tune never saw and spends its token budget reasoning. The
# served checkpoint inverts the default: suppressed unless a request opts in
# with enable_thinking=true. Both constants are verbatim template text, where
# `\n` is the two-character Jinja escape, not a newline.
_UPSTREAM_THINKING_BRANCH = (
    "{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}"
)
_SUPPRESSED_THINKING_BRANCH = (
    "{%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}"
)


def _patch_thinking_default(template: str) -> tuple[str | None, str]:
    """Invert a chat template's thinking default. Returns (patched, status)."""
    if _SUPPRESSED_THINKING_BRANCH in template:
        return None, "already defaults to thinking off"
    if _UPSTREAM_THINKING_BRANCH in template:
        patched = template.replace(_UPSTREAM_THINKING_BRANCH, _SUPPRESSED_THINKING_BRANCH, 1)
        return patched, "patched: thinking is now opt-in"
    if "enable_thinking" not in template:
        return None, "no enable_thinking switch to invert"
    raise RuntimeError(
        "The chat template has an enable_thinking switch in an unrecognised form, "
        "so thinking-off cannot be made the default. Serving it as-is would let "
        "any client that omits chat_template_kwargs get a reasoning preamble. "
        "Re-read the template and update _UPSTREAM_THINKING_BRANCH."
    )


def default_thinking_off(output_dir: str) -> str:
    """Make thinking opt-in in a saved checkpoint's chat template.

    vLLM loads the chat template from the model directory, so patching the file
    makes suppression a property of the endpoint: the AI Playground, curl, and
    any client that forgets ``chat_template_kwargs`` get the render training
    used. Requests can still opt back in with ``enable_thinking: true``.
    """
    import json

    out = Path(output_dir)
    statuses = []

    jinja_path = out / "chat_template.jinja"
    if jinja_path.exists():
        patched, status = _patch_thinking_default(jinja_path.read_text())
        if patched is not None:
            jinja_path.write_text(patched)
        statuses.append(f"{jinja_path.name}: {status}")

    # The upstream snapshot also embeds the template in tokenizer_config.json;
    # patch every copy so no loader can pick up the thinking-on default.
    config_path = out / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        template = config.get("chat_template")
        if isinstance(template, str):
            patched, status = _patch_thinking_default(template)
            if patched is not None:
                config["chat_template"] = patched
                config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
            statuses.append(f"{config_path.name}: {status}")

    if not statuses:
        raise RuntimeError(
            f"No chat template found under {output_dir}; serving would fall back to "
            "vLLM's default and lose thinking suppression."
        )
    return "; ".join(statuses)


def _graft_text_weights(text_state_dict, composite_model) -> tuple[int, list[str]]:
    """Copy a merged text backbone's tensors onto a composite model in place.

    The composite model nests the backbone one level deeper
    (``model.layers.*`` -> ``model.language_model.layers.*``), so keys are
    remapped by trying the plausible prefixes and confirming the shape.
    """
    composite_sd = composite_model.state_dict()
    remapped = {}
    unmatched = []
    for key, tensor in text_state_dict.items():
        for candidate in (
            key,
            key.replace("model.", "model.language_model.", 1),
            f"model.{key}",
        ):
            if candidate in composite_sd and composite_sd[candidate].shape == tensor.shape:
                remapped[candidate] = tensor
                break
        else:
            unmatched.append(key)

    # Keys left alone keep the base checkpoint's pretrained values. That is
    # intended for the vision tower and Qwen3.5's MTP head (neither exists in
    # the text backbone; vLLM reads MTP only for speculative decoding). A text
    # key left behind would silently serve base weights, so that fails loudly.
    carried_from_base = ("model.visual.", "mtp.")
    untouched = sorted(set(composite_sd) - set(remapped))
    stray = [key for key in untouched if not key.startswith(carried_from_base)]
    if unmatched or stray:
        raise RuntimeError(
            f"Could not map the merged adapter onto {type(composite_model).__name__}. "
            f"Unmapped merged tensors: {unmatched[:10]} ({len(unmatched)} total). "
            f"Non-vision tensors left at base values: {stray[:10]} ({len(stray)} total)."
        )

    composite_model.load_state_dict(remapped, strict=False)
    return len(remapped), untouched


def merge_adapter_to_serving_checkpoint(adapter_output_dir: str, output_dir: str) -> dict:
    """Merge the LoRA adapter and write the checkpoint vLLM can actually serve.

    Training and merging both use Qwen3.5's text backbone, but vLLM 0.24
    implements ``Qwen3_5ForCausalLM`` without registering it: architecture
    normalisation rewrites the ``ForCausalLM`` suffix until a registered name
    matches, so a text-only checkpoint builds ``Qwen3_5ForConditionalGeneration``
    and dies with ``'Qwen3_5TextConfig' object has no attribute 'vision_config'``.

    So the merged backbone is saved back into the base model's composite
    (vision + text) shape. ``--language-model-only`` keeps the vision tower idle
    at serve time, but its weights must still ship: vLLM's loader raises on any
    parameter missing from the checkpoint.
    """
    import json
    import shutil

    import transformers
    from transformers import AutoConfig

    output_path = Path(output_dir)

    # The saved adapter records the durable base path (restored after
    # training), so the merge does not depend on the current train.yaml.
    adapter_config = json.loads((Path(adapter_output_dir) / "adapter_config.json").read_text())
    base_source = str(adapter_config["base_model_name_or_path"])
    if base_source.startswith(VOLUME_PATH_PREFIX):
        base_source = stage_model_locally(base_source)

    model, tokenizer = load_model_for_merge(adapter_output_dir)
    merged = model.merge_and_unload()

    base_config = AutoConfig.from_pretrained(base_source)
    if not hasattr(base_config, "vision_config"):
        # Text-only base: nothing to wrap, and vLLM registers whatever
        # architecture the checkpoint declares.
        merged.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        return {
            "architecture": merged.config.architectures[0],
            "grafted_tensors": len(merged.state_dict()),
            "base_tensors_carried": 0,
            "chat_template": default_thinking_off(output_dir),
        }

    composite_arch = (getattr(base_config, "architectures", None) or [None])[0]
    composite_class = getattr(transformers, composite_arch, None) if composite_arch else None
    if composite_class is None:
        raise RuntimeError(
            f"The installed Transformers has no class for the base architecture "
            f"{composite_arch!r}; check the transformers pin in requirements.txt."
        )
    composite = composite_class.from_pretrained(
        base_source, dtype=preferred_dtype(), low_cpu_mem_usage=True
    )
    grafted, untouched = _graft_text_weights(merged.state_dict(), composite)
    del merged, model

    composite.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    # Copied verbatim rather than round-tripped through AutoProcessor, which
    # needs torchvision and would also rewrite the tokenizer files.
    if Path(base_source).is_dir():
        for file_name in PROCESSOR_CONFIG_FILES:
            if (Path(base_source) / file_name).exists():
                shutil.copy2(Path(base_source) / file_name, output_path / file_name)
    if not (output_path / "preprocessor_config.json").exists():
        raise RuntimeError(
            f"No preprocessor_config.json found in the base snapshot {base_source}; "
            "vLLM cannot build Qwen3.5's processor without it."
        )

    return {
        "architecture": composite_arch,
        "grafted_tensors": grafted,
        "base_tensors_carried": len(untouched),
        "chat_template": default_thinking_off(output_dir),
    }


def train_qwen35_sft(
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

    eval_pdf = None
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

    model, tokenizer = load_base_model_and_tokenizer(MODEL_LOAD_PATH, device_map=device_map)

    dataset = Dataset.from_list(prompt_completion_records(examples_pdf, tokenizer))
    eval_dataset = (
        Dataset.from_list(prompt_completion_records(eval_pdf, tokenizer))
        if eval_pdf is not None
        else None
    )

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

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        # TRL's chunked_nll path omits logits and returns scalar counters,
        # which makes decoded classification metrics impossible.
        loss_type="nll",
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16,
        bf16=use_bf16,
        # Non-reentrant checkpointing is required under multi-GPU DDP: the
        # reentrant implementation fires each LoRA parameter's gradient hook
        # twice ("Expected to mark a variable ready only once").
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
        # Environment v6 removed bitsandbytes, so the 8-bit Adam used by the
        # Unsloth projects is unavailable; LoRA optimizer state is tiny anyway.
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=local_output_dir,
        report_to=["mlflow"],
        run_name=run_name,
        save_strategy="steps",
        save_steps=max(5, MAX_STEPS // 2),
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        dataset_num_proc=1,
        packing=False,
        # Every LoRA parameter gets a gradient each step, and the
        # unused-parameter scan conflicts with checkpointing.
        ddp_find_unused_parameters=False,
    )

    # Hand SFTTrainer the LoRA config rather than a pre-wrapped PEFT model: it
    # applies the adapter itself and rejects an already-wrapped model when
    # peft_config is set. trainer.model is therefore the PEFT model.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=build_peft_config(),
        args=training_args,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[build_mlflow_metrics_callback(eval_dataset is not None)],
    )
    if is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

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
                    "training_stack": "trl-sft-peft-lora",
                    "loss_masking": "completion_only",
                    "enable_thinking": CHAT_TEMPLATE_KWARGS["enable_thinking"],
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
                fraud_metrics = evaluate_fraud_classification(trainer.model, tokenizer, eval_pdf)
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


def _warn_single_process_multi_gpu(world_size: int) -> None:
    """Flag the launcher setup that silently trains on 1/world_size of the data.

    Reached when serverless_gpu reports a multi-GPU workload but no per-process
    launcher exported RANK — one process is holding every GPU. Trainer then
    wraps the model in ``nn.DataParallel``, which crashes Qwen3.5's Gated
    DeltaNet path and leaves this process reading only rank 0's shards.
    """
    try:
        import torch

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        # A diagnostic must never be the thing that fails the run.
        visible = 0

    if visible > 1:
        print(
            f"WARNING: serverless_gpu reports world_size={world_size}, no RANK is "
            f"set, and this process sees {visible} GPUs. Launch one process per "
            "GPU (torchrun --nproc_per_node=gpu) — otherwise Trainer falls back to "
            "nn.DataParallel and only rank 0's shard slice is read."
        )


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) under any launcher.

    Per-process launcher variables win. Under ``train.yaml``'s torchrun command
    serverless_gpu reports the *workload's* rank and size, so it would answer
    rank 0 in every process and each would claim rank 0's shards. AIR sets
    WORLD_SIZE but never RANK, so requiring both detects a per-GPU launcher.
    The serverless_gpu runtime remains the fallback for the notebook
    ``@distributed`` path, and a lone process is the last resort.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank

    try:
        from serverless_gpu import runtime as rt

        rank, world_size = rt.get_global_rank(), rt.get_world_size()
    except Exception:
        return 0, 1, local_rank

    if world_size > 1:
        _warn_single_process_multi_gpu(world_size)
    return rank, world_size, local_rank


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
    if examples_pdf.empty:
        raise ValueError(
            "Training sampling produced no rows. Increase training_sample_fraction."
        )

    run_suffix = "-air-cli" if LAUNCHED_VIA_AIR_CLI else ""
    data_mode = "full_dataset" if IGNORE_PARTITIONS else "rank_sharded"

    try:
        return train_qwen35_sft(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-trl-{world_size}gpu{run_suffix}",
            training_mode=f"trl_lora_{world_size}_gpu_{data_mode}_sample",
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
