"""Standalone training entrypoint for the AIR fraud fine-tuning demo.

This module owns the Hugging Face LoRA supervised fine-tuning implementation
(TRL ``SFTTrainer`` + PEFT ``LoraConfig``) and runs two ways:

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import nullcontext

import pandas as pd
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from pathlib import Path

# The helpers module is deliberately NOT named `utils`: the Databricks AI base
# environment's nvidia_cutlass_dsl package registers its own top-level `utils`
# module once the torch/CUDA stack loads (importing trl pulls in transformers
# and torch), which shadows any local utils.py.
from training_utils import load_training_config, resolve_experiment_path

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


# LoRA shape for the demo: 16-bit base weights with rank-16 adapters on the
# attention and MLP projections (no quantization — the 4B model fits in GPU
# memory unquantized, which keeps the merge step for serving lossless).
#
# On Qwen3.5's 3:1 hybrid stack these names only match where they exist: the
# Gated Attention layers (every 4th, per the checkpoint's `layer_types`) expose
# q/k/v/o_proj, while the Gated DeltaNet layers expose `linear_attn.*` instead
# and are deliberately left unadapted — recurrent linear-attention projections
# do not decompose into a low-rank update the way dense ones do. The MLP
# projections match on every layer.
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
# Qwen3.5 runs in thinking mode by default and ships no non-thinking variant, so
# every render disables it. The served endpoint must send the same thing
# (chat_template_kwargs) or the fine-tune's direct-JSON behaviour will not match
# the serving template. Harmless on templates that ignore the flag (Qwen3).
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

_TEMPLATE_SPLIT_WARNED = False


def _warn_once_template_split() -> None:
    """Flag the generation-prompt fallback once instead of per row."""
    global _TEMPLATE_SPLIT_WARNED
    if not _TEMPLATE_SPLIT_WARNED:
        _TEMPLATE_SPLIT_WARNED = True
        print(
            "Chat template's generation prompt is not a prefix of the full render; "
            "falling back to bare response + eos_token for the completion column."
        )


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class MLflowStepMetricsCallback(TrainerCallback):
    """Mirror the Trainer's per-step logs into the active MLflow run.

    ``report_to="none"`` keeps Hugging Face's own ``MLflowCallback`` out of the
    way: it re-logs every ``TrainingArguments`` field as a parameter and manages
    the run's lifecycle, which collides with the run this module opens. This
    forwards only metrics, keyed on ``global_step``.

    Metric names are deliberately not hardcoded — whatever the installed
    transformers/TRL emit per step is forwarded as-is, so ``loss``,
    ``mean_token_accuracy``, ``grad_norm``, ``learning_rate``, ``entropy``,
    ``num_tokens`` and ``epoch`` all land in MLflow (plus ``aux_loss`` on MoE
    checkpoints) without this callback tracking TRL's metric set per version.
    Two derived series are added because they are what a GPU-scaling demo is
    actually judged on: tokens/second and CUDA memory.
    """

    def __init__(self):
        self._last_tokens = None
        self._last_time = None
        self._warned = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only rank 0 has an MLflow run open (see train_qwen3_sft); the other
        # ranks would log into whatever run their process happened to inherit.
        if not logs or not state.is_world_process_zero:
            return

        import math
        import time

        import mlflow

        now = time.time()
        metrics = {}
        for key, value in logs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(value):
                metrics[key] = float(value)

        # Throughput from the cumulative token counter TRL maintains. The first
        # log has no previous sample to difference against.
        tokens = metrics.get("num_tokens")
        if tokens is not None and self._last_tokens is not None:
            elapsed = now - self._last_time
            if elapsed > 0 and tokens >= self._last_tokens:
                metrics["tokens_per_second"] = (tokens - self._last_tokens) / elapsed
        if tokens is not None:
            self._last_tokens, self._last_time = tokens, now

        try:
            import torch

            if torch.cuda.is_available():
                metrics["cuda_memory_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1024**3
                )
                metrics["cuda_max_memory_allocated_gb"] = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )
        except Exception:  # noqa: BLE001 - telemetry must not fail a run
            pass

        if not metrics:
            return

        try:
            mlflow.log_metrics(metrics, step=state.global_step)
        except Exception as err:  # noqa: BLE001 - see above
            if not self._warned:
                self._warned = True
                print(f"WARNING: per-step MLflow metric logging failed: {err}")


def preferred_dtype():
    """bfloat16 where the GPU supports it, else float16 (matches SFTConfig flags)."""
    import torch

    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def load_base_model_and_tokenizer(model_name: str, device_map=None):
    """Load the base model and tokenizer for LoRA fine-tuning.

    ``AutoModelForCausalLM`` deliberately loads the **text backbone only**. The
    Qwen3.5 checkpoints are natively multimodal (``Qwen3_5ForConditionalGeneration``
    plus a ``vision_config``), but the transformers docs prescribe
    ``Qwen3_5ForCausalLM`` + ``Qwen3_5TextConfig`` for text-only generation, which
    is what the auto class resolves to. This fraud task never sees an image, and
    it mirrors serving, where the entrypoint passes ``--language-model-only``.

    The model is returned unwrapped: ``SFTTrainer`` applies the
    :class:`~peft.LoraConfig` itself, which is also what makes it call
    ``enable_input_require_grads()`` for gradient checkpointing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {"dtype": preferred_dtype()}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_adapter_model(adapter_dir: str, device_map=None):
    """Load a trained LoRA adapter directory and its tokenizer.

    Used by the registration cell, which calls ``merge_and_unload()`` on the
    returned model to fuse the adapter into the base weights before logging
    plain Hugging Face weights for vLLM serving.
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    load_kwargs = {"dtype": preferred_dtype()}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    return model, tokenizer


# The upstream Qwen3.5 chat template opens a `<think>` block unless the caller
# passes enable_thinking=False, so any client that forgets the kwarg gets a
# prompt ending in '<think>\n' -- a state the fine-tune never saw, and the model
# spends its whole token budget reasoning instead of emitting JSON. Rather than
# relying on every caller, invert the default in the served checkpoint:
# suppressed unless the request explicitly opts in with enable_thinking=true.
# Both constants are the verbatim template text, where `\n` is the two-character
# sequence a Jinja string literal escapes, not a newline.
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
        patched = template.replace(
            _UPSTREAM_THINKING_BRANCH, _SUPPRESSED_THINKING_BRANCH, 1
        )
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

    vLLM loads the chat template straight out of the model directory, so
    patching the file here makes suppression a property of the endpoint: the AI
    Playground, curl, and any client that forgets ``chat_template_kwargs`` all
    get the same render training used. Callers can still opt back in per request
    with ``chat_template_kwargs={"enable_thinking": true}``.
    """
    import json

    out = Path(output_dir)
    jinja_path = out / "chat_template.jinja"
    config_path = out / "tokenizer_config.json"

    if jinja_path.exists():
        patched, status = _patch_thinking_default(jinja_path.read_text())
        if patched is not None:
            jinja_path.write_text(patched)
        return f"{jinja_path.name}: {status}"

    # Older transformers embedded the template in tokenizer_config.json instead.
    if config_path.exists():
        config = json.loads(config_path.read_text())
        template = config.get("chat_template")
        if isinstance(template, str):
            patched, status = _patch_thinking_default(template)
            if patched is not None:
                config["chat_template"] = patched
                config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
            return f"tokenizer_config.json: {status}"

    raise RuntimeError(
        f"No chat template found under {output_dir}; serving would fall back to "
        "vLLM's default and lose thinking suppression."
    )


def _graft_text_weights(text_state_dict, composite_model):
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

    # Keys left alone keep the base checkpoint's pretrained values, so they are
    # always initialized and never a load-time failure. That is intended for the
    # vision tower and for Qwen3.5's MTP head (neither exists in the text
    # backbone, and vLLM only reads the MTP weights for speculative decoding).
    # A *text* key left behind would silently serve base weights in place of the
    # fine-tune, so that still fails loudly.
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


def merge_adapter_to_serving_checkpoint(adapter_dir: str, output_dir: str) -> dict:
    """Merge the LoRA adapter and write the checkpoint vLLM can actually serve.

    Training and merging both work on Qwen3.5's text backbone
    (``AutoPeftModelForCausalLM`` -> ``Qwen3_5ForCausalLM``), but vLLM 0.24.0
    implements that class without registering it: architecture normalisation
    rewrites the ``ForCausalLM`` suffix until a registered name matches, so a
    text-only checkpoint silently builds ``Qwen3_5ForConditionalGeneration`` and
    dies with ``'Qwen3_5TextConfig' object has no attribute 'vision_config'``.

    So the merged backbone is saved back into the base model's composite
    (vision + text) shape, which the registry does know. ``--language-model-only``
    then keeps the vision tower idle at serve time, but its weights must still be
    present: vLLM's loader raises on any parameter missing from the checkpoint.
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor

    model, tokenizer = load_adapter_model(adapter_dir)
    merged = model.merge_and_unload()

    base_config = AutoConfig.from_pretrained(MODEL_NAME)
    composite_arch = (getattr(base_config, "architectures", None) or [None])[0]
    if not hasattr(base_config, "vision_config"):
        # Text-only base: nothing to wrap, and vLLM will have a registry entry
        # for whatever architecture the checkpoint declares.
        merged.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        return {
            "architecture": merged.config.architectures[0],
            "grafted_tensors": len(merged.state_dict()),
            "vision_tensors_from_base": 0,
            "chat_template": default_thinking_off(output_dir),
        }

    composite = getattr(transformers, composite_arch).from_pretrained(
        MODEL_NAME, dtype=preferred_dtype()
    )
    grafted, untouched = _graft_text_weights(merged.state_dict(), composite)

    composite.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    # vLLM builds the multimodal processor even with the modality limits zeroed,
    # so the preprocessor configs have to travel with the weights.
    try:
        AutoProcessor.from_pretrained(MODEL_NAME).save_pretrained(output_dir)
    except Exception as err:  # noqa: BLE001 - optional scaffolding, never fatal
        print(f"WARNING: could not save the multimodal processor config: {err}")

    return {
        "architecture": composite_arch,
        "grafted_tensors": grafted,
        "vision_tensors_from_base": len(untouched),
        "chat_template": default_thinking_off(output_dir),
    }


def train_qwen3_sft(
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
    experiment_path: str | None = None,
) -> str | None:
    import mlflow
    import torch
    from datasets import Dataset

    mlflow.set_registry_uri("databricks-uc")
    is_main_process = rank == 0
    save_artifacts = save_artifacts and is_main_process

    if experiment_path and is_main_process:
        # Notebook runs pass the experiment path their driver resolved from
        # train.yaml's `experiment_name`, so rank 0 logs to the same experiment
        # as AIR CLI runs rather than depending on the launcher to propagate the
        # driver's active experiment. AIR CLI runs leave this None: the
        # pre-created workload run already targets that experiment.
        mlflow.set_experiment(experiment_path)

    model, tokenizer = load_base_model_and_tokenizer(MODEL_NAME, device_map=device_map)

    dataset = Dataset.from_pandas(
        examples_pdf[["prompt", "assistant_response"]],
        preserve_index=False,
    )

    # TRL's prompt-completion format with `completion_only_loss` puts the loss on
    # the assistant response alone. The chat template is applied HERE rather than
    # by TRL so that CHAT_TEMPLATE_KWARGS (enable_thinking=False) reaches it —
    # Qwen3.5 renders a thinking preamble otherwise, and the training render must
    # match what the endpoint serves.
    def to_prompt_completion(examples):
        prompts, completions = [], []
        for prompt, assistant_response in zip(
            examples["prompt"], examples["assistant_response"]
        ):
            user_turn = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                user_turn,
                tokenize=False,
                add_generation_prompt=True,
                **CHAT_TEMPLATE_KWARGS,
            )
            full_text = tokenizer.apply_chat_template(
                user_turn + [{"role": "assistant", "content": assistant_response}],
                tokenize=False,
                add_generation_prompt=False,
                **CHAT_TEMPLATE_KWARGS,
            )
            # Splitting the full render keeps the completion byte-identical to what
            # the template produces, including its end-of-turn token. Some
            # templates inject tokens into the generation prompt that the full
            # render places differently, so fall back to the bare response.
            if full_text.startswith(prompt_text):
                completion_text = full_text[len(prompt_text):]
            else:
                completion_text = assistant_response + (tokenizer.eos_token or "")
                _warn_once_template_split()
            prompts.append(prompt_text)
            completions.append(completion_text)
        return {"prompt": prompts, "completion": completions}

    dataset = dataset.map(
        to_prompt_completion,
        batched=True,
        remove_columns=dataset.column_names,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not use_bf16,
        bf16=use_bf16,
        # Non-reentrant checkpointing is required under multi-GPU DDP: the
        # reentrant implementation fires each LoRA parameter's gradient hook
        # twice ("Expected to mark a variable ready only once").
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Every LoRA parameter gets a gradient on every step, so DDP does not
        # need the unused-parameter scan (which also conflicts with
        # checkpointing).
        ddp_find_unused_parameters=False,
        logging_steps=1,
        # Serverless GPU environment v6 removed bitsandbytes from the base image,
        # so the 8-bit Adam path is gone. Fused AdamW costs nothing here: LoRA
        # trains ~0.7% of the parameters, so optimizer state is tiny either way.
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=output_dir,
        # Per-step metrics reach MLflow through MLflowStepMetricsCallback rather
        # than report_to="mlflow": HF's built-in callback also re-logs every
        # TrainingArguments field as a param and owns the run lifecycle, which
        # collides with the run opened below.
        report_to="none",
        run_name=run_name,
        save_strategy="steps",
        save_steps=max(5, MAX_STEPS // 2),
        # TRL renamed SFTConfig's `max_seq_length` to `max_length` (trl >= 0.20).
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        dataset_num_proc=1,
        packing=False,
    )

    # Hand SFTTrainer the LoRA config rather than a pre-wrapped PEFT model: it
    # applies the adapter itself and calls enable_input_require_grads() so
    # gradient checkpointing produces gradients for the adapter weights. TRL
    # rejects a model that is already PEFT-wrapped when peft_config is set.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        # logging_steps=1 above means this fires every step; the callback is a
        # no-op on non-zero ranks.
        callbacks=[MLflowStepMetricsCallback()],
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
                    "training_mode": training_mode,
                    "num_gpus": num_gpus,
                    "rank": rank,
                    "world_size": world_size,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "max_steps": MAX_STEPS,
                    "rank_0_training_record_count": len(examples_pdf),
                    "source_table": SOURCE_TABLE,
                    "sft_table": SFT_TABLE,
                    "training_stack": "trl-sft-peft-lora",
                    "loss_masking": "completion_only",
                    "enable_thinking": CHAT_TEMPLATE_KWARGS["enable_thinking"],
                    "lora_r": LORA_RANK,
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
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.log_param("adapter_output_dir", output_dir)

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            mlflow.log_metric("peak_cuda_memory_allocated_gb", peak_memory_gb)
            print(f"Peak CUDA memory allocated: {peak_memory_gb:.2f} GB")

        return run.info.run_id


def _warn_single_process_multi_gpu(world_size: int) -> None:
    """Flag the launcher setup that silently trains on 1/world_size of the data.

    Reached when serverless_gpu reports a multi-GPU workload but no per-process
    launcher exported RANK/WORLD_SIZE — i.e. one process is holding every GPU.
    Hugging Face Trainer then wraps the model in ``nn.DataParallel``, which both
    crashes Qwen3.5's Gated DeltaNet reference path ("lazy wrapper should be
    called at most once") and leaves this process claiming only rank 0's shard
    slice, so the other ranks' data is never read.
    """
    try:
        import torch

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        # A diagnostic must never be the thing that fails the run.
        visible = 0

    if visible > 1:
        print(
            f"WARNING: serverless_gpu reports world_size={world_size}, no "
            f"RANK/WORLD_SIZE is set, and this process sees {visible} GPUs. "
            "Launch one process per GPU (torchrun --nproc_per_node=gpu) — "
            "otherwise Trainer falls back to nn.DataParallel and only rank 0's "
            "shard slice is read."
        )


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) under any launcher.

    Precedence matters. A per-process launcher (torchrun / torch elastic, which
    ``train.yaml``'s CLI command uses) exports RANK/WORLD_SIZE/LOCAL_RANK, and
    those must win: ``serverless_gpu.runtime`` reports the *workload's* rank and
    size, so under torchrun it answers rank 0 / world_size N in **every**
    process — each rank would then claim rank 0's shards and act as the main
    process. The serverless_gpu runtime stays the fallback for the notebook
    ``@distributed`` path, and a lone process is the last resort.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank

    try:
        from serverless_gpu import runtime as rt

        rank, world_size = rt.get_global_rank(), rt.get_world_size()
    except Exception:
        return (
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
            local_rank,
        )

    if world_size > 1:
        _warn_single_process_multi_gpu(world_size)
    return rank, world_size, local_rank


def run_rank_training(
    sample_fraction: float | None = None,
    experiment_path: str | None = None,
) -> str | None:
    """Train this rank's shard slice of the exported SFT parquet files.

    ``sample_fraction`` overrides the ``training_sample_fraction`` value from
    the config — the runner notebook passes it from its training cell so the
    fraction can be changed live during the demo; the AIR CLI path leaves it
    ``None`` and uses the config value.

    ``experiment_path`` is the resolved MLflow experiment rank 0 should log to
    (``/Users/<user>/<experiment_name>``). The runner notebook passes the path
    it resolved from ``train.yaml``'s ``experiment_name``; AIR CLI runs leave it
    ``None`` because AIR already created the run in that experiment.

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
        return train_qwen3_sft(
            examples_pdf=examples_pdf,
            output_dir=f"{TRAINING_OUTPUT_DIR}/{world_size}gpu",
            run_name=f"{TRAINING_RUN_NAME}-{world_size}gpu{run_suffix}",
            training_mode=f"{world_size}_gpu_rank_sharded_sample",
            num_gpus=world_size,
            device_map={"": local_rank},
            save_artifacts=rank == 0,
            rank=rank,
            world_size=world_size,
            experiment_path=experiment_path,
        )
    finally:
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    rank, world_size, _ = get_distributed_context()
    # Under the AIR CLI the launcher already points the workload's run at
    # /Users/<user>/<experiment_name>; a bare `python train.py` resolves the
    # same name here so it lands in that experiment too.
    experiment_path = None if LAUNCHED_VIA_AIR_CLI else resolve_experiment_path(EXPERIMENT_NAME)
    run_id = run_rank_training(experiment_path=experiment_path)
    if rank == 0:
        print(f"Training MLflow run ID: {run_id}")
        print(f"Trained adapter output dir: {TRAINING_OUTPUT_DIR}/{world_size}gpu")


if __name__ == "__main__":
    main()
