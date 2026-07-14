"""Shared helpers for the AIR fine-tuning pipeline.

This is a plain Python module (NOT a Databricks notebook): the workspace
refuses ordinary imports of notebook-formatted files (NotebookImportException),
and ``train.py`` must import these helpers under the runner notebook, the AI
Runtime CLI, and local execution alike. Notebooks consume it the same way —
insert this file's directory into ``sys.path`` and import; do not ``%run`` it.

The module is deliberately not named ``utils``: GPU base environments (via
``nvidia_cutlass_dsl``) register their own top-level ``utils`` module once the
torch/CUDA stack loads, which shadows any local ``utils.py`` on import.
"""

from pathlib import Path

import yaml

MODULE_DIR = Path(__file__).resolve().parent


def load_yaml_config(config_filename: str, base_dir: Path | None = None) -> tuple[Path, dict]:
    """Load a YAML mapping from ``base_dir`` (default: this module's folder)."""
    config_path = (base_dir or MODULE_DIR) / config_filename

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(config).__name__}")

    return config_path, config


GLOBAL_CONFIG_FILENAME = "global.yaml"


def load_global_config() -> tuple[Path, dict]:
    """Load the repo-root ``global.yaml`` (pipeline-wide shared parameters).

    Searched at this module's parent directory (the repository root) with a
    cwd fallback, which covers workspace notebooks, local scripts, and AIR
    CLI runs — the code snapshot roots at the repository (train.yaml's
    ``root_path: ..``) precisely so this file ships with ``train/``.
    """
    candidates = [
        MODULE_DIR.parent / GLOBAL_CONFIG_FILENAME,
        Path.cwd() / GLOBAL_CONFIG_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_yaml_config(GLOBAL_CONFIG_FILENAME, base_dir=candidate.parent)
    raise FileNotFoundError(
        "global.yaml not found (searched: "
        + ", ".join(str(candidate) for candidate in candidates)
        + "). It lives at the repository root; AIR CLI snapshots include it "
        "because train.yaml's code_source roots at the repo (root_path: ..)."
    )


def config_value(config: dict, key: str):
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def config_str(config: dict, key: str) -> str:
    value = str(config_value(config, key)).strip()
    if not value:
        raise ValueError(f"Config key cannot be empty: {key}")
    return value


def config_int(config: dict, key: str) -> int:
    return int(config_value(config, key))


def config_float(config: dict, key: str) -> float:
    return float(config_value(config, key))


def config_bool(config: dict, key: str) -> bool:
    value = config_value(config, key)
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Config key must be boolean-like: {key}")


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def full_name(*parts: str) -> str:
    return ".".join(quote_identifier(part) for part in parts)


def get_spark_session():
    """Return the active Spark session, attaching a serverless Databricks
    Connect session when none exists yet (local scripts, GPU workers).

    Workspace notebooks and jobs already have a session — reuse it instead of
    building one (serverless job environments bundle a databricks-connect
    whose builder has no ``serverless()`` attribute, so building would fail
    there anyway)."""
    try:
        from pyspark.sql import SparkSession

        active_session = SparkSession.getActiveSession()
        if active_session is not None:
            return active_session
    except Exception:
        pass

    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    if hasattr(builder, "serverless"):
        return builder.serverless().getOrCreate()
    return builder.getOrCreate()


VOLUME_PATH_PREFIX = "/Volumes/"


def _staging_fingerprint(source_dir: Path, source_files: list[Path]) -> str:
    """Hash the source path plus per-file (name, size, mtime) so a retrained
    adapter written to the same volume path stages a fresh copy."""
    import hashlib

    manifest = [str(source_dir)]
    for source_file in source_files:
        file_stat = source_file.stat()
        manifest.append(f"{source_file.name}:{file_stat.st_size}:{file_stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(manifest).encode()).hexdigest()[:12]


def stage_model_locally(source_dir: str) -> str:
    """Copy a model directory from a UC volume to node-local disk; return the copy.

    safetensors loading memory-maps the weight files, and mmap page faults
    against the /Volumes FUSE mount turn into thousands of small,
    latency-bound object-store reads — minutes for multi-GB weights. The
    mount is fast at sequential streaming, so one copy to local disk plus a
    local mmap load is far faster than loading from the volume directly.

    Only top-level regular files are copied — everything from_pretrained
    reads — which skips training checkpoint-*/ subdirectories and Hugging
    Face .cache folders. Concurrent callers (GPU ranks sharing a node) are
    serialized with an flock plus a completion marker so the copy happens
    once per node. When the staged directory is a LoRA adapter whose recorded
    base model is itself a volume path, the base weights are staged too and
    the local copy of adapter_config.json is pointed at them (the volume copy
    is never modified).
    """
    import fcntl
    import json
    import shutil
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    source = Path(source_dir)
    source_files = sorted(path for path in source.iterdir() if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No files to stage in {source}")
    total_gb = sum(source_file.stat().st_size for source_file in source_files) / 1024**3

    local_disk_tmp = Path("/local_disk0/tmp")
    staging_base = local_disk_tmp if local_disk_tmp.exists() else Path(tempfile.gettempdir())
    staging_root = staging_base / "air-model-staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    destination = staging_root / f"{source.name}-{_staging_fingerprint(source, source_files)}"
    marker = destination.with_name(destination.name + ".complete")
    lock_path = destination.with_name(destination.name + ".lock")

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if marker.exists():
            print(f"Reusing staged model copy: {destination}")
        else:
            start = time.monotonic()
            destination.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=min(8, len(source_files))) as pool:
                list(
                    pool.map(
                        lambda source_file: shutil.copy2(source_file, destination / source_file.name),
                        source_files,
                    )
                )

            adapter_config_path = destination / "adapter_config.json"
            if adapter_config_path.exists():
                adapter_config = json.loads(adapter_config_path.read_text())
                base_model_path = str(adapter_config.get("base_model_name_or_path") or "")
                if base_model_path.startswith(VOLUME_PATH_PREFIX):
                    adapter_config["base_model_name_or_path"] = stage_model_locally(base_model_path)
                    adapter_config_path.write_text(json.dumps(adapter_config, indent=2))

            print(
                f"Staged {len(source_files)} files ({total_gb:.2f} GB) from {source} "
                f"to {destination} in {time.monotonic() - start:.1f}s"
            )
            marker.touch()

    return str(destination)


def parse_compute_block(workload_config: dict) -> tuple[int, str]:
    """Derive (gpus, gpu_type) for ``@distributed`` from the workload's
    ``compute`` block, so the notebook and AIR CLI paths are sized by one
    setting.

    ``num_accelerators`` is authoritative for the count. ``accelerator_type``
    accepts the AIR spellings (``GPU_8xH100``, ``8xH100``, ``GPU_1xA10``) or
    a bare chip name (``H100``); when a count is embedded it must agree with
    ``num_accelerators``.
    """
    import re

    compute = workload_config.get("compute") or {}
    gpus = int(compute.get("num_accelerators") or 1)
    raw_accelerator_type = str(compute.get("accelerator_type") or "A10").strip()
    chip = raw_accelerator_type
    if chip.upper().startswith("GPU_"):
        chip = chip[len("GPU_"):]
    embedded_count_match = re.match(r"^(\d+)x(.+)$", chip, re.IGNORECASE)
    if embedded_count_match:
        embedded_count = int(embedded_count_match.group(1))
        chip = embedded_count_match.group(2)
        if embedded_count != gpus:
            raise ValueError(
                f"compute.accelerator_type {raw_accelerator_type!r} embeds "
                f"{embedded_count} accelerators but num_accelerators is {gpus} "
                "— make the two agree."
            )
    return gpus, chip


# ---- Fraud prompt/response rendering (single source of the SFT contract) ----
# Setup stages RAW transaction records (setup/02 writes no prompt text).
# train/00_prep_sft.py renders them into SFT prompt/response records with the
# functions below (the load test renders live payloads with the same
# functions), and the training loop applies the per-model step — the loaded
# model's own chat template. The prompt's "- key: value" block is a
# monitoring contract: monitor.yaml's prompt_fields are extracted by matching
# these exact line prefixes (scripts/validate_config.py checks coverage
# against this template).

# The raw staged columns the renderers read.
FRAUD_RECORD_COLUMNS = (
    "user_id_text",
    "card_id_text",
    "transaction_ts_text",
    "amount_usd",
    "use_chip_text",
    "merchant_city_text",
    "merchant_state_text",
    "mcc_text",
    "errors_text",
    "is_fraud",
    "has_error_signal",
)


def render_fraud_prompt(record) -> str:
    """Render the user prompt from one raw transaction record.

    Reproduces byte-for-byte the prompt shape the pipeline has always trained
    and served with (previously built with Spark expressions in setup/02): a
    fixed instruction header plus the "- key: value" transaction block the
    monitoring stage parses prompt_fields from.
    """
    return (
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        f"- user_id: {record['user_id_text']}\n"
        f"- card_id: {record['card_id_text']}\n"
        f"- timestamp: {record['transaction_ts_text']}\n"
        f"- amount_usd: {float(record['amount_usd']):.2f}\n"
        f"- use_chip: {record['use_chip_text']}\n"
        f"- merchant_city: {record['merchant_city_text']}\n"
        f"- merchant_state: {record['merchant_state_text']}\n"
        f"- merchant_category_code: {record['mcc_text']}\n"
        f"- errors: {record['errors_text']}"
    )


def render_fraud_response(record, suspicious_amount_threshold: float) -> str:
    """Render the assistant target: compact JSON with keys risk/action/reason.

    The labeling heuristic (the historical fraud label wins; error signals or
    a large amount mean review) moved here from setup/02 together with the
    prompt template; ``suspicious_amount_threshold`` comes from the training
    config.
    """
    import json

    needs_review = bool(record["has_error_signal"]) or (
        float(record["amount_usd"]) >= suspicious_amount_threshold
    )
    if int(record["is_fraud"] or 0) == 1:
        risk, action = "likely_fraud", "decline_and_escalate"
        reason = "The historical label marks this transaction as fraud."
    elif needs_review:
        risk, action = "suspicious", "step_up_authentication"
        reason = (
            "The transaction is not labeled fraud, but amount or error signals "
            "warrant review."
        )
    else:
        risk, action = "legitimate", "approve"
        reason = "The historical label is non-fraud and no strong review signal is present."
    return json.dumps(
        {"risk": risk, "action": action, "reason": reason}, separators=(",", ":")
    )


# ---- Staged-export access (split=train|eval / shard_id=N parquet) -----------


def split_shard_dirs(files_root: str, split: str, hint: str | None = None) -> list[Path]:
    """List the shard_id=N directories of one split of a staged export.

    Both staged exports use this layout (``split=train/shard_id=N/``,
    ``split=eval/shard_id=N/``): setup/02's raw-record export and
    train/00_prep_sft.py's SFT staging. Fails fast with guidance — ``hint``
    overrides the default (which assumes the training modules' SFT staging,
    the only callers that reach this through the helpers below).
    """
    hint = hint or (
        "Run train/00_prep_sft.py first (after the setup notebooks) to render "
        "and stage the SFT-format export."
    )
    root = Path(files_root)
    shard_dirs = sorted((root / f"split={split}").glob("shard_id=*"))
    if not shard_dirs:
        if sorted(root.glob("shard_id=*")):
            raise FileNotFoundError(
                f"{files_root} holds a pre-split export (shard_id=* directories "
                f"at the root) — it must be restaged with train/eval splits. {hint}"
            )
        raise FileNotFoundError(
            f"No split={split} parquet shards found under {files_root}. {hint}"
        )
    return shard_dirs


def claim_rank_shard_files(
    files_root: str,
    split: str,
    rank: int,
    world_size: int,
    sample_fraction: float,
    seed: int,
) -> tuple[list[str], float]:
    """Claim this rank's parquet files for one split of the staged export.

    Each rank takes the ``shard_id=N`` directories where
    ``N % world_size == rank`` (the AIR rank-sharding contract). Two-level
    sampling: for ``sample_fraction < 1`` only a seeded subset of the rank's
    shard directories is loaded (hash shards are uniform, so this is
    statistically equivalent to row sampling) and the returned within-shard
    fraction lands the caller's row-level sample on the exact requested
    fraction — keeping the HF ``datasets`` Arrow conversion proportional to
    the fraction instead of always materializing the rank's full slice.

    Returns ``(parquet file paths, within-shard row fraction)``.
    """
    import math
    import random

    rank_shard_dirs = [
        shard_dir
        for shard_dir in split_shard_dirs(files_root, split)
        if int(shard_dir.name.split("=", 1)[1]) % world_size == rank
    ]

    within_shard_fraction = 1.0
    if sample_fraction < 1.0 and rank_shard_dirs:
        total_rank_dirs = len(rank_shard_dirs)
        dirs_to_load = max(1, math.ceil(total_rank_dirs * sample_fraction))
        rank_shard_dirs = sorted(random.Random(seed).sample(rank_shard_dirs, dirs_to_load))
        within_shard_fraction = min(1.0, sample_fraction * total_rank_dirs / dirs_to_load)

    parquet_files = [
        str(parquet_file)
        for shard_dir in rank_shard_dirs
        for parquet_file in sorted(shard_dir.glob("*.parquet"))
    ]
    return parquet_files, within_shard_fraction


def sample_eval_records(
    files_root: str,
    sample_size: int,
    seed: int,
    stratify_column: str | None = None,
):
    """Draw a deterministic sample of records from the staged eval split.

    Shard directories are read in seeded order and reading stops as soon as
    the accumulated rows can satisfy the request, so a small sample never
    materializes the whole eval split. With ``stratify_column`` the draw is
    half positive (column == 1) / half rest — at the ~1% natural fraud rate
    an unstratified sample would carry almost no positives, leaving
    precision/recall meaningless.

    Returns a pandas DataFrame (fewer than ``sample_size`` rows when the
    eval split itself is smaller than the request).
    """
    import random

    import pandas as pd

    shard_dirs = list(split_shard_dirs(files_root, "eval"))
    random.Random(seed).shuffle(shard_dirs)

    positive_needed = sample_size // 2 if stratify_column else 0
    rest_needed = sample_size - positive_needed
    frames = []
    positive_count = total_count = 0
    for shard_dir in shard_dirs:
        frame = pd.read_parquet(shard_dir)
        if frame.empty:
            continue
        frames.append(frame)
        total_count += len(frame)
        if stratify_column:
            positive_count += int((frame[stratify_column] == 1).sum())
            if positive_count >= positive_needed and (total_count - positive_count) >= rest_needed:
                break
        elif total_count >= sample_size:
            break

    if not frames:
        raise ValueError(f"The eval split under {files_root} holds no rows.")
    records_pdf = pd.concat(frames, ignore_index=True)

    if not stratify_column:
        if len(records_pdf) <= sample_size:
            return records_pdf
        return records_pdf.sample(n=sample_size, random_state=seed)

    positive_pdf = records_pdf[records_pdf[stratify_column] == 1]
    rest_pdf = records_pdf[records_pdf[stratify_column] != 1]
    positive_n = min(sample_size // 2, len(positive_pdf))
    rest_n = min(sample_size - positive_n, len(rest_pdf))
    return pd.concat(
        [
            positive_pdf.sample(n=positive_n, random_state=seed),
            rest_pdf.sample(n=rest_n, random_state=seed),
        ]
    )


def load_training_config(config_filename: str = "train.yaml") -> dict:
    """Load the ``parameters.training_config`` section and derive shared names.

    ``config_filename`` selects the workload file next to this module —
    ``train.yaml`` (Unsloth LoRA DDP) by default, or ``train_fsdp.yaml`` for
    the TRL+FSDP variant. Under an AIR CLI run, ``$HYPERPARAMETERS_PATH``
    carries whichever workload file was submitted, so the filename only
    matters for notebook and local execution.

    Each pipeline stage owns its full configuration so the modules run
    standalone; the values ``train.yaml`` shares with the setup and load-test
    YAMLs (catalog, schema, table/volume/endpoint names) are checked for
    agreement by ``scripts/validate_config.py``.

    Under an AI Runtime CLI workload the parameters arrive via the YAML file
    at ``$HYPERPARAMETERS_PATH`` (which reflects ``air run --override``
    values); otherwise they are read from ``train.yaml`` next to this file.

    Returns a flat dict of typed config values, derived UC names/paths, and
    quoted SQL identifiers, intended to be bound into the caller's namespace
    with ``globals().update(load_training_config())``. Used by the training
    runner notebook and by train.py; deliberately a function (not top-level
    code) so that importing this module has no side effects.
    """
    import os

    hyperparameters_path = os.environ.get("HYPERPARAMETERS_PATH")
    workload_config: dict = {}
    if hyperparameters_path:
        config_path = Path(hyperparameters_path)
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        # The AIR CLI docs say HYPERPARAMETERS_PATH holds just the
        # `parameters` dict, but v0.1.0b1 points it at the full workload
        # YAML — accept either shape.
        parameters = loaded.get("parameters", loaded)
        if "parameters" in loaded:
            workload_config = loaded
    else:
        config_path, workload_config = load_yaml_config(config_filename)
        parameters = config_value(workload_config, "parameters")
    config = config_value(parameters, "training_config")

    # catalog/schema come from the repo-root global.yaml; every other key in
    # this loader is stage-owned (training_config or the workload level).
    _, global_config = load_global_config()
    uc_catalog = config_str(global_config, "catalog")
    uc_schema = config_str(global_config, "schema")

    source_table_name = config_str(config, "source_table")
    sft_table_name = config_str(config, "sft_table")
    sft_volume = config_str(config, "sft_volume")
    uc_model_name = config_str(config, "uc_model_name")
    model_name = config_str(config, "model_name")
    # Optional UC volume snapshot of the base weights; empty or missing means
    # download from Hugging Face by model_name.
    model_volume_path = str(config.get("model_volume_path") or "").strip()
    # The workspace experiment name comes from the workload-level
    # experiment_name (the same value the AIR CLI uses); it is absent when
    # HYPERPARAMETERS_PATH carries only the parameters dict, so fall back to
    # a model-derived name.
    experiment_name = str(
        workload_config.get("experiment_name") or f"{uc_model_name}_finetuning"
    ).strip()

    uc_volume = config_str(config, "checkpoint_volume")
    sft_staging_volume = config_str(config, "sft_staging_volume")
    max_steps = config_int(config, "max_steps")
    notebook_gpus, notebook_gpu_type = parse_compute_block(workload_config)
    output_root = f"/Volumes/{uc_catalog}/{uc_schema}/{uc_volume}/{uc_model_name}"

    lora_target_modules = config_value(config, "lora_target_modules")
    if not isinstance(lora_target_modules, list) or not lora_target_modules:
        raise ValueError("lora_target_modules must be a non-empty list in training_config")

    # Chat-template markers are read without strip(): trailing newlines are
    # part of the text train_on_responses_only matches against.
    response_instruction_part = str(config_value(config, "response_instruction_part"))
    response_part = str(config_value(config, "response_part"))

    return {
        "CONFIG_PATH": config_path,
        "UC_CATALOG": uc_catalog,
        "UC_SCHEMA": uc_schema,
        "SOURCE_TABLE_NAME": source_table_name,
        "SFT_TABLE_NAME": sft_table_name,
        "UC_VOLUME": uc_volume,
        "UC_MODEL_NAME": uc_model_name,
        "MODEL_NAME": model_name,
        "MODEL_VOLUME_PATH": model_volume_path,
        # Where FastLanguageModel.from_pretrained reads the base weights:
        # the UC volume snapshot when configured, else the HF repo id.
        "MODEL_LOAD_PATH": model_volume_path or model_name,
        "MAX_SEQ_LENGTH": config_int(config, "max_seq_length"),
        "MAX_STEPS": max_steps,
        "PER_DEVICE_TRAIN_BATCH_SIZE": config_int(config, "per_device_train_batch_size"),
        "GRADIENT_ACCUMULATION_STEPS": config_int(config, "gradient_accumulation_steps"),
        "LEARNING_RATE": config_float(config, "learning_rate"),
        "WARMUP_STEPS": config_int(config, "warmup_steps"),
        "TRAINING_SAMPLE_FRACTION": config_float(config, "training_sample_fraction"),
        "EVAL_SAMPLE_SIZE": config_int(config, "eval_sample_size"),
        # Labeling heuristic for the assistant responses rendered in the
        # training loop (render_fraud_response above).
        "SUSPICIOUS_AMOUNT_THRESHOLD": config_float(config, "suspicious_amount_threshold"),
        "LORA_R": config_int(config, "lora_r"),
        "LORA_ALPHA": config_int(config, "lora_alpha"),
        "LORA_DROPOUT": config_float(config, "lora_dropout"),
        "LORA_TARGET_MODULES": [str(module) for module in lora_target_modules],
        "RESPONSE_INSTRUCTION_PART": response_instruction_part,
        "RESPONSE_PART": response_part,
        # The runner notebook's @distributed cell is sized from the same
        # workload-level `compute` block the AIR CLI uses — one setting for
        # both launch paths. (Defaults 1xA10 when HYPERPARAMETERS_PATH
        # carries only the parameters dict; irrelevant there, since AIR
        # already provisioned the compute.)
        "NOTEBOOK_GPUS": notebook_gpus,
        "NOTEBOOK_GPU_TYPE": notebook_gpu_type,
        "EXPERIMENT_NAME": experiment_name,
        "SEED": config_int(config, "seed"),
        "SFT_VOLUME": sft_volume,
        "SFT_STAGING_VOLUME": sft_staging_volume,
        # Raw-record parquet export written by setup/02 (split=train|eval /
        # shard_id=N) — the input train/00_prep_sft.py renders.
        "RAW_SPLIT_FILES_DIR": f"/Volumes/{uc_catalog}/{uc_schema}/{sft_volume}/{sft_table_name}",
        # SFT-format parquet staging written by train/00_prep_sft.py (rendered
        # prompt/assistant_response records, same split/shard layout) — what
        # the training loop reads instead of querying Delta through Spark on
        # the GPU workers (per the AIR data-loading guidance).
        "SFT_FILES_DIR": f"/Volumes/{uc_catalog}/{uc_schema}/{sft_staging_volume}/{sft_table_name}",
        "SOURCE_TABLE": f"{uc_catalog}.{uc_schema}.{source_table_name}",
        "SFT_TABLE": f"{uc_catalog}.{uc_schema}.{sft_table_name}",
        "OUTPUT_ROOT": output_root,
        "TRAINING_OUTPUT_DIR": f"{output_root}/training_demo",
        "TRAINING_RUN_NAME": f"{uc_model_name}-training-steps{max_steps}",
        "schema_q": full_name(uc_catalog, uc_schema),
        "volume_q": full_name(uc_catalog, uc_schema, uc_volume),
        "sft_staging_volume_q": full_name(uc_catalog, uc_schema, sft_staging_volume),
        "source_table_q": full_name(uc_catalog, uc_schema, source_table_name),
        "sft_table_q": full_name(uc_catalog, uc_schema, sft_table_name),
    }


def load_deploy_config() -> dict:
    """Load ``parameters.deploy_config`` (registration/serving settings).

    The deployment notebook (``02_register_and_deploy.py``) shares
    ``train.yaml`` with training: catalog/schema come from the repo-root
    ``global.yaml`` and the experiment from the workload level (both via
    :func:`load_training_config`), while ``deploy_config`` holds the
    deployment-stage keys. Returns a flat dict intended for
    ``globals().update(...)``, like the training loader.
    """
    config_path, workload_config = load_yaml_config("train.yaml")
    parameters = config_value(workload_config, "parameters")
    config = config_value(parameters, "deploy_config")

    training_context = load_training_config()
    uc_catalog = training_context["UC_CATALOG"]
    uc_schema = training_context["UC_SCHEMA"]

    uc_model_name = config_str(config, "uc_model_name")
    endpoint_name = config_str(config, "endpoint_name")
    inference_table_prefix = config_str(config, "inference_table_prefix")

    best_run_metric_goal = config_str(config, "best_run_metric_goal").lower()
    if best_run_metric_goal not in {"minimize", "maximize"}:
        raise ValueError(
            "best_run_metric_goal must be 'minimize' or 'maximize', "
            f"got {best_run_metric_goal!r} (train.yaml deploy_config)."
        )

    # The serving container's packages come from a requirements file (the
    # consolidated requirements.txt by default) instead of an inline YAML
    # list; read it here so registration can pass the parsed list to
    # log_model.
    serving_requirements_file = config_str(config, "serving_requirements_file")
    serving_requirements_path = Path(serving_requirements_file)
    if not serving_requirements_path.is_absolute():
        serving_requirements_path = MODULE_DIR / serving_requirements_path
    if not serving_requirements_path.exists():
        raise FileNotFoundError(
            f"serving_requirements_file not found: {serving_requirements_path} "
            "(deploy_config in train.yaml; relative paths resolve against the "
            "train.yaml directory)."
        )
    serving_pip_requirements = [
        line.strip()
        for line in serving_requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not serving_pip_requirements:
        raise ValueError(f"{serving_requirements_path} contains no requirements.")

    return {
        "DEPLOY_CONFIG_PATH": config_path,
        "UC_CATALOG": uc_catalog,
        "UC_SCHEMA": uc_schema,
        "EXPERIMENT_NAME": training_context["EXPERIMENT_NAME"],
        "RUN_ID": str(config.get("run_id") or "").strip(),
        "BEST_RUN_METRIC": config_str(config, "best_run_metric"),
        "BEST_RUN_METRIC_GOAL": best_run_metric_goal,
        "UC_MODEL_NAME": uc_model_name,
        "SERVED_MODEL_NAME": config_str(config, "served_model_name"),
        "SERVING_PIP_REQUIREMENTS": [str(requirement) for requirement in serving_pip_requirements],
        "VLLM_DTYPE": config_str(config, "vllm_dtype"),
        "VLLM_MAX_MODEL_LEN": config_int(config, "vllm_max_model_len"),
        "VLLM_GPU_MEMORY_UTILIZATION": config_float(config, "vllm_gpu_memory_utilization"),
        "INFERENCE_TABLE_PREFIX": inference_table_prefix,
        "ENDPOINT_NAME": endpoint_name,
        "ENDPOINT_DESCRIPTION": config_str(config, "endpoint_description"),
        "SERVING_WORKLOAD_TYPE": config_str(config, "serving_workload_type"),
        "SERVING_PROVISIONED_CONCURRENCY": config_int(config, "serving_provisioned_concurrency"),
        # Optional legacy sizing knob; passed through only when non-empty.
        "SERVING_WORKLOAD_SIZE": str(config.get("serving_workload_size") or "").strip(),
        "SERVING_SCALE_TO_ZERO": config_bool(config, "serving_scale_to_zero"),
        "FULL_MODEL_NAME": f"{uc_catalog}.{uc_schema}.{uc_model_name}",
    }


def ensure_uc_object(spark, ddl_statement: str) -> None:
    """Run a ``CREATE ... IF NOT EXISTS`` statement with a permission-friendly error.

    The pipeline creates schemas and volumes on first run; in customer
    workspaces the most common failure is a missing Unity Catalog privilege,
    which surfaces from Spark as a generic AnalysisException. Reframe it with
    the grants to ask for.
    """
    try:
        spark.sql(ddl_statement)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run `{ddl_statement}`. If the object does not already "
            "exist, the current user likely lacks the required Unity Catalog "
            "privilege — ask an admin for USE CATALOG plus CREATE SCHEMA / "
            "CREATE VOLUME / CREATE TABLE on the configured catalog and "
            "schema, or point train/train.yaml's training_config at objects "
            "that already exist."
        ) from exc


def init_training_workspace(training_context: dict):
    """Initialize Spark and ensure the training schema/checkpoint volume exist.

    Takes the dict returned by :func:`load_training_config` and returns the
    Spark session.
    """
    spark = get_spark_session()

    ensure_uc_object(spark, f"CREATE SCHEMA IF NOT EXISTS {training_context['schema_q']}")
    ensure_uc_object(spark, f"CREATE VOLUME IF NOT EXISTS {training_context['volume_q']}")

    Path(training_context["TRAINING_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    return spark
