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
    Connect session when none exists yet (local scripts, GPU workers)."""
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.serverless().getOrCreate()


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
    max_steps = config_int(config, "max_steps")
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
        "LORA_R": config_int(config, "lora_r"),
        "LORA_ALPHA": config_int(config, "lora_alpha"),
        "LORA_DROPOUT": config_float(config, "lora_dropout"),
        "LORA_TARGET_MODULES": [str(module) for module in lora_target_modules],
        "RESPONSE_INSTRUCTION_PART": response_instruction_part,
        "RESPONSE_PART": response_part,
        # GPUs for the runner notebook's @distributed cell; the AIR CLI path
        # sizes compute from the workload-level `compute` block instead.
        "NOTEBOOK_GPUS": config_int(config, "notebook_gpus"),
        "NOTEBOOK_GPU_TYPE": config_str(config, "notebook_gpu_type"),
        "EXPERIMENT_NAME": experiment_name,
        "SEED": config_int(config, "seed"),
        "SFT_VOLUME": sft_volume,
        # Parquet export of the SFT table, written by setup per the AIR
        # data-loading guidance; training reads these files instead of
        # querying Delta through Spark on the GPU workers.
        "SFT_FILES_DIR": f"/Volumes/{uc_catalog}/{uc_schema}/{sft_volume}/{sft_table_name}",
        "SOURCE_TABLE": f"{uc_catalog}.{uc_schema}.{source_table_name}",
        "SFT_TABLE": f"{uc_catalog}.{uc_schema}.{sft_table_name}",
        "OUTPUT_ROOT": output_root,
        "TRAINING_OUTPUT_DIR": f"{output_root}/training_demo",
        "TRAINING_RUN_NAME": f"{uc_model_name}-training-steps{max_steps}",
        "schema_q": full_name(uc_catalog, uc_schema),
        "volume_q": full_name(uc_catalog, uc_schema, uc_volume),
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
