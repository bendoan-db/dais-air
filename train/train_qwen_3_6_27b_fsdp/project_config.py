"""Project-local configuration and I/O helpers for an AIR trainer."""

from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
VOLUME_PATH_PREFIX = "/Volumes/"


def _required(config: dict, key: str):
    if key not in config:
        raise KeyError(f"Missing required training_config key: {key}")
    return config[key]


def _str(config: dict, key: str) -> str:
    value = str(_required(config, key)).strip()
    if not value:
        raise ValueError(f"training_config.{key} cannot be empty")
    return value


def _int(config: dict, key: str) -> int:
    return int(_required(config, key))


def _float(config: dict, key: str) -> float:
    return float(_required(config, key))


def _bool(config: dict, key: str) -> bool:
    value = _required(config, key)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"training_config.{key} must be boolean-like")


def _parse_compute(workload_config: dict) -> tuple[int, str]:
    import re

    compute = workload_config.get("compute") or {}
    gpus = int(compute.get("num_accelerators") or 1)
    raw_type = str(compute.get("accelerator_type") or "A10").strip()
    gpu_type = raw_type.removeprefix("GPU_")
    embedded_count = re.match(r"^(\d+)x(.+)$", gpu_type, re.IGNORECASE)
    if embedded_count:
        if int(embedded_count.group(1)) != gpus:
            raise ValueError(
                f"compute.accelerator_type={raw_type!r} disagrees with "
                f"compute.num_accelerators={gpus}"
            )
        gpu_type = embedded_count.group(2)
    return gpus, gpu_type


def _load_workload(config_filename: str) -> tuple[Path, dict, dict]:
    import os

    hyperparameters_path = os.environ.get("HYPERPARAMETERS_PATH")
    if hyperparameters_path:
        config_path = Path(hyperparameters_path)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        workload_config = loaded if "parameters" in loaded else {}
        parameters = loaded.get("parameters", loaded)
    else:
        config_path = PROJECT_DIR / config_filename
        workload_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(workload_config, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        parameters = _required(workload_config, "parameters")

    if not isinstance(parameters, dict):
        raise ValueError(f"Expected parameters to be a mapping in {config_path}")
    training_config = _required(parameters, "training_config")
    if not isinstance(training_config, dict):
        raise ValueError(f"Expected parameters.training_config to be a mapping in {config_path}")
    return config_path, workload_config, training_config


def _volume_path(config: dict, key: str) -> str:
    value = _str(config, key).rstrip("/")
    if not value.startswith(VOLUME_PATH_PREFIX) or len(Path(value).parts) < 5:
        raise ValueError(
            f"training_config.{key} must look like "
            f"/Volumes/<catalog>/<schema>/<volume>/...; got {value!r}"
        )
    return value


def load_project_config(config_filename: str = "train.yaml") -> dict:
    """Return the self-contained project configuration as typed constants."""
    config_path, workload_config, config = _load_workload(config_filename)

    train_data_path = _volume_path(config, "train_data_path")
    eval_data_path = _volume_path(config, "eval_data_path")
    if train_data_path == eval_data_path:
        raise ValueError("train_data_path and eval_data_path must be different")

    sample_fraction = _float(config, "training_sample_fraction")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("training_sample_fraction must be in (0, 1]")

    project_name = _str(config, "project_name")
    max_steps = _int(config, "max_steps")
    notebook_gpus, notebook_gpu_type = _parse_compute(workload_config)

    model_weights_path = _volume_path(config, "model_weights_path")
    return {
        "CONFIG_PATH": config_path,
        "PROJECT_NAME": project_name,
        "UC_CATALOG": _str(config, "catalog"),
        "UC_SCHEMA": _str(config, "schema"),
        "EXPERIMENT_PATH": _str(config, "experiment_path"),
        "MODEL_NAME": _str(config, "model_name"),
        "MODEL_WEIGHTS_PATH": model_weights_path,
        "MODEL_LOAD_PATH": model_weights_path,
        "TRAIN_DATA_PATH": train_data_path,
        "EVAL_DATA_PATH": eval_data_path,
        "CONVERT_SFT": _bool(config, "convert_sft"),
        "IGNORE_PARTITIONS": _bool(config, "ignore_partitions"),
        "SUSPICIOUS_AMOUNT_THRESHOLD": _float(config, "suspicious_amount_threshold"),
        "TRAINING_OUTPUT_DIR": _volume_path(config, "output_dir"),
        "TRAINING_RUN_NAME": f"{project_name}-steps{max_steps}",
        "MAX_SEQ_LENGTH": _int(config, "max_seq_length"),
        "MAX_STEPS": max_steps,
        "PER_DEVICE_TRAIN_BATCH_SIZE": _int(config, "per_device_train_batch_size"),
        "PER_DEVICE_EVAL_BATCH_SIZE": _int(config, "per_device_eval_batch_size"),
        "GRADIENT_ACCUMULATION_STEPS": _int(config, "gradient_accumulation_steps"),
        "LEARNING_RATE": _float(config, "learning_rate"),
        "WARMUP_STEPS": _int(config, "warmup_steps"),
        "TRAINING_SAMPLE_FRACTION": sample_fraction,
        "EVAL_SAMPLE_SIZE": _int(config, "eval_sample_size"),
        "LOGGING_STEPS": _int(config, "logging_steps"),
        "EVAL_STEPS": _int(config, "eval_steps"),
        "SEED": _int(config, "seed"),
        "NOTEBOOK_GPUS": notebook_gpus,
        "NOTEBOOK_GPU_TYPE": notebook_gpu_type,
    }


def load_deploy_config(config_filename: str = "train.yaml") -> dict:
    """Return this project's registration and serving configuration."""
    config_path = PROJECT_DIR / config_filename
    workload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(workload, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    parameters = _required(workload, "parameters")
    training_config = _required(parameters, "training_config")
    config = _required(parameters, "deploy_config")

    metric_goal = _str(config, "best_run_metric_goal").lower()
    if metric_goal not in {"minimize", "maximize"}:
        raise ValueError("best_run_metric_goal must be 'minimize' or 'maximize'")
    serving_reasoning = _str(config, "serving_reasoning").lower()
    if serving_reasoning not in {"on", "off", "auto"}:
        raise ValueError("serving_reasoning must be 'on', 'off', or 'auto'")

    requirements_path = Path(_str(config, "serving_requirements_file"))
    if not requirements_path.is_absolute():
        requirements_path = PROJECT_DIR / requirements_path
    if not requirements_path.exists():
        raise FileNotFoundError(f"Serving requirements file not found: {requirements_path}")
    serving_requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not serving_requirements:
        raise ValueError(f"{requirements_path} contains no requirements")

    catalog = _str(training_config, "catalog")
    schema = _str(training_config, "schema")
    model_name = _str(config, "uc_model_name")
    return {
        "DEPLOY_CONFIG_PATH": config_path,
        "PROJECT_NAME": _str(training_config, "project_name"),
        "UC_CATALOG": catalog,
        "UC_SCHEMA": schema,
        "EXPERIMENT_PATH": _str(training_config, "experiment_path"),
        "RUN_ID": str(config.get("run_id") or "").strip(),
        "BEST_RUN_METRIC": _str(config, "best_run_metric"),
        "BEST_RUN_METRIC_GOAL": metric_goal,
        "UC_MODEL_NAME": model_name,
        "SERVED_MODEL_NAME": _str(config, "served_model_name"),
        "SERVING_PIP_REQUIREMENTS": serving_requirements,
        "SERVING_DTYPE": _str(config, "serving_dtype"),
        "SERVING_CONTINUOUS_BATCHING": _bool(config, "serving_continuous_batching"),
        "SERVING_REASONING": serving_reasoning,
        "INFERENCE_TABLE_PREFIX": _str(config, "inference_table_prefix"),
        "ENDPOINT_NAME": _str(config, "endpoint_name"),
        "ENDPOINT_DESCRIPTION": _str(config, "endpoint_description"),
        "SERVING_WORKLOAD_TYPE": _str(config, "serving_workload_type"),
        "SERVING_WORKLOAD_SIZE": _str(config, "serving_workload_size"),
        "SERVING_SCALE_TO_ZERO": _bool(config, "serving_scale_to_zero"),
        "FULL_MODEL_NAME": f"{catalog}.{schema}.{model_name}",
    }


def _staging_fingerprint(source_dir: Path, source_files: list[Path]) -> str:
    import hashlib

    manifest = [str(source_dir)]
    for source_file in source_files:
        stat = source_file.stat()
        manifest.append(f"{source_file.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(manifest).encode()).hexdigest()[:12]


def stage_model_locally(source_dir: str) -> str:
    """Copy volume-hosted model files once per node for fast mmap loading."""
    import fcntl
    import shutil
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    source = Path(source_dir)
    source_files = sorted(path for path in source.iterdir() if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No model files found in {source}")

    local_tmp = Path("/local_disk0/tmp")
    staging_root = (local_tmp if local_tmp.exists() else Path(tempfile.gettempdir())) / "air-model-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = staging_root / f"{source.name}-{_staging_fingerprint(source, source_files)}"
    marker = destination.with_name(destination.name + ".complete")

    with destination.with_name(destination.name + ".lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not marker.exists():
            started = time.monotonic()
            destination.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=min(8, len(source_files))) as pool:
                list(pool.map(lambda path: shutil.copy2(path, destination / path.name), source_files))

            marker.touch()
            print(f"Staged {source} to {destination} in {time.monotonic() - started:.1f}s")
        else:
            print(f"Reusing staged model copy: {destination}")
    return str(destination)


def _shard_dirs(data_path: str) -> list[Path]:
    shard_dirs = sorted(Path(data_path).glob("shard_id=*"))
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard_id=N parquet directories found under {data_path}. "
            "Point the project YAML at one prepared train or eval split."
        )
    return shard_dirs


def _all_parquet_files(data_path: str) -> list[Path]:
    parquet_files = sorted(Path(data_path).rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_path}")
    return parquet_files


def claim_rank_shard_files(
    data_path: str,
    rank: int,
    world_size: int,
    sample_fraction: float,
    seed: int,
    ignore_partitions: bool = False,
) -> tuple[list[str], float]:
    """Return this rank's input files and post-load sampling fraction."""
    import math
    import random

    if ignore_partitions:
        return [str(path) for path in _all_parquet_files(data_path)], sample_fraction

    rank_dirs = [
        path
        for path in _shard_dirs(data_path)
        if int(path.name.split("=", 1)[1]) % world_size == rank
    ]
    within_shard_fraction = 1.0
    if sample_fraction < 1.0 and rank_dirs:
        total_dirs = len(rank_dirs)
        dirs_to_load = max(1, math.ceil(total_dirs * sample_fraction))
        rank_dirs = sorted(random.Random(seed).sample(rank_dirs, dirs_to_load))
        within_shard_fraction = min(1.0, sample_fraction * total_dirs / dirs_to_load)

    parquet_files = [
        str(parquet_file)
        for shard_dir in rank_dirs
        for parquet_file in sorted(shard_dir.glob("*.parquet"))
    ]
    return parquet_files, within_shard_fraction


def sample_eval_records(
    eval_data_path: str,
    sample_size: int,
    seed: int,
    stratify_column: str | None = None,
    ignore_partitions: bool = False,
):
    """Read enough eval inputs to produce a deterministic sample."""
    import random

    import pandas as pd

    sources = (
        _all_parquet_files(eval_data_path)
        if ignore_partitions
        else _shard_dirs(eval_data_path)
    )
    random.Random(seed).shuffle(sources)
    positive_needed = sample_size // 2 if stratify_column else 0
    rest_needed = sample_size - positive_needed
    frames = []
    positive_count = total_count = 0
    for source in sources:
        frame = pd.read_parquet(source)
        if frame.empty:
            continue
        frames.append(frame)
        total_count += len(frame)
        if stratify_column:
            positive_count += int((frame[stratify_column] == 1).sum())
            if positive_count >= positive_needed and total_count - positive_count >= rest_needed:
                break
        elif total_count >= sample_size:
            break

    if not frames:
        raise ValueError(f"The eval data at {eval_data_path} contains no rows")
    records = pd.concat(frames, ignore_index=True)
    if not stratify_column:
        return records if len(records) <= sample_size else records.sample(sample_size, random_state=seed)

    positives = records[records[stratify_column] == 1]
    rest = records[records[stratify_column] != 1]
    positive_n = min(sample_size // 2, len(positives))
    rest_n = min(sample_size - positive_n, len(rest))
    return pd.concat(
        [
            positives.sample(positive_n, random_state=seed),
            rest.sample(rest_n, random_state=seed),
        ],
        ignore_index=True,
    )
