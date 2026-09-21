"""Project-local configuration and I/O helpers for an AIR trainer.

Everything this project needs to resolve its inputs lives here: the typed
configuration objects loaded from ``train.yaml``, node-local model staging, and
rank-aware Parquet shard selection. Nothing outside this directory is imported,
so the whole project can be copied into another workspace unchanged.

Only ``pandas`` is imported lazily, inside the one function that needs it. The
heavy GPU packages are deferred in ``train.py`` instead, where the reason for
deferring them (environment variables must be set before import) applies.
"""

import fcntl
import hashlib
import math
import os
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent
VOLUME_PATH_PREFIX = "/Volumes/"


# --------------------------------------------------------------------------
# YAML coercion helpers
# --------------------------------------------------------------------------
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


def _str_tuple(config: dict, key: str) -> tuple[str, ...]:
    value = _required(config, key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"training_config.{key} must be a non-empty list")
    return tuple(str(item).strip() for item in value)


def _volume_path(config: dict, key: str) -> str:
    value = _str(config, key).rstrip("/")
    if not value.startswith(VOLUME_PATH_PREFIX) or len(Path(value).parts) < 5:
        raise ValueError(
            f"training_config.{key} must look like "
            f"/Volumes/<catalog>/<schema>/<volume>/...; got {value!r}"
        )
    return value


def _read_workload_file(config_filename: str) -> tuple[Path, dict]:
    """Read the project's own workload YAML from disk."""
    config_path = PROJECT_DIR / config_filename
    workload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(workload, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config_path, workload


def _load_parameters(config_filename: str, section: str) -> tuple[Path, dict]:
    """Return one ``parameters`` section, from either launch path.

    Under ``air run`` the CLI writes the ``parameters`` block to the file named
    by ``HYPERPARAMETERS_PATH``; some versions write the whole workload. Both
    shapes are accepted. In a notebook the project's own YAML is read directly.
    """
    hyperparameters_path = os.environ.get("HYPERPARAMETERS_PATH")
    if hyperparameters_path:
        config_path = Path(hyperparameters_path)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        parameters = loaded.get("parameters", loaded)
    else:
        config_path, workload = _read_workload_file(config_filename)
        parameters = _required(workload, "parameters")

    if not isinstance(parameters, dict):
        raise ValueError(f"Expected parameters to be a mapping in {config_path}")
    section_config = _required(parameters, section)
    if not isinstance(section_config, dict):
        raise ValueError(f"Expected parameters.{section} to be a mapping in {config_path}")
    return config_path, section_config


# --------------------------------------------------------------------------
# Typed configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    """Every runtime input for one training run, resolved from train.yaml."""

    config_path: Path
    experiment_path: str
    run_name: str

    model_name: str
    model_weights_path: str
    expected_model_classes: tuple[str, ...]
    fsdp_wrap_classes: tuple[str, ...]

    train_data_path: str
    eval_data_path: str
    convert_sft: bool
    ignore_partitions: bool
    suspicious_amount_threshold: float
    output_dir: str

    max_seq_length: int
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    training_sample_fraction: float
    eval_sample_size: int
    logging_steps: int
    eval_steps: int
    seed: int

    def output_dir_for(self, world_size: int) -> str:
        """The one place the trained-checkpoint path convention is defined.

        ``02_register_and_deploy.py`` finds this path through the training
        run's ``model_output_dir`` parameter, so it is written once here.
        """
        return f"{self.output_dir}/{world_size}gpu"


@dataclass(frozen=True)
class DeployConfig:
    """Registration and serving inputs for this project's checkpoint."""

    config_path: Path
    uc_catalog: str
    uc_schema: str
    experiment_path: str

    run_id: str
    best_run_metric: str
    best_run_metric_goal: str

    uc_model_name: str
    serving_pip_requirements: list[str]
    serving_dtype: str
    serving_continuous_batching: bool
    serving_reasoning: str

    inference_table_prefix: str
    endpoint_name: str
    endpoint_description: str
    serving_workload_type: str
    serving_workload_size: str
    serving_scale_to_zero: bool

    @property
    def full_model_name(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.uc_model_name}"

    @property
    def inference_payload_table(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.inference_table_prefix}_payload"


def load_project_config(config_filename: str = "train.yaml") -> TrainingConfig:
    """Return this project's typed training configuration."""
    config_path, config = _load_parameters(config_filename, "training_config")

    train_data_path = _volume_path(config, "train_data_path")
    eval_data_path = _volume_path(config, "eval_data_path")
    if train_data_path == eval_data_path:
        raise ValueError("train_data_path and eval_data_path must be different")

    sample_fraction = _float(config, "training_sample_fraction")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("training_sample_fraction must be in (0, 1]")

    max_steps = _int(config, "max_steps")
    return TrainingConfig(
        config_path=config_path,
        experiment_path=_str(config, "experiment_path"),
        run_name=f"{_str(config, 'project_name')}-steps{max_steps}",
        model_name=_str(config, "model_name"),
        model_weights_path=_volume_path(config, "model_weights_path"),
        expected_model_classes=_str_tuple(config, "expected_model_classes"),
        fsdp_wrap_classes=_str_tuple(config, "fsdp_wrap_classes"),
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        convert_sft=_bool(config, "convert_sft"),
        ignore_partitions=_bool(config, "ignore_partitions"),
        suspicious_amount_threshold=_float(config, "suspicious_amount_threshold"),
        output_dir=_volume_path(config, "output_dir"),
        max_seq_length=_int(config, "max_seq_length"),
        max_steps=max_steps,
        per_device_train_batch_size=_int(config, "per_device_train_batch_size"),
        per_device_eval_batch_size=_int(config, "per_device_eval_batch_size"),
        gradient_accumulation_steps=_int(config, "gradient_accumulation_steps"),
        learning_rate=_float(config, "learning_rate"),
        warmup_steps=_int(config, "warmup_steps"),
        training_sample_fraction=sample_fraction,
        eval_sample_size=_int(config, "eval_sample_size"),
        logging_steps=_int(config, "logging_steps"),
        eval_steps=_int(config, "eval_steps"),
        seed=_int(config, "seed"),
    )


def load_deploy_config(config_filename: str = "train.yaml") -> DeployConfig:
    """Return this project's typed registration and serving configuration."""
    config_path, workload = _read_workload_file(config_filename)
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

    return DeployConfig(
        config_path=config_path,
        uc_catalog=_str(training_config, "catalog"),
        uc_schema=_str(training_config, "schema"),
        experiment_path=_str(training_config, "experiment_path"),
        run_id=str(config.get("run_id") or "").strip(),
        best_run_metric=_str(config, "best_run_metric"),
        best_run_metric_goal=metric_goal,
        uc_model_name=_str(config, "uc_model_name"),
        serving_pip_requirements=serving_requirements,
        serving_dtype=_str(config, "serving_dtype"),
        serving_continuous_batching=_bool(config, "serving_continuous_batching"),
        serving_reasoning=serving_reasoning,
        inference_table_prefix=_str(config, "inference_table_prefix"),
        endpoint_name=_str(config, "endpoint_name"),
        endpoint_description=_str(config, "endpoint_description"),
        serving_workload_type=_str(config, "serving_workload_type"),
        serving_workload_size=_str(config, "serving_workload_size"),
        serving_scale_to_zero=_bool(config, "serving_scale_to_zero"),
    )


def load_notebook_compute(config_filename: str = "train.yaml") -> tuple[int, str]:
    """Return ``(gpus, gpu_type)`` for the notebook's ``@distributed`` call.

    Read from the project's own YAML rather than from ``HYPERPARAMETERS_PATH``,
    which carries only the ``parameters`` block and therefore has no ``compute``
    section to fall back on.
    """
    _, workload = _read_workload_file(config_filename)
    compute = workload.get("compute") or {}
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


# --------------------------------------------------------------------------
# Node-local staging
# --------------------------------------------------------------------------
def local_staging_root() -> Path:
    """Fastest writable scratch directory on this node.

    Trainer output and staged model weights both go here: writing safetensors
    straight to /Volumes fails, and mmap reads through FUSE are slow.
    """
    local_disk = Path("/local_disk0/tmp")
    return local_disk if local_disk.exists() else Path(tempfile.gettempdir())


def _staging_fingerprint(source_dir: Path, source_files: list[Path]) -> str:
    manifest = [str(source_dir)]
    for source_file in source_files:
        stat = source_file.stat()
        manifest.append(f"{source_file.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(manifest).encode()).hexdigest()[:12]


def stage_model_locally(source_dir: str) -> str:
    """Copy volume-hosted model files once per node for fast mmap loading."""
    source = Path(source_dir)
    source_files = sorted(path for path in source.iterdir() if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No model files found in {source}")

    staging_root = local_staging_root() / "air-model-staging"
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


# --------------------------------------------------------------------------
# Rank-aware data selection
# --------------------------------------------------------------------------
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
    """Read enough eval inputs to produce a deterministic sample.

    ``stratify_column`` is best-effort: pre-converted SFT data is only required
    to carry ``prompt`` and ``assistant_response``, so when the column is absent
    this falls back to unstratified sampling instead of failing.
    """
    import pandas as pd

    sources = (
        _all_parquet_files(eval_data_path)
        if ignore_partitions
        else _shard_dirs(eval_data_path)
    )
    random.Random(seed).shuffle(sources)
    frames = []
    positive_count = total_count = 0
    for source in sources:
        frame = pd.read_parquet(source)
        if frame.empty:
            continue
        if stratify_column and stratify_column not in frame.columns:
            print(
                f"Eval data has no {stratify_column!r} column; sampling "
                f"{sample_size} records without stratification."
            )
            stratify_column = None
        frames.append(frame)
        total_count += len(frame)
        if stratify_column:
            positive_needed = sample_size // 2
            positive_count += int((frame[stratify_column] == 1).sum())
            if (
                positive_count >= positive_needed
                and total_count - positive_count >= sample_size - positive_needed
            ):
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
