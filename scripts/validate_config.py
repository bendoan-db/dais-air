#!/usr/bin/env python3
"""Validate the pipeline's configuration contracts.

Run from anywhere (no Databricks connection needed):

    python scripts/validate_config.py

The repo-root global.yaml is the single source of truth for every parameter
shared across the pipeline modules (setup, train+deploy, load test,
monitor); each stage YAML holds only stage-specific keys. This script checks
that structure before any workspace time is spent:

1. global.yaml parses and defines every required shared key (catalog,
   schema, experiment_name, source_table, sft_table, sft_volume,
   uc_model_name, model_name, endpoint_name, inference_table_prefix;
   model_volume_path may be empty).
2. train/train.yaml's training_config and deploy_config sections load
   through the real loaders (types, required keys, lists, metric goal,
   serving_requirements_file resolution) — which also proves global.yaml
   resolves from the train module.
3. No stage YAML re-introduces a global key (shadowed copies would be
   silently ignored by the notebooks and drift).
4. global.yaml's experiment_name equals train.yaml's top-level
   experiment_name (the AIR CLI schema requires the key there too — the one
   unavoidable duplication).
5. When model_volume_path is set, setup.yaml's models list contains a
   matching entry (volume_path == model_volume_path and huggingface_path ==
   model_name), so the weights setup downloads are the weights training loads.
6. experiment_name stays alphanumeric/-/_ (it becomes a Jobs API task key).
7. Every `-r <file>` reference in train.yaml's environment resolves.
8. The monitor stage's own keys are present and well-typed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))

GLOBAL_YAML = "global.yaml"
SETUP_YAML = "setup/setup.yaml"
LOAD_TEST_YAML = "load_test/serving_load_test.yaml"
MONITOR_YAML = "monitor/monitor.yaml"
TRAIN_YAML = "train/train.yaml"

# Keys owned by global.yaml. Stage YAMLs must not redefine them (the
# notebooks would ignore the copies). Includes retired aliases (`table`,
# `inference_table`) so old-style keys cannot creep back in.
GLOBAL_KEYS = {
    "catalog",
    "schema",
    "experiment_name",
    "source_table",
    "table",
    "sft_table",
    "sft_volume",
    "uc_model_name",
    "model_name",
    "model_volume_path",
    "endpoint_name",
    "inference_table_prefix",
    "inference_table",
}

REQUIRED_GLOBAL_KEYS = (
    "catalog",
    "schema",
    "experiment_name",
    "source_table",
    "sft_table",
    "sft_volume",
    "uc_model_name",
    "model_name",
    "endpoint_name",
    "inference_table_prefix",
)

errors: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        fail(f"{path}: file not found.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        fail(f"{path}: expected a YAML mapping, got {type(loaded).__name__}")
        return {}
    return loaded


def check_global_config() -> dict:
    global_config = load_yaml(REPO_ROOT / GLOBAL_YAML)
    for key in REQUIRED_GLOBAL_KEYS:
        if not str(global_config.get(key) or "").strip():
            fail(f"{GLOBAL_YAML}: missing required key `{key}`.")
    return global_config


def check_training_config() -> dict:
    """Exercise the real loader so type/requirement errors surface here."""
    from training_utils import load_training_config

    try:
        return load_training_config()
    except Exception as exc:
        fail(f"{TRAIN_YAML} / {GLOBAL_YAML}: load_training_config() failed: {exc}")
        return {}


def check_deploy_context() -> dict:
    """Exercise the deploy loader; it validates types, presence, the metric
    goal, and serving_requirements_file resolution itself."""
    from training_utils import load_deploy_config

    try:
        return load_deploy_config()
    except Exception as exc:
        fail(f"{TRAIN_YAML} (deploy_config): load_deploy_config() failed: {exc}")
        return {}


def check_no_global_shadowing(path: str, config: dict) -> None:
    shadowed = sorted(GLOBAL_KEYS & set(config))
    if shadowed:
        fail(
            f"{path}: defines {shadowed}, but these parameters live only in "
            f"{GLOBAL_YAML} — the notebooks read them from there, so these "
            "copies would be silently ignored. Remove them."
        )


def check_experiment_name_duplicate(global_config: dict) -> None:
    """The AIR CLI schema requires experiment_name at train.yaml's top level;
    it must equal global.yaml's copy."""
    train_yaml = load_yaml(REPO_ROOT / "train" / "train.yaml")
    workload_experiment = str(train_yaml.get("experiment_name") or "").strip()
    global_experiment = str(global_config.get("experiment_name") or "").strip()
    if not workload_experiment:
        fail(f"{TRAIN_YAML}: missing top-level experiment_name (required by the AIR CLI).")
    elif global_experiment and workload_experiment != global_experiment:
        fail(
            f"experiment_name mismatch: {GLOBAL_YAML} has {global_experiment!r} "
            f"but {TRAIN_YAML}'s top-level key has {workload_experiment!r} — "
            "AIR CLI runs would log to a different experiment than the one "
            "deployment searches. Keep the two equal."
        )


def check_experiment_name_charset(global_config: dict) -> None:
    experiment_name = str(global_config.get("experiment_name") or "").strip()
    if experiment_name and not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name):
        fail(
            f"{GLOBAL_YAML}: experiment_name {experiment_name!r} must be "
            "alphanumeric/-/_ (it becomes a Jobs API task key under AIR)."
        )


def check_models_contract(setup_config: dict, global_config: dict) -> None:
    models = setup_config.get("models") or []
    if not isinstance(models, list) or not models:
        fail(f"{SETUP_YAML}: `models:` must be a non-empty list.")
        return

    volume_paths = {}
    for entry in models:
        if not isinstance(entry, dict):
            fail(f"{SETUP_YAML}: models entries must be mappings, got {entry!r}")
            continue
        volume_path = str(entry.get("volume_path") or "").strip().rstrip("/")
        huggingface_path = str(entry.get("huggingface_path") or "").strip()
        if not volume_path or not huggingface_path:
            fail(f"{SETUP_YAML}: models entry missing huggingface_path/volume_path: {entry!r}")
            continue
        if not volume_path.startswith("/Volumes/") or len(Path(volume_path).parts) < 5:
            fail(
                f"{SETUP_YAML}: volume_path must look like "
                f"/Volumes/<catalog>/<schema>/<volume>/...: {volume_path}"
            )
        if volume_path in volume_paths:
            fail(f"{SETUP_YAML}: duplicate volume_path in models: {volume_path}")
        volume_paths[volume_path] = huggingface_path

    model_volume_path = str(global_config.get("model_volume_path") or "").rstrip("/")
    model_name = str(global_config.get("model_name") or "").strip()
    if model_volume_path:
        if model_volume_path not in volume_paths:
            fail(
                f"{GLOBAL_YAML}: model_volume_path ({model_volume_path}) has no "
                f"matching volume_path in {SETUP_YAML}'s models list — training "
                "would load weights setup never downloads."
            )
        elif volume_paths[model_volume_path] != model_name:
            fail(
                f"{SETUP_YAML}: the models entry for {model_volume_path} downloads "
                f"{volume_paths[model_volume_path]!r}, but {GLOBAL_YAML}'s model_name "
                f"is {model_name!r} — the snapshot would not match the training base model."
            )


def check_environment_requirements() -> None:
    """Every `-r <file>` entry in train.yaml's environment must resolve."""
    train_yaml = load_yaml(REPO_ROOT / "train" / "train.yaml")
    dependencies = (train_yaml.get("environment", {}) or {}).get("dependencies", []) or []
    for dependency in dependencies:
        dependency = str(dependency).strip()
        if not dependency.startswith("-r"):
            continue
        reference = dependency[2:].strip().strip("'\"")
        reference_path = Path(reference)
        if not reference_path.is_absolute():
            reference_path = REPO_ROOT / "train" / reference_path
        if not reference_path.exists():
            fail(
                f"{TRAIN_YAML}: environment dependency {dependency!r} "
                f"references a missing file ({reference_path})."
            )


def check_monitor_config(monitor_config: dict) -> None:
    for key in ("unpacked_table", "checkpoint_volume"):
        if not str(monitor_config.get(key) or "").strip():
            fail(f"{MONITOR_YAML}: missing required key `{key}`.")

    response_json_fields = monitor_config.get("response_json_fields")
    if response_json_fields is not None and not isinstance(response_json_fields, list):
        fail(f"{MONITOR_YAML}: response_json_fields must be a list (or omitted).")


def main() -> int:
    global_config = check_global_config()
    training_context = check_training_config()
    deploy_context = check_deploy_context()

    setup_config = load_yaml(REPO_ROOT / "setup" / "setup.yaml")
    load_test_config = load_yaml(REPO_ROOT / "load_test" / "serving_load_test.yaml")
    monitor_config = load_yaml(REPO_ROOT / "monitor" / "monitor.yaml")

    for path, config in (
        (SETUP_YAML, setup_config),
        (LOAD_TEST_YAML, load_test_config),
        (MONITOR_YAML, monitor_config),
    ):
        check_no_global_shadowing(path, config)

    if global_config:
        check_experiment_name_duplicate(global_config)
        check_experiment_name_charset(global_config)
        check_models_contract(setup_config, global_config)

    check_environment_requirements()
    check_monitor_config(monitor_config)

    if errors:
        print(f"Configuration validation FAILED ({len(errors)} error(s)):\n")
        for index, message in enumerate(errors, 1):
            print(f"{index}. {message}\n")
        return 1

    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
