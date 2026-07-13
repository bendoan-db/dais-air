#!/usr/bin/env python3
"""Validate the pipeline's cross-file configuration contracts.

Run from anywhere (no Databricks connection needed):

    python scripts/validate_config.py

Each stage (setup, train, load test) owns a self-contained YAML so the
modules run standalone. The values they share must agree across files, and
this script checks every agreement before any workspace time is spent:

1. 01_train/train.yaml parses and its training_config section loads through
   training_utils.load_training_config() (types, required keys, lists).
2. catalog and schema agree across all five YAMLs (setup, train, deploy,
   load test, monitor).
3. setup.yaml's `table` equals train.yaml's `source_table` (setup writes the
   table training reads).
4. `sft_table` agrees across setup, train, and load test; `sft_volume` agrees
   between setup.yaml and train.yaml (together they locate the Parquet export
   training reads and the prompts the load test samples).
5. `endpoint_name` agrees between deploy.yaml and serving_load_test.yaml
   (deploy creates the endpoint the load test targets).
6. deploy.yaml's `experiment_name` agrees with train.yaml's (run selection
   searches the experiment training logs to), unless deploy.yaml uses an
   absolute workspace path; `best_run_metric_goal` is minimize|maximize and
   the registration/serving keys are present.
7. monitor.yaml's `inference_table` equals deploy.yaml's
   `inference_table_prefix` + '_payload' (inference logging is always enabled
   at deployment, and the monitor must unpack the table the endpoint actually
   writes), and the monitor stage keys are present.
8. When model_volume_path is set, setup.yaml's models list contains a
   matching entry (volume_path == model_volume_path and huggingface_path ==
   model_name), so the weights setup downloads are the weights training loads.
9. experiment_name stays alphanumeric/-/_ (it becomes a Jobs API task key).
10. train.yaml's environment.dependencies and 01_train/requirements.txt stay
    in sync (same requirement lines, order-insensitive).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "01_train"))

SETUP_YAML = "00_setup/setup.yaml"
LOAD_TEST_YAML = "02_deploy/load_test/serving_load_test.yaml"
DEPLOY_YAML = "02_deploy/deploy.yaml"
MONITOR_YAML = "03_monitor/monitor.yaml"
TRAIN_YAML = "01_train/train.yaml (training_config)"

errors: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        fail(f"{path}: expected a YAML mapping, got {type(loaded).__name__}")
        return {}
    return loaded


def check_training_config() -> dict:
    """Exercise the real loader so type/requirement errors surface here."""
    from training_utils import load_training_config

    try:
        return load_training_config()
    except Exception as exc:
        fail(f"01_train/train.yaml: load_training_config() failed: {exc}")
        return {}


def stage_value(config: dict, key: str, location: str) -> str | None:
    """Fetch a required stage-YAML key, recording an error when missing."""
    value = str(config.get(key) or "").strip()
    if not value:
        fail(f"{location}: missing required key `{key}`.")
        return None
    return value


def check_agreement(description: str, entries: list[tuple[str, str | None]]) -> None:
    """All present values must be identical (missing ones already errored)."""
    present = [(location, value) for location, value in entries if value is not None]
    if len({value for _, value in present}) > 1:
        details = "; ".join(f"{location} has {value!r}" for location, value in present)
        fail(
            f"{description} must agree across files ({details}) — the stages "
            "would otherwise read/write different Unity Catalog objects."
        )


def check_shared_identity(
    setup_config: dict,
    load_test_config: dict,
    deploy_config: dict,
    monitor_config: dict,
    training_context: dict,
) -> None:
    check_agreement(
        "catalog",
        [
            (SETUP_YAML, stage_value(setup_config, "catalog", SETUP_YAML)),
            (LOAD_TEST_YAML, stage_value(load_test_config, "catalog", LOAD_TEST_YAML)),
            (DEPLOY_YAML, stage_value(deploy_config, "catalog", DEPLOY_YAML)),
            (MONITOR_YAML, stage_value(monitor_config, "catalog", MONITOR_YAML)),
            (TRAIN_YAML, training_context.get("UC_CATALOG")),
        ],
    )
    check_agreement(
        "schema",
        [
            (SETUP_YAML, stage_value(setup_config, "schema", SETUP_YAML)),
            (LOAD_TEST_YAML, stage_value(load_test_config, "schema", LOAD_TEST_YAML)),
            (DEPLOY_YAML, stage_value(deploy_config, "schema", DEPLOY_YAML)),
            (MONITOR_YAML, stage_value(monitor_config, "schema", MONITOR_YAML)),
            (TRAIN_YAML, training_context.get("UC_SCHEMA")),
        ],
    )
    check_agreement(
        "source table (setup.yaml `table` / train.yaml `source_table`)",
        [
            (SETUP_YAML, stage_value(setup_config, "table", SETUP_YAML)),
            (TRAIN_YAML, training_context.get("SOURCE_TABLE_NAME")),
        ],
    )
    check_agreement(
        "sft_table",
        [
            (SETUP_YAML, stage_value(setup_config, "sft_table", SETUP_YAML)),
            (LOAD_TEST_YAML, stage_value(load_test_config, "sft_table", LOAD_TEST_YAML)),
            (TRAIN_YAML, training_context.get("SFT_TABLE_NAME")),
        ],
    )
    check_agreement(
        "sft_volume",
        [
            (SETUP_YAML, stage_value(setup_config, "sft_volume", SETUP_YAML)),
            (TRAIN_YAML, training_context.get("SFT_VOLUME")),
        ],
    )
    check_agreement(
        "endpoint_name",
        [
            (DEPLOY_YAML, stage_value(deploy_config, "endpoint_name", DEPLOY_YAML)),
            (LOAD_TEST_YAML, stage_value(load_test_config, "endpoint_name", LOAD_TEST_YAML)),
        ],
    )


def check_deploy_config(deploy_config: dict, training_context: dict) -> None:
    """Deploy-stage keys: presence, metric goal, and experiment agreement."""
    for key in (
        "uc_model_name",
        "served_model_name",
        "serving_workload_type",
        "endpoint_description",
        "vllm_dtype",
        "best_run_metric",
        # Inference logging is unconditional at deployment; the prefix names
        # the payload table the monitoring stage unpacks.
        "inference_table_prefix",
    ):
        stage_value(deploy_config, key, DEPLOY_YAML)

    goal = str(deploy_config.get("best_run_metric_goal") or "").strip().lower()
    if goal not in {"minimize", "maximize"}:
        fail(
            f"{DEPLOY_YAML}: best_run_metric_goal must be 'minimize' or "
            f"'maximize', got {goal!r}."
        )

    serving_pip_requirements = deploy_config.get("serving_pip_requirements")
    if not isinstance(serving_pip_requirements, list) or not serving_pip_requirements:
        fail(f"{DEPLOY_YAML}: serving_pip_requirements must be a non-empty list.")

    # Run selection searches the experiment training logs to; a bare name must
    # match train.yaml's experiment_name (absolute workspace paths pass — the
    # notebook resolves them as-is).
    deploy_experiment = stage_value(deploy_config, "experiment_name", DEPLOY_YAML)
    if deploy_experiment and not deploy_experiment.startswith("/"):
        check_agreement(
            "experiment_name",
            [
                (DEPLOY_YAML, deploy_experiment),
                (TRAIN_YAML, training_context.get("EXPERIMENT_NAME")),
            ],
        )


def check_monitor_config(monitor_config: dict, deploy_config: dict) -> None:
    """Monitor-stage keys, and the payload-table contract with deploy.yaml."""
    for key in ("inference_table", "unpacked_table", "checkpoint_volume"):
        stage_value(monitor_config, key, MONITOR_YAML)

    response_json_fields = monitor_config.get("response_json_fields")
    if response_json_fields is not None and not isinstance(response_json_fields, list):
        fail(f"{MONITOR_YAML}: response_json_fields must be a list (or omitted).")

    # Deployment always enables inference logging to
    # <inference_table_prefix>_payload; the monitor must unpack that exact
    # table.
    prefix = str(deploy_config.get("inference_table_prefix") or "").strip()
    monitor_table = str(monitor_config.get("inference_table") or "").strip()
    if prefix and monitor_table:
        expected = f"{prefix}_payload"
        if monitor_table != expected:
            fail(
                f"{MONITOR_YAML}: inference_table is {monitor_table!r} but the "
                f"endpoint logs to {expected!r} ({DEPLOY_YAML}'s "
                "inference_table_prefix + '_payload') — the monitor would unpack "
                "a table the endpoint never writes."
            )


def check_models_contract(setup_config: dict, training_context: dict) -> None:
    models = setup_config.get("models") or []
    if not isinstance(models, list) or not models:
        fail("00_setup/setup.yaml: `models:` must be a non-empty list.")
        return

    volume_paths = {}
    for entry in models:
        if not isinstance(entry, dict):
            fail(f"00_setup/setup.yaml: models entries must be mappings, got {entry!r}")
            continue
        volume_path = str(entry.get("volume_path") or "").strip().rstrip("/")
        huggingface_path = str(entry.get("huggingface_path") or "").strip()
        if not volume_path or not huggingface_path:
            fail(f"00_setup/setup.yaml: models entry missing huggingface_path/volume_path: {entry!r}")
            continue
        if not volume_path.startswith("/Volumes/") or len(Path(volume_path).parts) < 5:
            fail(
                "00_setup/setup.yaml: volume_path must look like "
                f"/Volumes/<catalog>/<schema>/<volume>/...: {volume_path}"
            )
        if volume_path in volume_paths:
            fail(f"00_setup/setup.yaml: duplicate volume_path in models: {volume_path}")
        volume_paths[volume_path] = huggingface_path

    model_volume_path = str(training_context.get("MODEL_VOLUME_PATH") or "").rstrip("/")
    model_name = training_context.get("MODEL_NAME")
    if model_volume_path:
        if model_volume_path not in volume_paths:
            fail(
                f"01_train/train.yaml: model_volume_path ({model_volume_path}) has no "
                "matching volume_path in 00_setup/setup.yaml's models list — training "
                "would load weights setup never downloads."
            )
        elif volume_paths[model_volume_path] != model_name:
            fail(
                f"00_setup/setup.yaml: the models entry for {model_volume_path} downloads "
                f"{volume_paths[model_volume_path]!r}, but 01_train/train.yaml's model_name "
                f"is {model_name!r} — the snapshot would not match the training base model."
            )


def check_experiment_name(training_context: dict) -> None:
    experiment_name = training_context.get("EXPERIMENT_NAME", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name or ""):
        fail(
            f"01_train/train.yaml: experiment_name {experiment_name!r} must be "
            "alphanumeric/-/_ (it becomes a Jobs API task key under AIR)."
        )


def check_dependency_sync() -> None:
    train_yaml = load_yaml(REPO_ROOT / "01_train" / "train.yaml")
    yaml_deps = set(
        str(dep).strip()
        for dep in (train_yaml.get("environment", {}) or {}).get("dependencies", [])
    )
    requirements = set(
        line.strip()
        for line in (REPO_ROOT / "01_train" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    if yaml_deps != requirements:
        only_yaml = sorted(yaml_deps - requirements)
        only_requirements = sorted(requirements - yaml_deps)
        fail(
            "01_train/train.yaml environment.dependencies and 01_train/requirements.txt "
            f"are out of sync. Only in train.yaml: {only_yaml or '[]'}; only in "
            f"requirements.txt: {only_requirements or '[]'}."
        )


def main() -> int:
    training_context = check_training_config()

    setup_config = load_yaml(REPO_ROOT / "00_setup" / "setup.yaml")
    load_test_config = load_yaml(REPO_ROOT / "02_deploy" / "load_test" / "serving_load_test.yaml")
    deploy_config = load_yaml(REPO_ROOT / "02_deploy" / "deploy.yaml")
    monitor_config = load_yaml(REPO_ROOT / "03_monitor" / "monitor.yaml")

    if training_context:
        check_shared_identity(
            setup_config, load_test_config, deploy_config, monitor_config, training_context
        )
        check_deploy_config(deploy_config, training_context)
        check_monitor_config(monitor_config, deploy_config)
        check_models_contract(setup_config, training_context)
        check_experiment_name(training_context)

    shard_key_columns = setup_config.get("sft_shard_key_columns")
    if not isinstance(shard_key_columns, list) or not shard_key_columns:
        fail("00_setup/setup.yaml: sft_shard_key_columns must be a non-empty list.")

    check_dependency_sync()

    if errors:
        print(f"Configuration validation FAILED ({len(errors)} error(s)):\n")
        for index, message in enumerate(errors, 1):
            print(f"{index}. {message}\n")
        return 1

    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
