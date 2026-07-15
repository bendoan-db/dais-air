#!/usr/bin/env python3
"""Validate standalone trainer projects and remaining pipeline contracts."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "monitor"))

GLOBAL_YAML = "global.yaml"
SETUP_YAML = "setup/setup.yaml"
SETUP_UTILS_PATH = REPO_ROOT / "setup" / "utils.py"
LOAD_TEST_UTILS_PATH = REPO_ROOT / "load_test" / "utils.py"
LOAD_TEST_YAML = "load_test/serving_load_test.yaml"
MONITOR_YAML = "monitor/monitor.yaml"
PROJECTS = {
    "qwen_unsloth": "train/train_qwen_unsloth",
    "phi_4_unsloth": "train/train_phi_4_unsloth",
    "gpt_oss_fsdp": "train/train_gpt_oss_fsdp",
}

errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def warn(message: str) -> None:
    warnings.append(message)


def load_yaml(relative_path: str) -> dict:
    path = REPO_ROOT / relative_path
    if not path.exists():
        fail(f"{relative_path}: file not found")
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        fail(f"{relative_path}: expected a YAML mapping")
        return {}
    return loaded


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def required(config: dict, key: str, location: str) -> str | None:
    value = str(config.get(key) or "").strip()
    if not value:
        fail(f"{location}: missing required key `{key}`")
        return None
    return value


def check_agreement(description: str, entries: list[tuple[str, str | None]]) -> None:
    present = [(location, value) for location, value in entries if value is not None]
    if len({value for _, value in present}) > 1:
        details = "; ".join(f"{location} has {value!r}" for location, value in present)
        fail(f"{description} must agree across files ({details})")


def load_project_context(project_name: str, relative_dir: str) -> tuple[dict, dict, dict]:
    project_dir = REPO_ROOT / relative_dir
    for filename in (
        "01_runner.py",
        "02_register_and_deploy.py",
        "train.py",
        "train.yaml",
        "project_config.py",
        "sft_conversion.py",
        "requirements.txt",
    ):
        if not (project_dir / filename).exists():
            fail(f"{relative_dir}: missing required project file {filename}")

    workload = load_yaml(f"{relative_dir}/train.yaml")
    config_module_path = project_dir / "project_config.py"
    if not config_module_path.exists():
        return workload, {}, {}

    spec = importlib.util.spec_from_file_location(
        f"{project_name}_project_config", config_module_path
    )
    if spec is None or spec.loader is None:
        fail(f"{relative_dir}/project_config.py: could not create import spec")
        return workload, {}, {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        context = module.load_project_config()
    except Exception as exc:
        fail(f"{relative_dir}/train.yaml: load_project_config() failed: {exc}")
        context = {}
    try:
        deploy_context = module.load_deploy_config()
    except Exception as exc:
        fail(f"{relative_dir}/train.yaml: load_deploy_config() failed: {exc}")
        deploy_context = {}

    experiment_name = str(workload.get("experiment_name") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name):
        fail(
            f"{relative_dir}/train.yaml: experiment_name must be non-empty and "
            "contain only letters, numbers, hyphens, and underscores"
        )

    snapshot = ((workload.get("code_source") or {}).get("snapshot") or {})
    if snapshot.get("root_path") != ".":
        fail(f"{relative_dir}/train.yaml: code_source.snapshot.root_path must be '.'")
    command = str(workload.get("command") or "")
    if "$CODE_SOURCE_PATH/train.py" not in command:
        fail(f"{relative_dir}/train.yaml: command must execute the project-local train.py")

    dependencies = (workload.get("environment") or {}).get("dependencies") or []
    for dependency in dependencies:
        dependency = str(dependency).strip()
        if dependency.startswith("-r"):
            reference = dependency[2:].strip().strip("'\"")
            if not (project_dir / reference).exists():
                fail(
                    f"{relative_dir}/train.yaml: {dependency!r} references missing "
                    f"file {project_dir / reference}"
                )
    return workload, context, deploy_context


def check_model_snapshots(setup_config: dict, project_contexts: dict[str, dict]) -> None:
    model_entries = setup_config.get("models") or []
    if not isinstance(model_entries, list) or not model_entries:
        fail(f"{SETUP_YAML}: `models` must be a non-empty list")
        return

    snapshots: dict[str, str] = {}
    for entry in model_entries:
        if not isinstance(entry, dict):
            fail(f"{SETUP_YAML}: model entries must be mappings")
            continue
        volume_path = str(entry.get("volume_path") or "").strip().rstrip("/")
        model_name = str(entry.get("huggingface_path") or "").strip()
        if not volume_path.startswith("/Volumes/") or len(Path(volume_path).parts) < 5:
            fail(f"{SETUP_YAML}: invalid model volume_path {volume_path!r}")
            continue
        if not model_name:
            fail(f"{SETUP_YAML}: model entry for {volume_path!r} has no huggingface_path")
            continue
        if volume_path in snapshots:
            fail(f"{SETUP_YAML}: duplicate model volume_path {volume_path!r}")
        snapshots[volume_path] = model_name

    for project_name, context in project_contexts.items():
        weights_path = str(context.get("MODEL_WEIGHTS_PATH") or "").rstrip("/")
        model_name = str(context.get("MODEL_NAME") or "")
        if weights_path not in snapshots:
            warn(
                f"{PROJECTS[project_name]}/train.yaml: model_weights_path is not "
                "downloaded by setup/04_download_base_model_weights.py; ensure it is "
                f"pre-populated: {weights_path}"
            )
        elif snapshots[weights_path] != model_name:
            fail(
                f"{PROJECTS[project_name]}/train.yaml: model_name {model_name!r} "
                f"does not match setup snapshot {snapshots[weights_path]!r} at {weights_path}"
            )


def check_remaining_stage_contracts(
    setup_config: dict,
    load_test_config: dict,
    monitor_config: dict,
    deploy_context: dict,
) -> None:
    check_agreement(
        "sft_table",
        [
            (SETUP_YAML, required(setup_config, "sft_table", SETUP_YAML)),
            (LOAD_TEST_YAML, required(load_test_config, "sft_table", LOAD_TEST_YAML)),
            (MONITOR_YAML, required(monitor_config, "sft_table", MONITOR_YAML)),
        ],
    )
    check_agreement(
        "endpoint_name",
        [
            ("train/train_qwen_unsloth/train.yaml deploy_config", deploy_context.get("ENDPOINT_NAME")),
            (LOAD_TEST_YAML, required(load_test_config, "endpoint_name", LOAD_TEST_YAML)),
        ],
    )

    prefix = str(deploy_context.get("INFERENCE_TABLE_PREFIX") or "")
    inference_table = required(monitor_config, "inference_table", MONITOR_YAML)
    if prefix and inference_table and inference_table != f"{prefix}_payload":
        fail(
            f"{MONITOR_YAML}: inference_table must be {prefix + '_payload'!r}; "
            f"got {inference_table!r}"
        )


def check_monitor_config(monitor_config: dict) -> None:
    for key in ("unpacked_table", "checkpoint_volume"):
        required(monitor_config, key, MONITOR_YAML)

    from monitoring_utils import parse_quality_monitor_config

    try:
        settings = parse_quality_monitor_config(monitor_config)
    except Exception as exc:
        fail(f"{MONITOR_YAML}: parse_quality_monitor_config() failed: {exc}")
        return

    template = SETUP_UTILS_PATH.read_text(encoding="utf-8")
    for name, _ in settings["prompt_fields"]:
        if f"- {name}: " not in template:
            warn(
                f"{MONITOR_YAML}: prompt field {name!r} is absent from "
                "setup/utils.py render_fraud_prompt and will extract as null"
            )


def check_sft_conversion_contract() -> None:
    """Keep inline trainer conversion byte-identical to setup conversion."""
    try:
        setup_utils = load_module("setup_stage_utils", SETUP_UTILS_PATH)
        load_test_utils = load_module("load_test_stage_utils", LOAD_TEST_UTILS_PATH)
    except Exception as exc:
        fail(f"Could not load stage-local utils modules: {exc}")
        return

    records = [
        {
            "user_id_text": "1",
            "card_id_text": "2",
            "transaction_ts_text": "2026-01-01T00:00:00Z",
            "amount_usd": 10.0,
            "use_chip_text": "Chip",
            "merchant_city_text": "Boston",
            "merchant_state_text": "MA",
            "mcc_text": "5411",
            "errors_text": "",
            "is_fraud": 0,
            "has_error_signal": False,
        },
        {
            "user_id_text": "3",
            "card_id_text": "4",
            "transaction_ts_text": "2026-01-02T00:00:00Z",
            "amount_usd": 800.0,
            "use_chip_text": "Online",
            "merchant_city_text": "Seattle",
            "merchant_state_text": "WA",
            "mcc_text": "5732",
            "errors_text": "Bad PIN",
            "is_fraud": 1,
            "has_error_signal": True,
        },
    ]
    threshold = 500.0
    for record in records:
        if load_test_utils.render_fraud_prompt(record) != setup_utils.render_fraud_prompt(
            record
        ):
            fail("load_test/utils.py: prompt renderer differs from setup/utils.py")

    for project_name, relative_dir in PROJECTS.items():
        module_path = REPO_ROOT / relative_dir / "sft_conversion.py"
        try:
            module = load_module(f"{project_name}_sft_conversion", module_path)
        except Exception as exc:
            fail(f"{module_path}: could not load module: {exc}")
            continue
        for record in records:
            if module.render_fraud_prompt(record) != setup_utils.render_fraud_prompt(record):
                fail(f"{relative_dir}/sft_conversion.py: prompt renderer differs from setup")
            if module.render_fraud_response(
                record, threshold
            ) != setup_utils.render_fraud_response(record, threshold):
                fail(f"{relative_dir}/sft_conversion.py: response renderer differs from setup")


def main() -> int:
    global_config = load_yaml(GLOBAL_YAML)
    for key in ("catalog", "schema"):
        required(global_config, key, GLOBAL_YAML)

    setup_config = load_yaml(SETUP_YAML)
    load_test_config = load_yaml(LOAD_TEST_YAML)
    monitor_config = load_yaml(MONITOR_YAML)
    for location, config in (
        (SETUP_YAML, setup_config),
        (LOAD_TEST_YAML, load_test_config),
        (MONITOR_YAML, monitor_config),
    ):
        shadowed = sorted({"catalog", "schema"} & set(config))
        if shadowed:
            fail(f"{location}: catalog/schema remain owned by {GLOBAL_YAML}; remove {shadowed}")

    project_contexts = {}
    deploy_contexts = {}
    for name, relative_dir in PROJECTS.items():
        _, project_contexts[name], deploy_contexts[name] = load_project_context(
            name, relative_dir
        )

    endpoint_names = [
        context.get("ENDPOINT_NAME") for context in deploy_contexts.values() if context
    ]
    if len(endpoint_names) != len(set(endpoint_names)):
        fail("Training projects must use distinct deploy_config endpoint_name values")

    check_model_snapshots(setup_config, project_contexts)
    check_remaining_stage_contracts(
        setup_config,
        load_test_config,
        monitor_config,
        deploy_contexts.get("qwen_unsloth", {}),
    )
    check_sft_conversion_contract()
    check_monitor_config(monitor_config)

    if warnings:
        print(f"Configuration warnings ({len(warnings)}):\n")
        for index, message in enumerate(warnings, 1):
            print(f"{index}. {message}\n")
    if errors:
        print(f"Configuration validation FAILED ({len(errors)} error(s)):\n")
        for index, message in enumerate(errors, 1):
            print(f"{index}. {message}\n")
        return 1

    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
