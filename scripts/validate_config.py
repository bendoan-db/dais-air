#!/usr/bin/env python3
"""Validate the pipeline's configuration contracts.

Run from anywhere (no Databricks connection needed):

    python scripts/validate_config.py

The repo-root global.yaml holds the Unity Catalog identity every module
shares (catalog, schema); all other parameters are stage-owned in each
module's YAML, with the cross-stage agreements validated here before any
workspace time is spent:

1. global.yaml parses, defines catalog and schema, and no stage YAML
   re-introduces either key (shadowed copies would be silently ignored by
   the notebooks and drift).
2. train/train.yaml's training_config and deploy_config sections load
   through the real loaders (types, required keys, lists, metric goal,
   serving_requirements_file resolution) — which also proves global.yaml
   resolves from the train module.
3. Cross-stage values agree: setup.yaml's source_table/sft_table/sft_volume
   with train.yaml's training_config; sft_table also with the load test and
   the monitor (its baseline source); endpoint_name between train.yaml's
   deploy_config and the load test; and monitor.yaml's inference_table
   equals deploy_config's inference_table_prefix + '_payload' (the monitor
   must unpack the table the endpoint actually writes).
4. When model_volume_path is set (training_config), setup.yaml's models
   list contains a matching entry (volume_path == model_volume_path and
   huggingface_path == model_name), so the weights setup downloads are the
   weights training loads.
5. experiment_name (train.yaml top level — the key the AIR CLI reads) stays
   alphanumeric/-/_ (it becomes a Jobs API task key).
6. Every `-r <file>` reference in train.yaml's environment resolves.
7. The monitor stage's own keys are present and well-typed: the field
   extraction and quality-monitor settings load through the real parser
   (monitor/monitoring_utils.parse_quality_monitor_config — field name and
   type validity, prediction_field membership in response_json_fields,
   slicing_fields resolving to extracted fields, granularity spellings,
   baseline settings). A prompt_fields entry missing from the prompt
   template built by setup/02_stage_training_data.py warns (the txn_ column
   would be 100% null) without failing — the template is example-specific.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(REPO_ROOT / "monitor"))

GLOBAL_YAML = "global.yaml"
SETUP_YAML = "setup/setup.yaml"
LOAD_TEST_YAML = "load_test/serving_load_test.yaml"
MONITOR_YAML = "monitor/monitor.yaml"
TRAIN_YAML = "train/train.yaml"

# Keys owned by global.yaml. Stage YAMLs must not redefine them (the
# notebooks read them only from global.yaml, so copies would be ignored).
GLOBAL_KEYS = {
    "catalog",
    "schema",
}

REQUIRED_GLOBAL_KEYS = (
    "catalog",
    "schema",
)

errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def warn(message: str) -> None:
    warnings.append(message)


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


def check_stage_agreements(
    setup_config: dict,
    load_test_config: dict,
    monitor_config: dict,
    training_context: dict,
    deploy_context: dict,
) -> None:
    check_agreement(
        "source_table",
        [
            (SETUP_YAML, stage_value(setup_config, "source_table", SETUP_YAML)),
            (TRAIN_YAML, training_context.get("SOURCE_TABLE_NAME")),
        ],
    )
    check_agreement(
        "sft_table",
        [
            (SETUP_YAML, stage_value(setup_config, "sft_table", SETUP_YAML)),
            (LOAD_TEST_YAML, stage_value(load_test_config, "sft_table", LOAD_TEST_YAML)),
            (MONITOR_YAML, stage_value(monitor_config, "sft_table", MONITOR_YAML)),
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
            (f"{TRAIN_YAML} (deploy_config)", deploy_context.get("ENDPOINT_NAME")),
            (LOAD_TEST_YAML, stage_value(load_test_config, "endpoint_name", LOAD_TEST_YAML)),
        ],
    )

    # Deployment always enables inference logging to <prefix>_payload; the
    # monitor must unpack that exact table.
    prefix = str(deploy_context.get("INFERENCE_TABLE_PREFIX") or "").strip()
    monitor_table = str(monitor_config.get("inference_table") or "").strip()
    if not monitor_table:
        fail(f"{MONITOR_YAML}: missing required key `inference_table`.")
    elif prefix and monitor_table != f"{prefix}_payload":
        fail(
            f"{MONITOR_YAML}: inference_table is {monitor_table!r} but the "
            f"endpoint logs to {prefix + '_payload'!r} ({TRAIN_YAML} "
            "deploy_config's inference_table_prefix + '_payload') — the monitor "
            "would unpack a table the endpoint never writes."
        )


def check_no_global_shadowing(path: str, config: dict) -> None:
    shadowed = sorted(GLOBAL_KEYS & set(config))
    if shadowed:
        fail(
            f"{path}: defines {shadowed}, but these parameters live only in "
            f"{GLOBAL_YAML} — the notebooks read them from there, so these "
            "copies would be silently ignored. Remove them."
        )


def check_experiment_name() -> None:
    """train.yaml's top-level experiment_name is the key the AIR CLI reads
    and the loaders resolve for notebook runs."""
    train_yaml = load_yaml(REPO_ROOT / "train" / "train.yaml")
    experiment_name = str(train_yaml.get("experiment_name") or "").strip()
    if not experiment_name:
        fail(f"{TRAIN_YAML}: missing top-level experiment_name (required by the AIR CLI).")
    elif not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_name):
        fail(
            f"{TRAIN_YAML}: experiment_name {experiment_name!r} must be "
            "alphanumeric/-/_ (it becomes a Jobs API task key under AIR)."
        )


def check_models_contract(setup_config: dict, training_context: dict) -> None:
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

    model_volume_path = str(training_context.get("MODEL_VOLUME_PATH") or "").rstrip("/")
    model_name = str(training_context.get("MODEL_NAME") or "").strip()
    if model_volume_path:
        if model_volume_path not in volume_paths:
            fail(
                f"{TRAIN_YAML}: model_volume_path ({model_volume_path}) has no "
                f"matching volume_path in {SETUP_YAML}'s models list — training "
                "would load weights setup never downloads."
            )
        elif volume_paths[model_volume_path] != model_name:
            fail(
                f"{SETUP_YAML}: the models entry for {model_volume_path} downloads "
                f"{volume_paths[model_volume_path]!r}, but {TRAIN_YAML}'s model_name "
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

    # Exercise the real parser (the same one both monitor notebooks use) so
    # field/type errors surface here instead of at workspace runtime.
    from monitoring_utils import parse_quality_monitor_config

    try:
        monitor_settings = parse_quality_monitor_config(monitor_config)
    except Exception as exc:
        fail(f"{MONITOR_YAML}: parse_quality_monitor_config() failed: {exc}")
        return

    check_prompt_template_coverage(monitor_settings)


def check_prompt_template_coverage(monitor_settings: dict) -> None:
    """Warn when a prompt_fields entry is absent from the prompt template.

    setup/02_stage_training_data.py builds the prompt with literal
    "- <field>: " line prefixes; a prompt_fields name that never appears
    there extracts a 100% null txn_ column. Warning, not error — the
    template is example-specific and a customer's own prompts may differ.
    """
    staging_source_path = REPO_ROOT / "setup" / "02_stage_training_data.py"
    if not staging_source_path.exists():
        return
    staging_source = staging_source_path.read_text(encoding="utf-8")
    for name, _ in monitor_settings["prompt_fields"]:
        if f"- {name}: " not in staging_source:
            warn(
                f"{MONITOR_YAML}: prompt_fields entry `{name}` has no "
                f"'- {name}: ' line in setup/02_stage_training_data.py's "
                "prompt template — its txn_ column will be 100% null on "
                "prompts built by this repo's setup stage."
            )


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

    if training_context and deploy_context:
        check_stage_agreements(
            setup_config, load_test_config, monitor_config, training_context, deploy_context
        )
    if training_context:
        check_models_contract(setup_config, training_context)
    check_experiment_name()

    check_environment_requirements()
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
