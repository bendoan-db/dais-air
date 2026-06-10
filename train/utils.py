# Databricks notebook source
# DBTITLE 1,AIR fraud fine-tuning utilities
# MAGIC %md
# MAGIC # AIR fraud fine-tuning utilities
# MAGIC
# MAGIC Shared setup helpers for the demo. The training notebook loads this file with `%run ./utils` and the load-test notebook with `%run ../train/utils`, so configuration loading and Spark naming are defined in the notebook session. `train.py` (and the setup notebook in local-script mode) imports it as a regular Python module instead.

# COMMAND ----------

from pathlib import Path

import yaml

# COMMAND ----------


def get_notebook_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        notebook_path = notebook_context.notebookPath().get()
        return Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]


def load_yaml_config(config_filename: str) -> tuple[Path, dict]:
    config_path = get_notebook_dir() / config_filename

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}, got {type(config).__name__}")

    return config_path, config


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

# COMMAND ----------


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def full_name(*parts: str) -> str:
    return ".".join(quote_identifier(part) for part in parts)


def get_spark_session():
    if "spark" in globals():
        return globals()["spark"]

    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.serverless().getOrCreate()
