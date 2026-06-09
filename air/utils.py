# Databricks notebook source
# DBTITLE 1,AIR fraud fine-tuning utilities
# MAGIC %md
# MAGIC # AIR fraud fine-tuning utilities
# MAGIC
# MAGIC Shared setup helpers for the fraud fine-tuning notebook. The training notebook loads this file with `%run ./utils` so configuration loading and Spark naming are defined in the notebook session.

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

# COMMAND ----------

from contextlib import contextmanager

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

import mlflow

@contextmanager
def start_mlflow_run(mlflow_module, run_name: str):
    try:
        with mlflow_module.start_run(run_name=run_name, log_system_metrics=True) as run:
            yield run
    except TypeError:
        with mlflow_module.start_run(run_name=run_name) as run:
            yield run


def render_chat_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

