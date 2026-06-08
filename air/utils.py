# Databricks notebook source
# DBTITLE 1,AIR fraud fine-tuning utilities
# MAGIC %md
# MAGIC # AIR fraud fine-tuning utilities
# MAGIC
# MAGIC Shared setup and data helpers for the fraud fine-tuning notebook. The training notebook loads this file with `%run ./utils` so configuration loading, prompt construction, dataset writing, and Spark naming are defined in the notebook session.

# COMMAND ----------

from pathlib import Path
import json

import pandas as pd
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


def transaction_prompt(row: pd.Series) -> str:
    amount = float(row["amount_usd"])
    return (
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        f"- user_id: {row['user_id_text']}\n"
        f"- card_id: {row['card_id_text']}\n"
        f"- timestamp: {row['transaction_ts_text']}\n"
        f"- amount_usd: {amount:.2f}\n"
        f"- use_chip: {row['use_chip_text']}\n"
        f"- merchant_city: {row['merchant_city_text']}\n"
        f"- merchant_state: {row['merchant_state_text']}\n"
        f"- merchant_category_code: {row['mcc_text']}\n"
        f"- errors: {row['errors_text']}"
    )


def transaction_answer(row: pd.Series, suspicious_amount_threshold: float) -> str:
    is_fraud = int(row["is_fraud"])
    amount = float(row["amount_usd"])
    has_error_signal = bool(row["has_error_signal"])

    if is_fraud == 1:
        payload = {
            "risk": "likely_fraud",
            "action": "decline_and_escalate",
            "reason": "The historical label marks this transaction as fraud.",
        }
    elif has_error_signal or amount >= suspicious_amount_threshold:
        payload = {
            "risk": "suspicious",
            "action": "step_up_authentication",
            "reason": "The transaction is not labeled fraud, but amount or error signals warrant review.",
        }
    else:
        payload = {
            "risk": "legitimate",
            "action": "approve",
            "reason": "The historical label is non-fraud and no strong review signal is present.",
        }

    return json.dumps(payload, separators=(",", ": "))


def make_chat_record(row: pd.Series, suspicious_amount_threshold: float) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": transaction_prompt(row)},
            {
                "role": "assistant",
                "content": transaction_answer(row, suspicious_amount_threshold),
            },
        ],
        "label": row["fraud_label"],
        "transaction": {
            "user_id": row["user_id_text"],
            "card_id": row["card_id_text"],
            "amount": float(row["amount_usd"]),
            "merchant_city": row["merchant_city_text"],
            "merchant_state": row["merchant_state_text"],
            "mcc": row["mcc_text"],
            "is_fraud": int(row["is_fraud"]),
        },
    }


def write_jsonl(records: list[dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as dataset_file:
        for record in records:
            dataset_file.write(json.dumps(record, ensure_ascii=True) + "\n")
