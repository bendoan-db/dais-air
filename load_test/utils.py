"""Configuration, prompt, and Unity Catalog helpers for the load-test stage."""

from pathlib import Path

import yaml

MODULE_DIR = Path(__file__).resolve().parent
GLOBAL_CONFIG_FILENAME = "global.yaml"


def load_yaml_config(
    config_filename: str, base_dir: Path | None = None
) -> tuple[Path, dict]:
    config_path = (base_dir or MODULE_DIR) / config_filename
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected YAML mapping in {config_path}, got {type(config).__name__}"
        )
    return config_path, config


def load_global_config() -> tuple[Path, dict]:
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
        + ")"
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
    try:
        from pyspark.sql import SparkSession

        active_session = SparkSession.getActiveSession()
        if active_session is not None:
            return active_session
    except Exception:
        pass

    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    if hasattr(builder, "serverless"):
        return builder.serverless().getOrCreate()
    return builder.getOrCreate()


FRAUD_RECORD_COLUMNS = (
    "user_id_text",
    "card_id_text",
    "transaction_ts_text",
    "amount_usd",
    "use_chip_text",
    "merchant_city_text",
    "merchant_state_text",
    "mcc_text",
    "errors_text",
    "is_fraud",
    "has_error_signal",
)


def render_fraud_prompt(record) -> str:
    return (
        "You are a fraud decision model for a credit-card transaction stream. "
        "Classify the transaction as legitimate, suspicious, or likely_fraud. "
        "Return only compact JSON with keys risk, action, and reason.\n\n"
        "Transaction:\n"
        f"- user_id: {record['user_id_text']}\n"
        f"- card_id: {record['card_id_text']}\n"
        f"- timestamp: {record['transaction_ts_text']}\n"
        f"- amount_usd: {float(record['amount_usd']):.2f}\n"
        f"- use_chip: {record['use_chip_text']}\n"
        f"- merchant_city: {record['merchant_city_text']}\n"
        f"- merchant_state: {record['merchant_state_text']}\n"
        f"- merchant_category_code: {record['mcc_text']}\n"
        f"- errors: {record['errors_text']}"
    )


def ensure_uc_object(spark, ddl_statement: str) -> None:
    try:
        spark.sql(ddl_statement)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run `{ddl_statement}`. Verify Unity Catalog privileges "
            "for the configured catalog and schema."
        ) from exc
