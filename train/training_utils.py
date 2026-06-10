# Databricks notebook source
# DBTITLE 1,AIR fraud fine-tuning utilities
# MAGIC %md
# MAGIC # AIR fraud fine-tuning utilities
# MAGIC
# MAGIC Shared setup helpers for the demo. The training notebook loads this file with `%run ./training_utils` and the load-test notebook with `%run ../train/training_utils`, so configuration loading and Spark naming are defined in the notebook session. `train.py` (and the setup notebook in local-script mode) imports it as a regular Python module instead.
# MAGIC
# MAGIC The module is deliberately not named `utils`: GPU base environments (via `nvidia_cutlass_dsl`) register their own top-level `utils` module once the torch/CUDA stack loads, which shadows any local `utils.py` on import.

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


def load_training_config() -> dict:
    """Load the ``parameters.training_config`` section and derive shared names.

    Under an AI Runtime CLI workload the parameters arrive via the YAML file
    at ``$HYPERPARAMETERS_PATH`` (which reflects ``air run --override``
    values); otherwise they are read from ``train.yaml`` next to this file.

    Returns a flat dict of typed config values, derived UC names/paths, and
    quoted SQL identifiers, intended to be bound into the caller's namespace
    with ``globals().update(load_training_config())``. Used by the training
    runner notebook and by train.py; deliberately a function (not top-level
    code) so that other consumers of this file are unaffected.
    """
    import os

    hyperparameters_path = os.environ.get("HYPERPARAMETERS_PATH")
    if hyperparameters_path:
        config_path = Path(hyperparameters_path)
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        # The AIR CLI docs say HYPERPARAMETERS_PATH holds just the
        # `parameters` dict, but v0.1.0b1 points it at the full workload
        # YAML — accept either shape.
        parameters = loaded.get("parameters", loaded)
    else:
        config_path, workload_config = load_yaml_config("train.yaml")
        parameters = config_value(workload_config, "parameters")
    config = config_value(parameters, "training_config")

    uc_catalog = config_str(config, "catalog")
    uc_schema = config_str(config, "schema")
    source_table_name = config_str(config, "source_table")
    sft_table_name = config_str(config, "sft_table")
    sft_volume = config_str(config, "sft_volume")
    uc_volume = config_str(config, "checkpoint_volume")
    uc_model_name = config_str(config, "uc_model_name")
    max_steps = config_int(config, "max_steps")
    output_root = f"/Volumes/{uc_catalog}/{uc_schema}/{uc_volume}/{uc_model_name}"

    return {
        "CONFIG_PATH": config_path,
        "UC_CATALOG": uc_catalog,
        "UC_SCHEMA": uc_schema,
        "SOURCE_TABLE_NAME": source_table_name,
        "SFT_TABLE_NAME": sft_table_name,
        "UC_VOLUME": uc_volume,
        "UC_MODEL_NAME": uc_model_name,
        "ENDPOINT_NAME": config_str(config, "endpoint_name"),
        "MODEL_NAME": config_str(config, "model_name"),
        "MAX_SEQ_LENGTH": config_int(config, "max_seq_length"),
        "MAX_STEPS": max_steps,
        "PER_DEVICE_TRAIN_BATCH_SIZE": config_int(config, "per_device_train_batch_size"),
        "GRADIENT_ACCUMULATION_STEPS": config_int(config, "gradient_accumulation_steps"),
        "LEARNING_RATE": config_float(config, "learning_rate"),
        "TRAINING_SAMPLE_FRACTION": config_float(config, "training_sample_fraction"),
        "REGISTER_MODEL": config_bool(config, "register_model"),
        "DEPLOY_ENDPOINT": config_bool(config, "deploy_endpoint"),
        "SERVING_WORKLOAD_TYPE": config_str(config, "serving_workload_type"),
        "SERVING_WORKLOAD_SIZE": config_str(config, "serving_workload_size"),
        "SERVING_SCALE_TO_ZERO": config_bool(config, "serving_scale_to_zero"),
        "SERVED_MODEL_NAME": config_str(config, "served_model_name"),
        "VLLM_DTYPE": config_str(config, "vllm_dtype"),
        "VLLM_MAX_MODEL_LEN": config_int(config, "vllm_max_model_len"),
        "VLLM_GPU_MEMORY_UTILIZATION": config_float(config, "vllm_gpu_memory_utilization"),
        "SEED": config_int(config, "seed"),
        "SFT_VOLUME": sft_volume,
        # Parquet export of the SFT table, written by setup per the AIR
        # data-loading guidance; training reads these files instead of
        # querying Delta through Spark on the GPU workers.
        "SFT_FILES_DIR": f"/Volumes/{uc_catalog}/{uc_schema}/{sft_volume}/{sft_table_name}",
        "SOURCE_TABLE": f"{uc_catalog}.{uc_schema}.{source_table_name}",
        "SFT_TABLE": f"{uc_catalog}.{uc_schema}.{sft_table_name}",
        "FULL_MODEL_NAME": f"{uc_catalog}.{uc_schema}.{uc_model_name}",
        "OUTPUT_ROOT": output_root,
        "TRAINING_OUTPUT_DIR": f"{output_root}/training_demo",
        "TRAINING_RUN_NAME": f"air-demo-{uc_model_name}-training-steps{max_steps}",
        "schema_q": full_name(uc_catalog, uc_schema),
        "volume_q": full_name(uc_catalog, uc_schema, uc_volume),
        "source_table_q": full_name(uc_catalog, uc_schema, source_table_name),
        "sft_table_q": full_name(uc_catalog, uc_schema, sft_table_name),
    }


def init_training_workspace(training_context: dict):
    """Initialize Spark and ensure the training schema/checkpoint volume exist.

    Takes the dict returned by :func:`load_training_config` and returns the
    Spark session.
    """
    spark = get_spark_session()

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {training_context['schema_q']}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {training_context['volume_q']}")

    Path(training_context["TRAINING_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    return spark
