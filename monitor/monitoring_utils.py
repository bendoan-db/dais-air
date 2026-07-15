"""Shared helpers for the monitoring stage.

This is a plain Python module (NOT a Databricks notebook), like
``monitor/utils.py``: the monitor notebooks insert this directory into
``sys.path`` and import it — never ``%run`` it, and never add the notebook
header (notebook-formatted files cannot be imported in the workspace).

Both monitor notebooks parse model traffic through these helpers:
``01_unpack_inference_table.py`` applies them to serving requests and
``02_create_quality_monitor.py`` applies them to the SFT training records
when building the monitor's baseline table — one implementation, so the two
tables are guaranteed to extract identical columns, the property baseline
drift comparisons depend on.

Pyspark imports stay inside the DataFrame helpers so the configuration parsers
remain importable without a Spark session.
"""

from __future__ import annotations

import re

# Extracted prompt fields become txn_<name>; extracted response JSON fields
# become response_<name>. The prefixes keep them clear of the built-in
# unpacked columns and of each other.
PROMPT_COLUMN_PREFIX = "txn_"
RESPONSE_COLUMN_PREFIX = "response_"

# Spark cast targets allowed for prompt_fields entries. Categorical codes
# (e.g. merchant_category_code) should stay ``string``: casting a code to a
# number would give it meaningless numeric drift statistics.
PROMPT_FIELD_TYPES = {"string", "int", "double", "timestamp"}

# response_json_fields that would collide with the built-in response_*
# columns of the unpacked table (response_text, response_model,
# response_chars).
RESERVED_RESPONSE_FIELDS = {"text", "model", "chars"}

# Time windows the data quality monitor accepts.
_GRANULARITY_PATTERN = re.compile(
    r"5 minutes|30 minutes|1 hour|1 day|[1-4] weeks?|1 month|1 year"
)

_FIELD_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_response_json_fields(config: dict) -> list[str]:
    """Validate ``response_json_fields`` from monitor.yaml; return the names."""
    fields = config.get("response_json_fields") or []
    if not isinstance(fields, list):
        raise ValueError("response_json_fields must be a list (or empty) in monitor.yaml")

    parsed: list[str] = []
    for field in fields:
        name = str(field).strip()
        if not _FIELD_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"response_json_fields entry {name!r} must be a valid identifier "
                "(it becomes a column name)"
            )
        if name in RESERVED_RESPONSE_FIELDS:
            raise ValueError(
                f"response_json_fields entry {name!r} collides with the built-in "
                f"{RESPONSE_COLUMN_PREFIX}{name} column of the unpacked table"
            )
        if name in parsed:
            raise ValueError(f"duplicate response_json_fields entry: {name!r}")
        parsed.append(name)
    return parsed


def parse_prompt_fields(config: dict) -> list[tuple[str, str]]:
    """Validate ``prompt_fields`` from monitor.yaml; return (name, cast type) pairs."""
    entries = config.get("prompt_fields") or []
    if not isinstance(entries, list):
        raise ValueError("prompt_fields must be a list (or empty) in monitor.yaml")

    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "type"}:
            raise ValueError(
                f"prompt_fields entries must be mappings with exactly the keys "
                f"name and type, got {entry!r}"
            )
        name = str(entry["name"]).strip()
        cast_type = str(entry["type"]).strip()
        if not _FIELD_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"prompt_fields entry {name!r} must be a valid identifier "
                "(it becomes a column name)"
            )
        if cast_type not in PROMPT_FIELD_TYPES:
            raise ValueError(
                f"prompt_fields entry {name!r} has type {cast_type!r}; "
                f"expected one of {sorted(PROMPT_FIELD_TYPES)}"
            )
        if name in seen:
            raise ValueError(f"duplicate prompt_fields entry: {name!r}")
        seen.add(name)
        parsed.append((name, cast_type))
    return parsed


def parse_quality_monitor_config(config: dict) -> dict:
    """Validate and normalize the monitor.yaml keys module 02 consumes.

    Returns a dict of normalized values; slicing fields are resolved to the
    extracted column names they refer to. Module 02 calls this before creating
    or updating the monitor.
    """
    response_json_fields = parse_response_json_fields(config)
    prompt_fields = parse_prompt_fields(config)
    prompt_names = [name for name, _ in prompt_fields]

    prediction_field = str(config.get("prediction_field") or "").strip()
    if not prediction_field:
        raise ValueError(
            "prediction_field is required (the response field the monitor "
            "treats as the predicted label)"
        )
    if prediction_field not in response_json_fields:
        raise ValueError(
            f"prediction_field {prediction_field!r} must be one of "
            f"response_json_fields {response_json_fields} — the prediction "
            "column has to be extracted before it can be monitored"
        )

    raw_values = config.get("expected_prediction_values") or []
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("expected_prediction_values must be a non-empty list")
    expected_prediction_values: list[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if not value:
            raise ValueError("expected_prediction_values entries cannot be empty")
        if value in expected_prediction_values:
            raise ValueError(f"duplicate expected_prediction_values entry: {value!r}")
        expected_prediction_values.append(value)

    raw_granularities = config.get("granularities") or []
    if not isinstance(raw_granularities, list) or not raw_granularities:
        raise ValueError("granularities must be a non-empty list")
    granularities: list[str] = []
    for raw_granularity in raw_granularities:
        granularity = str(raw_granularity).strip()
        if not _GRANULARITY_PATTERN.fullmatch(granularity):
            raise ValueError(
                f"granularities entry {granularity!r} is not a supported monitor "
                "window (5 minutes, 30 minutes, 1 hour, 1 day, 1-4 weeks, "
                "1 month, 1 year)"
            )
        granularities.append(granularity)

    raw_slicing = config.get("slicing_fields") or []
    if not isinstance(raw_slicing, list):
        raise ValueError("slicing_fields must be a list (or empty)")
    slicing_exprs: list[str] = []
    for raw_field in raw_slicing:
        field = str(raw_field).strip()
        if field in response_json_fields:
            slicing_exprs.append(f"{RESPONSE_COLUMN_PREFIX}{field}")
        elif field in prompt_names:
            slicing_exprs.append(f"{PROMPT_COLUMN_PREFIX}{field}")
        else:
            raise ValueError(
                f"slicing_fields entry {field!r} is neither a "
                "response_json_fields nor a prompt_fields entry — only "
                "extracted fields can slice the monitor"
            )

    baseline_table = str(config.get("baseline_table") or "").strip()
    if not baseline_table:
        raise ValueError("baseline_table is required (module 02 writes it)")

    try:
        baseline_sample_fraction = float(config.get("baseline_sample_fraction", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_sample_fraction must be a number") from exc
    if not 0 < baseline_sample_fraction <= 1:
        raise ValueError("baseline_sample_fraction must be in (0, 1]")

    return {
        "response_json_fields": response_json_fields,
        "prompt_fields": prompt_fields,
        "prediction_field": prediction_field,
        "prediction_col": f"{RESPONSE_COLUMN_PREFIX}{prediction_field}",
        "expected_prediction_values": expected_prediction_values,
        "granularities": granularities,
        "slicing_exprs": slicing_exprs,
        "baseline_table": baseline_table,
        "baseline_sample_fraction": baseline_sample_fraction,
        "monitor_output_schema": str(config.get("monitor_output_schema") or "").strip(),
        "label_field": str(config.get("label_field") or "").strip(),
    }


def prompt_field_pattern(name: str) -> str:
    """Regex extracting one ``- <name>: <value>`` line from the prompt block."""
    return rf"(?m)^-\s*{re.escape(name)}:\s*(.*?)\s*$"


def with_prompt_fields(df, prompt_fields: list[tuple[str, str]], prompt_column: str = "prompt"):
    """Add a typed ``txn_<name>`` column per prompt field.

    ``regexp_extract`` returns ``""`` when the line is absent (and when its
    value is empty); that is normalized to null, and a failed cast also
    yields null — a renamed field, a reordered template, or a malformed
    value produces nulls, never a failure. Rising txn_* null rates are
    themselves a monitored signal (the prompt template changed).
    """
    from pyspark.sql import functions as F

    for name, cast_type in prompt_fields:
        raw = F.regexp_extract(F.col(prompt_column), prompt_field_pattern(name), 1)
        df = df.withColumn(
            f"{PROMPT_COLUMN_PREFIX}{name}",
            F.when(raw == "", None).otherwise(raw).cast(cast_type),
        )
    return df


def with_response_fields(df, response_json_fields: list[str], source_column: str):
    """Add a string ``response_<field>`` column per configured JSON field.

    ``from_json`` yields null when the completion is not valid JSON or the
    field is absent, so contract breakage surfaces as null rates rather than
    failures.
    """
    from pyspark.sql import functions as F, types as T

    if not response_json_fields:
        return df

    fields_schema = T.StructType(
        [T.StructField(field, T.StringType()) for field in response_json_fields]
    )
    df = df.withColumn("_response_fields", F.from_json(F.col(source_column), fields_schema))
    for field in response_json_fields:
        df = df.withColumn(
            f"{RESPONSE_COLUMN_PREFIX}{field}", F.col(f"_response_fields.{field}")
        )
    return df.drop("_response_fields")
