"""Normalize raw fraud records or validate pre-converted SFT records."""

import json

RAW_RECORD_COLUMNS = (
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
SFT_COLUMNS = ("prompt", "assistant_response")


def _require_columns(records_pdf, columns: tuple[str, ...], mode: str) -> None:
    missing = [column for column in columns if column not in records_pdf.columns]
    if missing:
        raise ValueError(
            f"{mode} input is missing required columns: {missing}. "
            "Update train_data_path/eval_data_path or convert_sft in train.yaml."
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


def render_fraud_response(record, suspicious_amount_threshold: float) -> str:
    needs_review = bool(record["has_error_signal"]) or (
        float(record["amount_usd"]) >= suspicious_amount_threshold
    )
    if int(record["is_fraud"] or 0) == 1:
        risk, action = "likely_fraud", "decline_and_escalate"
        reason = "The historical label marks this transaction as fraud."
    elif needs_review:
        risk, action = "suspicious", "step_up_authentication"
        reason = (
            "The transaction is not labeled fraud, but amount or error signals "
            "warrant review."
        )
    else:
        risk, action = "legitimate", "approve"
        reason = "The historical label is non-fraud and no strong review signal is present."
    return json.dumps(
        {"risk": risk, "action": action, "reason": reason}, separators=(",", ":")
    )


def prepare_sft_records(records_pdf, convert_sft: bool, suspicious_amount_threshold: float):
    """Convert raw records once or validate an already-converted SFT frame."""
    if not convert_sft:
        _require_columns(records_pdf, SFT_COLUMNS, "Pre-converted SFT")
        return records_pdf

    _require_columns(records_pdf, RAW_RECORD_COLUMNS, "Raw")
    converted_pdf = records_pdf.copy()
    records = converted_pdf.to_dict("records")
    converted_pdf["prompt"] = [render_fraud_prompt(record) for record in records]
    converted_pdf["assistant_response"] = [
        render_fraud_response(record, suspicious_amount_threshold) for record in records
    ]
    return converted_pdf
