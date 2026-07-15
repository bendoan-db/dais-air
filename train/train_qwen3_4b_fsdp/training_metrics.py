"""Memory-safe step evaluation metrics for causal-language-model SFT."""

import re
from collections.abc import Callable

RISK_LABELS = ("legitimate", "suspicious", "likely_fraud")
RISK_PATTERN = re.compile(
    r"""["']risk["']\s*:\s*["']([^"']+)["']""", re.IGNORECASE
)


def preprocess_logits_for_metrics(logits, _labels):
    """Reduce vocabulary logits to token IDs before Trainer gathers them."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _extract_risk(text: str) -> str | None:
    matches = RISK_PATTERN.findall(text)
    if not matches:
        return None
    risk = matches[-1].strip().lower()
    return risk if risk in RISK_LABELS else None


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(
    expected: list[str], predicted: list[str | None]
) -> dict[str, float]:
    parsed_pairs = [
        (truth, prediction)
        for truth, prediction in zip(expected, predicted)
        if prediction is not None
    ]
    metrics = {
        "classification_sample_count": float(len(expected)),
        "classification_parse_rate": _safe_divide(len(parsed_pairs), len(expected)),
        # Unparseable outputs count as incorrect, matching serving evaluation.
        "classification_accuracy": _safe_divide(
            sum(truth == prediction for truth, prediction in zip(expected, predicted)),
            len(expected),
        ),
    }

    precisions = []
    recalls = []
    f1_scores = []
    for label in RISK_LABELS:
        true_positives = sum(
            truth == label and prediction == label
            for truth, prediction in zip(expected, predicted)
        )
        false_positives = sum(
            truth != label and prediction == label
            for truth, prediction in zip(expected, predicted)
        )
        false_negatives = sum(
            truth == label and prediction != label
            for truth, prediction in zip(expected, predicted)
        )
        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        metric_label = label.replace("likely_fraud", "fraud")
        metrics[f"classification_{metric_label}_precision"] = precision
        metrics[f"classification_{metric_label}_recall"] = recall
        metrics[f"classification_{metric_label}_f1"] = f1

    metrics.update(
        {
            "classification_macro_precision": sum(precisions) / len(precisions),
            "classification_macro_recall": sum(recalls) / len(recalls),
            "classification_macro_f1": sum(f1_scores) / len(f1_scores),
        }
    )
    return metrics


def build_compute_metrics(tokenizer) -> Callable:
    """Build Trainer metrics over shifted, teacher-forced token predictions.

    The risk metrics measure whether each next-token prediction reconstructs
    the target response's JSON ``risk`` field. They are not autoregressive
    generation metrics; the separate post-training generation evaluation in
    the Unsloth projects remains the serving-like quality check.
    """

    def compute_metrics(eval_prediction) -> dict[str, float]:
        import numpy as np

        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        labels = np.asarray(eval_prediction.label_ids)
        if predictions.ndim != 2 or labels.ndim != 2:
            raise ValueError(
                "Expected token-id predictions and labels with shape "
                f"[batch, sequence], got {predictions.shape} and {labels.shape}"
            )

        shifted_predictions = predictions[:, :-1]
        shifted_labels = labels[:, 1:]
        valid_mask = shifted_labels != -100
        valid_token_count = int(valid_mask.sum())
        metrics = {
            "token_accuracy": _safe_divide(
                int(((shifted_predictions == shifted_labels) & valid_mask).sum()),
                valid_token_count,
            )
        }

        expected_risks = []
        predicted_risks = []
        for predicted_row, label_row, row_mask in zip(
            shifted_predictions, shifted_labels, valid_mask
        ):
            expected_text = tokenizer.decode(
                label_row[row_mask].tolist(), skip_special_tokens=True
            )
            expected_risk = _extract_risk(expected_text)
            if expected_risk is None:
                continue
            predicted_text = tokenizer.decode(
                predicted_row[row_mask].tolist(), skip_special_tokens=True
            )
            expected_risks.append(expected_risk)
            predicted_risks.append(_extract_risk(predicted_text))

        # Always return the complete schema. Missing targets are a data-quality
        # signal, not a reason for accuracy/F1 metrics to disappear from MLflow.
        metrics.update(_classification_metrics(expected_risks, predicted_risks))
        metrics["classification_target_coverage"] = _safe_divide(
            len(expected_risks), labels.shape[0]
        )
        return metrics

    return compute_metrics


def build_mlflow_metrics_callback(evaluation_enabled: bool):
    """Log and verify custom evaluation metrics on Trainer evaluation events."""
    from transformers import TrainerCallback

    class MlflowClassificationMetricsCallback(TrainerCallback):
        def __init__(self):
            self.evaluation_count = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not evaluation_enabled:
                return

            self.evaluation_count += 1
            metrics = metrics or {}
            coverage_key = "eval_classification_target_coverage"
            if coverage_key not in metrics:
                raise RuntimeError(
                    "Evaluation completed without custom classification metrics. "
                    "Ensure compute_metrics is configured and "
                    "prediction_loss_only is false."
                )

            if not state.is_world_process_zero:
                return

            import mlflow

            custom_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if key == "eval_token_accuracy"
                or key.startswith("eval_classification_")
            }
            if mlflow.active_run() is None:
                raise RuntimeError(
                    "Custom evaluation metrics were computed without an active "
                    "MLflow run."
                )
            # Log explicitly instead of relying only on the Transformers
            # integration so custom metrics survive integration/version changes.
            mlflow.log_metrics(custom_metrics, step=int(state.global_step))

            if custom_metrics[coverage_key] == 0.0:
                print(
                    "WARNING: Evaluation targets contain no recognized risk "
                    "labels; classification metrics were logged as zero."
                )

        def on_train_end(self, args, state, control, **kwargs):
            if evaluation_enabled and self.evaluation_count == 0:
                raise RuntimeError(
                    "Training completed without evaluation, so no classification "
                    "metrics were logged. Check eval_steps and eval_sample_size."
                )

    return MlflowClassificationMetricsCallback()
