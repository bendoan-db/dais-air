"""Memory-safe step evaluation metrics for causal-language-model SFT."""

import re
from collections.abc import Callable

RISK_LABELS = ("legitimate", "suspicious", "likely_fraud")
RISK_PATTERN = re.compile(
    r"""["']risk["']\s*:\s*["']([^"']+)["']""", re.IGNORECASE
)


def preprocess_logits_for_metrics(logits, labels):
    """Select vocabulary logits and reduce them before Trainer gathers them."""
    candidates = []

    def collect_tensors(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                collect_tensors(item)
        elif hasattr(value, "ndim") and hasattr(value, "shape"):
            candidates.append(value)

    collect_tensors(logits)
    label_shape = tuple(getattr(labels, "shape", ()))
    vocabulary_logits = [
        candidate
        for candidate in candidates
        if candidate.ndim == 3
        and (len(label_shape) < 2 or tuple(candidate.shape[:2]) == label_shape[:2])
    ]
    if vocabulary_logits:
        # Router/auxiliary tensors may also be 3-D. Vocabulary logits have the
        # largest final dimension by a wide margin.
        vocabulary_logits = max(
            vocabulary_logits, key=lambda candidate: candidate.shape[-1]
        )
        return vocabulary_logits.argmax(dim=-1)

    token_ids = [
        candidate
        for candidate in candidates
        if candidate.ndim == 2
        and (not label_shape or tuple(candidate.shape) == label_shape)
    ]
    if token_ids:
        return token_ids[0]

    candidate_shapes = [tuple(candidate.shape) for candidate in candidates]
    raise ValueError(
        "Could not find vocabulary logits or token IDs matching labels "
        f"{label_shape}; model output shapes: {candidate_shapes}"
    )


def _as_token_id_matrix(values, name: str):
    """Flatten dense or ragged Trainer outputs into a padded token-id matrix."""
    import numpy as np

    def as_array(value):
        try:
            return np.asarray(value)
        except ValueError:
            if not isinstance(value, (list, tuple)):
                raise
            object_array = np.empty(len(value), dtype=object)
            object_array[:] = list(value)
            return object_array

    # Causal LM outputs can be wrapped in a tuple with auxiliary model outputs.
    if isinstance(values, tuple):
        values = values[0]

    array = as_array(values)
    if array.dtype != object:
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim >= 2:
            return array.reshape(-1, array.shape[-1])

    rows = []

    def collect_rows(value):
        nested = as_array(value)
        if nested.dtype != object:
            if nested.ndim == 0:
                raise ValueError(f"{name} contains a scalar instead of token IDs")
            if nested.ndim == 1:
                rows.append(nested.astype(np.int64, copy=False))
                return
            rows.extend(
                row.astype(np.int64, copy=False)
                for row in nested.reshape(-1, nested.shape[-1])
            )
            return

        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Could not normalize {name} item of type {type(value).__name__}"
            )
        for item in value:
            collect_rows(item)

    collect_rows(values)
    if not rows:
        raise ValueError(f"{name} contains no token-id rows")

    width = max(len(row) for row in rows)
    matrix = np.full((len(rows), width), -100, dtype=np.int64)
    for row_index, row in enumerate(rows):
        matrix[row_index, : len(row)] = row
    return matrix


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

        predictions = _as_token_id_matrix(
            eval_prediction.predictions, "predictions"
        )
        labels = _as_token_id_matrix(eval_prediction.label_ids, "labels")
        if predictions.shape[0] != labels.shape[0]:
            raise ValueError(
                "Predictions and labels contain different row counts: "
                f"{predictions.shape[0]} and {labels.shape[0]}"
            )

        sequence_length = max(predictions.shape[1], labels.shape[1])
        if predictions.shape[1] < sequence_length:
            predictions = np.pad(
                predictions,
                ((0, 0), (0, sequence_length - predictions.shape[1])),
                constant_values=-100,
            )
        if labels.shape[1] < sequence_length:
            labels = np.pad(
                labels,
                ((0, 0), (0, sequence_length - labels.shape[1])),
                constant_values=-100,
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
