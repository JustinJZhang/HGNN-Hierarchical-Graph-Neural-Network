import numpy as np


METRIC_KEYS = (
    "sens_severity",
    "spec_severity",
    "prec_severity",
    "acc_severity_ovr",
    "acc_severity",
    "sens_asymmetry",
    "spec_asymmetry",
    "prec_asymmetry",
    "acc_asymmetry_ovr",
    "acc_asymmetry",
)


def _compute_confusion(ground_truth, predictions, num_classes):
    """Convert class indices [N] into TP, TN, FP, and FN arrays with shape [C]."""
    true_positive, false_positive, false_negative, true_negative = [
        np.zeros(num_classes) for _ in range(4)
    ]
    for target, prediction in zip(ground_truth, predictions):
        for class_index in range(num_classes):
            if target == class_index and prediction == class_index:
                true_positive[class_index] += 1
            elif target != class_index and prediction != class_index:
                true_negative[class_index] += 1
            elif target == class_index and prediction != class_index:
                false_negative[class_index] += 1
            else:
                false_positive[class_index] += 1
    return true_positive, true_negative, false_positive, false_negative


def _safe_divide(numerator, denominator):
    result = np.zeros_like(numerator, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator != 0)


def _classification_metrics(ground_truth, predictions, num_classes):
    """Return per-class metrics [C] and multiclass accuracy.

    Args:
        ground_truth: Integer ground-truth class indices with shape [N].
        predictions: Integer predicted class indices with shape [N].
        num_classes: Integer number of classes C.
    Returns:
        Sensitivity, specificity, precision, and one-vs-rest accuracy arrays
        with shape [C], followed by multiclass task accuracy.
    """
    true_positive, true_negative, false_positive, false_negative = _compute_confusion(
        ground_truth, predictions, num_classes
    )
    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    one_vs_rest_accuracy = _safe_divide(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    task_accuracy = np.mean(ground_truth == predictions)
    return sensitivity, specificity, precision, one_vs_rest_accuracy, task_accuracy


def get_classification_metrics(labels, predictions, num_classes=3):
    severity_labels = np.argmax(labels[:, 0, :], axis=1)
    asymmetry_labels = np.argmax(labels[:, 1, :], axis=1)
    severity_predictions = np.argmax(predictions[:, 0, :], axis=1)
    asymmetry_predictions = np.argmax(predictions[:, 1, :], axis=1)
    severity_metrics = _classification_metrics(
        severity_labels, severity_predictions, num_classes
    )
    asymmetry_metrics = _classification_metrics(
        asymmetry_labels, asymmetry_predictions, num_classes
    )
    return severity_metrics + asymmetry_metrics
