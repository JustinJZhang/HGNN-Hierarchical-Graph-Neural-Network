import numpy as np


METRIC_KEYS = (
    "sens_score",
    "spec_score",
    "prec_score",
    "acc_score",
    "sens_side",
    "spec_side",
    "prec_side",
    "acc_side",
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
    """Divide arrays with shape [C] and return finite values with shape [C]."""
    result = np.zeros_like(numerator, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator != 0)


def _classification_metrics(ground_truth, predictions, num_classes):
    """Calculate sensitivity, specificity, precision, and accuracy arrays with shape [C]."""
    true_positive, true_negative, false_positive, false_negative = _compute_confusion(
        ground_truth, predictions, num_classes
    )
    sensitivity = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    accuracy = _safe_divide(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    return sensitivity, specificity, precision, accuracy


def get_classification_metrics(labels, predictions, num_classes=3):
    """Calculate eight per-class metric arrays from labels and logits with shape [N,2,C]."""
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
