import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import NUM_EXPRESSIONS
from train_eval import FOLD_DEFAULTS, SCALER
from metrics import METRIC_KEYS, get_classification_metrics
from util import load_obj, parse_experiment_args, preprocess_features, print_classification_metrics


MODEL_TYPES = ("xgb", "rf", "svm", "threshold")
MODEL_NAMES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "svm": "SVM",
    "threshold": "Threshold Baseline",
}


def prepare_ml_features(X_train_au, X_train_fa, X_test_au, X_test_fa, apply_scaler=SCALER):
    """Apply the preprocessing and flatten each sample for ML models.

    Args:
        X_train_au: Training AU array with shape [N_train, 8, 17].
        X_train_fa: Training FA array with shape [N_train, 8, 24].
        X_test_au: Test AU array with shape [N_test, 8, 17].
        X_test_fa: Test FA array with shape [N_test, 8, 24].
        apply_scaler: Boolean controlling training-only standardization.

    Returns:
        Tuple of training [N_train, 328] and test [N_test, 328] arrays.
    """
    X_train_au, X_train_fa, X_test_au, X_test_fa = preprocess_features(
        X_train_au, X_train_fa, X_test_au, X_test_fa, apply_scaler
    )

    X_train = np.concatenate((X_train_au, X_train_fa), axis=1)
    X_test = np.concatenate((X_test_au, X_test_fa), axis=1)
    return (
        X_train.reshape(X_train.shape[0] // NUM_EXPRESSIONS, -1),
        X_test.reshape(X_test.shape[0] // NUM_EXPRESSIONS, -1),
    )


def build_model(model_type):
    """Build one three-class classifier.

    Args:
        model_type: String model identifier: ``xgb``, ``rf``, or ``svm``.

    Returns:
        Classifier mapping feature arrays [N, 328] to probabilities [N, 3].
    """
    if model_type == "xgb":
        return XGBClassifier(random_state=42)
    if model_type == "rf":
        return RandomForestClassifier(
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "svm":
        return SVC(
            probability=True,
            random_state=42,
            cache_size=500,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def predict_class_probabilities(model, X):
    """Align a fitted classifier's probabilities to class index order.

    Args:
        model: Fitted three-class classifier.
        X: Floating-point array with shape [N, 328].

    Returns:
        Floating-point class-probability array with shape [N, 3].
    """
    probabilities = model.predict_proba(X)
    aligned_probabilities = np.zeros((X.shape[0], 3), dtype=probabilities.dtype)
    aligned_probabilities[:, model.classes_.astype(int)] = probabilities
    return aligned_probabilities


def train_and_predict_threshold(X_train, y_train, X_test):
    """Fit class-specific threshold rules and predict by equal-weight voting.

    Args:
        X_train: Preprocessed training features [N_train, 328].
        y_train: One-hot training labels [N_train, 2, 3].
        X_test: Preprocessed test features [N_test, 328].

    Returns:
        Float32 vote scores [N_test, 2, 3].
    """
    X_train = X_train.astype(np.float32).reshape(len(X_train), NUM_EXPRESSIONS, -1)
    X_test = X_test.astype(np.float32).reshape(len(X_test), NUM_EXPRESSIONS, -1)
    votes = np.zeros((len(X_test), 2, 3), dtype=np.float32)
    for feature_index in range(X_train.shape[2]):
        train_values = X_train[:, :, feature_index]
        test_values = X_test[:, :, feature_index]
        thresholds = np.unique(np.quantile(train_values, np.linspace(0.05, 0.95, 19)))
        train_votes, test_votes = [], []
        for threshold in thresholds:
            train_hits = (train_values <= threshold).sum(axis=1)
            test_hits = (test_values <= threshold).sum(axis=1)
            train_votes.extend((train_hits, NUM_EXPRESSIONS - train_hits))
            test_votes.extend((test_hits, NUM_EXPRESSIONS - test_hits))

        for task_index in range(2):
            for class_index in range(3):
                best_rule = np.argmax([
                    roc_auc_score(y_train[:, task_index, class_index], candidate)
                    for candidate in train_votes
                ])
                votes[:, task_index, class_index] += test_votes[best_rule]

    total_votes = votes.sum(axis=2, keepdims=True)
    return np.divide(
        votes, total_votes, out=np.full_like(votes, 1.0 / 3.0), where=total_votes != 0
    )


def train_and_predict_ml(X_train, y_train, X_test, model_type):
    """Train severity/asymmetry classifiers and predict both tasks.

    Args:
        X_train: Floating-point training features with shape [N_train, 328].
        y_train: One-hot training labels with shape [N_train, 2, 3].
        X_test: Floating-point test features with shape [N_test, 328].
        model_type: String identifier: ``xgb``, ``rf``, ``svm``, or ``threshold``.

    Returns:
        Class probabilities or threshold votes [N_test, 2, 3].
    """
    if model_type == "threshold":
        return train_and_predict_threshold(X_train, y_train, X_test)

    y_train_severity = np.argmax(y_train[:, 0, :], axis=1)
    y_train_asymmetry = np.argmax(y_train[:, 1, :], axis=1)

    weights_severity = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=y_train_severity
    )
    weights_asymmetry = compute_class_weight(
        "balanced", classes=np.array([0, 1, 2]), y=y_train_asymmetry
    )

    model_severity = build_model(model_type)
    model_asymmetry = build_model(model_type)
    model_severity.fit(X_train, y_train_severity, sample_weight=weights_severity[y_train_severity])
    model_asymmetry.fit(X_train, y_train_asymmetry, sample_weight=weights_asymmetry[y_train_asymmetry])

    probability_severity = predict_class_probabilities(model_severity, X_test)
    probability_asymmetry = predict_class_probabilities(model_asymmetry, X_test)
    predictions = np.stack((probability_severity, probability_asymmetry), axis=1)

    return predictions


def main(argv=None):
    args = parse_experiment_args(
        "Train and evaluate ML methods and threshold baseline.",
        {**{key: FOLD_DEFAULTS[key] for key in ("checkpoint_dir", "k_fold", "scaler")},
         "models": MODEL_TYPES},
        argv,
    )
    for fold in range(args.k_fold):
        print(f"Fold {fold + 1}")
        fold_data = load_obj(
            os.path.join(args.checkpoint_dir, f"data_{fold + 1}")
        )
        y_train = fold_data["train_data"]["y"]
        X_train_au = fold_data["train_data"]["X"]["AU"]
        X_train_fa = fold_data["train_data"]["X"]["FA"]
        y_test = fold_data["test_data"]["y"]
        X_test_au = fold_data["test_data"]["X"]["AU"]
        X_test_fa = fold_data["test_data"]["X"]["FA"]
        X_train, X_test = prepare_ml_features(
            X_train_au, X_train_fa, X_test_au, X_test_fa, args.scaler
        )

        for model_type in args.models:
            model_name = MODEL_NAMES[model_type]
            print(f"Training {model_name} on Fold {fold + 1}...")
            predictions = train_and_predict_ml(
                X_train, y_train, X_test, model_type
            )
            fold_metrics = get_classification_metrics(y_test, predictions)
            metrics_dict = {
                key: value for key, value in zip(METRIC_KEYS, fold_metrics)
            }
            print(f"\nFold {fold + 1} Evaluation Metrics ({model_name}):")
            print_classification_metrics(metrics_dict)


if __name__ == "__main__":
    main()
