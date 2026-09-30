import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def parse_experiment_args(description, defaults, argv=None):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=False,
        allow_abbrev=False,
    )
    for name, default in defaults.items():
        flag = "--" + name.replace("_", "-")
        options = {"default": default}
        if isinstance(default, bool):
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, dest=name, action="store_true", **options)
            group.add_argument(
                "--no-" + name.replace("_", "-"), dest=name, action="store_false",
                default=argparse.SUPPRESS,
            )
        else:
            options["type"] = str if isinstance(default, (tuple, list)) else type(default)
            if isinstance(default, (tuple, list)):
                options.update(nargs="+", choices=default)
            parser.add_argument(flag, **options)

    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if name.endswith("_path") or name == "checkpoint_dir":
            setattr(args, name, os.path.expanduser(value))

    print("Run configuration:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    return args


def save_obj(obj, name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)


def preprocess_features(
    X_train_au, X_train_fa, X_test_au, X_test_fa, apply_scaler, return_scalers=False
):
    """Reshape and standardize AU and FA features.

    Args:
        X_train_au: Training AU array with shape [N_train, 8, 17].
        X_train_fa: Training FA array with shape [N_train, 8, 24].
        X_test_au: Test AU array with shape [N_test, 8, 17].
        X_test_fa: Test FA array with shape [N_test, 8, 24].
        apply_scaler: Boolean indicating whether to fit training-only z-score scalers.
        return_scalers: Boolean indicating whether to also return the fitted scalers.

    Returns:
        Tuple containing training AU [N_train*8, 17], training FA
        [N_train*8, 24], test AU [N_test*8, 17], and test FA [N_test*8, 24].
    """
    au_scaler = None
    fa_scaler = None
    if apply_scaler:
        au_scaler = StandardScaler()
        fa_scaler = StandardScaler()
        X_train_au = au_scaler.fit_transform(
            X_train_au.reshape(-1, X_train_au.shape[2])
        )
        X_train_fa = fa_scaler.fit_transform(
            X_train_fa.reshape(-1, X_train_fa.shape[2])
        )
        X_test_au = au_scaler.transform(
            X_test_au.reshape(-1, X_test_au.shape[2])
        )
        X_test_fa = fa_scaler.transform(
            X_test_fa.reshape(-1, X_test_fa.shape[2])
        )
    else:
        X_train_au = X_train_au.reshape(-1, X_train_au.shape[2])
        X_train_fa = X_train_fa.reshape(-1, X_train_fa.shape[2])
        X_test_au = X_test_au.reshape(-1, X_test_au.shape[2])
        X_test_fa = X_test_fa.reshape(-1, X_test_fa.shape[2])
    features = (X_train_au, X_train_fa, X_test_au, X_test_fa)
    if return_scalers:
        return (*features, {"AU": au_scaler, "FA": fa_scaler})
    return features


def prepare_fold_data(fold, checkpoint_dir, device, apply_scaler, loss_class=None):
    fold_data = load_obj(os.path.join(checkpoint_dir, f"data_{fold}"))
    train_data = fold_data["train_data"]
    test_data = fold_data["test_data"]
    X_train_au, X_train_fa, X_test_au, X_test_fa = preprocess_features(
        train_data["X"]["AU"],
        train_data["X"]["FA"],
        test_data["X"]["AU"],
        test_data["X"]["FA"],
        apply_scaler,
    )
    X_test_au_t = torch.from_numpy(X_test_au).float().to(device)
    X_test_fa_t = torch.from_numpy(X_test_fa).float().to(device)
    test_input = torch.cat((X_test_au_t, X_test_fa_t), dim=1)
    y_test = torch.from_numpy(test_data["y"]).float().to(device)
    if loss_class is None:
        return test_input, y_test

    X_train_au_t = torch.from_numpy(X_train_au).float().to(device)
    X_train_fa_t = torch.from_numpy(X_train_fa).float().to(device)
    train_input = torch.cat((X_train_au_t, X_train_fa_t), dim=1)
    y_train = torch.from_numpy(train_data["y"]).float().to(device)
    weights_severity = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.argmax(train_data["y"][:, 0, :], axis=1),
    )
    weights_asymmetry = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=np.argmax(train_data["y"][:, 1, :], axis=1),
    )
    loss_fn = loss_class(
        alpha_severity=weights_severity.tolist(),
        alpha_asymmetry=weights_asymmetry.tolist(),
    ).to(device)
    return train_input, y_train, test_input, y_test, loss_fn


def update_training_schedule(
    optimizer, epochs_no_improve, patience_lr, patience_early_stop, decay_factor, min_lr
):
    """Adjust learning rates after an optimizer step and report early stopping.

    Args:
        optimizer: PyTorch optimizer with scalar learning rates in its groups.
        epochs_no_improve: Integer consecutive non-improving epoch count.
        patience_lr: Positive integer interval between learning-rate reductions.
        patience_early_stop: Positive integer consecutive-epoch stopping limit.
        decay_factor: Scalar multiplicative learning-rate reduction factor.
        min_lr: Scalar lower bound for each parameter group's learning rate.

    Returns:
        Boolean scalar indicating whether training should stop.
    """
    if epochs_no_improve > 0 and epochs_no_improve % patience_lr == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"] * decay_factor, min_lr)
    return epochs_no_improve >= patience_early_stop


def print_classification_metrics(metrics):
    metric_pairs = {
        "One-vs-Rest Accuracy": ("acc_severity_ovr", "acc_asymmetry_ovr"),
        "Sensitivity": ("sens_severity", "sens_asymmetry"),
        "Specificity": ("spec_severity", "spec_asymmetry"),
        "Precision": ("prec_severity", "prec_asymmetry"),
    }
    for metric_name, keys in metric_pairs.items():
        for task_name, key in zip(("Severity Grading", "Asymmetry Prediction"), keys):
            values = metrics[key]
            formatted_values = ", ".join(f"{value:.3f}" for value in values)
            print(
                f"  {task_name} {metric_name}: Per-class [{formatted_values}] | "
                f"Mean: {np.mean(values):.3f} ± {np.std(values):.3f}"
            )
    print(f"  Severity Grading Multiclass Accuracy: {metrics['acc_severity']:.3f}")
    print(f"  Asymmetry Prediction Multiclass Accuracy: {metrics['acc_asymmetry']:.3f}")
