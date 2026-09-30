import os

import numpy as np
import torch

from ablation_model import ABLATION_DEFINITIONS, build_ablation_model
from config import NUM_ACTION_UNITS, NUM_ASYMMETRY_FEATURES, NUM_EXPRESSIONS
from metrics import METRIC_KEYS, get_classification_metrics
from train_eval import TRAINING_DEFAULTS, CombinedLoss
from util import (
    parse_experiment_args,
    prepare_fold_data,
    print_classification_metrics,
    update_training_schedule,
)


ABLATION_VARIANTS = (
    "ge",
    "ge_gau",
    "ge_gfa",
    "gau_gfa",
    "without_gga",
    "without_cross_constraint",
    "depth_1",
    "depth_2",
)


def train_ablation_model(variant, train_input, y_train, loss_fn, fold, args):
    """Train one ablation model.

    Args:
        variant: String key in ``ABLATION_DEFINITIONS``.
        train_input: Floating-point tensor with shape [N_train*8, 41].
        y_train: One-hot labels with shape [N_train, 2, 3].
        loss_fn: ``CombinedLoss`` module returning two scalar losses.
        fold: One-based integer fold index.
        args: Namespace of scalar device and training hyperparameters.

    Returns:
        Trained ``AblationHGNN`` with the lowest training loss.
    """
    device = torch.device(args.device)
    model = build_ablation_model(
        variant,
        NUM_ACTION_UNITS,
        NUM_ASYMMETRY_FEATURES,
        NUM_EXPRESSIONS,
        device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(args.max_epoch):
        model.train()
        train_pred = model(train_input)
        loss_severity, loss_asymmetry = loss_fn(train_pred, y_train)
        optimizer.zero_grad()
        (loss_severity + loss_asymmetry).backward()

        total_loss = loss_severity.item() + loss_asymmetry.item()
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_no_improve = 0
            best_model_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1

        optimizer.step()

        if update_training_schedule(
            optimizer,
            epochs_no_improve,
            args.patience_lr,
            args.patience_early_stop,
            args.decay_factor,
            args.min_lr,
        ):
            print(
                f"Early stopping for {variant}, fold {fold}, at epoch {epoch + 1}"
            )
            break

    model.load_state_dict(best_model_state)
    return model


def evaluate_ablation_model(model, eval_input, y_test):
    """Evaluate one model and return metrics.

    Args:
        model: Trained ablation model mapping [N*8, 41] to logits [N, 2, 3].
        eval_input: Floating-point evaluation tensor with shape [N*8, 41].
        y_test: One-hot label tensor with shape [N, 2, 3].

    Returns:
        Tuple containing a metric dictionary and a float32 probability
        array with shape [N, 2, 3].
    """
    model.eval()
    with torch.no_grad():
        y_probability = torch.softmax(model(eval_input), dim=2)

    y_true = y_test.cpu().numpy()
    y_probability = y_probability.cpu().numpy()
    best_fold_metrics = get_classification_metrics(y_true, y_probability)
    metrics = {
        key: value
        for key, value in zip(METRIC_KEYS, best_fold_metrics)
    }
    return metrics, y_probability


def main(argv=None):
    args = parse_experiment_args(
        "Train and evaluate HGNN ablations.",
        {**TRAINING_DEFAULTS, "variants": ABLATION_VARIANTS},
        argv,
    )
    device = torch.device(args.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    fold_labels = []
    fold_probabilities = {
        variant: []
        for variant in args.variants
    }

    for fold in range(1, args.k_fold + 1):
        print(f"Fold {fold}")
        train_input, y_train, eval_input, y_test, loss_fn = prepare_fold_data(
            fold, args.checkpoint_dir, device, args.scaler, loss_class=CombinedLoss
        )
        fold_labels.append(y_test.cpu().numpy())

        for variant in args.variants:
            definition = ABLATION_DEFINITIONS[variant]
            print(
                f"Training {variant}: "
                f"graphs=({definition.use_expression_graph}, "
                f"{definition.use_action_graph}, "
                f"{definition.use_asymmetry_graph}), "
                f"layers={definition.num_gcn_layers}, "
                f"GGA={definition.use_gga}, "
                f"constraint={definition.use_cross_constraint}"
            )
            model = train_ablation_model(
                variant,
                train_input,
                y_train,
                loss_fn,
                fold,
                args,
            )
            metrics, y_probability = evaluate_ablation_model(
                model,
                eval_input,
                y_test,
            )
            fold_probabilities[variant].append(y_probability)
            print(f"\nAblation Variant: {variant}")
            print(f"Fold {fold} Best Evaluation Metrics:")
            print_classification_metrics(metrics)

    y_true = np.concatenate(fold_labels, axis=0)
    for variant in args.variants:
        result_dir = os.path.join(
            args.checkpoint_dir,
            "experiment_results",
            variant,
        )
        os.makedirs(result_dir, exist_ok=True)
        np.savez(
            os.path.join(result_dir, "predictions.npz"),
            y_true=y_true,
            y_probability=np.concatenate(
                fold_probabilities[variant],
                axis=0,
            ),
        )


if __name__ == "__main__":
    main()
