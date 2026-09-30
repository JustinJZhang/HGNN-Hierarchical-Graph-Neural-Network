import torch

from compare_model import (
    CNN1D_Model,
    CNN_Model,
    FT_Transformer_Model,
    MLP_Model,
)
from config import NUM_ACTION_UNITS, NUM_ASYMMETRY_FEATURES, NUM_EXPRESSIONS
from metrics import METRIC_KEYS, get_classification_metrics
from train_eval import TRAINING_DEFAULTS, CombinedLoss
from util import (
    parse_experiment_args,
    prepare_fold_data,
    print_classification_metrics,
    update_training_schedule,
)


MODEL_TYPES = ("cnn2d", "cnn1d", "mlp", "ft_transformer")
MODEL_NAMES = {
    "cnn2d": "2D CNN",
    "cnn1d": "1D CNN",
    "mlp": "MLP",
    "ft_transformer": "FT-Transformer",
}
MODEL_CLASSES = {
    "cnn2d": CNN_Model,
    "cnn1d": CNN1D_Model,
    "mlp": MLP_Model,
    "ft_transformer": FT_Transformer_Model,
}


def build_dl_model(model_type, device):
    """Map a model-name string and device to a model [N*8,41] -> [N,2,3]."""
    model_class = MODEL_CLASSES[model_type]
    return model_class(
        NUM_ACTION_UNITS,
        NUM_ASYMMETRY_FEATURES,
        NUM_EXPRESSIONS,
        device,
    ).to(device)


def train_dl_model(model_type, train_input, y_train, loss_fn, fold, args):
    """Train one DL comparison model.

    Args:
        model_type: String key in ``MODEL_CLASSES``.
        train_input: Floating-point tensor with shape [N_train*8, 41].
        y_train: One-hot floating-point labels with shape [N_train, 2, 3].
        loss_fn: ``CombinedLoss`` module returning two scalar losses.
        fold: One-based integer fold index used in progress messages.
        args: Namespace of scalar device and training hyperparameters.

    Returns:
        Trained comparison model.
    """
    device = torch.device(args.device)
    model = build_dl_model(model_type, device)
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
                f"Early stopping for {MODEL_NAMES[model_type]}, fold {fold}, at epoch {epoch + 1}"
            )
            break

    model.load_state_dict(best_model_state)
    return model


def evaluate_dl_model(model, eval_input, y_test):
    """Evaluate one comparison model.

    Args:
        model: Trained model mapping input [N_test*8, 41] to [N_test, 2, 3].
        eval_input: Floating-point tensor with shape [N_test*8, 41].
        y_test: One-hot floating-point labels with shape [N_test, 2, 3].

    Returns:
        METRIC_KEYS dictionary containing eight per-class arrays [3]
        and two task accuracies.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(eval_input)

    y_true = y_test.cpu().numpy()
    y_logits = predictions.cpu().numpy()
    fold_metrics = get_classification_metrics(y_true, y_logits)
    return {
        key: value
        for key, value in zip(METRIC_KEYS, fold_metrics)
    }


def main(argv=None):
    args = parse_experiment_args(
        "Train and evaluate DL comparison models.",
        {**TRAINING_DEFAULTS, "models": MODEL_TYPES},
        argv,
    )
    device = torch.device(args.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    for fold in range(1, args.k_fold + 1):
        print(f"Fold {fold}")
        train_input, y_train, eval_input, y_test, loss_fn = prepare_fold_data(
            fold, args.checkpoint_dir, device, args.scaler, loss_class=CombinedLoss
        )

        for model_type in args.models:
            model_name = MODEL_NAMES[model_type]
            print(f"Training {model_name} on Fold {fold}...")
            model = train_dl_model(
                model_type,
                train_input,
                y_train,
                loss_fn,
                fold,
                args,
            )
            metrics = evaluate_dl_model(model, eval_input, y_test)
            print(f"\nModel: {model_name}")
            print(f"Fold {fold} Best Evaluation Metrics:")
            print_classification_metrics(metrics)


if __name__ == "__main__":
    main()
