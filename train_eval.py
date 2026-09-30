import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from config import (
    ACTION_UNIT_KEYS,
    ASYMMETRY_DEVIATION_KEYS,
    ASYMMETRY_SIDE_KEYS,
    EXPRESSIONS,
    NUM_ACTION_UNITS,
    NUM_ASYMMETRY_FEATURES,
    NUM_EXPRESSIONS,
)
from metrics import METRIC_KEYS, get_classification_metrics
from model import HGNN
from util import (
    load_obj,
    parse_experiment_args,
    preprocess_features,
    print_classification_metrics,
    save_obj,
    update_training_schedule,
)

TASK_NAME = "EXP"
CHECKPOINT_DIR = os.path.join("/data/FSHD/checkpoints", TASK_NAME)
MODEL_FILE_PATTERN = "model_{fold}.pth"
DEVICE = torch.device("cuda:0")

DATA_PATH = "AU_asymmetry_data.pkl"

K_FOLD = 10
MAX_EPOCH = 10000
INITIAL_LR = 0.0001
MIN_LR = INITIAL_LR * 0.1
PATIENCE_LR = 50
PATIENCE_EARLY_STOP = 200
DECAY_FACTOR = 0.95
FLIP_AUG = True
SCALER = True

FOLD_DEFAULTS = {
    "checkpoint_dir": CHECKPOINT_DIR,
    "device": str(DEVICE),
    "k_fold": K_FOLD,
    "scaler": SCALER,
}
TRAINING_DEFAULTS = {
    **FOLD_DEFAULTS,
    "max_epoch": MAX_EPOCH,
    "initial_lr": INITIAL_LR,
    "min_lr": MIN_LR,
    "patience_lr": PATIENCE_LR,
    "patience_early_stop": PATIENCE_EARLY_STOP,
    "decay_factor": DECAY_FACTOR,
}

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        alpha = self.alpha
        if alpha is not None and alpha.device != targets.device:
            alpha = alpha.to(targets.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha_severity=None, alpha_asymmetry=None, gamma=2.0):
        super().__init__()
        self.loss_fl_severity = FocalLoss(alpha=alpha_severity, gamma=gamma)
        self.loss_fl_asymmetry = FocalLoss(alpha=alpha_asymmetry, gamma=gamma)

    def forward(self, pred, target):
        pred_severity, pred_asymmetry = pred[:, 0, :], pred[:, 1, :]
        target_severity, target_asymmetry = target[:, 0, :], target[:, 1, :]

        severity_idx = target_severity.argmax(dim=1)
        asymmetry_idx = target_asymmetry.argmax(dim=1)

        fl_severity = self.loss_fl_severity(pred_severity, severity_idx)
        fl_asymmetry = self.loss_fl_asymmetry(pred_asymmetry, asymmetry_idx)

        return fl_severity, fl_asymmetry


def extract_subject_features(subject, data):
    label = data[subject]["label"]
    au_feat = [
        [data[subject][act]["Action Unit"][key] for key in ACTION_UNIT_KEYS]
        for act in EXPRESSIONS
    ]
    fa_feat = [
        [data[subject][act]["Facial Asymmetry"]["Deviation"][k] for k in ASYMMETRY_DEVIATION_KEYS] +
        [data[subject][act]["Facial Asymmetry"]["Left"][k] for k in ASYMMETRY_SIDE_KEYS] +
        [data[subject][act]["Facial Asymmetry"]["Right"][k] for k in ASYMMETRY_SIDE_KEYS]
        for act in EXPRESSIONS
    ]
    return label, np.array(au_feat), np.array(fa_feat)


def create_flipped_sample(label, au_feat, fa_feat):
    fa_flip = fa_feat.copy()
    num_unilateral_features = len(ASYMMETRY_SIDE_KEYS)
    fa_flip[:, 10:10 + num_unilateral_features], fa_flip[:, 10 + num_unilateral_features:] = \
        fa_feat[:, 10 + num_unilateral_features:].copy(), fa_feat[:, 10:10 + num_unilateral_features].copy()

    label_flip = label.copy()
    label_flip[1, :] = label[1, :][[0, 2, 1]]
    return label_flip, au_feat, fa_flip


def collect_data(sub_list, data, apply_flip):
    """Collect subject records into labels [N,2,3], AU [N,8,17], and FA [N,8,24]."""
    y_list, au_list, fa_list, name_list = [], [], [], []
    for sub in sub_list:
        y, au, fa = extract_subject_features(sub, data)
        y_list.append(y)
        au_list.append(au)
        fa_list.append(fa)
        name_list.append(sub)
        if apply_flip:
            y_f, au_f, fa_f = create_flipped_sample(y, au, fa)
            y_list.append(y_f)
            au_list.append(au_f)
            fa_list.append(fa_f)
            name_list.append(f"{sub}_flip")
    return np.array(y_list), np.array(au_list), np.array(fa_list), np.array(name_list)


def save_fold_data(
    fold,
    y_train,
    X_train_au,
    X_train_fa,
    train_names,
    y_test,
    X_test_au,
    X_test_fa,
    test_names,
    checkpoint_dir=CHECKPOINT_DIR,
):
    save_obj(
        {
            "train_data": {
                "X": {"AU": X_train_au, "FA": X_train_fa},
                "y": y_train,
            },
            "test_data": {
                "X": {"AU": X_test_au, "FA": X_test_fa},
                "y": y_test,
            },
            "train_names": train_names,
            "test_names": test_names,
        },
        os.path.join(checkpoint_dir, f"data_{fold}"),
    )


def main(argv=None):
    args = parse_experiment_args(
        "Train and evaluate the HGNN.",
        {**TRAINING_DEFAULTS, "data_path": DATA_PATH, "flip_aug": FLIP_AUG},
        argv,
    )
    device = torch.device(args.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    data = load_obj(args.data_path)
    subjects = list(data.keys())
    print(f"Total Subjects: {len(subjects)}")

    labels = np.array([data[sub]["label"][0] for sub in subjects])
    labels = np.argmax(labels, axis=1)
    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True)

    for fold, (train_sub_idx, test_sub_idx) in enumerate(skf.split(subjects, labels)):
        print(f"Fold {fold + 1}")
        train_subs = np.array(subjects)[train_sub_idx]
        test_subs = np.array(subjects)[test_sub_idx]
        y_train, X_train_au, X_train_fa, train_names = collect_data(
            train_subs, data, args.flip_aug
        )
        y_test, X_test_au, X_test_fa, test_names = collect_data(
            test_subs, data, False
        )

        save_fold_data(
            fold + 1,
            y_train,
            X_train_au,
            X_train_fa,
            train_names,
            y_test,
            X_test_au,
            X_test_fa,
            test_names,
            checkpoint_dir=args.checkpoint_dir,
        )

        X_train_au, X_train_fa, X_test_au, X_test_fa, scalers = preprocess_features(
            X_train_au, X_train_fa, X_test_au, X_test_fa, args.scaler,
            return_scalers=True,
        )

        X_train_au_t = torch.from_numpy(X_train_au).float().to(device)
        X_test_au_t = torch.from_numpy(X_test_au).float().to(device)
        X_train_fa_t = torch.from_numpy(X_train_fa).float().to(device)
        X_test_fa_t = torch.from_numpy(X_test_fa).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        Xy_train = {"X": {"AU": X_train_au_t, "FA": X_train_fa_t}, "y": y_train_t}
        Xy_test = {"X": {"AU": X_test_au_t, "FA": X_test_fa_t}, "y": y_test_t}
        y_train_severity = np.argmax(y_train[:, 0, :], axis=1)
        weights_severity = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_severity)
        y_train_asymmetry = np.argmax(y_train[:, 1, :], axis=1)
        weights_asymmetry = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_asymmetry)
        loss_fn = CombinedLoss(
            alpha_severity=weights_severity.tolist(), alpha_asymmetry=weights_asymmetry.tolist()
        ).to(device)

        model = HGNN(
            NUM_ACTION_UNITS,
            NUM_ASYMMETRY_FEATURES,
            NUM_EXPRESSIONS,
            device,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
        best_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None
        train_input = torch.cat((Xy_train["X"]["AU"], Xy_train["X"]["FA"]), dim=1)
        eval_input = torch.cat((Xy_test["X"]["AU"], Xy_test["X"]["FA"]), dim=1)

        for epoch in range(args.max_epoch):
            model.train()
            train_pred = model(train_input)
            loss_severity, loss_asymmetry = loss_fn(train_pred, Xy_train["y"])
            optimizer.zero_grad()
            (loss_severity + loss_asymmetry).backward()

            total_loss = loss_severity.item() + loss_asymmetry.item()
            if total_loss < best_loss:
                best_loss = total_loss
                epochs_no_improve = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
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
                print(f"Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(best_model_state)
        torch.save(
            model.state_dict(),
            os.path.join(
                args.checkpoint_dir,
                MODEL_FILE_PATTERN.format(fold=fold + 1),
            ),
        )
        save_obj(scalers, os.path.join(args.checkpoint_dir, f"scalers_{fold + 1}.pkl"))
        model.eval()
        with torch.no_grad():
            best_pred = model(eval_input)

        y_true = Xy_test["y"].cpu().numpy()
        y_logits = best_pred.cpu().numpy()
        best_fold_metrics = get_classification_metrics(y_true, y_logits)
        best_metrics_dict = {key: value for key, value in zip(METRIC_KEYS, best_fold_metrics)}
        print(f"\nFold {fold + 1} Best Evaluation Metrics:")
        print_classification_metrics(best_metrics_dict)


if __name__ == "__main__":
    main()
