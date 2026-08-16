import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

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
from util import load_obj

TASK_NAME = "EXP"
CHECKPOINT_DIR = os.path.join("/data/FSHD/checkpoints", TASK_NAME)
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
    def __init__(self, alpha_score=None, alpha_side=None, gamma=2.0):
        super().__init__()
        self.loss_fl_score = FocalLoss(alpha=alpha_score, gamma=gamma)
        self.loss_fl_side = FocalLoss(alpha=alpha_side, gamma=gamma)

    def forward(self, pred, target):
        pred_score, pred_side = pred[:, 0, :], pred[:, 1, :]
        target_score, target_side = target[:, 0, :], target[:, 1, :]

        score_idx = target_score.argmax(dim=1)
        side_idx = target_side.argmax(dim=1)

        fl_score = self.loss_fl_score(pred_score, score_idx)
        fl_side = self.loss_fl_side(pred_side, side_idx)

        return fl_score, fl_side


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
    n_side = len(ASYMMETRY_SIDE_KEYS)
    fa_flip[:, 10:10 + n_side], fa_flip[:, 10 + n_side:] = \
        fa_feat[:, 10 + n_side:].copy(), fa_feat[:, 10:10 + n_side].copy()

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


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = load_obj(DATA_PATH)
    subjects = list(data.keys())
    print(f"Total Subjects: {len(subjects)}")

    labels = np.array([data[sub]["label"][0] for sub in subjects])
    labels = np.argmax(labels, axis=1)
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True)

    for fold, (train_sub_idx, test_sub_idx) in enumerate(skf.split(subjects, labels)):
        print(f"Fold {fold + 1}")
        train_subs = np.array(subjects)[train_sub_idx]
        test_subs = np.array(subjects)[test_sub_idx]
        y_train, X_train_au, X_train_fa, train_names = collect_data(
            train_subs, data, FLIP_AUG
        )
        y_test, X_test_au, X_test_fa, test_names = collect_data(
            test_subs, data, False
        )

        if SCALER:
            au_scaler = StandardScaler()
            fa_scaler = StandardScaler()
            X_train_au = au_scaler.fit_transform(X_train_au.reshape(-1, X_train_au.shape[2]))
            X_train_fa = fa_scaler.fit_transform(X_train_fa.reshape(-1, X_train_fa.shape[2]))
            X_test_au = au_scaler.transform(X_test_au.reshape(-1, X_test_au.shape[2]))
            X_test_fa = fa_scaler.transform(X_test_fa.reshape(-1, X_test_fa.shape[2]))
        else:
            X_train_au = X_train_au.reshape(-1, X_train_au.shape[2])
            X_train_fa = X_train_fa.reshape(-1, X_train_fa.shape[2])
            X_test_au = X_test_au.reshape(-1, X_test_au.shape[2])
            X_test_fa = X_test_fa.reshape(-1, X_test_fa.shape[2])

        X_train_au_t = torch.from_numpy(X_train_au).float().to(DEVICE)
        X_test_au_t = torch.from_numpy(X_test_au).float().to(DEVICE)
        X_train_fa_t = torch.from_numpy(X_train_fa).float().to(DEVICE)
        X_test_fa_t = torch.from_numpy(X_test_fa).float().to(DEVICE)
        y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
        y_test_t = torch.from_numpy(y_test).float().to(DEVICE)

        Xy_train = {"X": {"AU": X_train_au_t, "FA": X_train_fa_t}, "y": y_train_t}
        Xy_test = {"X": {"AU": X_test_au_t, "FA": X_test_fa_t}, "y": y_test_t}
        y_train_score = np.argmax(y_train[:, 0, :], axis=1)
        weights_score = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_score)
        y_train_side = np.argmax(y_train[:, 1, :], axis=1)
        weights_side = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_side)
        loss_fn = CombinedLoss(
            alpha_score=weights_score.tolist(), alpha_side=weights_side.tolist()
        ).to(DEVICE)

        model = HGNN(
            NUM_ACTION_UNITS,
            NUM_ASYMMETRY_FEATURES,
            NUM_EXPRESSIONS,
            DEVICE,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
        best_loss = float("inf")
        epochs_no_improve_lr = 0
        epochs_no_improve_early_stop = 0
        best_model_state = None
        train_input = torch.cat((Xy_train["X"]["AU"], Xy_train["X"]["FA"]), dim=1)
        eval_input = torch.cat((Xy_test["X"]["AU"], Xy_test["X"]["FA"]), dim=1)

        for epoch in range(MAX_EPOCH):
            model.train()
            train_pred = model(train_input)
            loss_s, loss_si = loss_fn(train_pred, Xy_train["y"])
            optimizer.zero_grad()
            (loss_s + loss_si).backward()
            optimizer.step()

            total_loss = loss_s.item() + loss_si.item()
            if total_loss < best_loss:
                best_loss = total_loss
                epochs_no_improve_lr = 0
                epochs_no_improve_early_stop = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve_lr += 1
                epochs_no_improve_early_stop += 1

            if epochs_no_improve_lr >= PATIENCE_LR:
                for param_group in optimizer.param_groups:
                    new_lr = param_group["lr"] * DECAY_FACTOR
                    param_group["lr"] = max(new_lr, MIN_LR)
                epochs_no_improve_lr = 0

            if epochs_no_improve_early_stop >= PATIENCE_EARLY_STOP:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            best_pred = model(eval_input)
            absence_mask = torch.argmax(best_pred[:, 0, :], dim=1) == 0
            best_pred[absence_mask, 1, :] = torch.tensor(
                [1.0, 0.0, 0.0], device=DEVICE
            )

        y_true = Xy_test["y"].cpu().numpy()
        y_logits = best_pred.cpu().numpy()
        best_fold_metrics = get_classification_metrics(y_true, y_logits)
        best_metrics_dict = {key: value for key, value in zip(METRIC_KEYS, best_fold_metrics)}
        fmt = lambda arr: ", ".join([f"{v:.3f}" for v in arr])
        print(f"\nFold {fold + 1} Best Evaluation Metrics:")
        metric_pairs = {
            "Sensitivity": ("sens_score", "sens_side"),
            "Specificity": ("spec_score", "spec_side"),
            "Precision": ("prec_score", "prec_side"),
            "Accuracy": ("acc_score", "acc_side"),
        }
        for metric_name, (score_key, side_key) in metric_pairs.items():
            score_values = best_metrics_dict[score_key]
            side_values = best_metrics_dict[side_key]
            print(
                f"  Score {metric_name}: Per-class [{fmt(score_values)}] | "
                f"Mean: {np.mean(score_values):.3f} ± {np.std(score_values):.3f}"
            )
            print(
                f"  Side  {metric_name}: Per-class [{fmt(side_values)}] | "
                f"Mean: {np.mean(side_values):.3f} ± {np.std(side_values):.3f}"
            )


if __name__ == "__main__":
    main()
