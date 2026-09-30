import os

import matplotlib
import numpy as np
import torch
from openpyxl import Workbook
from sklearn.metrics import auc, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ASYMMETRY_CLASSES,
    NUM_ACTION_UNITS,
    NUM_ASYMMETRY_FEATURES,
    NUM_EXPRESSIONS,
    SEVERITY_CLASSES,
)
from model import HGNN
from train_eval import (
    CHECKPOINT_DIR,
    DEVICE,
    FOLD_DEFAULTS,
    MODEL_FILE_PATTERN,
)
from util import parse_experiment_args, prepare_fold_data


ROC_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
TASK_NAMES = ("Severity Grading", "Asymmetry Prediction")


def load_fold_model(fold, checkpoint_dir=CHECKPOINT_DIR, device=DEVICE):
    """Load the HGNN parameters for one fold.

    Args:
        fold: One-based integer index of a saved fold.
        checkpoint_dir: Scalar string naming the saved-fold directory.
        device: PyTorch device or device string for model and weight loading.

    Returns:
        HGNN model mapping input [N*8, 41] to logits [N, 2, 3].
    """
    model = HGNN(
        NUM_ACTION_UNITS,
        NUM_ASYMMETRY_FEATURES,
        NUM_EXPRESSIONS,
        device,
    ).to(device)
    model_path = os.path.join(
        checkpoint_dir,
        MODEL_FILE_PATTERN.format(fold=fold),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def predict_fold_probabilities(model, test_input, y_test):
    """Generate probabilities for one test fold.

    Args:
        model: Trained HGNN.
        test_input: Floating-point tensor with shape [N_test*8, 41].
        y_test: One-hot tensor with shape [N_test, 2, 3].

    Returns:
        Tuple containing ground truth [N_test, 2, 3] and
        probabilities [N_test, 2, 3] as NumPy arrays.
    """
    model.eval()
    with torch.no_grad():
        logits = model(test_input)
        probabilities = torch.softmax(logits, dim=-1)

    return (
        y_test.cpu().numpy().astype(np.float64),
        probabilities.cpu().numpy().astype(np.float64),
    )


def compute_task_micro_roc(y_true, y_probability, task_index):
    """Compute one fold's ROC for a task.

    Args:
        y_true: One-hot ground truth with shape [N_test, 2, 3].
        y_probability: Class probabilities with shape [N_test, 2, 3].
        task_index: Integer scalar; 0 selects severity and 1 asymmetry.

    Returns:
        Tuple containing TPR [1001] on ROC_GRID and scalar AUC.
    """
    task_true = y_true[:, task_index, :].reshape(-1)
    task_probability = y_probability[:, task_index, :].reshape(-1)
    fpr, tpr, _ = roc_curve(
        task_true,
        task_probability,
    )
    tpr = np.interp(ROC_GRID, fpr, tpr)
    tpr[0] = 0.0
    tpr[-1] = 1.0
    roc_auc = auc(ROC_GRID, tpr)
    return tpr, roc_auc


def plot_roc(task_results, output_path):
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(12.27, 4.65))

        for task_index, axis in enumerate(axes):
            fold_results = task_results[task_index]
            tracked_title = "\u2009".join(TASK_NAMES[task_index])
            fold_tpr = np.stack(
                [result[0] for result in fold_results],
                axis=0,
            )
            fold_auc = np.array(
                [result[1] for result in fold_results],
                dtype=np.float64,
            )
            mean_tpr = np.mean(fold_tpr, axis=0)
            std_tpr = np.std(fold_tpr, axis=0)

            for fold_index, result in enumerate(fold_results):
                tpr, roc_auc = result
                axis.plot(
                    ROC_GRID,
                    tpr,
                    color=(
                        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                    )[fold_index % 10],
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"Fold {fold_index + 1} (AUC={roc_auc:.3f})",
                )

            axis.plot(
                ROC_GRID,
                mean_tpr,
                color="black",
                linewidth=3.0,
                label=(
                    f"Mean (AUC={np.mean(fold_auc):.3f} "
                    f"± {np.std(fold_auc):.3f})"
                ),
            )
            axis.fill_between(
                ROC_GRID,
                np.clip(mean_tpr - std_tpr, 0.0, 1.0),
                np.clip(mean_tpr + std_tpr, 0.0, 1.0),
                color="gray",
                alpha=0.2,
                label="±1 STD",
            )
            axis.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.05)
            axis.set_xlabel("FPR", fontsize=12)
            axis.set_ylabel("TPR", fontsize=12)
            axis.set_title(
                f"({chr(ord('a') + task_index)}) {tracked_title}",
                fontsize=13,
                fontweight="bold",
            )
            axis.tick_params(axis="both", labelsize=10)
            axis.set_axisbelow(True)
            axis.grid(
                True,
                color="#b0b0b0",
                linewidth=0.8,
                alpha=0.3,
            )
            axis.legend(
                loc="lower right",
                bbox_to_anchor=(0.98, 0.01),
                fontsize=8,
                labelspacing=0.33,
                frameon=True,
                framealpha=0.8,
                facecolor="white",
                edgecolor="#cccccc",
            )

        figure.tight_layout()
        figure.savefig(
            output_path,
            format="pdf",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(figure)


def save_roc_results(task_results, output_path):
    severity_tpr = np.stack(
        [result[0] for result in task_results[0]],
        axis=0,
    )
    asymmetry_tpr = np.stack(
        [result[0] for result in task_results[1]],
        axis=0,
    )
    severity_auc = np.array(
        [result[1] for result in task_results[0]],
        dtype=np.float64,
    )
    asymmetry_auc = np.array(
        [result[1] for result in task_results[1]],
        dtype=np.float64,
    )
    np.savez(
        output_path,
        fpr=ROC_GRID,
        severity_tpr=severity_tpr,
        severity_mean_tpr=np.mean(severity_tpr, axis=0),
        severity_std_tpr=np.std(severity_tpr, axis=0),
        severity_auc=severity_auc,
        severity_mean_auc=np.mean(severity_auc),
        severity_std_auc=np.std(severity_auc),
        asymmetry_tpr=asymmetry_tpr,
        asymmetry_mean_tpr=np.mean(asymmetry_tpr, axis=0),
        asymmetry_std_tpr=np.std(asymmetry_tpr, axis=0),
        asymmetry_auc=asymmetry_auc,
        asymmetry_mean_auc=np.mean(asymmetry_auc),
        asymmetry_std_auc=np.std(asymmetry_auc),
    )


def save_roc_spreadsheet(task_results, output_path):
    workbook = Workbook()

    for task_index, sheet_name in enumerate(TASK_NAMES):
        if task_index == 0:
            worksheet = workbook.active
            worksheet.title = sheet_name
        else:
            worksheet = workbook.create_sheet(title=sheet_name)
        worksheet.append(("curve", "fpr", "tpr"))

        for fold_index, fold_result in enumerate(
            task_results[task_index],
            start=1,
        ):
            curve_name = f"fold_{fold_index}"
            tpr = fold_result[0]
            for false_positive_rate, true_positive_rate in zip(
                ROC_GRID,
                tpr,
            ):
                worksheet.append(
                    (
                        curve_name,
                        float(false_positive_rate),
                        float(true_positive_rate),
                    )
                )

    workbook.save(output_path)


def main(argv=None):
    """Generate task-level ROC outputs and class-wise one-vs-rest AUC summaries."""
    args = parse_experiment_args("Generate ROC curves.", FOLD_DEFAULTS, argv)
    device = torch.device(args.device)
    figure_path = os.path.join(args.checkpoint_dir, "roc.pdf")
    data_path = os.path.join(args.checkpoint_dir, "roc_results.npz")
    spreadsheet_path = os.path.join(args.checkpoint_dir, "roc.xlsx")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    task_results = [[], []]
    class_auc_results = np.empty(
        (args.k_fold, len(TASK_NAMES), len(SEVERITY_CLASSES))
    )

    for fold in range(1, args.k_fold + 1):
        print(f"Evaluating fold {fold}")
        test_input, y_test = prepare_fold_data(fold, args.checkpoint_dir, device, args.scaler)
        model = load_fold_model(fold, args.checkpoint_dir, device)
        y_true, y_probability = predict_fold_probabilities(
            model,
            test_input,
            y_test,
        )

        for task_index in range(len(TASK_NAMES)):
            task_results[task_index].append(
                compute_task_micro_roc(
                    y_true,
                    y_probability,
                    task_index,
                )
            )
            for class_index in range(y_true.shape[2]):
                class_auc_results[fold - 1, task_index, class_index] = roc_auc_score(
                    y_true[:, task_index, class_index],
                    y_probability[:, task_index, class_index],
                )

    plot_roc(task_results, figure_path)
    save_roc_results(task_results, data_path)
    save_roc_spreadsheet(task_results, spreadsheet_path)
    for task_index, fold_results in enumerate(task_results):
        fold_auc = np.array(
            [result[1] for result in fold_results],
            dtype=np.float64,
        )
        formatted_auc = ", ".join(f"{value:.3f}" for value in fold_auc)
        print(f"{TASK_NAMES[task_index]} fold AUC: [{formatted_auc}]")
        print(
            f"{TASK_NAMES[task_index]} mean AUC: "
            f"{np.mean(fold_auc):.3f} ± {np.std(fold_auc):.3f}"
        )
        for class_index, class_name in enumerate(
            (SEVERITY_CLASSES, ASYMMETRY_CLASSES)[task_index]
        ):
            class_fold_auc = class_auc_results[:, task_index, class_index]
            formatted_auc = ", ".join(f"{value:.3f}" for value in class_fold_auc)
            print(
                f"{TASK_NAMES[task_index]} - {class_name} OvR fold AUC: "
                f"[{formatted_auc}]"
            )
            print(
                f"{TASK_NAMES[task_index]} - {class_name} OvR mean AUC: "
                f"{np.mean(class_fold_auc):.3f} ± {np.std(class_fold_auc):.3f}"
            )
    print(f"ROC figure saved to {figure_path}")
    print(f"ROC data saved to {data_path}")
    print(f"ROC spreadsheet saved to {spreadsheet_path}")


if __name__ == "__main__":
    main()
