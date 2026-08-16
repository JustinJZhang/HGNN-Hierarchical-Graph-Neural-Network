import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap

from config import (
    ACTION_UNIT_DISPLAY_NAMES,
    ASYMMETRY_DEVIATION_DISPLAY_NAMES,
    ASYMMETRY_LEFT_DISPLAY_NAMES,
    ASYMMETRY_RIGHT_DISPLAY_NAMES,
    EXPRESSION_DISPLAY_NAMES,
    NUM_ACTION_UNITS,
    NUM_ASYMMETRY_FEATURES,
    NUM_EXPRESSIONS,
)
from model import HGNN

SPLIT_PATH = "your_data.pkl"
MODEL_PATH = "model.pkl"


def show_attribution(values, errors, labels, xlabel):
    figure = plt.figure(figsize=(10, 6), dpi=300)
    labels = np.asarray(labels)
    sort_locations = np.argsort(values)[::-1]
    plt.bar(labels[sort_locations], values[sort_locations])
    plt.errorbar(
        labels[sort_locations],
        values[sort_locations],
        yerr=errors[sort_locations],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=2,
        capsize=4,
    )
    figure.autofmt_xdate(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel("Shapley Value")
    plt.show()
    plt.close(figure)


def main():
    """Load one HGNN checkpoint and display AU [17], FA [24], and expression [8] attributions."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(SPLIT_PATH, "rb") as stream:
        split_data = pickle.load(stream)
    train_data = split_data["train_data"]

    model = HGNN(
        NUM_ACTION_UNITS,
        NUM_ASYMMETRY_FEATURES,
        NUM_EXPRESSIONS,
        device,
    ).to(device)
    weight = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(weight["model_state_dict"])
    model.eval()
    inputs = torch.cat((train_data["X"]["AU"], train_data["X"]["FA"]), dim=1)
    explainer = shap.DeepExplainer(model, inputs)
    shap_values = np.abs(explainer.shap_values(inputs, check_additivity=False))

    feature_values = np.mean(shap_values, axis=0).squeeze()
    feature_errors = np.std(shap_values, axis=0).squeeze()
    au_values = feature_values[:NUM_ACTION_UNITS]
    au_errors = feature_errors[:NUM_ACTION_UNITS]
    fa_values = feature_values[NUM_ACTION_UNITS:]
    fa_errors = feature_errors[NUM_ACTION_UNITS:]
    expression_samples = np.mean(shap_values, axis=1).squeeze().reshape(-1, NUM_EXPRESSIONS)
    expression_values = np.mean(expression_samples, axis=0).squeeze()
    expression_errors = np.std(expression_samples, axis=0).squeeze()
    asymmetry_labels = (
        ASYMMETRY_DEVIATION_DISPLAY_NAMES
        + ASYMMETRY_LEFT_DISPLAY_NAMES
        + ASYMMETRY_RIGHT_DISPLAY_NAMES
    )

    show_attribution(au_values, au_errors, ACTION_UNIT_DISPLAY_NAMES, "Action")
    show_attribution(fa_values, fa_errors, asymmetry_labels, "Asymmetry")
    show_attribution(
        expression_values,
        expression_errors,
        EXPRESSION_DISPLAY_NAMES,
        "Expressions",
    )


if __name__ == "__main__":
    main()
