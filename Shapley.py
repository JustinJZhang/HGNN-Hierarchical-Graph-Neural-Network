import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn

from config import (
    ACTION_UNIT_DISPLAY_NAMES,
    ASYMMETRY_CLASSES,
    ASYMMETRY_DEVIATION_DISPLAY_NAMES,
    ASYMMETRY_LEFT_DISPLAY_NAMES,
    ASYMMETRY_RIGHT_DISPLAY_NAMES,
    EXPRESSION_DISPLAY_NAMES,
    NUM_ACTION_UNITS,
    NUM_ASYMMETRY_FEATURES,
    NUM_EXPRESSIONS,
    SEVERITY_CLASSES,
)
from model import HGNN
from train_eval import collect_data
from util import load_obj, parse_experiment_args

DATA_PATH = "AU_asymmetry_data.pkl"
MODEL_PATH = "model.pth"
SCALERS_PATH = "scalers.pkl"


def show_attribution(values, errors, labels, xlabel, title):
    """Print and plot SHAP values with shape [F].

    Args:
        values: SHAP values with shape [F].
        errors: Standard deviations with shape [F].
        labels: Sequence of F feature-name strings.
        xlabel: String naming the feature group.
        title: String naming the task and output class.

    Returns:
        Prints F values and displays one bar chart.
    """
    figure = plt.figure(figsize=(10, 6), dpi=300)
    labels = np.asarray(labels)
    sort_locations = np.argsort(values)[::-1]
    print(f"\n{title} - {xlabel}")
    for index in sort_locations:
        print(f"  {labels[index]}: mean={values[index]:.6f}, std={errors[index]:.6f}")
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("SHAP value")
    plt.show()
    plt.close(figure)


def main(argv=None):
    args = parse_experiment_args(
        "Explain HGNN logits using Shapley value.",
        {"data_path": DATA_PATH, "model_path": MODEL_PATH, "scalers_path": SCALERS_PATH,
         "device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        argv,
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(f"Using device: {device}")

    data = load_obj(args.data_path)
    _, X_au, X_fa, _ = collect_data(list(data), data, False)
    scalers = load_obj(args.scalers_path)
    X_au = X_au.reshape(-1, NUM_ACTION_UNITS)
    X_fa = X_fa.reshape(-1, NUM_ASYMMETRY_FEATURES)
    if scalers["AU"] is not None:
        X_au = scalers["AU"].transform(X_au)
    if scalers["FA"] is not None:
        X_fa = scalers["FA"].transform(X_fa)

    model = HGNN(
        NUM_ACTION_UNITS,
        NUM_ASYMMETRY_FEATURES,
        NUM_EXPRESSIONS,
        device,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model.use_cross_constraint = False

    au_inputs = torch.from_numpy(X_au).float().to(device).reshape(
        -1, NUM_EXPRESSIONS, NUM_ACTION_UNITS
    )
    fa_inputs = torch.from_numpy(X_fa).float().to(device).reshape(
        -1, NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES
    )
    inputs = torch.cat((au_inputs, fa_inputs), dim=2)

    with torch.no_grad():
        predicted_classes = model(inputs.flatten(0, 1)).argmax(dim=-1).cpu().numpy()
    explainer = shap.GradientExplainer(
        nn.Sequential(nn.Flatten(0, 1), model, nn.Flatten(1)).eval(),
        inputs,
    )
    shap_values = explainer.shap_values(inputs)

    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    shap_values = np.abs(shap_values)
    asymmetry_labels = (
        ASYMMETRY_DEVIATION_DISPLAY_NAMES
        + ASYMMETRY_LEFT_DISPLAY_NAMES
        + ASYMMETRY_RIGHT_DISPLAY_NAMES
    )

    tasks = (
        ("Severity Grading", SEVERITY_CLASSES),
        ("Asymmetry Prediction", ASYMMETRY_CLASSES),
    )
    for task_index, (task_name, class_names) in enumerate(tasks):
        task_values = shap_values[..., task_index * 3 : (task_index + 1) * 3]
        groups = [
            (np.mean(task_values, axis=2), EXPRESSION_DISPLAY_NAMES, "Expressions"),
        ]
        if task_index == 0:
            groups.append((
                np.mean(task_values[:, :, :NUM_ACTION_UNITS], axis=1),
                ACTION_UNIT_DISPLAY_NAMES,
                "Action",
            ))
        groups.append((
            np.mean(task_values[:, :, NUM_ACTION_UNITS:], axis=1),
            asymmetry_labels,
            "Asymmetry",
        ))

        for output_index in range(4):
            output_name = "Aggregated" if output_index == 0 else class_names[output_index - 1]
            title = f"{task_name} - {output_name}"
            for samples, labels, xlabel in groups:
                attributions = (
                    samples[np.arange(len(samples)), :, predicted_classes[:, task_index]]
                    if output_index == 0
                    else samples[..., output_index - 1]
                )
                maximum = np.max(samples)
                attributions = attributions / maximum if maximum > 0 else attributions
                show_attribution(
                    np.mean(attributions, axis=0),
                    np.std(attributions, axis=0),
                    labels,
                    xlabel,
                    title,
                )


if __name__ == "__main__":
    main()
