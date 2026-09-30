from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test

from roc import load_fold_model, predict_fold_probabilities
from train_eval import CHECKPOINT_DIR, FOLD_DEFAULTS
from util import parse_experiment_args, prepare_fold_data


@dataclass(frozen=True)
class TestDefinition:
    """Store one comparison label and result-directory name.

    Data format:
        Input and output are two scalar strings: ``label`` and ``variant``.
    """

    label: str
    variant: str


TEST_DEFINITIONS = (
    TestDefinition("w/o G_e", "gau_gfa"),
    TestDefinition("w/o G_au", "ge_gfa"),
    TestDefinition("w/o G_fa", "ge_gau"),
    TestDefinition("w/o G_au and G_fa", "ge"),
    TestDefinition("w/o GGA", "without_gga"),
    TestDefinition(
        "w/o Cross Constraint",
        "without_cross_constraint",
    ),
    TestDefinition("1 Layer", "depth_1"),
    TestDefinition("2 Layers", "depth_2"),
)


def load_experiment_results(variant, checkpoint_dir=CHECKPOINT_DIR):
    """Load labels and probabilities for one experiment variant.

    Args:
        checkpoint_dir: Scalar string naming the main experiment directory.
        variant: Scalar string naming a result directory containing arrays
            ``y_true`` [N, 2, 3] and ``y_probability`` [N, 2, 3].

    Returns:
        Dictionary containing float64 arrays ``y_true`` [N, 2, 3] and
        ``y_probability`` [N, 2, 3].
    """
    result_path = (
        Path(checkpoint_dir)
        / "experiment_results"
        / variant
        / "predictions.npz"
    )
    with np.load(result_path, allow_pickle=False) as result_file:
        y_true = np.asarray(result_file["y_true"], dtype=np.float64)
        y_probability = np.asarray(
            result_file["y_probability"],
            dtype=np.float64,
        )
    return {
        "y_true": y_true,
        "y_probability": y_probability,
    }


def compute_task_cross_entropy(y_true, y_probability):
    """Compute subject-wise cross-entropy for both tasks.

    Args:
        y_true: One-hot float array with shape [N, 2, 3].
        y_probability: Class-probability float array with shape [N, 2, 3].

    Returns:
        Cross-entropy array with shape [N, 2].
    """
    true_classes = np.argmax(y_true, axis=2)
    true_probabilities = np.take_along_axis(
        y_probability,
        true_classes[:, :, None],
        axis=2,
    ).squeeze(axis=2)
    return -np.log(np.clip(true_probabilities, 1.0e-12, 1.0))


def significance_label(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def run_statistical_test(
    reference_results,
    ablation_results,
    n_resamples=100000,
):
    reference_task_loss = compute_task_cross_entropy(
        reference_results["y_true"],
        reference_results["y_probability"],
    )
    ablation_task_loss = compute_task_cross_entropy(
        ablation_results["y_true"],
        ablation_results["y_probability"],
    )
    paired_difference = np.mean(
        ablation_task_loss - reference_task_loss,
        axis=1,
    )
    permutation_result = permutation_test(
        data=(paired_difference,),
        statistic=np.mean,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative="greater",
    )
    return float(permutation_result.pvalue)


def main(argv=None):
    args = parse_experiment_args(
        "Compare saved ablation predictions with the main HGNN.",
        {**FOLD_DEFAULTS, "n_resamples": 100000,
         "variants": tuple(definition.variant for definition in TEST_DEFINITIONS)},
        argv,
    )
    fold_labels = []
    fold_probabilities = []
    for fold in range(1, args.k_fold + 1):
        test_input, y_test = prepare_fold_data(fold, args.checkpoint_dir, args.device, args.scaler)
        model = load_fold_model(fold, args.checkpoint_dir, args.device)
        y_true, y_probability = predict_fold_probabilities(
            model, test_input, y_test
        )
        fold_labels.append(y_true)
        fold_probabilities.append(y_probability)
    reference_results = {
        "y_true": np.concatenate(fold_labels, axis=0),
        "y_probability": np.concatenate(fold_probabilities, axis=0),
    }

    for definition in TEST_DEFINITIONS:
        if definition.variant not in args.variants:
            continue
        ablation_results = load_experiment_results(definition.variant, args.checkpoint_dir)
        p_value = run_statistical_test(
            reference_results,
            ablation_results,
            n_resamples=args.n_resamples,
        )
        print(
            f"{definition.label}: p={p_value:.6f}, "
            f"significance={significance_label(p_value)}"
        )


if __name__ == "__main__":
    main()
