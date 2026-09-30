import numpy as np
import pandas as pd

from config import (
    ACTION_UNIT_KEYS,
    ASYMMETRY_DEVIATION_KEYS,
    ASYMMETRY_SIDE_KEYS,
    EXPRESSIONS,
    NUM_ACTION_UNITS,
    NUM_ASYMMETRY_FEATURES,
    NUM_EXPRESSIONS,
)
from util import load_obj, parse_experiment_args

DATA_PATH = "AU_asymmetry_data.pkl"
OUTPUT_PATH = "data_statistic.xlsx"
NUM_CLASSES = 3


def main(argv=None):
    args = parse_experiment_args(
        "Export class-wise feature summaries.",
        {"data_path": DATA_PATH, "output_path": OUTPUT_PATH},
        argv,
    )
    data = load_obj(args.data_path)
    subjects = list(data.keys())
    X_au = np.zeros((len(subjects), NUM_EXPRESSIONS, NUM_ACTION_UNITS))
    X_fa = np.zeros((len(subjects), NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES))
    y = np.zeros((len(subjects), 2, NUM_CLASSES))

    for subject_index, name in enumerate(subjects):
        y[subject_index] = data[name]["label"]
        X_au[subject_index] = np.array([
            [data[name][expression]["Action Unit"][key] for key in ACTION_UNIT_KEYS]
            for expression in EXPRESSIONS
        ])
        X_fa[subject_index] = np.array([
            [data[name][expression]["Facial Asymmetry"]["Deviation"][key]
             for key in ASYMMETRY_DEVIATION_KEYS]
            + [data[name][expression]["Facial Asymmetry"]["Left"][key]
               for key in ASYMMETRY_SIDE_KEYS]
            + [data[name][expression]["Facial Asymmetry"]["Right"][key]
               for key in ASYMMETRY_SIDE_KEYS]
            for expression in EXPRESSIONS
        ])

    severity_exp_fa = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES))
    severity_exp_au = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ACTION_UNITS))
    asymmetry_exp_fa = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES))
    asymmetry_exp_au = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ACTION_UNITS))
    count_severity = np.zeros(NUM_CLASSES)
    count_asymmetry = np.zeros(NUM_CLASSES)

    for subject_index in range(len(subjects)):
        severity_class = np.argmax(y[subject_index, 0])
        asymmetry_class = np.argmax(y[subject_index, 1])
        severity_exp_fa[severity_class] += X_fa[subject_index]
        severity_exp_au[severity_class] += X_au[subject_index]
        asymmetry_exp_fa[asymmetry_class] += X_fa[subject_index]
        asymmetry_exp_au[asymmetry_class] += X_au[subject_index]
        count_severity[severity_class] += 1
        count_asymmetry[asymmetry_class] += 1

    for class_index in range(NUM_CLASSES):
        severity_exp_fa[class_index] /= count_severity[class_index]
        severity_exp_au[class_index] /= count_severity[class_index]
        asymmetry_exp_fa[class_index] /= count_asymmetry[class_index]
        asymmetry_exp_au[class_index] /= count_asymmetry[class_index]

    severity_au = np.mean(severity_exp_au, axis=1)
    severity_fa = np.mean(severity_exp_fa, axis=1)
    severity_exp = np.mean(severity_exp_fa, axis=2) + np.mean(severity_exp_au, axis=2)
    asymmetry_au = np.mean(asymmetry_exp_au, axis=1)
    asymmetry_fa = np.mean(asymmetry_exp_fa, axis=1)
    asymmetry_exp = np.mean(asymmetry_exp_fa, axis=2) + np.mean(asymmetry_exp_au, axis=2)

    severity_au_relative = severity_au[0] - severity_au
    severity_fa_relative = severity_fa[0] - severity_fa
    severity_exp_relative = severity_exp[0] - severity_exp
    asymmetry_au_relative = asymmetry_au[0] - asymmetry_au
    asymmetry_fa_relative = asymmetry_fa[0] - asymmetry_fa
    asymmetry_exp_relative = asymmetry_exp[0] - asymmetry_exp

    with pd.ExcelWriter(args.output_path) as writer:
        pd.DataFrame(severity_au).to_excel(writer, sheet_name="Severity AU")
        pd.DataFrame(severity_fa).to_excel(writer, sheet_name="Severity FA")
        pd.DataFrame(severity_exp).to_excel(writer, sheet_name="Severity EXP")
        pd.DataFrame(asymmetry_au).to_excel(writer, sheet_name="Asymmetry AU")
        pd.DataFrame(asymmetry_fa).to_excel(writer, sheet_name="Asymmetry FA")
        pd.DataFrame(asymmetry_exp).to_excel(writer, sheet_name="Asymmetry EXP")
        pd.DataFrame(severity_au_relative).to_excel(writer, sheet_name="Severity AU vs Absence")
        pd.DataFrame(severity_fa_relative).to_excel(writer, sheet_name="Severity FA vs Absence")
        pd.DataFrame(severity_exp_relative).to_excel(writer, sheet_name="Severity EXP vs Absence")
        pd.DataFrame(asymmetry_au_relative).to_excel(writer, sheet_name="Asymmetry AU vs Non-Asym")
        pd.DataFrame(asymmetry_fa_relative).to_excel(writer, sheet_name="Asymmetry FA vs Non-Asym")
        pd.DataFrame(asymmetry_exp_relative).to_excel(writer, sheet_name="Asymmetry EXP vs Non-Asym")


if __name__ == "__main__":
    main()
