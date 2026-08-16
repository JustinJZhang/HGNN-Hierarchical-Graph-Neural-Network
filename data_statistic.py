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
from util import load_obj

DATA_PATH = "AU_asymmetry_data.pkl"
OUTPUT_PATH = "data_statistic.xlsx"
NUM_CLASSES = 3


def main():
    """Aggregate labels [N,2,3], AU [N,8,17], and FA [N,8,24] into class means."""
    data = load_obj(DATA_PATH)
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

    score_exp_fa = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES))
    score_exp_au = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ACTION_UNITS))
    side_exp_fa = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ASYMMETRY_FEATURES))
    side_exp_au = np.zeros((NUM_CLASSES, NUM_EXPRESSIONS, NUM_ACTION_UNITS))
    count_score = np.zeros(NUM_CLASSES)
    count_side = np.zeros(NUM_CLASSES)

    for subject_index in range(len(subjects)):
        score_class = np.argmax(y[subject_index, 0])
        side_class = np.argmax(y[subject_index, 1])
        score_exp_fa[score_class] += X_fa[subject_index]
        score_exp_au[score_class] += X_au[subject_index]
        side_exp_fa[side_class] += X_fa[subject_index]
        side_exp_au[side_class] += X_au[subject_index]
        count_score[score_class] += 1
        count_side[side_class] += 1

    for class_index in range(NUM_CLASSES):
        score_exp_fa[class_index] /= count_score[class_index]
        score_exp_au[class_index] /= count_score[class_index]
        side_exp_fa[class_index] /= count_side[class_index]
        side_exp_au[class_index] /= count_side[class_index]

    score_au = np.mean(score_exp_au, axis=1)
    score_fa = np.mean(score_exp_fa, axis=1)
    score_exp = np.mean(score_exp_fa, axis=2) + np.mean(score_exp_au, axis=2)
    side_au = np.mean(side_exp_au, axis=1)
    side_fa = np.mean(side_exp_fa, axis=1)
    side_exp = np.mean(side_exp_fa, axis=2) + np.mean(side_exp_au, axis=2)

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        pd.DataFrame(score_au).to_excel(writer, sheet_name="Score AU")
        pd.DataFrame(score_fa).to_excel(writer, sheet_name="Score FA")
        pd.DataFrame(score_exp).to_excel(writer, sheet_name="Score EXP")
        pd.DataFrame(side_au).to_excel(writer, sheet_name="Side AU")
        pd.DataFrame(side_fa).to_excel(writer, sheet_name="Side FA")
        pd.DataFrame(side_exp).to_excel(writer, sheet_name="Side EXP")


if __name__ == "__main__":
    main()
