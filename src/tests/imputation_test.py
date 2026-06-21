
import sys
from pathlib import Path

import pandas as pd

# make src/ importable regardless of how the test is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils.dataloader import (list_cohorts, load_continuous_values, load_fold,
                              load_cohort, load_merged_ranges)
from utils.pipeline import _knn_impute_ranges

REPORT_PATH = OUTPUT_DIR / "reports" / "imputation_report.csv"


def _measured_pairs(cont_df):
    """Long table of (hadm_id, itemid) pairs that have a measured value."""

    lab_cols = [c for c in cont_df.columns if c != "hadm_id"]
    measured = cont_df.melt(id_vars="hadm_id", value_vars=lab_cols,
                            var_name="itemid", value_name="value")
    measured["itemid"] = measured["itemid"].astype(int)
    measured["hadm_id"] = measured["hadm_id"].astype(int)
    return measured.dropna(subset=["value"])  # only pairs with a value


def main():
    cohorts = list_cohorts()
    rows = []

    print("Imputation coverage report")
    print(f"KNN imputation uses k = {BEST_KNN} nearest donor(s).")
    print(f"Mapping: {BEST_MERGE_STRATEGY} (best merge strategy).")
    print(f"Counts are over (hadm_id, itemid) pairs that have a measured value.")
    print(f"'imputable' is the mean over {N_FOLDS} folds (depends on the training split).\n")

    for cohort in cohorts:
        print(f"Processing cohort {cohort}")
        cont_df = load_continuous_values(cohort)
        demo = load_cohort(cohort).set_index("hadm_id")[["age", "gender"]]
        measured = _measured_pairs(cont_df)

        # load the fold training splits once
        train_splits = [load_fold(cohort, f)[0] for f in range(N_FOLDS)]

        merged_ranges = load_merged_ranges(cohort, BEST_MERGE_STRATEGY)
        # pairs that already have at least one range bound
        real_ranges = {k: v for k, v in merged_ranges.items()
                       if v[0] is not None or v[1] is not None}
        real_keys = set(real_ranges.keys())

        # split measured pairs into those with / without a range
        has_range = [(h, i) in real_keys
                     for h, i in zip(measured["hadm_id"], measured["itemid"])]
        value_and_range = int(sum(has_range))
        targets_all = measured.loc[[not b for b in has_range], ["hadm_id", "itemid"]]
        value_no_range = len(targets_all)

        # attach demographics once; donors are chosen per fold from the train split
        targets = targets_all.merge(demo, left_on="hadm_id", right_index=True)

        imputable_per_fold = []
        for idx, train_split in enumerate(train_splits):
            print(f"Fold {idx + 1}...")
            train_hadms = {int(h) for h in train_split[:, 1]}
            imputed = _knn_impute_ranges(real_ranges, train_hadms, demo,
                                         targets, BEST_KNN)
            imputable_per_fold.append(len(imputed))

        imputable_mean = (sum(imputable_per_fold) / len(imputable_per_fold)
                          if imputable_per_fold else 0)
        n_measured = len(measured)
        value_and_range_pct = (100 * value_and_range / n_measured) if n_measured else 0.0
        value_no_range_pct = (100 * value_no_range / n_measured) if n_measured else 0.0
        imputable_pct_of_measured = (100 * imputable_mean / n_measured) if n_measured else 0.0
        imputable_pct_of_no_range = (100 * imputable_mean / value_no_range) if value_no_range else 0.0

        row = {
            "cohort": cohort,
            "knn_k": BEST_KNN,
            "measured_pairs": n_measured,
            "value_and_range": value_and_range,
            "value_and_range_pct": round(value_and_range_pct, 1),
            "value_no_range": value_no_range,
            "value_no_range_pct": round(value_no_range_pct, 1),
            "imputable_mean": round(imputable_mean, 1),
            "imputable_pct_of_measured": round(imputable_pct_of_measured, 1),
            "imputable_pct_of_no_range": round(imputable_pct_of_no_range, 1),
        }
        rows.append(row)


    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)
    print(f"\nReport written.")


if __name__ == "__main__":
    main()