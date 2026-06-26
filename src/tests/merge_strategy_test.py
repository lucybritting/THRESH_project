
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# make src/ importable regardless of how the test is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils.dataloader import (list_cohorts, load_continuous_values, load_fold,
                              load_cohort, load_merged_ranges)
from utils.pipeline import _discretise, CLASSIFIERS
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

SUMMARY_PATH = OUTPUT_DIR / "reports" / "merge_strategy_comparison.csv"
CHANGE_PATH = OUTPUT_DIR / "reports" / "merge_strategy_disc_change.csv"

# columns produced by _disc_change (used for grouping/averaging)
CHANGE_COLS = ["comparable_cells", "pct_changed",
               "pct_within_to_out", "pct_out_to_within"]


def _disc_change(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """
    Cell-level comparison of two discretised frames (identical layout).

    Looks at cells that are discretised (non-NaN) under BOTH strategies and reports,
    as a percentage of those comparable cells, how the -1/0/1 category changes:
      - pct_within_to_out : was within (0)   -> below/above (+-1)  (range tightened)
      - pct_out_to_within : was below/above  -> within (0)         (range widened)
    pct_changed counts every category change (incl. rare below<->above sign flips).
    """
    av = a.to_numpy().ravel()
    bv = b.to_numpy().ravel()
    both = ~np.isnan(av) & ~np.isnan(bv)
    av, bv = av[both], bv[both]
    n = av.size
    if n == 0:
        return {c: 0.0 for c in CHANGE_COLS}
    within_to_out = np.count_nonzero((av == 0) & (bv != 0))
    out_to_within = np.count_nonzero((av != 0) & (bv == 0))
    sign_flip = np.count_nonzero((av != 0) & (bv != 0) & (av != bv))
    changed = within_to_out + out_to_within + sign_flip
    return {
        "comparable_cells": float(n),
        "pct_changed": 100 * changed / n,
        "pct_within_to_out": 100 * within_to_out / n,
        "pct_out_to_within": 100 * out_to_within / n,
    }


def _score(y_test, y_prob, y_pred) -> dict:
    """All five metrics for one fold."""
    return {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "avg_prec": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
    }


def main():
    cohorts = list_cohorts()
    rows = []
    change_rows = []  # per-cohort cell-level disc comparison between strategy pairs

    print("Range-merge strategy comparison")
    print(f"Feature set: disc (no imputation). Metrics are the mean over {N_FOLDS} folds.")
    print(f"Classifiers: {', '.join(CLASSIFIERS)}\n")

    for cohort in cohorts:
        print(f"Processing cohort {cohort}")
        cont_df = load_continuous_values(cohort)
        labels = load_cohort(cohort).set_index("hadm_id")["label"]
        folds = [load_fold(cohort, f) for f in range(N_FOLDS)]
        disc_by_strategy = {}  # strategy -> discretised frame, for the cell-level comparison

        for strategy in ALL_MERGE_STRATEGIES:
            print(f"  strategy: {strategy}...")
            merged_ranges = load_merged_ranges(cohort, strategy)
            # disc is fold-independent -> build once per (cohort, strategy)
            disc = _discretise(cont_df, merged_ranges)  # wide, indexed by hadm_id
            disc_by_strategy[strategy] = disc

            per_clf = {name: [] for name in CLASSIFIERS}  # name -> list of per-fold metric dicts
            for train_split, test_split in folds:
                train_ids = [int(h) for h in train_split[:, 1]]
                test_ids = [int(h) for h in test_split[:, 1]]
                X_train, X_test = disc.reindex(train_ids), disc.reindex(test_ids)
                y_train, y_test = labels.loc[train_ids], labels.loc[test_ids]

                for name, factory in CLASSIFIERS.items():
                    model = factory().fit(X_train, y_train)  # fresh instance each fold
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                    per_clf[name].append(_score(y_test, y_prob, y_pred))

            # average each metric over folds, one row per classifier
            for name in CLASSIFIERS:
                mean_metrics = {m: float(np.mean([f[m] for f in per_clf[name]])) for m in METRICS}
                row = {"cohort": cohort, "strategy": strategy, "classifier": name}
                row.update({m: round(mean_metrics[m], 4) for m in METRICS})
                rows.append(row)

        # cell-level disc comparison for each strategy pair (this cohort)
        for sa, sb in combinations(ALL_MERGE_STRATEGIES, 2):
            change = _disc_change(disc_by_strategy[sa], disc_by_strategy[sb])
            change_rows.append({"cohort": cohort, "strategy_a": sa, "strategy_b": sb, **change})

    df = pd.DataFrame(rows)

    # averaged over cohorts: one row per (strategy, classifier) to compare strategies
    summary = (df.groupby(["strategy", "classifier"], sort=False)[METRICS]
                 .mean()
                 .round(4)
                 .reset_index())
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    # disc-change comparison averaged over cohorts: one row per strategy pair
    change_df = pd.DataFrame(change_rows)
    change_summary = (change_df.groupby(["strategy_a", "strategy_b"], sort=False)[CHANGE_COLS]
                       .mean()
                       .round(4)
                       .reset_index())
    change_summary.to_csv(CHANGE_PATH, index=False)

    print(f"\nReports written.")


if __name__ == "__main__":
    main()