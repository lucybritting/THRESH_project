
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make src/ importable regardless of how the test is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from utils.dataloader import (list_cohorts, load_continuous_values, load_fold,
                              load_cohort, load_merged_ranges)
from utils.pipeline import _impute_disc, CLASSIFIERS
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

SUMMARY_PATH = OUTPUT_DIR / "reports" / "knn_k_comparison.csv"


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

    print("KNN imputation k comparison")
    print(f"Feature set: disc_imp. Merge strategy: {BEST_MERGE_STRATEGY}.")
    print(f"k values: {KNN_KS}. Metrics are the mean over {N_FOLDS} folds.")
    print(f"Classifiers: {', '.join(CLASSIFIERS)}\n")

    for cohort in cohorts:
        print(f"Processing cohort {cohort}")
        cont_df = load_continuous_values(cohort)
        cohort_info = load_cohort(cohort).set_index("hadm_id")
        labels = cohort_info["label"]
        demo = cohort_info[["age", "gender"]]
        folds = [load_fold(cohort, f) for f in range(N_FOLDS)]
        merged_ranges = load_merged_ranges(cohort, BEST_MERGE_STRATEGY)

        for k in KNN_KS:
            print(f"  k = {k}...")
            per_clf = {name: [] for name in CLASSIFIERS}  # name -> list of per-fold metric dicts
            for train_split, test_split in folds:
                train_ids = [int(h) for h in train_split[:, 1]]
                test_ids = [int(h) for h in test_split[:, 1]]
                # disc_imp depends on the train split -> rebuild per fold
                disc_imp = _impute_disc(cont_df, train_split, demo, merged_ranges, k=k)
                X_train, X_test = disc_imp.reindex(train_ids), disc_imp.reindex(test_ids)
                y_train, y_test = labels.loc[train_ids], labels.loc[test_ids]

                for name, factory in CLASSIFIERS.items():
                    model = factory().fit(X_train, y_train)  # fresh instance each fold
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                    per_clf[name].append(_score(y_test, y_prob, y_pred))

            # average each metric over folds, one row per classifier
            for name in CLASSIFIERS:
                mean_metrics = {m: float(np.mean([f[m] for f in per_clf[name]])) for m in METRICS}
                row = {"cohort": cohort, "k": k, "classifier": name}
                row.update({m: round(mean_metrics[m], 4) for m in METRICS})
                rows.append(row)

    df = pd.DataFrame(rows)

    # averaged over cohorts: one row per (k, classifier) to compare k values
    summary = (df.groupby(["k", "classifier"], sort=False)[METRICS]
                 .mean()
                 .round(4)
                 .reset_index())
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nReport written.")


if __name__ == "__main__":
    main()