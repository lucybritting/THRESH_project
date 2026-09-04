

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# make src/ importable regardless of how the test is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from data_io import (list_cohorts, load_continuous_values, load_discrete_values,
                              load_fold, load_cohort, load_merged_ranges, reports_path,
                              set_cohort_folder, continuous_mapping_path, ranges_path)
from prep.folds import run_fold_step
from prep.mapping import run_mapping_step
from prep.ranges import run_merge_step
from prep.imputation import _impute_disc
from model.classifiers import CLASSIFIERS
from plots import (plot_representation_boxplot, SERIES, UNCHANGED, LOST, GAINED)
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

REPORT_SECTION = "test_knn_k"
RESULTS_PATH = reports_path(REPORT_SECTION) / "knn_k_results.csv"
SUMMARY_PATH = reports_path(REPORT_SECTION) / "knn_k_comparison.csv"
CHANGE_PATH = reports_path(REPORT_SECTION) / "knn_k_change.csv"

# the representations trained
REFERENCE = "disc"
REPRESENTATIONS = [REFERENCE] + [f"disc_imp_k{k}" for k in KNN_KS]
REPRESENTATION_TITLES = {REFERENCE: "discretised (no imputation)",
                         **{f"disc_imp_k{k}": f"imputed, k = {k}" for k in KNN_KS}}

KNN_COLOURS = {REFERENCE: UNCHANGED,
               **{f"disc_imp_k{k}": colour
                  for k, colour in zip(KNN_KS, ["#2a78d6", "#008300", "#eda100"])}}

# columns produced by _disc_change
CHANGE_COLS = ["comparable_cells", "pct_changed",
               "pct_within_to_out", "pct_out_to_within"]
# Only the total is plotted
CHANGE_MEASURES = ["pct_changed"]
CHANGE_TITLES = {"pct_changed": "changed category"}
CHANGE_COLOURS = {"pct_changed": UNCHANGED}


def _disc_change(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """
    Cell-level comparison of two discretised frames (identical layout). Kept in step with
    the copy in merge_strategy_test.py; both scripts are standalone experiments.

    Looks at cells discretised (non-NaN) under BOTH frames and reports, as a percentage of
    those comparable cells, how the -1/0/1 category changes.
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
    return {
        "comparable_cells": float(n),
        "pct_changed": 100 * (within_to_out + out_to_within + sign_flip) / n,
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


def _require_scan(cohorts: list[str]) -> None:
    """
    Stop before doing any work if the labevents scan has not produced what the merge and
    mapping steps read
    """
    missing = [c for c in cohorts
               if not continuous_mapping_path(c).exists() or not ranges_path(c).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(cohorts)} cohorts have no continuous mapping or ranges "
            f"file: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}\n"
            f"Run main.py on this cohort folder first; this script does not scan labevents.")


def main(cohort_folder: str) -> None:
    set_cohort_folder(cohort_folder)
    cohorts = list_cohorts()
    _require_scan(cohorts)

    classifier = CLASSIFIERS[BEST_CLASSIFIER]
    print(f"KNN imputation k comparison over the '{cohort_folder}' cohort folder "
          f"({len(cohorts)} cohorts).")
    print(f"k values: {KNN_KS}. Merge strategy held at {BEST_MERGE_STRATEGY}.")
    print(f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}. "
          f"Metrics are per fold over {N_FOLDS} folds.\n")

    rows = []          # one per (cohort, representation, fold)
    change_rows = []   # one per (cohort, fold, k pair)

    for index, cohort in enumerate(cohorts, start=1):
        print(f"[{index}/{len(cohorts)}] {cohort}", flush=True)
        # both steps skip whatever already exists, so a rerun costs nothing
        run_merge_step(cohort)
        run_mapping_step(cohort)
        run_fold_step(cohort)

        cont_df = load_continuous_values(cohort)
        cohort_info = load_cohort(cohort).set_index("hadm_id")
        labels, demo = cohort_info["label"], cohort_info[["age", "gender"]]
        merged_ranges = load_merged_ranges(cohort, BEST_MERGE_STRATEGY)
        # the unimputed reference is fold-independent, it is built once per cohort
        reference_frame = load_discrete_values(cohort, BEST_MERGE_STRATEGY).set_index("hadm_id")

        for fold_idx in range(N_FOLDS):
            train_split, test_split = load_fold(cohort, fold_idx)
            train_ids = [int(h) for h in train_split[:, 1]]
            test_ids = [int(h) for h in test_split[:, 1]]
            y_train, y_test = labels.loc[train_ids], labels.loc[test_ids]

            # new imputation per fold and per k
            frames = {REFERENCE: reference_frame}
            for k in KNN_KS:
                print(f"    {cohort} | fold {fold_idx + 1}/{N_FOLDS} | imputing k = {k}",
                      flush=True)
                frames[f"disc_imp_k{k}"] = _impute_disc(cont_df, train_split, demo,
                                                        merged_ranges, k=k)

            # how far apart the k values put the data
            for k_a, k_b in combinations(KNN_KS, 2):
                change = _disc_change(frames[f"disc_imp_k{k_a}"], frames[f"disc_imp_k{k_b}"])
                change_rows.append({"cohort": cohort, "fold": fold_idx,
                                    "k_a": k_a, "k_b": k_b, **change})

            for representation in REPRESENTATIONS:
                print(f"    {cohort} | fold {fold_idx + 1}/{N_FOLDS} | training "
                      f"{REPRESENTATION_TITLES[representation]}", flush=True)
                frame = frames[representation]
                model = classifier().fit(frame.reindex(train_ids), y_train)  # fresh each fold
                X_test = frame.reindex(test_ids)
                metrics = _score(y_test, model.predict_proba(X_test)[:, 1],
                                 model.predict(X_test))
                rows.append({"cohort": cohort, "representation": representation,
                             "fold": fold_idx, **{m: round(metrics[m], 4) for m in METRICS}})

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    # per cohort, averaged over the folds
    per_cohort = (results.groupby(["cohort", "representation"], sort=False)[METRICS]
                  .mean().round(4).reset_index())
    per_cohort.to_csv(SUMMARY_PATH, index=False)

    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(CHANGE_PATH, index=False)

    # one box per representation, one dot per cohort
    wide = per_cohort.pivot(index="cohort", columns="representation", values=METRICS)
    for measure in METRICS:
        plot_representation_boxplot(
            wide[measure].reindex(columns=REPRESENTATIONS),
            groups=None, group_order=[],
            series=[(r, REPRESENTATION_TITLES[r]) for r in REPRESENTATIONS],
            colours=KNN_COLOURS,
            title=f"{METRIC_LABELS[measure]} by imputation neighbour count",
            subtitle=f"One dot per cohort, averaged over the {N_FOLDS} folds, for different k "
                     f"and without imputation. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}, merge "
                     f"strategy {BEST_MERGE_STRATEGY}, {cohort_folder} cohort folder "
                     f"({len(cohorts)} cohorts).",
            ylabel=METRIC_LABELS[measure],
            path=reports_path(REPORT_SECTION) / f"knn_k_boxplot_{measure}.png")


    difference = wide.copy()
    for measure in METRICS:
        for representation in REPRESENTATIONS:
            difference[(measure, representation)] = (wide[(measure, representation)]
                                                     - wide[(measure, REFERENCE)])
    imputed_reps = [r for r in REPRESENTATIONS if r != REFERENCE]  # reference is 0 throughout
    stacked = pd.concat(
        [difference[measure][imputed_reps].assign(measure=METRIC_LABELS[measure])
         for measure in METRICS])
    measure_of = stacked.pop("measure")
    stacked.index = [f"{cohort} · {measure}"
                     for cohort, measure in zip(stacked.index, measure_of)]
    plot_representation_boxplot(
        stacked, groups=pd.Series(list(measure_of), index=stacked.index),
        group_order=[METRIC_LABELS[m] for m in METRICS],
        series=[(r, REPRESENTATION_TITLES[r]) for r in imputed_reps],
        colours=KNN_COLOURS,
        title="Difference of performance between imputed and unimputed discretisation",
        subtitle=f"Each imputed representation minus the unimputed discretised one on the "
                 f"same cohort, averaged over the {N_FOLDS} folds. Above 0 imputation "
                 f"increases performance, below 0 it decreases performance. One dot per "
                 f"cohort. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}, {cohort_folder} cohort "
                 f"folder ({len(cohorts)} cohorts).",
        ylabel="difference from the unimputed representation",
        path=reports_path(REPORT_SECTION) / "knn_k_difference_boxplot.png",
        hline=(0.0, ""))

    # how far apart the k values put the data itself, averaged over the folds
    change_df["pair"] = "k=" + change_df["k_a"].astype(str) + " vs k=" + change_df["k_b"].astype(str)
    pair_order = [f"k={a} vs k={b}" for a, b in combinations(KNN_KS, 2)]
    per_cohort_change = (change_df.groupby(["cohort", "pair"], sort=False)[CHANGE_MEASURES]
                         .mean().reset_index())
    mean_changed = per_cohort_change.groupby("pair", sort=False)["pct_changed"].mean()

    change_frame = per_cohort_change[CHANGE_MEASURES].copy()
    change_frame.index = [f"{cohort} · {pair}" for cohort, pair
                          in zip(per_cohort_change["cohort"], per_cohort_change["pair"])]
    plot_representation_boxplot(
        change_frame,
        groups=pd.Series(list(per_cohort_change["pair"]), index=change_frame.index),
        group_order=pair_order,
        series=[(m, CHANGE_TITLES[m]) for m in CHANGE_MEASURES],
        colours=CHANGE_COLOURS,
        title="Data change by imputation",
        subtitle="Percentage of the cells imputed under both k values whose -1/0/1 category "
                 f"differs, averaged over the {N_FOLDS} folds. One dot per cohort. Mean "
                 "changed: "
                 + "; ".join(f"{pair} {mean_changed[pair]:.3f}%" for pair in pair_order)
                 + f". '{cohort_folder}' cohort folder ({len(cohorts)} cohorts).",
        ylabel="percentage of comparable cells",
        path=reports_path(REPORT_SECTION) / "knn_k_change_boxplot.png")

    print("\nMean percentage of comparable cells that change category, across cohorts:")
    for pair in pair_order:
        rows_for_pair = per_cohort_change[per_cohort_change["pair"] == pair]
        print(f"   {pair:18} {rows_for_pair['pct_changed'].mean():7.4f}%  "
              f"(worst cohort {rows_for_pair['pct_changed'].max():.3f}%)")

    print(f"\nWrote {len(results)} result rows, {len(per_cohort)} per-cohort rows, "
          f"{len(change_df)} change rows and {len(METRICS) + 2} box plots to "
          f"{reports_path(REPORT_SECTION)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_folder", choices=list(COHORT_FOLDERS), required=True,
                        help="which cohort folder to run over, as in main.py")
    main(parser.parse_args().cohort_folder)
