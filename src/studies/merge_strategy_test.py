

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
                              load_fold, load_cohort, reports_path, set_cohort_folder,
                              continuous_mapping_path, ranges_path)
from prep.folds import run_fold_step
from prep.mapping import run_mapping_step
from prep.ranges import run_merge_step
from model.classifiers import CLASSIFIERS
from plots import (plot_representation_boxplot, SERIES, REPRESENTATION_COLOUR,
                         UNCHANGED, LOST, GAINED)
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)


REPORT_SECTION = "test_range_merge"
RESULTS_PATH = reports_path(REPORT_SECTION) / "merge_strategy_results.csv"
SUMMARY_PATH = reports_path(REPORT_SECTION) / "merge_strategy_comparison.csv"
CHANGE_PATH = reports_path(REPORT_SECTION) / "merge_strategy_disc_change.csv"

# the representations trained, in plot order
CONTINUOUS = "cont"
REPRESENTATIONS = [CONTINUOUS] + [f"disc_{s}" for s in ALL_MERGE_STRATEGIES]
REPRESENTATION_TITLES = {CONTINUOUS: "continuous",
                         **{f"disc_{s}": f"discretised ({s})" for s in ALL_MERGE_STRATEGIES}}

STRATEGY_COLOURS = {CONTINUOUS: REPRESENTATION_COLOUR[CONTINUOUS],
                    **{f"disc_{s}": SERIES[i + 1]
                       for i, s in enumerate(ALL_MERGE_STRATEGIES)}}

# columns produced by _disc_change (used for grouping/averaging)
CHANGE_COLS = ["comparable_cells", "pct_changed",
               "pct_within_to_out", "pct_out_to_within"]

# Only one of the three change columns is ever non-zero
CHANGE_MEASURES = ["pct_within_to_out"]
CHANGE_TITLES = {"pct_within_to_out": "moved out of range when tightened"}
CHANGE_COLOURS = {"pct_within_to_out": LOST}


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


def _representation_frames(cohort: str) -> dict[str, pd.DataFrame]:
    """The continuous frame and one discretised frame per merge strategy, indexed by hadm_id."""
    frames = {CONTINUOUS: load_continuous_values(cohort).set_index("hadm_id")}
    for strategy in ALL_MERGE_STRATEGIES:
        frames[f"disc_{strategy}"] = load_discrete_values(cohort, strategy).set_index("hadm_id")
    return frames


def main(cohort_folder: str) -> None:
    set_cohort_folder(cohort_folder)
    cohorts = list_cohorts()
    _require_scan(cohorts)

    classifier = CLASSIFIERS[BEST_CLASSIFIER]
    print(f"Range-merge strategy comparison over the '{cohort_folder}' cohort folder "
          f"({len(cohorts)} cohorts).")
    print(f"Representations: {', '.join(REPRESENTATION_TITLES[r] for r in REPRESENTATIONS)}.")
    print(f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}. "
          f"Metrics are per fold over {N_FOLDS} folds.\n")

    rows = []          # one per (cohort, representation, fold)
    change_rows = []   # one per (cohort, strategy pair)

    for index, cohort in enumerate(cohorts, start=1):
        print(f"[{index}/{len(cohorts)}] {cohort}")
        # build the ranges and the discretised mapping for every strategy.
        run_merge_step(cohort, all_mappings=True)
        run_mapping_step(cohort, all_mappings=True)
        run_fold_step(cohort)

        frames = _representation_frames(cohort)
        labels = load_cohort(cohort).set_index("hadm_id")["label"]

        for fold_idx in range(N_FOLDS):
            train_split, test_split = load_fold(cohort, fold_idx)
            train_ids = [int(h) for h in train_split[:, 1]]
            test_ids = [int(h) for h in test_split[:, 1]]
            y_train, y_test = labels.loc[train_ids], labels.loc[test_ids]

            for representation in REPRESENTATIONS:
                # flushed, so a long run shows progress
                print(f"    {cohort} | fold {fold_idx + 1}/{N_FOLDS} | "
                      f"{REPRESENTATION_TITLES[representation]}", flush=True)
                frame = frames[representation]
                model = classifier().fit(frame.reindex(train_ids), y_train)  # fresh each fold
                X_test = frame.reindex(test_ids)
                metrics = _score(y_test, model.predict_proba(X_test)[:, 1],
                                 model.predict(X_test))
                rows.append({"cohort": cohort, "representation": representation,
                             "fold": fold_idx, **{m: round(metrics[m], 4) for m in METRICS}})

        # cell-level comparison of the discretised frames
        for strategy_a, strategy_b in combinations(ALL_MERGE_STRATEGIES, 2):
            change = _disc_change(frames[f"disc_{strategy_a}"], frames[f"disc_{strategy_b}"])
            change_rows.append({"cohort": cohort, "strategy_a": strategy_a,
                                "strategy_b": strategy_b, **change})

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
            colours=STRATEGY_COLOURS,
            title=f"{METRIC_LABELS[measure]} by range-merge strategy",
            subtitle=f"One dot per cohort, averaged over the {N_FOLDS} folds. Continuous is "
                     f"the same data before any thresholding, so the distance from it is what "
                     f"discretising costs. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}, "
                     f"'{cohort_folder}' cohort folder ({len(cohorts)} cohorts).",
            ylabel=METRIC_LABELS[measure],
            path=reports_path(REPORT_SECTION) / f"merge_strategy_boxplot_{measure}.png")

    # Every measure at once, as the distance from the continuous representation.
    difference = wide.copy()
    for measure in METRICS:
        for representation in REPRESENTATIONS:
            difference[(measure, representation)] = (wide[(measure, representation)]
                                                     - wide[(measure, CONTINUOUS)])
    strategies = [r for r in REPRESENTATIONS if r != CONTINUOUS]  # cont is 0 by construction
    stacked = pd.concat(
        [difference[measure][strategies].assign(measure=METRIC_LABELS[measure])
         for measure in METRICS])
    measure_of = stacked.pop("measure")
    stacked.index = [f"{cohort} · {measure}" for cohort, measure in zip(stacked.index, measure_of)]

    plot_representation_boxplot(
        stacked, groups=pd.Series(list(measure_of), index=stacked.index),
        group_order=[METRIC_LABELS[m] for m in METRICS],
        series=[(r, REPRESENTATION_TITLES[r]) for r in strategies],
        colours=STRATEGY_COLOURS,
        title="Cost of discretising, by measure and range-merge strategy",
        subtitle=f"Each discretised representation minus the continuous one on the same "
                 f"cohort, averaged over the {N_FOLDS} folds. Above 0 = better than "
                 f"continuous, below = worse. One dot per cohort. "
                 f"{CLASSIFIER_LABELS[BEST_CLASSIFIER]}, '{cohort_folder}' cohort folder "
                 f"({len(cohorts)} cohorts).",
        ylabel="difference from the continuous representation",
        path=reports_path(REPORT_SECTION) / "merge_strategy_difference_boxplot.png",
        hline=(0.0, "same as continuous"))


    change_df["pair"] = change_df["strategy_a"] + " vs " + change_df["strategy_b"]
    pair_order = [f"{a} vs {b}" for a, b in combinations(ALL_MERGE_STRATEGIES, 2)]
    mean_changed = change_df.groupby("pair", sort=False)["pct_changed"].mean()

    change_frame = change_df[CHANGE_MEASURES].copy()
    change_frame.index = [f"{cohort} · {pair}"
                          for cohort, pair in zip(change_df["cohort"], change_df["pair"])]
    plot_representation_boxplot(
        change_frame, groups=pd.Series(list(change_df["pair"]), index=change_frame.index),
        group_order=pair_order,
        series=[(m, CHANGE_TITLES[m]) for m in CHANGE_MEASURES],
        colours=CHANGE_COLOURS,
        title="How much the discretised data changes between range-merge strategies",
        subtitle="Of the cells discretised under both strategies, the percentage that were "
                 "within range under the wider one and fall outside it under the tighter. "
                 "Nothing moves the other way: the bands are nested, so tightening can only "
                 "push values out. One dot per cohort. Mean: "
                 + "; ".join(f"{pair} {mean_changed[pair]:.3f}%" for pair in pair_order)
                 + f". '{cohort_folder}' cohort folder ({len(cohorts)} cohorts).",
        ylabel="percentage of comparable cells",
        path=reports_path(REPORT_SECTION) / "merge_strategy_disc_change_boxplot.png")

    print("\nMean percentage of comparable cells pushed out of range by tightening:")
    for pair in pair_order:
        rows_for_pair = change_df[change_df["pair"] == pair]
        print(f"   {pair:24} {rows_for_pair['pct_within_to_out'].mean():7.4f}%  "
              f"(worst cohort {rows_for_pair['pct_within_to_out'].max():.3f}%)")

    print(f"\nWrote {len(results)} result rows, {len(per_cohort)} per-cohort rows and "
          f"{len(METRICS) + 2} box plots to {reports_path(REPORT_SECTION)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_folder", choices=list(COHORT_FOLDERS), required=True,
                        help="which cohort folder to run over, as in main.py")
    main(parser.parse_args().cohort_folder)
