

import argparse

import numpy as np
import pandas as pd

from config import *
from data_io import (list_cohorts, load_labitem_labels, load_cohort, reports_path,
                     set_cohort_folder, continuous_mapping_path, merged_ranges_path)

REPORT_SECTION = "cohort_analysis"
PER_COHORT_PATH = reports_path(REPORT_SECTION) / "lab_coverage_per_cohort.csv"
AVERAGE_PATH = reports_path(REPORT_SECTION) / "lab_coverage.csv"
LABEL_PATH = reports_path(REPORT_SECTION) / "label_distribution.csv"

# the two coverages summarised per cohort
VALUE_COVERAGE = "value coverage"
RANGE_COVERAGE = "range coverage"

# the counts, per (cohort, lab), over that cohort's admissions
COUNT_COLS = ["n_admissions", "n_value", "n_no_value",
              "n_value_and_range", "n_value_no_range", "n_range_no_value"]


def _require_inputs(cohorts: list[str]) -> None:
    """
    Stop before doing any work if the scan and merge steps have not run, naming every
    missing cohort at once rather than dying on a FileNotFoundError partway through.
    """
    missing = [c for c in cohorts
               if not continuous_mapping_path(c).exists()
               or not merged_ranges_path(c, BEST_MERGE_STRATEGY).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(cohorts)} cohorts have no continuous mapping or no "
            f"{BEST_MERGE_STRATEGY} merged ranges: "
            f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}\n"
            f"Run main.py on this cohort folder first; this script builds nothing itself.")


def _cohort_coverage(cohort: str) -> pd.DataFrame:
    """
    One row per lab for this cohort, counting its admissions five ways.

    A lab is a column of the continuous mapping, so the labs are already the ones that
    survived the blacklist. "Has a range" means the merged range for that (admission, lab)
    carries at least one bound: a pair with both bounds None was recorded but is unusable,
    and counting it as ranged would overstate what can be discretised.
    """
    values = pd.read_csv(continuous_mapping_path(cohort)).set_index("hadm_id")
    values.columns = values.columns.astype(int)
    has_value = values.notna()

    ranges = pd.read_csv(merged_ranges_path(cohort, BEST_MERGE_STRATEGY))
    ranges = ranges[ranges["ref_range_lower"].notna() | ranges["ref_range_upper"].notna()]
    # crosstab then reindex, so labs and admissions with no range at all become False
    # rather than dropping out of the frame and misaligning the counts
    has_range = (pd.crosstab(ranges["hadm_id"], ranges["itemid"]) > 0).reindex(
        index=values.index, columns=values.columns, fill_value=False).astype(bool)

    return pd.DataFrame({
        "cohort": cohort,
        "n_admissions": len(values),
        "n_value": has_value.sum(axis=0),
        "n_no_value": (~has_value).sum(axis=0),
        "n_value_and_range": (has_value & has_range).sum(axis=0),
        "n_value_no_range": (has_value & ~has_range).sum(axis=0),
        "n_range_no_value": (~has_value & has_range).sum(axis=0),
    }).rename_axis("itemid").reset_index()


def _cohort_labels(cohort: str) -> dict:
    """
    What this cohort is asked to predict: its size and how much of it is positive.

    Subjects as well as admissions, because the folds are grouped by subject: a cohort with
    far fewer subjects than admissions has repeated patients, and its effective sample size
    is the smaller number.
    """
    cohort_df = load_cohort(cohort)
    n, n_positive = len(cohort_df), int(cohort_df["label"].sum())
    return {"cohort": cohort, "n": n, "n_positive": n_positive,
            "n_negative": n - n_positive, "n_subjects": cohort_df["subject_id"].nunique(),
            # a fraction, as the pipeline's own fairness outputs report it, not a percentage
            "prevalence": round(n_positive / n, 6)}


def _describe(series: pd.Series, name: str, decimals: int) -> str:
    """One row of the printed summaries: a named series as five statistics."""
    return (f"   {name:22}" + "".join(f"{value:>12,.{decimals}f}" for value in
            (series.mean(), series.median(), series.std(), series.min(), series.max())))


def _label_summary(label_df: pd.DataFrame) -> None:
    """
    The label side, described over the cohorts: how big the positive class is, and which
    cohorts are too thin to train on.
    """
    overall = label_df.set_index("cohort")

    print(f"\nLabel distribution, across {len(overall)} cohorts:")
    print(f"   {'':22}{'mean':>12}{'median':>12}{'std':>12}{'min':>12}{'max':>12}")
    print(_describe(overall["n"], "admissions", 1))
    print(_describe(overall["n_subjects"], "subjects", 1))
    print(_describe(overall["n_positive"], "positives", 1))
    print(_describe(100 * overall["prevalence"], "prevalence %", 2))
    print("   subjects:            distinct patients; folds are grouped by subject, so a "
          "cohort with\n                        fewer subjects than admissions repeats "
          "patients across its rows.")


def main(cohort_folder: str) -> None:
    set_cohort_folder(cohort_folder)
    cohorts = list_cohorts()
    _require_inputs(cohorts)
    labels = load_labitem_labels()

    print(f"Lab coverage across the '{cohort_folder}' cohort folder ({len(cohorts)} cohorts).")
    print(f"Ranges from the {BEST_MERGE_STRATEGY} merge strategy. Counts are over "
          f"(admission, lab) pairs.\n")

    frames, label_rows = [], []
    for index, cohort in enumerate(cohorts, start=1):
        print(f"[{index}/{len(cohorts)}] {cohort}", flush=True)
        frames.append(_cohort_coverage(cohort))
        label_rows.append(_cohort_labels(cohort))

    per_cohort = pd.concat(frames, ignore_index=True)
    per_cohort.insert(2, "label", [labels.get(i, i) for i in per_cohort["itemid"]])
    PER_COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_cohort.to_csv(PER_COHORT_PATH, index=False)

    average = per_cohort.groupby("itemid")[COUNT_COLS].mean().round(1)
    average.insert(0, "label", [labels.get(i, i) for i in average.index])
    average.insert(1, "n_cohorts", per_cohort.groupby("itemid").size())
    # share of admissions that have the lab at all, and share of those whose value is usable
    # by the discretised representation. The second is the ceiling on what discretising can
    # reach for this lab.
    average["pct_measured"] = (100 * average["n_value"] / average["n_admissions"]).round(1)
    average["pct_of_measured_with_range"] = (
        100 * average["n_value_and_range"] / average["n_value"].replace(0, np.nan)).round(1)
    average.sort_values("n_value", ascending=False).to_csv(AVERAGE_PATH)

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(LABEL_PATH, index=False)

    total_value = average["n_value"].sum()
    total_gap = average["n_value_no_range"].sum()
    print(f"\n{len(average)} labs over {len(cohorts)} cohorts, "
          f"{per_cohort['n_admissions'].groupby(per_cohort['cohort']).first().sum():,} admissions.")
    print(f"Per cohort, averaged: {total_value:,.0f} measured (admission, lab) pairs, of which "
          f"{total_gap:,.0f} ({100 * total_gap / total_value:.2f}%) carry no usable range and "
          f"so cannot be discretised.")
    print(f"{average['n_range_no_value'].sum():,.0f} pairs have a range but no numeric value, "
          f"which no representation can use.")

    # PER-COHORT SUMMARY. Size and the two coverages, described over the cohorts rather
    # than plotted: cohort size and a percentage share no useful axis, and four statistics
    # each say what a figure would.
    by_cohort = per_cohort.groupby("cohort", sort=False).agg(
        n_admissions=("n_admissions", "first"),
        n_labs=("itemid", "size"),
        n_value=("n_value", "sum"),
        n_value_and_range=("n_value_and_range", "sum"),
        n_value_no_range=("n_value_no_range", "sum"))
    coverage = pd.DataFrame({
        VALUE_COVERAGE: 100 * by_cohort["n_value"] / (by_cohort["n_admissions"]
                                                      * by_cohort["n_labs"]),
        RANGE_COVERAGE: 100 * by_cohort["n_value_and_range"] / by_cohort["n_value"],
    }).sort_values(VALUE_COVERAGE, ascending=False)

    print(f"\nPer cohort, across {len(by_cohort)} cohorts:")
    print(f"   {'':22}{'mean':>12}{'median':>12}{'std':>12}{'min':>12}{'max':>12}")
    print(_describe(by_cohort["n_admissions"], "admissions", 1))
    print(_describe(coverage[VALUE_COVERAGE], "value coverage %", 2))
    print(_describe(coverage[RANGE_COVERAGE], "range coverage %", 2))
    print(_describe(by_cohort["n_value_no_range"], "value, no range", 1))
    print("   admissions:          samples in the cohort, the rows of its feature matrix.")
    print("   value coverage:      share of (admission, lab) cells holding a measured value.")
    print("   range coverage:      share of those measured cells with a usable range.")
    print("   value, no range:     measured cells the discretised representation cannot use.")

    _label_summary(label_df)

    print(f"\nWrote {len(per_cohort):,} per-cohort rows, {len(average)} averaged rows and "
          f"{len(label_df):,} label rows to {reports_path(REPORT_SECTION)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort_folder", choices=list(COHORT_FOLDERS), required=True,
                        help="which cohort folder to report on, as in main.py")
    main(parser.parse_args().cohort_folder)
