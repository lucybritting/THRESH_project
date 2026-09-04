"""Every path the pipeline reads or writes, and the cohort folder state."""

import shutil
from pathlib import Path
from typing import Tuple

from config.paths import DATA_DIR, OUTPUT_DIR, COHORT_FOLDERS


# ---------- PATHS --------------------------------------
def analysis_path() -> Path:
    return OUTPUT_DIR / "mimic_analysis"


def blacklist_path() -> Path:
    return analysis_path() / "itemid_blacklist.txt"


def top100itemids_path() -> Path:
    return DATA_DIR / "top_features" / "all_mimic_top100_features_hadm.pkl"

# the cohort source folder, chosen at runtime from COHORT_FOLDERS (see config/paths.py)
_cohort_dir = DATA_DIR / COHORT_FOLDERS["all"]


def set_cohort_folder(which: str) -> None:
    global _cohort_dir
    _cohort_dir = DATA_DIR / COHORT_FOLDERS[which]


def cohort_path(cohort: str) -> Path:
    return _cohort_dir / f"{cohort}.csv.gz"


def binary_mapping_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "mapping" / f"{cohort}_binary.csv"


def discrete_mapping_path(cohort: str, strategy_name: str) -> Path:
    return OUTPUT_DIR / cohort / "mapping" / f"{cohort}_discrete_{strategy_name}.csv"


def continuous_mapping_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "mapping" / f"{cohort}_continuous.csv"


def ranges_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "ranges" / f"{cohort}_ranges.csv"


def merged_ranges_path(cohort: str, strategy_name: str) -> Path:
    return OUTPUT_DIR / cohort / "ranges" / f"{cohort}_ranges_{strategy_name}.csv"


def merge_warnings_path(cohort: str, strategy_name: str) -> Path:
    return OUTPUT_DIR / cohort / "ranges" / f"{cohort}_warnings_{strategy_name}.csv"


def fold_path(cohort: str, fold_idx: int) -> Path:
    return OUTPUT_DIR / cohort / "folds" / f"fold_{fold_idx}.pkl"

def metric_summary_path(cohort: str) -> Path:
  return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_performance_summary.csv"


def feature_importance_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_feature_importance.csv"


def feature_stability_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_feature_stability.csv"


def test_predictions_path(cohort: str, fold_idx: int) -> Path:
    return OUTPUT_DIR / cohort / "test_predictions" / f"{cohort}_test_predictions_fold_{fold_idx}.csv"


def fairness_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_fairness.csv"


def fairness_criteria_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_fairness_criteria.csv"


def feature_shift_path(cohort: str) -> Path:
    return OUTPUT_DIR / cohort / "metrics" / f"{cohort}_feature_shift.csv"


def figure_path(cohort: str, name: str) -> Path:
    return OUTPUT_DIR / cohort / "figures" / f"{cohort}_{name}.png"


def reports_path(section: str | None = None) -> Path:
    reports_dir = OUTPUT_DIR / "reports"
    return reports_dir if section is None else reports_dir / section


# ---------- CREATE DIRECTORIES -------------------------
def remove_directories(cohorts: list[str]) -> None:

    for cohort in cohorts:
        base = OUTPUT_DIR / cohort
        if base.exists():
            shutil.rmtree(base)
            print(f"Purged output directory for cohort {cohort}.")


def create_analysis_dir() -> Path:
    analysis_dir = analysis_path()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


REPORT_SECTIONS = ("performance", "fairness", "importance",
                   "test_range_merge", "test_knn_k", "cohort_analysis")


def create_reports_dir() -> Path:
    reports_dir = reports_path()
    for section in REPORT_SECTIONS:
        (reports_dir / section).mkdir(parents=True, exist_ok=True)
    return reports_dir


def create_output_directories(cohorts: list[str]) -> Tuple[list, list]:

    have_ranges = []
    need_ranges = []
    for cohort in cohorts:
        base = OUTPUT_DIR / cohort
        for subdir in ["ranges", "mapping", "folds", "metrics", "test_predictions"]:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        # check if range_file exists
        range_path = ranges_path(cohort)
        if range_path.exists():
            have_ranges.append(cohort)
        else:
            need_ranges.append(cohort)
    print(f"Output directories created for {len(cohorts)} cohorts.")
    return have_ranges, need_ranges


def list_cohorts() -> list[str]:
    """Cohort names in the selected cohort folder (file name without the .csv.gz suffix)."""
    return sorted(p.name.removesuffix(".csv.gz") for p in _cohort_dir.glob("*.csv.gz"))
