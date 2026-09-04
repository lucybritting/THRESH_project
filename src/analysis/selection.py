"""Which cohorts the cross-cohort analyses are allowed to see."""

from config.params import ROC_FILTER_REP, BEST_CLASSIFIER, MIN_COHORT_ROC
from data_io.paths import metric_summary_path
from data_io.store import load_metric_summary


def filter_cohorts_by_roc(cohorts: list[str]) -> tuple[list[str], list[str]]:
    """
    Split the cohorts into the ones good enough for the cross-cohort analyses
    """
    kept, dropped = [], []
    for cohort in cohorts:
        if not metric_summary_path(cohort).exists():
            print(f"Dropping {cohort} from the cross-cohort analyses: no performance summary.")
            dropped.append(cohort)
            continue
        summary = load_metric_summary(cohort)
        row = summary[(summary["model"] == BEST_CLASSIFIER)
                      & (summary["measure"] == "auc_roc")].iloc[0]
        auc = row[f"mean_{ROC_FILTER_REP}"]
        # NaN fails the comparison and lands in dropped, which is what we want
        if auc >= MIN_COHORT_ROC:
            kept.append(cohort)
        else:
            dropped.append(cohort)
            print(f"Dropping {cohort} from the cross-cohort analyses: "
                  f"{ROC_FILTER_REP} AUC-ROC {auc:.3f} < {MIN_COHORT_ROC}.")
    return kept, dropped
