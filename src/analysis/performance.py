"""Cross-cohort performance: how the representations score, overall and by cohort group."""

import numpy as np
import pandas as pd

from config.params import (ROC_FILTER_REP, BEST_CLASSIFIER, COHORT_SIZE_BINS,
                          COHORT_SIZE_LABELS, CONCATENATED_REPS, CROSS_COHORT_EXCLUDED_REPS,
                          MIN_COHORT_ROC, PREVALENCE_BINS, PREVALENCE_LABELS)
from config.labels import (CLASSIFICATION_HEATMAP_REPS, CLASSIFIER_LABELS, METRIC_LABELS,
                          REPRESENTATION_LABELS)
from data_io.paths import reports_path
from data_io.store import load_cohort, load_metric_summary
from plots import plot_performance_scatter, plot_representation_boxplot
from .grouping import ICD_GROUP_ORDER, _disease_group, _target_group


def performance_analysis(cohorts: list[str]) -> None:
    summaries = {cohort: load_metric_summary(cohort) for cohort in cohorts}
    # size and label distribution are not in the cohort name, so read them off the labels
    labels = {cohort: load_cohort(cohort)["label"] for cohort in cohorts}
    sizes = pd.Series({cohort: len(label) for cohort, label in labels.items()})
    prevalences = pd.Series({cohort: label.mean() for cohort, label in labels.items()})
    # ordered categoricals, so the heatmap rows come out in bin order and not alphabetically
    size_bins = pd.cut(sizes, COHORT_SIZE_BINS, labels=COHORT_SIZE_LABELS)
    prevalence_bins = pd.cut(prevalences, PREVALENCE_BINS, labels=PREVALENCE_LABELS)
    # every trained feature set, in the order they build on each other, minus the ones the
    # overviews leave out
    representations = [rep for rep in CLASSIFICATION_HEATMAP_REPS
                       if rep not in CROSS_COHORT_EXCLUDED_REPS]
    # only AUC-ROC is plotted here. f1 and mcc had one heatmap each and no other figure, so
    # with the heatmaps gone they have nothing to draw -- both stay in the per-cohort
    # performance summary csv files, which is where they are read from anyway
    measure, slug = "auc_roc", "roc"
    measure_label = METRIC_LABELS[measure]

    rows = {}
    for cohort in cohorts:
        summary = summaries[cohort]
        row = summary[(summary["model"] == BEST_CLASSIFIER)
                      & (summary["measure"] == measure)].iloc[0]
        # .get: a cohort trained before a feature set was added has no column for it,
        # which should read as a blank cell rather than kill the whole report
        rows[cohort] = [row.get(f"mean_{rep}", np.nan) for rep in representations]
    perf_df = pd.DataFrame.from_dict(rows, orient="index", columns=representations)
    perf_df = perf_df.loc[cohorts].rename(columns=REPRESENTATION_LABELS)

    # ordered categorical, so the groups follow ICD_CHAPTERS order rather than relying on
    # the group keys happening to sort into chapter order
    disease_groups = pd.Series(
        pd.Categorical([_disease_group(cohort) for cohort in perf_df.index],
                       categories=ICD_GROUP_ORDER, ordered=True),
        index=perf_df.index)
    # code2 read the same way, so the two sides of the cohort name are grouped and
    # ordered alike and a chapter means the same thing on either
    target_groups = pd.Series(
        pd.Categorical([_target_group(cohort) for cohort in perf_df.index],
                       categories=ICD_GROUP_ORDER, ordered=True),
        index=perf_df.index)

    # cohort size and label distribution unbinned, so a gradient can be told apart from a
    # threshold and single-cohort bins cannot masquerade as a trend
    plot_performance_scatter(
        pd.DataFrame(rows, index=representations).transpose(),  # keyed by representation, not the display labels
        panels=[(sizes, "cohort size (admissions, log scale)", True),
                (prevalences, "share of positive labels", False)],
        series=[(rep, REPRESENTATION_LABELS[rep]) for rep in representations],
        title=f"{measure_label} against cohort size and label distribution",
        subtitle=f"One dot per cohort, averaged over the 5 folds, with a least squares "
                 f"fit per representation. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        ylabel=measure_label,
        path=reports_path("performance") / f"performance_{slug}_vs_cohort_properties.png",
        hline=(MIN_COHORT_ROC, f"inclusion threshold ({MIN_COHORT_ROC})"),
        # the two x axes of the panels above against each other. They are strongly
        # related in this cohort set, so neither slope can be read as the effect of
        # its own property alone
        property_panel=(sizes, prevalences, "cohort size (admissions, log scale)",
                        "share of positive labels", True))

    # the spread across cohorts: one box per representation, and every cohort as a dot
    keyed_df = pd.DataFrame(rows, index=representations).transpose()  # representation keys, not display labels
    rep_series = [(rep, REPRESENTATION_LABELS[rep]) for rep in representations]
    # the grouped boxplots below drop the concatenated representations, see
    # CONCATENATED_REPS: their boxes would be too thin to read next to the groups
    single_rep_series = [(rep, label) for rep, label in rep_series
                         if rep not in CONCATENATED_REPS]
    box_note = ("Box spans the interquartile range, line is the median, whiskers reach "
                "min and max. One dot per cohort, averaged over the 5 folds. "
                f"{CLASSIFIER_LABELS[BEST_CLASSIFIER]}.")
    threshold_rule = (MIN_COHORT_ROC, f"inclusion threshold ({MIN_COHORT_ROC})")

    plot_representation_boxplot(
        keyed_df, groups=None, group_order=[], series=rep_series,
        title=f"{measure_label} by representation",
        subtitle=box_note + f" Rules carry the {REPRESENTATION_LABELS[ROC_FILTER_REP]} "
                            f"median and quartiles across the plot.",
        ylabel=measure_label,
        path=reports_path("performance") / f"performance_{slug}_boxplot.png",
        hline=threshold_rule,
        reference_series=ROC_FILTER_REP)

    # the same spread cut by the cohort properties, one figure per property
    for group_keys, group_order, group_name, group_slug in [
            (disease_groups, ICD_GROUP_ORDER, "disease group", "disease"),
            (target_groups, ICD_GROUP_ORDER, "target group", "target"),
            (size_bins, COHORT_SIZE_LABELS, "cohort size", "size"),
            (prevalence_bins, PREVALENCE_LABELS, "label distribution", "prevalence")]:
        keys = pd.Series(list(group_keys), index=perf_df.index)
        plot_representation_boxplot(
            keyed_df, groups=keys,
            group_order=group_order if group_order is not None else sorted(set(keys)),
            series=single_rep_series,
            title=f"{measure_label} by {group_name} and representation",
            subtitle=box_note + " Concatenated representations are left out.",
            ylabel=measure_label,
            path=reports_path("performance") / f"performance_{slug}_boxplot_{group_slug}.png",
            hline=threshold_rule)
