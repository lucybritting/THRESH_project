"""Per-group model performance and the fairness criteria gaps, per cohort and across them."""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

from config.params import (AGE_BANDS, AGE_BAND_LABELS, BEST_CLASSIFIER, CONCATENATED_REPS,
                          CROSS_COHORT_EXCLUDED_REPS, FAIRNESS_CRITERIA,
                          FAIRNESS_GROUP_SIZE_BINS, FAIRNESS_GROUP_SIZE_LABELS,
                          FAIRNESS_MIN_GROUP_SAMPLES, FAIRNESS_MIN_LABEL_SAMPLES, METRICS)
from config.labels import (CLASSIFICATION_HEATMAP_REPS, CLASSIFIER_LABELS,
                          FAIRNESS_ATTRIBUTE_LABELS, FAIRNESS_CRITERIA_LABELS,
                          METRIC_LABELS, REPRESENTATION_LABELS)
from data_io.paths import figure_path, reports_path
from data_io.store import (load_fairness, load_fairness_criteria, load_test_predictions,
                          save_fairness, save_fairness_criteria)
from plots import (plot_fairness, plot_fairness_eo_dumbbell,
                         plot_representation_boxplot, REPRESENTATION_COLOUR)
from .grouping import ICD_GROUP_ORDER, _disease_group


def fairness_analysis(cohorts: list[str]) -> None:
    # stack all fairness criteria dfs into one long
    parity = pd.concat([load_fairness_criteria(cohort) for cohort in cohorts], ignore_index=True)
    # per group frames needed for the exclusion note on the plot
    per_group = {cohort: load_fairness(cohort) for cohort in cohorts}
    # equalised odds only and only the 3 representations
    reps = ["cont", "disc", "bin"]
    eo = parity[(parity["criterion"] == "equalised_odds") & (parity["feature_set"].isin(reps))]

    for attribute in ["age_band", "gender"]:
        gaps = (eo[eo["attribute"] == attribute]
                .pivot(index="cohort", columns="feature_set", values="difference")
                .reindex(index=cohorts, columns=reps)) # for one attribute pivot to wide grid. Reindex fixes cohort row and column order

        # Check groups for exclusion
        excluded = {}
        for cohort in cohorts:
            fdf = per_group[cohort] # fairness df per group
            one_fs = fdf[(fdf["feature_set"] == fdf["feature_set"].iloc[0]) # filter for demographic attribute and keep only one representation (sample counts are identical)
                         & (fdf["attribute"] == attribute)]
            excluded[cohort] = [str(row.group) for row in one_fs.itertuples()
                                if not _group_supported({"n": row.n, "n_positive": row.n_positive})] # check for enough support
        # assemble note for the plot
        dropped = "; ".join(f"{cohort}: {', '.join(groups)}"
                            for cohort, groups in excluded.items() if groups)
        note = (f"Groups excluded from the gaps (fewer than {FAIRNESS_MIN_GROUP_SAMPLES} samples "
                f"or fewer than {FAIRNESS_MIN_LABEL_SAMPLES} of either label): "
                f"{dropped if dropped else 'none'}")

        plot_fairness_eo_dumbbell(
            gaps,
            baseline=("cont", "continuous"),
            targets=[("disc", "discretised"), ("bin", "binarised")],
            title=f"Equalised-odds gap across cohorts, by representation — {attribute}",
            subtitle=f"gap = worse of the tpr/fpr spread (max−min) across the {attribute} "
                     "groups; the bar is the change from the continuous representation. "
                     f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}",
            xlabel="equalised-odds gap (0 = equal across groups)",
            path=reports_path("fairness") / f"fairness_eo_{attribute}.png",
            note=note)


    eo_reps = [rep for rep in CLASSIFICATION_HEATMAP_REPS
               if rep not in CROSS_COHORT_EXCLUDED_REPS
               and rep not in CONCATENATED_REPS
               and rep in set(parity["feature_set"])]
    eo_all = parity[(parity["criterion"] == "equalised_odds")
                    & (parity["feature_set"].isin(eo_reps))]
    gap_df = (eo_all.pivot(index=["cohort", "attribute"], columns="feature_set",
                           values="difference")
              .reindex(columns=eo_reps))
    if gap_df.empty:
        print("No equalised-odds gaps across cohorts, so the boxplot is skipped.")
    else:
        # readable group headings, so the figure says "Gender" rather than the raw key
        attributes = [FAIRNESS_ATTRIBUTE_LABELS.get(attribute, attribute)
                      for attribute in gap_df.index.get_level_values("attribute")]
        gap_df.index = [f"{cohort} · {attribute}" for cohort, attribute in gap_df.index]
        plot_representation_boxplot(
            gap_df,
            groups=pd.Series(attributes, index=gap_df.index),
            group_order=[FAIRNESS_ATTRIBUTE_LABELS.get(a, a) for a in ("gender", "age_band")],
            series=[(rep, REPRESENTATION_LABELS[rep]) for rep in eo_reps],
            title="Equalised-odds gap across cohorts, by representation",
            subtitle="Gap = the worse of the tpr and fpr spread (max − min) across the groups "
                     "of the attribute; 0 = equal across groups, so lower is fairer. One dot "
                     f"per cohort. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
            ylabel="equalised-odds gap",
            path=reports_path("fairness") / "fairness_eo_boxplot.png")

        # gaps split by disease groups
        for attribute in ("gender", "age_band"):
            attribute_gaps = (eo_all[eo_all["attribute"] == attribute]
                              .pivot(index="cohort", columns="feature_set", values="difference")
                              .reindex(columns=eo_reps))
            attribute_gaps = attribute_gaps.dropna(how="all")
            if attribute_gaps.empty:
                continue
            disease = pd.Series(
                pd.Categorical([_disease_group(cohort) for cohort in attribute_gaps.index],
                               categories=ICD_GROUP_ORDER, ordered=True),
                index=attribute_gaps.index)
            plot_representation_boxplot(
                attribute_gaps, groups=disease, group_order=ICD_GROUP_ORDER,
                series=[(rep, REPRESENTATION_LABELS[rep]) for rep in eo_reps],
                title=f"Equalised-odds gap by disease group and representation — "
                      f"{FAIRNESS_ATTRIBUTE_LABELS.get(attribute, attribute)}",
                subtitle=f"Gap = the worse of the tpr and fpr spread (max − min) across the "
                         f"{FAIRNESS_ATTRIBUTE_LABELS.get(attribute, attribute).lower()} groups; "
                         f"0 = equal across groups, so lower is fairer. One dot per cohort, "
                         f"grouped by the ICD chapter. {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
                ylabel="equalised-odds gap",
                path=reports_path("fairness") / f"fairness_eo_boxplot_disease_{attribute}.png")

    # gaps per demographic group
    gap_frames = []
    for cohort, group_df in per_group.items():
        pooled = group_df[group_df["group"] == "all"].set_index("feature_set")
        rows = group_df[(group_df["group"] != "all")
                        & (group_df["feature_set"].isin(eo_reps))].copy()
        rows["gap"] = np.maximum(
            (rows["tpr"] - rows["feature_set"].map(pooled["tpr"])).abs(),
            (rows["fpr"] - rows["feature_set"].map(pooled["fpr"])).abs())
        rows["cohort"] = cohort
        gap_frames.append(rows[["cohort", "group", "feature_set", "n", "gap"]])

    group_gaps = pd.concat(gap_frames, ignore_index=True).dropna(subset=["gap"])
    if group_gaps.empty:
        print("No per-group equalised-odds gaps, so the group size boxplot is skipped.")
    else:
        size_df = (group_gaps.pivot_table(index=["cohort", "group"], columns="feature_set",
                                          values="gap")
                   .reindex(columns=eo_reps))
        # n is a property of the group, identical across representations
        group_n = (group_gaps.groupby(["cohort", "group"])["n"].first()
                   .reindex(size_df.index))
        size_bin = pd.cut(group_n, FAIRNESS_GROUP_SIZE_BINS,
                          labels=FAIRNESS_GROUP_SIZE_LABELS)
        size_df.index = [f"{cohort} · {group}" for cohort, group in size_df.index]
        plot_representation_boxplot(
            size_df, groups=pd.Series(list(size_bin), index=size_df.index),
            group_order=FAIRNESS_GROUP_SIZE_LABELS,
            series=[(rep, REPRESENTATION_LABELS[rep]) for rep in eo_reps],
            title="Equalised-odds gap by demographic group size and representation",
            subtitle="Per demographic group, the worse of the tpr and fpr distance from its "
                     "own cohort's pooled rate; 0 = matches the cohort, so lower is fairer. "
                     "One dot per (cohort, group). Groups of every size are shown, including "
                     f"those under the {FAIRNESS_MIN_GROUP_SAMPLES}-sample support floor the "
                     f"gap figures apply, since how the gap behaves at small n is the point. "
                     f"{CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
            ylabel="equalised-odds gap",
            path=reports_path("fairness") / "fairness_eo_boxplot_group_size.png")



def _group_performance(group_df: pd.DataFrame) -> dict:
    """
    Performance of one subgroup's held-out predictions. Measures that need positives and negatives are NaN for single class group.
    """
    y_true, y_prob, y_pred = group_df["y_true"], group_df["y_prob"], group_df["y_pred"]
    both_classes = y_true.nunique() == 2
    return {
        "n": len(group_df),
        "n_positive": int(y_true.sum()),
        "prevalence": y_true.mean(),
        "selection_rate": y_pred.mean(),
        "auc_roc": roc_auc_score(y_true, y_prob) if both_classes else np.nan,
        "avg_prec": average_precision_score(y_true, y_prob) if both_classes else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0) if both_classes else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if both_classes else np.nan,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) if both_classes else np.nan,
        "tpr": y_pred[y_true == 1].mean() if (y_true == 1).any() else np.nan,
        "fpr": y_pred[y_true == 0].mean() if (y_true == 0).any() else np.nan,
        "ppv": y_true[y_pred == 1].mean() if (y_pred == 1).any() else np.nan,
    }



def _group_supported(performance: dict) -> bool:
    """
    Filters groups by whether they have enough support to enter the fairness criteria gap calculation.
    """
    n, n_positive = performance["n"], performance["n_positive"]
    return (n >= FAIRNESS_MIN_GROUP_SAMPLES
            and n_positive >= FAIRNESS_MIN_LABEL_SAMPLES
            and n - n_positive >= FAIRNESS_MIN_LABEL_SAMPLES)



BIAS_METRIC_COLUMNS = ["cohort", "feature_set", "classifier", "attribute", "criterion",
                       "quantity", "difference", "ratio", "min_group", "max_group",
                       "n_groups", "n_groups_total"]


def _bias_metric_gaps(group_performance: pd.DataFrame) -> list[dict]:
    rows = []
    for criterion, quantities in FAIRNESS_CRITERIA.items():
        gaps = {}
        for quantity in quantities: # go through list of columns that are the quantities
            values = group_performance[quantity].dropna()
            if len(values) < 2:
                continue  # nothing to compare
            gaps[quantity] = {
                "difference": values.max() - values.min(),
                "ratio": values.min() / values.max() if values.max() > 0 else np.nan,
                "min_group": values.idxmin(),
                "max_group": values.idxmax(),
                "n_groups": len(values),
            }
        if not gaps:
            continue
        # equalised odds holds only as well as its worse half
        driver = max(gaps, key=lambda q: gaps[q]["difference"]) # picks the quantity with largest gap
        rows.append({"criterion": criterion, "quantity": driver, **gaps[driver],
                     "n_groups_total": len(group_performance)})
    return rows


def cohort_fairness_analysis(cohort: str) -> None:
    """
    Only performed for BEST_CLASSIFIER.
    Per cohort.
    :param cohort:
    :return:
    """
    predictions_df = load_test_predictions(cohort)  # all folds concatenated
    predictions_df["age_band"] = pd.cut(predictions_df["age"], bins=AGE_BANDS,
                                        right=False, labels=AGE_BAND_LABELS) # bins the entries into the age groups


    # BUILD THE METRIC TABLES
    fairness_rows = [] # performance for each group
    bias_metrics_rows = [] # gaps between bias metrics
    excluded_groups = {} # groups that are dropped from the bias metric gap calculation because of little support
    for feature_set, feature_set_df in predictions_df.groupby("feature_set"):
        # reference row over the whole cohort
        fairness_rows.append({"cohort": cohort,
                              "feature_set": feature_set,
                              "classifier": BEST_CLASSIFIER,
                              "attribute": "all",
                              "group": "all",
                              **_group_performance(feature_set_df)}) # metrics dict for the whole cohort
        # iterate over groups
        for attribute in ["gender", "age_band"]:
            per_group = {}
            for group, group_df in feature_set_df.groupby(attribute, observed=True): # splits cohort by attribute
                performance = _group_performance(group_df) # get performance metrics
                per_group[str(group)] = performance # stores the group's performance to per_group dict
                fairness_rows.append({"cohort": cohort,
                                      "feature_set": feature_set,
                                      "classifier": BEST_CLASSIFIER,
                                      "attribute": attribute,
                                      "group": str(group),
                                      **performance})
            # exclude groups with too little support
            supported = {g: p for g, p in per_group.items() if _group_supported(p)}
            excluded_groups[attribute] = [g for g in per_group if g not in supported]
            # get bias metrics rows
            # one row per attribute
            df_transposed = pd.DataFrame(supported).transpose() # makes each key of the outer dict (str(group)) a column in the df. transpose: groups becomes rows, metrics becomes columns
            for gap in _bias_metric_gaps(df_transposed): # get the bias metrics
                bias_metrics_rows.append({"cohort": cohort,
                                          "feature_set": feature_set,
                                          "classifier": BEST_CLASSIFIER,
                                          "attribute": attribute,
                                          **gap})

    fairness_df = pd.DataFrame(fairness_rows)
    bias_metric_df = pd.DataFrame(bias_metrics_rows, columns=BIAS_METRIC_COLUMNS)
    save_fairness(cohort, fairness_df)
    save_fairness_criteria(cohort, bias_metric_df)

    # PLOT FAIRNESS: MODEL PERFORMANCE PER DEMOGRAPHIC GROUP
    fairness_reps = [rep for rep in CLASSIFICATION_HEATMAP_REPS
                     if rep not in CONCATENATED_REPS]
    representation_colours = {label: REPRESENTATION_COLOUR[rep]
                              for rep, label in REPRESENTATION_LABELS.items()
                              if rep in REPRESENTATION_COLOUR}
    plot_df = fairness_df[fairness_df["feature_set"].isin(fairness_reps)]

    present = set(fairness_df["group"])
    genders = [g for g in ["F", "M"] if g in present]
    age_bands = [band for band in AGE_BAND_LABELS if band in present]
    group_order = ["all"] + genders + age_bands
    subgroups = genders + age_bands  # the demographic groups the average pools
    # group by performance metric: one panel per metric
    panels = []
    for measure in METRICS:
        panel_df = plot_df.pivot(index="group", columns="feature_set", values=measure)
        panel_df = panel_df.loc[group_order, fairness_reps].rename(columns=REPRESENTATION_LABELS)
        # average across all demographic groups (excludes the overall 'all' row)
        panel_df.loc["average"] = panel_df.loc[subgroups].mean(axis=0)
        panels.append((METRIC_LABELS[measure], panel_df))  # readable panel title

    # sample count and label distribution per group
    counts_src = plot_df[plot_df["feature_set"] == fairness_reps[0]].set_index("group")
    counts_df = pd.DataFrame({"n_negative": counts_src["n"] - counts_src["n_positive"],
                              "n_positive": counts_src["n_positive"]})

    plot_fairness(
        panels,
        separator_after=["all"] + genders[-1:] + age_bands[-1:],
        dashed_after=genders[:-1] + age_bands[:-1],
        counts_df=counts_df,  # extra panel: sample count and label balance per group
        colours=representation_colours,
        title=f"Performance by demographic group — {cohort}",
        subtitle=f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}. The top row is "
                 "the whole cohort, the bottom row the average across the demographic groups",
        path=figure_path(cohort, "fairness"))

    # PLOT GAPS IN BIAS METRICS BETWEEN GROUPS
    criteria_reps = fairness_reps  # same representations as the performance figure above
    # the criteria: one panel per attribute, one row per criterion
    bias_metric_df = bias_metric_df[bias_metric_df["feature_set"].isin(criteria_reps)]
    metric_panels = []
    for attribute, attribute_df in bias_metric_df.groupby("attribute"):
        panel_df = attribute_df.pivot(index="criterion", columns="feature_set",
                                      values="difference")
        panel_df = (panel_df.reindex(index=list(FAIRNESS_CRITERIA), columns=criteria_reps)
                    .rename(columns=REPRESENTATION_LABELS, index=FAIRNESS_CRITERIA_LABELS))  # readable rows
        metric_panels.append((FAIRNESS_ATTRIBUTE_LABELS[attribute], panel_df))  # readable header

    # name the under-supported groups left out of the gaps
    excluded_note = "; ".join(
        f"{FAIRNESS_ATTRIBUTE_LABELS[attr]}: {', '.join(excluded_groups[attr]) if excluded_groups.get(attr) else 'none'}"
        for attr in sorted(excluded_groups))
    note = (f"Groups with fewer than {FAIRNESS_MIN_GROUP_SAMPLES} samples or fewer than "
            f"{FAIRNESS_MIN_LABEL_SAMPLES} of either label are excluded. Excluded groups: {excluded_note}")


    if not metric_panels:
        print(f"No fairness criteria gaps for {cohort}: no attribute has two groups with "
              f"enough support, so the criteria figure is skipped.")
    else:
        plot_fairness(
            metric_panels,
            separator_after=[FAIRNESS_CRITERIA_LABELS[c] for c in list(FAIRNESS_CRITERIA)[:-1]],
            colours=representation_colours,
            title=f"Fairness criteria gaps — {cohort}",
            subtitle="Per criterion, the gap between the best- and worst-served group of the "
                     "attribute (max − min). "
                     f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
            path=figure_path(cohort, "fairness_criteria"),
            share_x=True,
            x_min=0.0,
            note=note)  # which demographic groups were dropped for too little support

    # PLOT EQUALISED ODDS GAP PER GROUP, ONE DOT PER REPRESENTATION
    eo_reps = [rep for rep in fairness_reps if rep != "disc_imp"]
    eo_df = fairness_df[fairness_df["feature_set"].isin(eo_reps)]
    reference = eo_df[eo_df["group"] == "all"].set_index("feature_set")  # pooled cohort rates
    # drop under-supported groups
    excluded = {g for groups in excluded_groups.values() for g in groups}
    eo_subgroups = [g for g in genders + age_bands if g not in excluded]
    eo_gaps = {}
    for rep in eo_reps:
        rep_groups = eo_df[(eo_df["feature_set"] == rep)
                           & (eo_df["group"].isin(eo_subgroups))].set_index("group")
        eo_gaps[rep] = np.maximum(
            (rep_groups["tpr"] - reference.loc[rep, "tpr"]).abs(),
            (rep_groups["fpr"] - reference.loc[rep, "fpr"]).abs())
    eo_gap_df = pd.DataFrame(eo_gaps).reindex(eo_subgroups)  # rows = groups, cols = reps

    # name the under-supported groups dropped from the gaps
    eo_excluded = [g for g in genders + age_bands if g in excluded]
    eo_note = (f"Groups with fewer than {FAIRNESS_MIN_GROUP_SAMPLES} samples or fewer than "
               f"{FAIRNESS_MIN_LABEL_SAMPLES} of either label are excluded. "
               f"Excluded groups: {', '.join(eo_excluded) if eo_excluded else 'none'}")

    if eo_gap_df.empty:
        print(f"No equalised odds gaps for {cohort}: every demographic group is "
              f"under-supported, so the equalised odds figure is skipped.")
    else:
        eo_panel = eo_gap_df.rename(columns=REPRESENTATION_LABELS)
        plot_fairness(
            [("Equalised odds gap", eo_panel)],
            separator_after=genders[-1:],
            dashed_after=[g for g in eo_subgroups if g not in genders[-1:]][:-1],
            colours=representation_colours,
            counts_df=counts_df,
            title=f"Equalised odds gap by group — {cohort}",
            subtitle="Per group, the worse of the tpr and fpr distance from the pooled cohort "
                     f"rate; 0 = matches the cohort. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
            path=figure_path(cohort, "equalised_odds"),
            x_min=0.0,
            note=eo_note)
