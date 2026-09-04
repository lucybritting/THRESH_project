"""Feature importance: its stability across folds per cohort, and its shifts across cohorts."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config.params import (BEST_CLASSIFIER, CROSS_COHORT_EXCLUDED_REPS,
                          RANK_DIFF_REPRESENTATIONS, RANK_DIFF_TOP_N,
                          STABILITY_FEATURE_SETS)
from config.labels import (CLASSIFICATION_HEATMAP_REPS, CLASSIFIER_LABELS,
                          REPRESENTATION_LABELS)
from data_io.paths import figure_path, reports_path
from data_io.store import (load_feature_importance, load_feature_shift,
                          load_feature_stability, load_labitem_labels,
                          load_labs_with_reference_range, save_feature_shift,
                          save_feature_stability)
from plots import (plot_rank_bump_chart, plot_rank_difference_heatmap,
                         plot_rank_shift_slope_chart, plot_representation_boxplot,
                         plot_value_heatmap)
from .grouping import ICD_GROUP_NAMES, ICD_GROUP_ORDER, _disease_group, _group_mean


def _kendalls_w(rank_matrix: np.ndarray) -> float: # (m folds, n features)
    m, n = rank_matrix.shape
    R = rank_matrix.sum(axis=0) # R_i: sum ranks over the folds for each feature
    S = ((R-R.mean())**2).sum() # S = sum of squared deviations
    # tie correction
    tie_term = 0.0
    for ranks in rank_matrix:
        _,counts = np.unique(ranks, return_counts=True)
        tie_term +=(counts**3-counts).sum()
    W = (12*S) / (m**2*(n**3-n)- m * tie_term)
    return W


def cohort_feature_importance_analysis(cohort: str) -> None:
    # FEATURE STABILITY ANALYSIS ACROSS 5 FOLDS
    # compare the feature importance between the 5 folds using Kendalls W
    all_importance_df = load_feature_importance(cohort)
    importance_df = all_importance_df[all_importance_df["feature_set"].isin(STABILITY_FEATURE_SETS)]

    # drop itemids that are never used: 0.0 importance in every fold of every representation
    max_per_itemid = importance_df.groupby("itemid")["importance"].transform("max") # group all rows with the same itemid, then take its max, then broadcast max back --> mask
    importance_df = importance_df[max_per_itemid > 0.0] # filter with mask


    # split into one frame per representation, all sharing the same itemids after the filter above
    by_representation = {rep: importance_df[importance_df["representation"] == rep]
                         for rep in STABILITY_FEATURE_SETS}
    # create rank matrix for each representation (m folds, n features)
    rank_matrices = {} # representation -> np.ndarray (m folds, n features)
    for rep, rep_df in by_representation.items():
        # wide: one row per fold, one column per itemid (columns sorted, so the same column is the same feature in every row)
        wide = rep_df.pivot(index="fold", columns="itemid", values="importance") # change from long format (one row per fold, itemid) to wide: rows = fold, columns = itemid
        # rank the features against each other within each fold; ties share their average rank
        rank_matrices[rep] = rankdata(wide.to_numpy(), axis=1)

    # KENDALLS W STABILITY ACROSS FOLDS PER REPRESENTATION
    kendalls_w = {rep: _kendalls_w(rank_matrix) for rep, rank_matrix in rank_matrices.items()}

    # one row per representation; n_folds/n_features record what W was computed over
    stability_df = pd.DataFrame([{"cohort": cohort,
                                  "representation": rep,
                                  "n_folds": rank_matrices[rep].shape[0],
                                  "n_features": rank_matrices[rep].shape[1],
                                  "kendalls_w": w}
                                 for rep, w in kendalls_w.items()])
    save_feature_stability(cohort, stability_df)


    # FEATURE IMPORTANCE MEAN AND STD PER REPRESENTATION
    mean_df = importance_df.pivot_table(index="itemid", columns="representation",
                                        values="importance", aggfunc="mean") # rows: itemids, columns: representations, values: mean importances
    std_df = importance_df.pivot_table(index="itemid", columns="representation",
                                       values="importance",
                                       aggfunc="std")
    mean_df.columns = [f"mean_{rep}" for rep in mean_df.columns]
    std_df.columns = [f"std_{rep}" for rep in std_df.columns]
    shift_df = pd.concat([mean_df, std_df], axis=1).reset_index()

    save_feature_shift(cohort, shift_df)

    # RANK SHIFT SLOPE CHART
    # turn shift_df into rankings per representation
    means = {rep: shift_df.set_index("itemid")[f"mean_{rep}"] for rep in STABILITY_FEATURE_SETS} # take mean importance per labitem per representaton across folds
    ranks = {rep: mean.rank(ascending=False, method="min") for rep, mean in means.items()} # rank the labitems by mean importance
    item_labels = load_labitem_labels() # load the labels

    # helper function to select the top 20 labs from each representation
    def slope_data(left_rep: str, right_rep: str, top_k: int = 20) -> pd.DataFrame:
        left, right = ranks[left_rep], ranks[right_rep]
        selected = left.nsmallest(top_k).index.union(right.nsmallest(top_k).index) # union of both selected top 20 labs
        return pd.DataFrame({"label": [item_labels.get(itemid,itemid) for itemid in selected],
                             "rank_left": left[selected].to_numpy(),
                             "rank_right": right[selected].to_numpy(),
                             "unused_left": (means[left_rep][selected] == 0).to_numpy(), # flag unused labitems instead of giving all of them the same last rank
                             "unused_right": (means[right_rep][selected] == 0).to_numpy()}) # flag unused labitems instead of giving all of them the same last rank

    cont_disc = slope_data("cont", "disc")
    disc_bin = slope_data("disc", "bin")
    cont_bin = slope_data("cont", "bin")

    plot_rank_shift_slope_chart(
        panels=[("continuous", "discretised", cont_disc),
                ("discretised", "binarised", disc_bin),
                ("continuous", "binarised", cont_bin)],
        title=f"Feature importance rank shift between representations — {cohort}",
        subtitle="Top 20 labs from each representation per comparison, ranked by mean "
                 "importance across the folds; 1 = most important. Ranks past 50 are "
                 f"compressed. Unused labs are labelled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        path=figure_path(cohort, "rank_shift"))


    disc_disc_imp = slope_data("disc", "disc_imp")
    plot_rank_shift_slope_chart(
        panels=[("discretised", "discretised + imputed", disc_disc_imp)],
        title=f"Feature importance rank shift from imputation — {cohort}",
        subtitle="Top 20 labs from each representation, ranked by mean importance across "
                 "the folds; 1 = most important. Ranks past 50 are compressed. Unused labs are labelled. "
                 f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        path=figure_path(cohort, "rank_shift_disc_imp"))

    # RANK BUMP CHART OF FEATURE IMPORTANCE ACROSS FOLDS
    per_lab = all_importance_df.groupby(["feature_set", "fold", "itemid"],
                                        as_index=False)["importance"].sum()

    # helper function to build one panel
    def bump_data(feature_set: str, top_k: int = 20) -> pd.DataFrame:
        wide = per_lab[per_lab["feature_set"] == feature_set].pivot(
            index="fold", columns="itemid", values="importance")
        # rank within each fold, 1 = most important, then one row per feature
        fold_ranks = wide.rank(axis=1, ascending=False, method="min").transpose() # rank within one fold
        selected = fold_ranks.mean(axis=1).nsmallest(top_k).index
        return fold_ranks.loc[selected].rename(index=lambda itemid: item_labels.get(itemid, itemid))

    bump_panels = []
    # build panels for each representation
    for rep in CLASSIFICATION_HEATMAP_REPS:
        fold_ranks = bump_data(rep)
        # each panel accents its own top 5
        top5 = list(fold_ranks.mean(axis=1).nsmallest(5).index)
        bump_panels.append((REPRESENTATION_LABELS[rep], fold_ranks, top5))

    plot_rank_bump_chart(
        panels=bump_panels,
        title=f"Feature importance rank across folds — {cohort}",
        subtitle="Top 20 labs per representation, coloured in importance blocks of 5 (grouped by mean across folds), "
                 "all named, top 5 in bold. In a concatenated representation a lab's importance is summed "
                 f"over its encodings. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        path=figure_path(cohort, "rank_bump"))


def feature_importance_analysis(cohorts: list[str]) -> None:
    # get per-cohort mean importances
    shift_dfs = {cohort: load_feature_shift(cohort).set_index("itemid") for cohort in cohorts} # dict of dfs: key= cohort name, value = cohorts df of feature shift
    item_labels = load_labitem_labels()

    # create one dict with keys: (cohort, representation), values: Series [itemid -> rank]
    rank_rows = {(cohort, rep): shift_dfs[cohort][f"mean_{rep}"].rank(ascending=False, method="min")
                 for cohort in cohorts for rep in STABILITY_FEATURE_SETS}
    rank_df = pd.DataFrame(rank_rows).transpose() # df with keys as columns, itemid as index, values is rank.

    # RANK DIFFERENCE HEATMAPS
    cont_ranks = rank_df.xs("cont", level=1)  # cross-section: select where the level equals "cont" and remove the level. rows = cohort, columns = itemid, values = continuous rank

    # columns: the union of the best RANK_DIFF_TOP_N mean ranks in each of the three
    # representations
    selected = set()
    for rep in RANK_DIFF_REPRESENTATIONS:
        selected |= set(rank_df.xs(rep, level=1).mean().nsmallest(RANK_DIFF_TOP_N).index)
    # a lab with no reference range in any cohort is NaN in every discretised cell
    with_range = set().union(*(load_labs_with_reference_range(cohort) for cohort in cohorts))
    dropped_labs = sorted(selected - with_range)
    selected = sorted(selected & with_range)
    print(f"Rank difference heatmaps: {len(selected)} labs "
          f"(top {RANK_DIFF_TOP_N} per representation, {len(dropped_labs)} dropped for "
          f"having no reference range in any cohort).")

    # get rank difference between continuous and other representation, one row per cohort
    def rank_difference(other_rep: str) -> pd.DataFrame:
        diff = (cont_ranks - rank_df.xs(other_rep, level=1))[selected]
        return diff.loc[cohorts].rename(columns=lambda i: item_labels.get(i, i)) # force to be in order given by cohorts, and insert labels

    def with_average(diff: pd.DataFrame) -> pd.DataFrame:
        # the across-cohort average per lab as a final summary row
        out = diff.copy()
        out.loc["average"] = diff.mean(axis=0)
        return out

    def by_disease_group(diff: pd.DataFrame) -> pd.DataFrame:
        groups = pd.Series(
            pd.Categorical([_disease_group(cohort) for cohort in diff.index],
                           categories=ICD_GROUP_ORDER, ordered=True),
            index=diff.index)
        grouped = _group_mean(diff, groups, "n", ICD_GROUP_NAMES)
        grouped.loc["average"] = diff.mean(axis=0)
        return grouped

    column_note = (f"The union of the {RANK_DIFF_TOP_N} labs with the best mean rank in each of the "
                   "continuous, discretised and binarised representations, minus the labs that carry "
                   "no reference range in any cohort; ")

    for other_rep, rep_word in (("disc", "discretised"), ("bin", "binarised")):
        diff = rank_difference(other_rep)
        shared = (f"blue = more important once {rep_word}, red = less important. "
                  f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.")
        colourbar = f"continuous rank − {rep_word} rank"

        plot_rank_difference_heatmap(
            with_average(diff),
            title=f"Rank shift from continuous to {rep_word} representation",
            subtitle=column_note + shared + " Bottom row is the average across cohorts.",
            colourbar_label=colourbar,
            path=reports_path("importance") / f"rank_difference_cont_{other_rep}_heatmap.png",
            separate_last_row=True)


        plot_rank_difference_heatmap(
            by_disease_group(diff),
            title=f"Rank shift from continuous to {rep_word} representation, by disease group",
            subtitle=column_note + shared +
                     " Each row is the mean over the cohorts of one ICD chapter of code1, "
                     "with the cohort count in the label. Bottom row is the average across "
                     "cohorts, so it is cohort-weighted and matches the per-cohort figure.",
            colourbar_label=colourbar,
            path=reports_path("importance") / f"rank_difference_cont_{other_rep}_disease_heatmap.png",
            separate_last_row=True)

    # KENDALL W
    stability_df = pd.concat([load_feature_stability(cohort) for cohort in cohorts]) # load each cohorts feature stability file
    w_keyed = stability_df.pivot(index="cohort", columns="representation", values="kendalls_w") # reshape to wide format

    w_reps = [rep for rep in STABILITY_FEATURE_SETS
              if rep not in CROSS_COHORT_EXCLUDED_REPS]
    w_keyed = w_keyed.loc[cohorts, w_reps]
    w_note = ("Kendall's W over the 5 folds; 1 = the folds rank the features identically. "
              f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.")

    w_df = w_keyed.rename(columns=REPRESENTATION_LABELS)
    # append the across-cohort average per representation as a final summary row
    w_df.loc["average"] = w_df.mean(axis=0)

    plot_value_heatmap(
        w_df,
        title="Fold agreement on the feature importance ranking",
        subtitle="Kendall's W per cohort and representation; 1 = the 5 folds rank the "
                 "features identically. Bottom row is the average across cohorts. "
                 f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        colourbar_label="Kendall's W",
        path=reports_path("importance") / "kendalls_w_heatmap.png",
        separate_last_row=True)


    plot_representation_boxplot(
        w_keyed, groups=None, group_order=[],
        series=[(rep, REPRESENTATION_LABELS[rep]) for rep in w_reps],
        title="Fold agreement on the feature importance ranking, by representation",
        subtitle=w_note + " Box spans the interquartile range, line is the median, whiskers "
                          "reach min and max. One dot per cohort.",
        ylabel="Kendall's W",
        path=reports_path("importance") / "kendalls_w_boxplot.png")
