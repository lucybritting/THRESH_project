"""load_* / save_* for the pipeline's own files."""

import pickle

import pandas as pd

from config.paths import D_LABITEMS_PATH
from config.params import N_FOLDS, BEST_MERGE_STRATEGY
from .paths import (binary_mapping_path, blacklist_path, cohort_path,
                    continuous_mapping_path, discrete_mapping_path, fairness_criteria_path,
                    fairness_path, feature_importance_path, feature_shift_path,
                    feature_stability_path, fold_path, merged_ranges_path,
                    metric_summary_path, ranges_path, test_predictions_path,
                    top100itemids_path)


# ---------- LOAD FUNCTIONS -----------------------------
def load_top100_itemids() -> set[int]:
    with open(top100itemids_path(), "rb") as f:
        itemids = set(pickle.load(f))
    return itemids - load_blacklist()  # excludes itemids with only non-numeric values


def load_all_itemids() -> set[int]:
    itemids = set(pd.read_csv(D_LABITEMS_PATH, usecols=["itemid"])["itemid"])
    return itemids - load_blacklist()  # excludes itemids with only non-numeric values


def load_continuous_values(cohort: str) -> pd.DataFrame:
    return pd.read_csv(continuous_mapping_path(cohort))

def load_discrete_values(cohort: str, range_merge: str) -> pd.DataFrame:
    return pd.read_csv(discrete_mapping_path(cohort, range_merge))

def load_binary_values(cohort: str) -> pd.DataFrame:
    return pd.read_csv(binary_mapping_path(cohort))

def load_feature_importance(cohort: str) -> pd.DataFrame:

    return pd.read_csv(feature_importance_path(cohort))



def load_labitem_labels() -> dict[int, str]:
    """
    itemid -> readable lab name, for labelling figures.

    """
    d_labitems = pd.read_csv(D_LABITEMS_PATH,
                             usecols=["itemid", "label", "fluid", "category"])
    label = d_labitems["label"].fillna("unnamed")
    fluid = d_labitems["fluid"].fillna("?").astype(str)
    category = d_labitems["category"].fillna("?").astype(str)
    itemid = d_labitems["itemid"].astype(str)

    plotted = ~d_labitems["itemid"].isin(load_blacklist())
    out = label.copy()
    for _, group in label[plotted].groupby(label[plotted]).groups.items():
        if len(group) == 1:
            continue  # unique among the plotted labs, nothing to qualify
        # the least verbose qualifier that separates this group; the loop leaves the
        # fullest one in place when none of them manages it on its own
        for qualifier in (fluid[group], category[group],
                          fluid[group] + ", " + category[group]):
            if qualifier.nunique() == len(group):
                break
        # same fluid AND category: only the itemid is left to tell them apart
        clash = qualifier.duplicated(keep=False)
        qualifier = qualifier.mask(clash, qualifier + ", " + itemid[group])
        out[group] = label[group] + " (" + qualifier + ")"
    return dict(zip(d_labitems["itemid"], out))


def load_test_predictions(cohort: str, fold_idx: int | None = None) -> pd.DataFrame:
    folds = range(N_FOLDS) if fold_idx is None else [fold_idx]
    return pd.concat([pd.read_csv(test_predictions_path(cohort, fold)) for fold in folds],
                     ignore_index=True)


def load_feature_shift(cohort: str) -> pd.DataFrame:
    return pd.read_csv(feature_shift_path(cohort))


def load_feature_stability(cohort: str) -> pd.DataFrame:
    return pd.read_csv(feature_stability_path(cohort))


def load_fairness(cohort: str) -> pd.DataFrame:
    return pd.read_csv(fairness_path(cohort))


def load_fairness_criteria(cohort: str) -> pd.DataFrame:
    return pd.read_csv(fairness_criteria_path(cohort))


def load_blacklist() -> set[int]:
    with open(blacklist_path()) as f:
        return {int(line) for line in f if line.strip()}


def load_ranges(cohort: str) -> dict[tuple, set]:
    df = pd.read_csv(ranges_path(cohort))
    ranges = {}
    for row in df.itertuples(index=False):  # itertuples iterates over the rows of the df. Each row is a named tuple.
        hadm_id = int(row.hadm_id)
        itemid = int(row.itemid)
        lower = None if pd.isna(row.ref_range_lower) else float(row.ref_range_lower)
        upper = None if pd.isna(row.ref_range_upper) else float(row.ref_range_upper)
        ranges.setdefault((hadm_id, itemid), set()).add((lower, upper))
    return ranges


def load_merged_ranges(cohort: str, strategy_name: str) -> dict[tuple, tuple]:
    df = pd.read_csv(merged_ranges_path(cohort, strategy_name))
    merged = {}
    for _, row in df.iterrows():
        hadm_id = int(row["hadm_id"])
        itemid = int(row["itemid"])
        lower = None if pd.isna(row["ref_range_lower"]) else float(row["ref_range_lower"])
        upper = None if pd.isna(row["ref_range_upper"]) else float(row["ref_range_upper"])
        merged[(hadm_id, itemid)] = (lower, upper)
    return merged


def load_labs_with_reference_range(cohort: str, strategy_name: str = BEST_MERGE_STRATEGY) -> set[int]:

    df = pd.read_csv(merged_ranges_path(cohort, strategy_name),
                     usecols=["itemid", "ref_range_lower", "ref_range_upper"])
    has_range = df["ref_range_lower"].notna() | df["ref_range_upper"].notna()
    return set(df.loc[has_range, "itemid"].astype(int))


def load_fold(cohort: str, fold_idx: int) -> tuple:
    with open(fold_path(cohort, fold_idx), "rb") as f:
        return pickle.load(f)


def load_cohort(cohort: str) -> pd.DataFrame:  # hadm_id, label, demographics
    return pd.read_csv(cohort_path(cohort), usecols=["subject_id", "hadm_id", "gender", "age", "label"])


# ---------- SAVE FUNCTIONS -----------------------------
def save_continuous_value(cohort: str,
                          cohort_avg: dict[tuple[int, int], float],
                          hadm_ids: list[int],
                          itemids: list[int]) -> None:
    # convert dict to Series with MultiIndex, then unstack to wide
    s = pd.Series(cohort_avg)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["hadm_id", "itemid"])
    df = s.unstack("itemid")  # pivots the itemid level from rows into columns --> gives DataFrame
    # fill in any missing hadm_ids or itemids with None
    df = df.reindex(index=sorted(hadm_ids), columns=sorted(
        itemids))  # ensures that all hadm_ids and itemids are present (if no measurement they get row or column of NaNs)
    df.reset_index().to_csv(continuous_mapping_path(cohort),
                            index=False)  # reset index to make hadm_id a column again


def save_binary_values(cohort: str, binary_df: pd.DataFrame) -> None:
    binary_df.to_csv(binary_mapping_path(cohort), index=False)
    print(f"Saved binary mapping for cohort {cohort}.")


def save_discrete_values(cohort: str, discrete_df: pd.DataFrame, strategy_name: str) -> None:
    discrete_df.to_csv(discrete_mapping_path(cohort, strategy_name), index=False)
    print(f"Saved discrete mapping ({strategy_name}) for cohort {cohort}.")


def save_ranges(cohort: str, cohort_ranges: dict) -> None:
    rows = []
    for (hadm_id, itemid), range_set in cohort_ranges.items():
        for (lower, upper) in range_set:
            rows.append({
                "hadm_id": hadm_id,
                "itemid": itemid,
                "ref_range_lower": lower,
                "ref_range_upper": upper,
            })
    df = pd.DataFrame(rows)
    df.to_csv(ranges_path(cohort), index=False)


def save_merged_ranges(cohort: str, merged: dict,
                       strategy_name: str) -> None:  # merged is dict: (hadm_id, itemid) -> (lower, upper)
    rows = [{"hadm_id": k[0], "itemid": k[1], "ref_range_lower": v[0], "ref_range_upper": v[1]}
            for k, v in
            merged.items()]  # list of dicts. Each dict gives a row in the output csv. k, v are key, values from the merged dict
    pd.DataFrame(rows).to_csv(merged_ranges_path(cohort, strategy_name), index=False)
    print(f"Saved merged ranges ({strategy_name}) for cohort {cohort}.")


def save_fold(cohort: str, fold_idx: int, train, test) -> None:
    """
    Saves one fold as a pickled (train, test) tuple. Each split is an (n, 2)
    numpy array of [subject_id, hadm_id].
    """
    path = fold_path(cohort, fold_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((train, test), f)

def save_metric_summary(cohort: str, summary_df: pd.DataFrame) -> None:
    summary_df.to_csv(metric_summary_path(cohort), index=False)
    print(f"Saved metric summary for cohort {cohort}.")


def load_metric_summary(cohort: str) -> pd.DataFrame:
    return pd.read_csv(metric_summary_path(cohort))


def save_feature_importance(cohort: str, importance_df: pd.DataFrame) -> None:
    importance_df.to_csv(feature_importance_path(cohort), index=False)
    print(f"Saved feature importance for cohort {cohort} ({len(importance_df)} rows).")


def save_feature_stability(cohort: str, stability_df: pd.DataFrame) -> None:
    stability_df.to_csv(feature_stability_path(cohort), index=False)
    print(f"Saved feature stability for cohort {cohort}.")


def save_fairness(cohort: str, fairness_df: pd.DataFrame) -> None:
    fairness_df.to_csv(fairness_path(cohort), index=False)
    print(f"Saved fairness analysis for cohort {cohort} ({len(fairness_df)} rows).")


def save_fairness_criteria(cohort: str, criteria_df: pd.DataFrame) -> None:
    criteria_df.to_csv(fairness_criteria_path(cohort), index=False)
    print(f"Saved fairness criteria for cohort {cohort} ({len(criteria_df)} rows).")


def save_test_predictions(cohort: str, fold_idx: int, predictions_df: pd.DataFrame) -> None:
    path = test_predictions_path(cohort, fold_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(path, index=False)
    print(f"Saved test predictions for cohort {cohort}, fold {fold_idx} ({len(predictions_df)} rows).")


def save_feature_shift(cohort: str, shift_df: pd.DataFrame) -> None:
    shift_df.to_csv(feature_shift_path(cohort), index=False)
    print(f"Saved feature shift for cohort {cohort} ({len(shift_df)} features).")
