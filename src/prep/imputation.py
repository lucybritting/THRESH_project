"""KNN imputation of missing reference ranges, and the representation built on it."""

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config.params import BEST_KNN
from .mapping import _discretise


def _knn_impute_ranges(real_ranges: dict, train_hadms: set, demo: pd.DataFrame,
                    targets: pd.DataFrame, k: int) -> dict:
    imputed = {}
    if targets.empty:
        return imputed

    # candidate donors: training pairs that have a real range, tagged with demographics + bounds
    cand_rows = [] # one row per valid donor
    for (hadm, itemid), (lower, upper) in real_ranges.items():
        if hadm not in train_hadms or hadm not in demo.index:
            continue  # donors must be training admissions we have demographics for
        cand_rows.append({
            "itemid": itemid,
            "gender": demo.at[hadm, "gender"],
            "age": demo.at[hadm, "age"],
            "lower": lower,
            "upper": upper,
        })
    cand = pd.DataFrame(cand_rows, columns=["itemid", "gender", "age", "lower", "upper"]) # df. Each dict becomes a row
    cand = cand.dropna(subset=["age", "gender"]) # remove the candidates for which age and gender are not given
    if cand.empty:
        return imputed

    cand_groups = dict(tuple(cand.groupby(["itemid", "gender"]))) # group the donor table by itemid and gender -> gives tuple ((itemid, gender), sub dataframe). Then make it as dict
    # cand_groups is dict: (itemid, gender) -> df of donors with this gender and itemid

    # remove targets with missing demographics
    targets = targets.dropna(subset=["age", "gender"])
    # group targets
    targets = targets.groupby(["itemid", "gender"])

    # for each (itemid, gender), find the nearest training donors by age and average their bounds
    for (itemid, gender), tg in targets: # key: (itemid, gender), value: tg (the sub dataframe of targets)
        cg = cand_groups.get((itemid, gender)) # get matching donors
        if cg is None or cg.empty:
            continue  # no same-(itemid, gender) donor -> leave missing (NaN)

        n = len(cg)
        cand_ages = cg["age"].to_numpy(dtype=float).reshape(-1, 1)
        cand_lower = cg["lower"].to_numpy(dtype=float)  # may contain NaN (one-sided ranges)
        cand_upper = cg["upper"].to_numpy(dtype=float)

        # fit KNN on age
        nn = NearestNeighbors(n_neighbors=min(k, n)).fit(cand_ages)
        _, neigh = nn.kneighbors(tg["age"].to_numpy(dtype=float).reshape(-1, 1))  # (n_targets, <=k)
        tgt_hadms = tg["hadm_id"].to_numpy()

        # average over the k donors
        with warnings.catch_warnings():  # all-NaN slice -> NaN bound, expected for one-sided ranges
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lower_avg = np.nanmean(cand_lower[neigh], axis=1)  # avg over the k donors, per target
            upper_avg = np.nanmean(cand_upper[neigh], axis=1)

        # write imputed range per target
        for j, hadm in enumerate(tgt_hadms):
            lo, up = lower_avg[j], upper_avg[j]
            imputed[(int(hadm), int(itemid))] = (
                None if np.isnan(lo) else float(lo),
                None if np.isnan(up) else float(up),
            )
    return imputed

def _impute_disc(cont_df: pd.DataFrame, train_split: np.array, demo: pd.DataFrame, merged_ranges, k: int = BEST_KNN) -> pd.DataFrame:
    # get existing ranges
    real_ranges = {key: val for key, val in merged_ranges.items()
                   if val[0] is not None or val[1] is not None} # dict (hadm_id, itemid) -> (lower, upper)
    real_keys = set(real_ranges.keys()) # set of (hadm_id, itemid)
    # get (hadm_id, itemid) pairs with measured value but missing range
    lab_cols = [c for c in cont_df.columns if c != "hadm_id"]
    measured = cont_df.melt(id_vars="hadm_id", value_vars=lab_cols, var_name="itemid", value_name="value") # long table (one row for each hadm_id, itemid pair)
    measured["itemid"] = measured["itemid"].astype(int)
    measured["hadm_id"] = measured["hadm_id"].astype(int)
    measured = measured.dropna(subset=["value"]) # drop those rows that have no value measured
    targets_bool = np.array([(h, i) not in real_keys for h, i in zip(measured["hadm_id"], measured["itemid"])]) # bool array. True where (hadm_id, itemid) pair is not in real keys
    targets = measured.loc[targets_bool, ["hadm_id", "itemid"]] # filter the measured array by the bool array to select the ones with missing range
    # add demographic info (age, gender)
    targets = targets.merge(demo, left_on="hadm_id", right_index=True)
    # get knn for the (hadm_id, itemid) pairs with missing range
    train_hadms = {int(h) for h in train_split[:, 1]}  # set of training hadm_ids
    imputed = _knn_impute_ranges(real_ranges, train_hadms, demo, targets, k) # dict: (hadm_id, itemid) -> (lower, upper)

    # merge real ranges with the imputed ones (keys are disjoint: targets have no real range),
    lookup = {**real_ranges, **imputed}
    # then discretise the full cohort using the combined lookup
    disc = _discretise(cont_df, lookup).add_prefix("disc_imp_")
    return disc
