"""The binary and discretised representations of a cohort's lab values."""

import numpy as np
import pandas as pd

from config.params import BEST_MERGE_STRATEGY
from data_io.paths import binary_mapping_path, discrete_mapping_path
from data_io.store import (load_continuous_values, load_merged_ranges, save_binary_values,
                           save_discrete_values)
from .ranges import MERGE_STRATEGIES_LIST


def _discretise(cont_df: pd.DataFrame, merged_ranges: dict) -> pd.DataFrame:
    """
    Discretise the wide continuous frame using a (hadm_id, itemid) -> (lower, upper) lookup.
    Returns a wide frame indexed by hadm_id: value < lower -> -1, > upper -> 1, else 0;
    NaN where the value is unmeasured or the pair has no (imputed or real) range.
    """
    lab_cols = [c for c in cont_df.columns if c != "hadm_id"] # all column names except the column "hadm_id" --> itemids
    long = cont_df.melt(id_vars="hadm_id", value_vars=lab_cols, var_name="itemid", value_name="value") # convert to long format. Each row becomes one (hadm_id, itemid, value)
    long["itemid"] = long["itemid"].astype(int) # convert itemid from string to int
    long["hadm_id"] = long["hadm_id"].astype(int)
    long = long.dropna(subset=["value"])  # drop triplets with Nan value (added again later) -> only measured pairs are discretised

    # vectorise as np arrays
    # look up the (lower, upper) for every measured pair; None -> np.nan via float cast
    keys = list(zip(long["hadm_id"], long["itemid"]))
    lower = np.array([merged_ranges.get(key, (None, None))[0] for key in keys], dtype=float) # 1D np array: lower bound for pair i or np.nan
    upper = np.array([merged_ranges.get(key, (None, None))[1] for key in keys], dtype=float) # 1D np array: upper bound for pair i or np.nan
    value = long["value"].to_numpy(dtype=float) # 1D np array: measured lab value for pair i

    # discretise using bool masks
    disc = np.zeros(len(value))
    disc[(~np.isnan(lower)) & (value < lower)] = -1 # ~ is element-wise bool NOT
    disc[(~np.isnan(upper)) & (value > upper)] = 1
    disc[np.isnan(lower) & np.isnan(upper)] = np.nan  # no range at all -> stays missing
    long = long.assign(disc_value=disc) # add disc_value as column
    long = long.pivot(index="hadm_id", columns="itemid", values="disc_value").reindex(index=cont_df["hadm_id"].astype(int),
                                                                               columns=[int(c) for c in
                                                                                        lab_cols])  # make wide format again (rows: hadmids, columns: itemids, values: discrete values) and add all hadm_id rows and itemid columns (string to int cast)
    return long


def run_mapping_step(cohort: str, all_mappings: bool = False) -> None:
    # ------------- BINARY MAPPING -------------------------------------
    if binary_mapping_path(cohort).exists():
        print(f"Binary mapping already exists for cohort {cohort}.")
    else:
        cont_df = load_continuous_values(cohort)
        lab_cols = [c for c in cont_df.columns if
                    c != "hadm_id"]  # all column names except the column "hadm_id" --> itemids
        # build in one shot so hadm_id is first and the frame is not fragmented
        binary_df = pd.concat(
            [cont_df[["hadm_id"]], cont_df[lab_cols].notna().astype(int)],  # 0 if NaN, 1 else
            axis=1,
        )
        save_binary_values(cohort, binary_df)

    # ------------- DISCRETE MAPPING -----------------------------------
    strategies = [s for s in MERGE_STRATEGIES_LIST
                  if all_mappings or s == BEST_MERGE_STRATEGY]
    for strategy in strategies:
        # check if file already exists
        if discrete_mapping_path(cohort, strategy).exists():
            print(f"Cohort {cohort}: discrete mapping for {strategy} already exists.")
            continue
        # load continuous dataframe and merged ranges
        cont_df = load_continuous_values(cohort)
        merged_ranges = load_merged_ranges(cohort, strategy)
        # discretise
        disc = _discretise(cont_df, merged_ranges).reset_index()
        save_discrete_values(cohort, disc, strategy)
