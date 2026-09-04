"""The one pass over labevents: continuous values and the raw reference ranges."""

from config.params import TOP100LABS
from data_io.labevents import scan_labevents
from data_io.paths import create_output_directories, remove_directories
from data_io.store import load_cohort, save_continuous_value, save_ranges


def run_scan_step(cohorts, purge: bool = False) -> None:
    # --------------------------- Create continuous and binary mapping, ranges from labevents ------------------------
    # read in the cohort files to get cohort
    cohort_dfs = {cohort: load_cohort(cohort) for cohort in
                  cohorts}  # dict: cohort_name --> DataFrame {string: dataframe}

    # delete existing cohort outputs so everything downstream is rebuilt from scratch
    if purge:
        remove_directories(cohorts)

    # create output directories for cohorts, check for which cohorts the ranges file already exist
    have_ranges, need_ranges = create_output_directories(cohorts)  # lists of strings

    if have_ranges:
        for cohort in have_ranges:
            print(f"Results already exist for cohort {cohort}.")

    # get values and range information for the cohorts that need ranges (files do not exist yet)
    if need_ranges:  # check if not empty
        # get union of all hadm_ids across need_range cohorts
        # build lookup: hadm_id -> set of cohorts it belongs to
        hadm_to_cohorts: dict[int, set[str]] = {}
        for cohort in need_ranges:
            for hadm_id in set(cohort_dfs[cohort]["hadm_id"]):
                hadm_to_cohorts.setdefault(hadm_id, set()).add(cohort)
        # union of all hadm_ids across need_ranges cohorts
        all_hadm_ids = set(hadm_to_cohorts.keys())

        # loads itemids without the blacklisted ones
        per_hadm_item_ranges, per_hadm_item_values, itemid_set = scan_labevents(
            all_hadm_ids,
            TOP100LABS)
        # average the values per (hadm_id, itemid) pair
        per_hadm_item_avg = {
            key: sum(vals) / len(vals)
            for key, vals in per_hadm_item_values.items()
        }

        # split dicts back into cohorts
        for cohort in need_ranges:
            # keep only the entries where hadm_id belongs to the current cohort
            # set of all hadm_ids that belong to the current cohort
            cohort_hadm_ids = set(cohort_dfs[cohort]["hadm_id"])
            cohort_ranges = {k: v for k, v in per_hadm_item_ranges.items() if k[
                0] in cohort_hadm_ids}  # dict: (hadm_id, itemid) -> set of (lower, upper) tuples (=set of ranges)
            cohort_avg = {k: v for k, v in per_hadm_item_avg.items() if
                          k[0] in cohort_hadm_ids}  # dict: (hadm_id, itemid) -> average value
            # save the results. NaN entries are implicitly added by the reindex in the save method
            hadm_ids = sorted(cohort_hadm_ids)
            itemids = sorted(itemid_set)
            save_continuous_value(cohort, cohort_avg, hadm_ids, itemids)
            save_ranges(cohort, cohort_ranges)
            print(f"Saved results for cohort {cohort}.")
