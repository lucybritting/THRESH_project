import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from utils.dataloader import *
from utils.range_merge import WidestRange, AverageRange, NarrowestRange, MergeWarningLog

RANGE_MERGE_STRATEGIES = [
    WidestRange(warning_log=MergeWarningLog()),
    AverageRange(warning_log=MergeWarningLog()),
    NarrowestRange(warning_log=MergeWarningLog()),
]

N_FOLDS = 5
FOLD_SEED = 42


def run_scan_step(args) -> None:
    # --------------------------- Create continuous and binary mapping, ranges from labevents ------------------------
    # read in the cohort files to get cohort
    cohort_dfs = {cohort: load_cohort(cohort) for cohort in
                  args.cohorts}  # dict: cohort_name --> DataFrame {string: dataframe}

    # create output directories for cohorts, check for which cohorts the ranges file already exist
    have_ranges, need_ranges = create_output_directories(args.cohorts)  # lists of strings

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
            args.top100labs)
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


def run_merge_step(args) -> None:
    for cohort in args.cohorts:
        # load the ranges file. ranges is dict: (hadm_id, itemid) -> set of tuples {(lower, upper), ...}
        ranges = load_ranges(cohort)
        strategies_to_run = [s for s in RANGE_MERGE_STRATEGIES
                             # only select merges that are requested by args and for which no files exist
                             if s.name in args.range_merge
                             and not merged_ranges_path(cohort, s.name).exists()]

        existing = [s.name for s in RANGE_MERGE_STRATEGIES
                    if s.name in args.range_merge
                    and merged_ranges_path(cohort, s.name).exists()]
        if existing:
            print(f"Cohort {cohort}: merged range files already exist for {', '.join(existing)}.")

        if strategies_to_run:
            merged = {s.name: {} for s in
                      strategies_to_run}  # dict to save the merged ranges. string strategy_name -> dict of ranges

            # go over all (hadm_id, itemid) pairs in this cohort
            for (hadm_id, itemid), range_set in ranges.items():  # iterate over the ranges dict for this cohort
                # for each (hadm_id, itemid) pair create lists for lower and upper thresholds
                lowers = [r[0] for r in range_set if r[0] is not None]
                uppers = [r[1] for r in range_set if r[1] is not None]
                # for each strategy get lower and upper threshold and add to dict
                for strategy in strategies_to_run:
                    lower, upper = strategy.merge(lowers, uppers, hadm_id=hadm_id, itemid=itemid)
                    merged[strategy.name][(hadm_id, itemid)] = (lower, upper)

            # save the merged ranges files
            for strategy in strategies_to_run:
                save_merged_ranges(cohort, merged[strategy.name], strategy.name)
                if strategy.warning_log is not None:
                    print(f"Saved warnings for cohort {cohort} and strategy {strategy.name}: ")
                    strategy.warning_log.save(merge_warnings_path(cohort, strategy.name))


def run_mapping_step(args) -> None:
    for cohort in args.cohorts:
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
        for strategy in args.range_merge:
            # check if file already exists
            if discrete_mapping_path(cohort, strategy).exists():
                print(f"Cohort {cohort}: discrete mapping for {strategy} already exists.")
                continue
            # load continuous dataframe and merged ranges
            cont_df = load_continuous_values(cohort)
            merged_ranges = load_merged_ranges(cohort, strategy)

            lab_cols = [c for c in cont_df.columns if
                        c != "hadm_id"]  # all column names except the column "hadm_id" --> itemids
            long = cont_df.melt(id_vars="hadm_id", value_vars=lab_cols, var_name="itemid",
                                value_name="value")  # convert to long format. Each row becomes one (hadm_id, itemid, value) triple
            long["itemid"] = long["itemid"].astype(int)  # convert itemid from string to int
            long = long.dropna(subset=["value"])  # drop triplets with Nan value (added again later)

            def discretise(row):  # row has hadm_id, itemid, value
                key = (int(row["hadm_id"]), int(row["itemid"]))  # lookup key
                lower, upper = merged_ranges.get(key, (None,
                                                       None))  # look up the merged (lower, upper) for this key. If pair has no range at all, defaults to (None, None)
                if lower is None and upper is None:  # if no range exists, it cannot discretise
                    return None
                value = float(row["value"])
                if lower is not None and value < lower: return -1
                if upper is not None and value > upper: return 1
                return 0

            long["disc_value"] = long.apply(discretise,
                                            axis=1)  # calls the discretise function for every row and stores result in new column "disc_value"
            disc_wide = long.pivot(index="hadm_id", columns="itemid",
                                   values="disc_value")  # converts back from long format to wide format: rows = hadm_ids, columns = itemids, values= discretised values
            disc_wide.reset_index()  # make hadm_id from index to column
            save_discrete_values(cohort, disc_wide, strategy)


def run_fold_step(args) -> None:
    # ---------------------- Split each cohort into N_FOLDS stratified, grouped folds ----------------------
    for cohort in args.cohorts:
        # skip if all fold files already exist
        if all(fold_path(cohort, i).exists() for i in range(N_FOLDS)):
            print(f"Folds already exist for cohort {cohort}.")
            continue

        df = load_cohort(cohort)
        subject_ids = df["subject_id"].astype(int).to_numpy()
        hadm_ids = df["hadm_id"].astype(int).to_numpy()
        labels = df["label"].astype(int).to_numpy()
        ids = np.column_stack([subject_ids, hadm_ids]) # (n, 2) array of [subject_id, hadm_id]

        # stratify by label, group by subject_id so a patient never spans train/test
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED) # splitter object

        # 80/20 layout: each fold uses 1 split as test (~20%) and the other 4 as train (~80%)
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(ids, labels, groups=subject_ids)):
            save_fold(cohort, fold_idx, ids[train_idx], ids[test_idx])

        print(f"Saved {N_FOLDS} folds for cohort {cohort}.")
