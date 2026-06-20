import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (roc_auc_score, average_precision_score,
                               f1_score, matthews_corrcoef, balanced_accuracy_score)

from utils.dataloader import *
from utils.range_merge import make_merge_strategies
from config import *


MERGE_STRATEGIES_LIST = [s.name for s in make_merge_strategies()]

# factory dict to build new instance each call
CLASSIFIERS = {
        "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "balanced_rf": lambda: BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "catboost": lambda: CatBoostClassifier(n_estimators=100, random_seed=42, verbose=False),
    }

FOLD_SEED = 42

def run_scan_step(cohorts) -> None:
    # --------------------------- Create continuous and binary mapping, ranges from labevents ------------------------
    # read in the cohort files to get cohort
    cohort_dfs = {cohort: load_cohort(cohort) for cohort in
                  cohorts}  # dict: cohort_name --> DataFrame {string: dataframe}

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


def run_merge_step(cohort: str) -> None:
    # load the ranges file. ranges is dict: (hadm_id, itemid) -> set of tuples {(lower, upper), ...}
    ranges = load_ranges(cohort)
    strategies = make_merge_strategies()
    strategies_to_run = [s for s in strategies
                         # only select merges for which no files exist
                         if not merged_ranges_path(cohort, s.name).exists()]

    existing = [s.name for s in strategies
                if merged_ranges_path(cohort, s.name).exists()]
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


def run_mapping_step(cohort: str) -> None:
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
    for strategy in MERGE_STRATEGIES_LIST:
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



def run_fold_step(cohort: str) -> None:
    # ---------------------- Split each cohort into N_FOLDS stratified, grouped folds ----------------------
    # skip if all fold files already exist
    if all(fold_path(cohort, i).exists() for i in range(N_FOLDS)):
        print(f"Folds already exist for cohort {cohort}.")
        return

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


def _knn_impute_ranges(real_ranges: dict, train_hadms: set, demo: pd.DataFrame,
                    targets: pd.DataFrame, k: int) -> dict:
    """
    Impute a range for each measured (hadm_id, itemid) pair lacking one, by averaging the
    bounds of the k nearest TRAINING donors (same itemid, same gender, closest age).
    No test information is used.

    :param real_ranges: (hadm_id, itemid) -> (lower, upper) for pairs that already have a range
    :param train_hadms: set of training hadm_ids (the only admissions allowed as donors)
    :param demo: DataFrame indexed by hadm_id with columns "age", "gender"
    :param targets: rows (hadm_id, itemid, age, gender) of measured pairs lacking a range
    :param k: number of nearest donors to average
    :return: {(hadm_id, itemid): (lower, upper)} with the imputed ranges
    """
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

def _impute_disc(cont_df: pd.DataFrame, train_split: np.array, demo: pd.DataFrame, merged_ranges) -> pd.DataFrame:
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
    imputed = _knn_impute_ranges(real_ranges, train_hadms, demo, targets, BEST_KNN) # dict: (hadm_id, itemid) -> (lower, upper)

    # merge real ranges with the imputed ones (keys are disjoint: targets have no real range),
    lookup = {**real_ranges, **imputed}
    # then discretise the full cohort using the combined lookup
    disc = _discretise(cont_df, lookup).add_prefix("disc_imp_")
    return disc

def train(cohort: str) -> None:
    print(f"Training {cohort}...")
    # load once per cohort:
    merged_ranges = load_merged_ranges(cohort, BEST_MERGE_STRATEGY)
    cont_raw = load_continuous_values(cohort)
    # reindex and set prefix to avoid ambiguous naming when concatenating
    cont = cont_raw.set_index("hadm_id").add_prefix("cont_")
    disc = load_discrete_values(cohort, BEST_MERGE_STRATEGY).set_index("hadm_id")
    bin = load_binary_values(cohort).set_index("hadm_id").add_prefix("bin_")
    cont_bin = pd.concat([bin, cont], axis=1)
    cohort_info = load_cohort(cohort).set_index("hadm_id")[["label", "age", "gender"]] # pd.Series indexed with hadm_id
    labels = cohort_info["label"]
    demo = cohort_info[["age", "gender"]]
    print(f"Label distribution for {cohort}: {labels.value_counts().to_dict()}")

    # dict: {feature set --> dict {model --> list [dict per fold {measure --> value}, ...]}}
    feature_list = ["cont", "disc", "disc_imp", "bin", "cont_bin", "disc_bin"]
    all_results = {feat_name: {name: [] for name in CLASSIFIERS} for feat_name in feature_list}

    # ----- cross validation loop (over 5 folds)---------------------------
    for fold_idx in range(N_FOLDS):
        print(f"Fold {fold_idx}...")
        train_split, test_split = load_fold(cohort, fold_idx) # load the (train_split, test) tuple, each np.array of [subject_id, hadm_id]
        train_ids = [int(h) for h in train_split[:, 1]]
        test_ids = [int(h) for h in test_split[:, 1]]
        disc_imp = _impute_disc(cont_raw, train_split, demo, merged_ranges)
        disc_bin = pd.concat([disc_imp, bin], axis=1)
        # split the labels for the fold
        y_train, y_test = labels.loc[train_ids], labels.loc[test_ids]
        # split the feature sets for the fold
        feature_sets = {
            "cont": (cont.reindex(train_ids), cont.reindex(test_ids)),
            "disc": (disc.reindex(train_ids), disc.reindex(test_ids)),
            "disc_imp": (disc_imp.reindex(train_ids), disc_imp.reindex(test_ids)),
            "bin": (bin.reindex(train_ids), bin.reindex(test_ids)),
            "cont_bin": (cont_bin.reindex(train_ids), cont_bin.reindex(test_ids)),
            "disc_bin": (disc_bin.reindex(train_ids), disc_bin.reindex(test_ids)),
        }
        # train_split each classifier for each feature set
        for feature_set_name, (X_train, X_test) in feature_sets.items():
            for classifier_name, classifier in CLASSIFIERS.items():
                model = classifier().fit(X_train, y_train) # build a fresh classifier instance in each call
                # predict
                y_prob = model.predict_proba(X_test)[:, 1]  # P(positive class) — for ranking metrics
                y_pred = model.predict(X_test)  # hard 0/1 labels   — for threshold metrics

                results = {
                    "auc_roc": roc_auc_score(y_test, y_prob),
                    "avg_prec": average_precision_score(y_test, y_prob),
                    "f1": f1_score(y_test, y_pred),
                    "mcc": matthews_corrcoef(y_test, y_pred),
                    "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                }
                all_results[feature_set_name][classifier_name].append(results)

    # build combined summary table
    rows = []
    for classifier_name in CLASSIFIERS:
        for metric in METRICS:
            row = {"model": classifier_name, "measure": metric}
            for feature_set_name in feature_list:
                fold_values = [fold[metric] for fold in all_results[feature_set_name][classifier_name]]
                row[f"mean_{feature_set_name}"] = np.mean(fold_values)
                #row[f"std_{feature_set_name}"] = np.std(fold_values)
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    save_metric_summary(cohort, summary_df)
