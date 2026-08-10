import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (roc_auc_score, average_precision_score,
                               f1_score, matthews_corrcoef, balanced_accuracy_score)

from utils.dataloader import *
from utils.plots import (plot_rank_shift_slope_chart, plot_rank_bump_chart,
                         plot_value_heatmap, plot_rank_difference_heatmap,
                         plot_fairness, plot_dumbbell_chart,
                         plot_classification_heatmap, plot_fairness_eo_dumbbell)
from utils.range_merge import make_merge_strategies
from config import *

# Names of reference range merge strategies
MERGE_STRATEGIES_LIST = [s.name for s in make_merge_strategies()]

# factory dict to build new instance each call
CLASSIFIERS = {
        "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "balanced_rf": lambda: BalancedRandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        "catboost": lambda: CatBoostClassifier(n_estimators=100, random_seed=42, verbose=False),
    }





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



def _split_feature_name(feature_name: str) -> tuple:
    """
    Splits a model feature name into representation and itemid. The prefixes are the ones added in
    train() ("cont_", "bin_", "disc_imp_"); the plain discrete frame keeps the bare itemid.
    :param feature_name: string, e.g. "cont_50902", "disc_imp_50902" or "50902"
    :return: tuple (representation, itemid), e.g. ("cont", 50902) or ("disc", 50902)
    """
    prefix, _, itemid = feature_name.rpartition("_")  # split on the LAST underscore -> keeps "disc_imp" intact
    if not prefix:
        return "disc", int(itemid)  # no prefix -> plain discrete frame
    return prefix, int(itemid)



def train(cohort: str, balanced_rf_only: bool = False) -> None:
    classifiers = {"balanced_rf": CLASSIFIERS["balanced_rf"]} if balanced_rf_only else CLASSIFIERS
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
    all_results = {feat_name: {name: [] for name in classifiers} for feat_name in feature_list}

    importance_rows = []

    # ----- cross validation loop (over 5 folds)---------------------------
    for fold_idx in range(N_FOLDS):
        print(f"Fold {fold_idx}...")
        # store predictions for fairness analysis
        prediction_rows = []  # reset per fold: one file per fold
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
            for classifier_name, classifier in classifiers.items():
                model = classifier().fit(X_train, y_train) # build a fresh classifier instance in each call
                # feature importance
                if classifier_name == BEST_CLASSIFIER and feature_set_name in STABILITY_FEATURE_SETS:
                    for feat_name, imp in zip(X_train.columns, model.feature_importances_):
                        representation, itemid = _split_feature_name(feat_name)
                        importance_rows.append({
                            "cohort": cohort,
                            "feature_set": feature_set_name,
                            "fold": fold_idx,
                            "feature": feat_name,
                            "itemid": itemid,
                            "representation": representation,
                            "importance": imp,
                        })
                # predict
                y_prob = model.predict_proba(X_test)[:, 1]  # probabilities. 1D np array. Order matches the rows in X_test
                y_pred = model.predict(X_test)  # hard 0/1 labels. 1D np array

                results = {
                    "auc_roc": roc_auc_score(y_test, y_prob),
                    "avg_prec": average_precision_score(y_test, y_prob),
                    "f1": f1_score(y_test, y_pred),
                    "mcc": matthews_corrcoef(y_test, y_pred),
                    "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                }
                all_results[feature_set_name][classifier_name].append(results)

                # Save predictions for fairness analysis only on BEST_CLASSIFIER
                if classifier_name == BEST_CLASSIFIER:
                    # get demographics for hadm_ids in X_test in the order how they appear in X_test --> matches y_prob
                    test_demo = demo.loc[X_test.index]
                    prediction_rows.append(pd.DataFrame({
                        "cohort": cohort,
                        "fold": fold_idx,
                        "feature_set": feature_set_name,
                        "classifier": classifier_name,
                        "hadm_id": X_test.index,
                        "age": test_demo["age"].to_numpy(),
                        "gender": test_demo["gender"].to_numpy(),
                        "y_true": y_test.to_numpy(),
                        "y_prob": y_prob,
                        "y_pred": y_pred,
                    }))
        # concatenate predictions on this fold for all featuer sets
        save_test_predictions(cohort, fold_idx, pd.concat(prediction_rows, ignore_index=True))

    if importance_rows:
        save_feature_importance(cohort, pd.DataFrame(importance_rows))

    # build combined summary table
    rows = []
    for classifier_name in classifiers:
        for metric in METRICS:
            row = {"model": classifier_name, "measure": metric}
            for feature_set_name in feature_list:
                fold_values = [fold[metric] for fold in all_results[feature_set_name][classifier_name]]
                row[f"mean_{feature_set_name}"] = np.mean(fold_values)
                row[f"std_{feature_set_name}"] = np.std(fold_values, ddof=1) # sample std, as in the shift analysis
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    save_metric_summary(cohort, summary_df)


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
    importance_df = load_feature_importance(cohort) # one row per (feature_set, fold, feature)

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
    # helper function to build one panel
    def bump_data(rep: str, top_k: int = 20) -> pd.DataFrame:
        wide = by_representation[rep].pivot(index="fold", columns="itemid", values="importance")
        # rank within each fold, 1 = most important, then one row per feature
        fold_ranks = wide.rank(axis=1, ascending=False, method="min").transpose() # rank within one fold
        selected = fold_ranks.mean(axis=1).nsmallest(top_k).index
        return fold_ranks.loc[selected].rename(index=lambda itemid: item_labels.get(itemid, itemid))

    bump_panels = []
    # build panels for each representation
    for rep in STABILITY_FEATURE_SETS:
        fold_ranks = bump_data(rep)
        # each panel accents its own top 5, so the accent means "top of this
        # representation" everywhere; the end labels carry which lab it is
        top5 = list(fold_ranks.mean(axis=1).nsmallest(5).index)
        bump_panels.append((REPRESENTATION_LABELS[rep], fold_ranks, top5))

    plot_rank_bump_chart(
        panels=bump_panels,
        title=f"Feature importance rank across folds — {cohort}",
        subtitle="Top 20 labs per representation, coloured in importance blocks of 5 (grouped by mean across folds), "
                 f"all named, top 5 in bold. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
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
    top100 = cont_ranks.mean().nsmallest(100).index  #select top 100 most important labs

    # get rank difference between continuous and other representation
    def rank_difference(other_rep: str) -> pd.DataFrame:
        diff = (cont_ranks - rank_df.xs(other_rep, level=1))[top100]
        diff = diff.loc[cohorts].rename(columns=lambda i: item_labels.get(i, i)) # force to be in order given by cohorts, and insert labels
        # append the across-cohort average per lab as a final summary row
        diff.loc["average"] = diff.mean(axis=0)
        return diff

    plot_rank_difference_heatmap(
        rank_difference("disc"),
        title="Rank shift from continuous to discretised representation",
        subtitle="The 100 labs with the best mean continuous rank across cohorts; "
                 "blue = more important once discretised, red = less important. Bottom row is average rank across cohorts. "
                 f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        colourbar_label="continuous rank − discretised rank",
        path=reports_path() / "rank_difference_cont_disc_heatmap.png",
        separate_last_row=True)

    plot_rank_difference_heatmap(
        rank_difference("bin"),
        title="Rank shift from continuous to binarised representation",
        subtitle="The 100 labs with the best mean continuous rank across cohorts; "
                 "blue = more important once binarised, red = less important. Bottom row is average rank across cohorts. "
                 f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        colourbar_label="continuous rank − binarised rank",
        path=reports_path() / "rank_difference_cont_bin_heatmap.png",
        separate_last_row=True)

    # KENDALL W HEATMAP
    stability_df = pd.concat([load_feature_stability(cohort) for cohort in cohorts]) # load each cohorts feature stability file
    w_df = stability_df.pivot(index="cohort", columns="representation", values="kendalls_w") # reshape to wide format
    # drop the imputed discrete representation from the plot
    shown_reps = ["cont", "disc", "bin"]
    w_df = w_df.loc[cohorts, shown_reps].rename(columns=REPRESENTATION_LABELS)
    # append the across-cohort average per representation as a final summary row
    w_df.loc["average"] = w_df.mean(axis=0)

    plot_value_heatmap(
        w_df,
        title="Fold agreement on the feature importance ranking",
        subtitle="Kendall's W per cohort and representation; 1 = the 5 folds rank the "
                 "features identically. Bottom row is the average across cohorts. "
                 f"Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        colourbar_label="Kendall's W",
        path=reports_path() / "kendalls_w_heatmap.png",
        separate_last_row=True)


def performance_analysis(cohorts: list[str]) -> None:
    summaries = {cohort: load_metric_summary(cohort) for cohort in cohorts}
    representations = ["cont", "disc", "disc_imp", "bin"]
    measures = [("auc_roc", "roc"),
                ("f1", "f1"),
                ("mcc", "mcc")]

    for measure, slug in measures:
        measure_label = METRIC_LABELS[measure]
        rows = {}
        for cohort in cohorts:
            summary = summaries[cohort]
            row = summary[(summary["model"] == BEST_CLASSIFIER)
                          & (summary["measure"] == measure)].iloc[0]
            rows[cohort] = [row[f"mean_{rep}"] for rep in representations]
        perf_df = pd.DataFrame.from_dict(rows, orient="index", columns=representations)
        perf_df = perf_df.loc[cohorts].rename(columns=REPRESENTATION_LABELS)
        # append the across-cohort average per representation as a final summary row
        perf_df.loc["average"] = perf_df.mean(axis=0)

        plot_value_heatmap(
            perf_df,
            title=f"{measure_label} by cohort and representation",
            subtitle=f"{measure_label} averaged over the 5 folds, {CLASSIFIER_LABELS[BEST_CLASSIFIER]}. "
                     "Bottom row is the average across cohorts.",
            colourbar_label=measure_label,
            path=reports_path() / f"performance_{slug}_heatmap.png",
            separate_last_row=True)


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
            path=reports_path() / f"fairness_eo_{attribute}.png",
            note=note)



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
    bias_metric_df = pd.DataFrame(bias_metrics_rows)
    save_fairness(cohort, fairness_df)
    save_fairness_criteria(cohort, bias_metric_df)

    # PLOT FAIRNESS: MODEL PERFORMANCE PER DEMOGRAPHIC GROUP
    plot_df = fairness_df[fairness_df["feature_set"].isin(STABILITY_FEATURE_SETS)] # take only feature sets in stability feature sets
    group_order = ["all", "F", "M"] + AGE_BAND_LABELS
    subgroups = ["F", "M"] + AGE_BAND_LABELS  # the demographic groups the average pools
    # group by performance metric: one panel per metric
    panels = []
    for measure in METRICS:
        panel_df = plot_df.pivot(index="group", columns="feature_set", values=measure)
        panel_df = panel_df.loc[group_order, STABILITY_FEATURE_SETS].rename(columns=REPRESENTATION_LABELS)
        # average across all demographic groups (excludes the overall 'all' row)
        panel_df.loc["average"] = panel_df.loc[subgroups].mean(axis=0)
        panels.append((METRIC_LABELS[measure], panel_df))  # human-readable panel title

    # sample count and label distribution per group
    counts_src = plot_df[plot_df["feature_set"] == STABILITY_FEATURE_SETS[0]].set_index("group")
    counts_df = pd.DataFrame({"n_negative": counts_src["n"] - counts_src["n_positive"],
                              "n_positive": counts_src["n_positive"]})

    plot_fairness(
        panels,
        separator_after=["all", "M", AGE_BAND_LABELS[-1]],
        dashed_after=["F"] + AGE_BAND_LABELS[:-1],
        counts_df=counts_df,  # extra panel: sample count and label balance per group
        title=f"Performance by demographic group — {cohort}",
        subtitle=f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}. The top row is "
                 "the whole cohort, the bottom row the average across the demographic groups",
        path=figure_path(cohort, "fairness"))

    # PLOT GAPS IN BIAS METRICS BETWEEN GROUPS
    # the criteria themselves: one panel per attribute, one row per criterion
    bias_metric_df = bias_metric_df[bias_metric_df["feature_set"].isin(STABILITY_FEATURE_SETS)] # take only feature sets in STABILITY FEATURES
    metric_panels = []
    for attribute, attribute_df in bias_metric_df.groupby("attribute"):
        panel_df = attribute_df.pivot(index="criterion", columns="feature_set",
                                      values="difference")
        panel_df = (panel_df.loc[list(FAIRNESS_CRITERIA), STABILITY_FEATURE_SETS]
                    .rename(columns=REPRESENTATION_LABELS, index=FAIRNESS_CRITERIA_LABELS))  # readable rows
        metric_panels.append((FAIRNESS_ATTRIBUTE_LABELS[attribute], panel_df))  # readable header

    # name the under-supported groups left out of the gaps, per attribute
    # look groups up by the raw attribute key, but display the readable label
    excluded_note = "; ".join(
        f"{FAIRNESS_ATTRIBUTE_LABELS[attr]}: {', '.join(excluded_groups[attr]) if excluded_groups.get(attr) else 'none'}"
        for attr in sorted(excluded_groups))
    note = (f"Groups with fewer than {FAIRNESS_MIN_GROUP_SAMPLES} samples or fewer than "
            f"{FAIRNESS_MIN_LABEL_SAMPLES} of either label are excluded. Excluded groups: {excluded_note}")

    plot_fairness(
        metric_panels,
        # a hairline between every criterion row (all but the last, which needs no rule);
        # must use the renamed row labels so it matches the panel index
        separator_after=[FAIRNESS_CRITERIA_LABELS[c] for c in list(FAIRNESS_CRITERIA)[:-1]],
        title=f"Fairness criteria gaps — {cohort}",
        subtitle="Per criterion, the gap between the best- and worst-served group of the "
                 "attribute (max − min). "
                 f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        path=figure_path(cohort, "fairness_criteria"),
        share_x=True,
        x_min=0.0,
        note=note)      # which demographic groups were dropped for too little support

    # PLOT EQUALISED ODDS DUMBBELL PLOT: DISC - DISC + IMP
    eo_reps = ["disc", "disc_imp"]
    eo_df = fairness_df[fairness_df["feature_set"].isin(eo_reps)]
    reference = eo_df[eo_df["group"] == "all"].set_index("feature_set")  # pooled cohort rates
    # drop under-supported groups, consistent with the criteria gaps
    excluded = {g for groups in excluded_groups.values() for g in groups}
    subgroups = [g for g in ["F", "M"] + AGE_BAND_LABELS if g not in excluded]
    eo_gaps = {}
    for rep in eo_reps:
        rep_groups = eo_df[(eo_df["feature_set"] == rep)
                           & (eo_df["group"].isin(subgroups))].set_index("group")
        eo_gaps[rep] = np.maximum(
            (rep_groups["tpr"] - reference.loc[rep, "tpr"]).abs(),
            (rep_groups["fpr"] - reference.loc[rep, "fpr"]).abs())
    eo_gap_df = pd.DataFrame(eo_gaps).reindex(subgroups)  # rows = groups, cols = reps

    # name the under-supported groups dropped from the gaps
    eo_excluded = [g for g in ["F", "M"] + AGE_BAND_LABELS if g in excluded]
    eo_note = (f"Groups with fewer than {FAIRNESS_MIN_GROUP_SAMPLES} samples or fewer than "
               f"{FAIRNESS_MIN_LABEL_SAMPLES} of either label are excluded. "
               f"Excluded groups: {', '.join(eo_excluded) if eo_excluded else 'none'}")

    plot_dumbbell_chart(
        eo_gap_df, left_col="disc", right_col="disc_imp",
        left_label="discretised", right_label="discretised + imputed",
        title=f"Equalised odds gap by group — {cohort}",
        subtitle="Per group, discretised and discretised + imputed against the pooled cohort rate; "
                 f"0 = matches the cohort. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}.",
        xlabel="Equalised odds gap (0 = fair)",
        path=figure_path(cohort, "equalised_odds"),
        note=eo_note)



def sample_classification_analysis(cohort: str) -> None:
    """
    Per-sample classification heatmap for BEST_CLASSIFIER.
    """
    predictions_df = load_test_predictions(cohort)  # all folds concatenated
    predictions_df = predictions_df[predictions_df["classifier"] == BEST_CLASSIFIER] # take only BEST_CLASSIFIER

    rep_order = [rep for rep in CLASSIFICATION_HEATMAP_REPS
                 if rep in predictions_df["feature_set"].unique()] # sort representations. Robust if one representation does not exist.

    # order samples by ground truth, then by mean prediction probabiltiy
    per_sample = predictions_df.groupby("hadm_id").agg(y_true=("y_true", "first"),
                                                       mean_prob=("y_prob", "mean"))
    sample_order = per_sample.sort_values(["y_true", "mean_prob"]).index

    # rows = representations, columns = samples, values = predicted class
    pred = predictions_df.pivot(index="feature_set", columns="hadm_id", values="y_pred")
    pred = pred.reindex(index=rep_order, columns=sample_order)

    truth_row = per_sample.loc[sample_order, "y_true"].to_numpy()[None, :]  # ground truth on top
    data = np.vstack([truth_row, pred.to_numpy()])
    row_labels = ["ground truth"] + [CLASSIFICATION_HEATMAP_REPS[rep] for rep in rep_order]

    # x-position where the sorted true label flips from 0 to 1
    n_negative = int((per_sample.loc[sample_order, "y_true"] == 0).sum())
    split_x = n_negative - 0.5 if 0 < n_negative < len(sample_order) else None

    plot_classification_heatmap(
        data, row_labels, n_ground_truth_rows=1, split_x=split_x,
        title=f"Per-sample classification by representation — {cohort}",
        subtitle=f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}",
        xlabel=f"{data.shape[1]:,} samples",
        path=figure_path(cohort, "classification_heatmap"))
