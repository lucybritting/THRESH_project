"""Cross-validated training over the feature representations of one cohort."""

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)

from config.params import BEST_CLASSIFIER, BEST_MERGE_STRATEGY, METRICS, N_FOLDS
from data_io.store import (load_binary_values, load_cohort, load_continuous_values,
                          load_discrete_values, load_fold, load_merged_ranges,
                          save_feature_importance, save_metric_summary,
                          save_test_predictions)
from prep.imputation import _impute_disc
from .classifiers import CLASSIFIERS


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
    disc_bin = pd.concat([disc, bin], axis=1)
    # all three base representations side by side. Uses the plain discretised frame, not the
    # imputed one, so it needs no per-fold rebuild and can be concatenated once here
    cont_disc_bin = pd.concat([cont, disc, bin], axis=1)
    cohort_info = load_cohort(cohort).set_index("hadm_id")[["label", "age", "gender"]] # pd.Series indexed with hadm_id
    labels = cohort_info["label"]
    demo = cohort_info[["age", "gender"]]
    print(f"Label distribution for {cohort}: {labels.value_counts().to_dict()}")

    # dict: {feature set --> dict {model --> list [dict per fold {measure --> value}, ...]}}
    feature_list = ["cont", "disc", "disc_imp", "bin", "cont_bin", "disc_bin", "cont_disc_bin"]
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
        #disc_bin = pd.concat([disc_imp, bin], axis=1)
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
            "cont_disc_bin": (cont_disc_bin.reindex(train_ids), cont_disc_bin.reindex(test_ids)),
        }
        # train_split each classifier for each feature set
        for feature_set_name, (X_train, X_test) in feature_sets.items():
            for classifier_name, classifier in classifiers.items():
                model = classifier().fit(X_train, y_train) # build a fresh classifier instance in each call
                # feature importance
                # recorded for every feature set. In a concatenated one a lab appears once per
                # encoding block, so "representation" says which block a row belongs to and
                # "feature_set" which model it was measured in; only together are they unique
                if classifier_name == BEST_CLASSIFIER:
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
