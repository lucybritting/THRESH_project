import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from catboost import CatBoostClassifier

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

COHORTS = {
    "aplasia": {
        "cont":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv",
        "disc_merged": OUTPUT_DIR / "range_mapping/aplasia_discretised_merged.csv",
        "disc_unmerged": OUTPUT_DIR / "range_mapping/aplasia_discretised_unmerged.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
        "folds":  DATA_DIR / "folds/mimic_cohort_aplasia_45_days",
    },
    "NF": {
        "cont":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_NF_30_days_agg.csv",
        "disc_merged": OUTPUT_DIR / "range_mapping/NF_discretised_merged.csv",
        "disc_unmerged": OUTPUT_DIR / "range_mapping/NF_discretised_unmerged.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
        "folds":  DATA_DIR / "folds/mimic_cohort_NF_30_days",
    },
}

N_FOLDS = 5

def load_data(cohort, discr):
    # aggregated file for continuous values
    cont = pd.read_csv(COHORTS[cohort]["cont"]).set_index("hadm_id")
    # load discretised data
    disc = pd.read_csv(COHORTS[cohort][f"disc_{discr}"]).set_index("hadm_id")
    feature_sets = {
        "cont": cont,
        "disc": disc,
    }
    # labels per hadm_id
    labels = (pd.read_csv(COHORTS[cohort]["cohort"], usecols=["hadm_id", "label"]).set_index("hadm_id"))
    return feature_sets, labels

# returns arrays of hadm_ids belonging to train, val, test part
def load_fold(cohort, fold_idx):
    with open(COHORTS[cohort]["folds"] / f"fold_{fold_idx}.pkl", "rb") as f:
        train, val, test = pickle.load(f)
    # each array is (n, 2): [subject_id, hadm_id] — extract hadm_id
    return train[:, 1], val[:, 1], test[:, 1]

# takes set of hadm_ids and returns the corresponding features and labels as np arrays
def get_features_labels(feat, labels, hadm_ids):
    # should not be the case.
    # drop any ids not present in the aggregated file
    hadm_ids = [h for h in hadm_ids if h in feat.index]
    # feature values
    X = feat.loc[hadm_ids].values # selects rows from file in order of the hadm_ids provided --> 2D np array of shape (n_admissions, n_features)
    y = labels.loc[hadm_ids]["label"].values # selects the labels of the provided hadm_ids in their order --> 1D np array of labels
    return X, y

def train(args):
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    feature_sets, labels = load_data(args.cohort, args.discretised)
    print(f"(label distribution: {labels['label'].value_counts().to_dict()})\n")

    # models:
    models = {
        "rf": RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "balanced_rf": BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "catboost": CatBoostClassifier(n_estimators=100, random_seed=42, verbose=False),
    }



    # dict: {feature set --> dict {model --> list [dict per fold {measure --> value}, ...]}}
    all_results = {feat_name: {name: [] for name in models} for feat_name in feature_sets}

    # iterate over all feature sets
    for feat_name, feat in feature_sets.items():
        print(f"\n=== Feature: {feat_name} ===")

        for fold_idx in range(N_FOLDS):
            train_ids, val_ids, test_ids = load_fold(args.cohort, fold_idx)
            X_train, y_train = get_features_labels(feat, labels, np.concatenate([train_ids, val_ids]))
            X_test, y_test = get_features_labels(feat, labels, test_ids)

            print(f"--- Fold {fold_idx} ---")

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)

                fold_results = {
                    "auc_roc":      roc_auc_score(y_test, y_prob),
                    "avg_prec":     average_precision_score(y_test, y_prob),
                    "f1":           f1_score(y_test, y_pred),
                    "mcc":          matthews_corrcoef(y_test, y_pred),
                    "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                }
                all_results[feat_name][model_name].append(fold_results)
                print(f"  [{model_name}]  "
                      f"AUC-ROC={fold_results['auc_roc']:.3f}  "
                      f"AP={fold_results['avg_prec']:.3f}  "
                      f"F1={fold_results['f1']:.3f}  "
                      f"MCC={fold_results['mcc']:.3f}  "
                      f"BalAcc={fold_results['balanced_acc']:.3f}")

    # build combined summary table
    rows = []
    metrics = ["auc_roc", "avg_prec", "f1", "mcc", "balanced_acc"]
    for model_name in models:
        for measure in metrics:
            row = {"model": model_name, "measure": measure}
            for feat_name in feature_sets:
                fold_vals = [f[measure] for f in all_results[feat_name][model_name]]
                row[f"mean_{feat_name}"] = np.mean(fold_vals)
                row[f"std_{feat_name}"]  = np.std(fold_vals)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(f"\n{summary_df.to_string(index=False)}")

    out_path = OUTPUT_DIR / "metrics" / f"{args.cohort}_{args.discretised}_performance_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=list(COHORTS), required=True)
    parser.add_argument("--discretised", choices=["merged", "unmerged"], required=True)
    args = parser.parse_args()

    train(args)









