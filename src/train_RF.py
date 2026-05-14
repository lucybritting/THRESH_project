import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from catboost import CatBoostClassifier

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "models"

# TODO: implement balancing for normal RF

# TODO: cohort file not necessary? Add discretised and binarised file paths
COHORTS = {
    "aplasia": {
        "agg":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
        "folds":  DATA_DIR / "folds/mimic_cohort_aplasia_45_days",
    },
    "NF": {
        "agg":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_NF_30_days_agg.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
        "folds":  DATA_DIR / "folds/mimic_cohort_NF_30_days",
    },
}

N_FOLDS = 5

def load_data(cohort):
    # TODO: directly handle missing values here? --> set to np.nan?
    # aggregated file for continuous values
    agg = pd.read_csv(COHORTS[cohort]["agg"]).set_index("hadm_id")
    # TODO: load discretised and binarised files
    # labels per hadm_id
    labels = (pd.read_csv(COHORTS[cohort]["cohort"], usecols=["hadm_id", "label"]).set_index("hadm_id"))
    return agg, labels

# returns arrays of hadm_ids belonging to train, val, test part
def load_fold(cohort, fold_idx):
    with open(COHORTS[cohort]["folds"] / f"fold_{fold_idx}.pkl", "rb") as f:
        train, val, test = pickle.load(f)
    # each array is (n, 2): [subject_id, hadm_id] — extract hadm_id
    return train[:, 1], val[:, 1], test[:, 1]

# takes set of hadm_ids and returns the corresponding features and labels as np arrays
def get_features_labels(agg, labels, hadm_ids):
    # TODO: also extract features from discretised and binarised files
    # should not be the case.
    # drop any ids not present in the aggregated file
    # TODO: Why can this happen??
    hadm_ids = [h for h in hadm_ids if h in agg.index]
    # continuous value from aggregated
    X_cont = agg.loc[hadm_ids].values # selects rows from agg in order of the hadm_ids provided --> 2D np array of shape (n_admissions, n_features)
    y = labels.loc[hadm_ids]["label"].values # selects the labels of the provided hadm_ids in their order --> 1D np array of labels
    return X_cont, y

def train(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg, labels = load_data(args.cohort)
    print(f"Loaded {len(agg)} admissions, {agg.shape[1]} features  "
          f"(label distribution: {labels['label'].value_counts().to_dict()})\n")

    # TODO: add the other models here
    # models:
    models = {
        "rf": RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "balanced_rf": BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "catboost": CatBoostClassifier(n_estimators=100, random_seed=42, verbose=False),
    }

    # results: dict, keys are names of the models, values are dicts, one dict per fold, each containing the metric scores for this fold
    results = {name: [] for name in models}

    # iterate over all folders
    for fold_idx in range(N_FOLDS):
        # get the hadm_ids
        train_ids, val_ids, test_ids = load_fold(args.cohort, fold_idx)

        # concatenate train and val and get features and labels
        # TODO: add the discretised and binarised features
        X_train_cont, y_train = get_features_labels(agg, labels, np.concatenate([train_ids, val_ids]))
        X_test_cont, y_test = get_features_labels(agg, labels, test_ids)

        print(f"--- Fold {fold_idx} ---")

        # go over all models
        for name, model in models.items():
            # TODO: add discretised and binarised data
            model.fit(X_train_cont, y_train)
            y_prob = model.predict_proba(X_test_cont)[:, 1] # predict_proba outputs 2D array of probabilities for each class
            y_pred = model.predict(X_test_cont) # predict outputs 1D array of 0.5 threshold on majority vote

            fold_results = {
                "fold_idx": fold_idx,
                "auc_roc": roc_auc_score(y_test, y_prob),
                "avg_prec": average_precision_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            }

            results[name].append(fold_results)
            print(f"[{name}]"
                  f"AUC-ROC={fold_results['auc_roc']:.3f}  "
                  f"AP={fold_results['avg_prec']:.3f}  "
                  f"F1={fold_results['f1']:.3f}  "
                  f"MCC={fold_results['mcc']:.3f}  "
                  f"BalAcc={fold_results['balanced_acc']:.3f}")

            # TODO: should model be stored?

    # get average scores and std
    # iterate over results
    for name, result in results.items():
        result_df = pd.DataFrame(result)
        mean = result_df.drop(columns="fold_idx").mean() # pd series where idx is metric names (auc, avg_prec etc) and the values are the means
        std = result_df.drop(columns="fold_idx").std() # same here
        print(f"\n[{name}] Mean ± std across {N_FOLDS} folds:")
        # print the values for the different metrics
        for col in mean.index:
            print(f"  {col:<22} {mean[col]:.3f} ± {std[col]:.3f}")

        # save performance metric results
        out_path = OUTPUT_DIR / f"{args.cohort}_{name}_cv_results.csv"
        result_df.to_csv(out_path, index=False)
        print(f"Saved {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=list(COHORTS), required=True)
    args = parser.parse_args()

    train(args)









