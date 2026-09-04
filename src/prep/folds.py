"""The stratified, subject-grouped cross-validation split."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config.params import N_FOLDS, FOLD_SEED
from data_io.paths import fold_path
from data_io.store import load_cohort, save_fold


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
