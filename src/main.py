import argparse
from prep.folds import run_fold_step
from prep.mapping import run_mapping_step
from prep.ranges import run_merge_step
from prep.scan import run_scan_step
from model.train import train
from analysis.performance import performance_analysis
from analysis.fairness import cohort_fairness_analysis, fairness_analysis
from analysis.importance import (cohort_feature_importance_analysis,
                                 feature_importance_analysis)
from analysis.classification import sample_classification_analysis
from analysis.selection import filter_cohorts_by_roc
from data_io import list_cohorts, set_cohort_folder, figure_path, blacklist_path
from config import ALL_MERGE_STRATEGIES, BEST_MERGE_STRATEGY, COHORT_FOLDERS



parser = argparse.ArgumentParser()
parser.add_argument("--cohort_folder", choices=list(COHORT_FOLDERS), required=True,
                    help="which cohort folder to run: all -> data/cohorts_100, "
                         "subset -> data/cohorts_50 (one cohort per code1), "
                         "test -> data/cohorts_test")
parser.add_argument("--purge", action='store_true')  # set to delete all cohort outputs and rebuild them from scratch
parser.add_argument("--all_classifiers", action='store_true',
                    help="train rf and catboost as well; without it only the "
                         "balanced random forest the analyses read is trained")
parser.add_argument("--all_mappings", action='store_true',
                    help=f"build merged ranges and discretised mappings for every merge strategy "
                         f"({', '.join(ALL_MERGE_STRATEGIES)}) instead of only {BEST_MERGE_STRATEGY}")
args = parser.parse_args()

if not blacklist_path().exists():
    raise SystemExit(f"No itemid blacklist at {blacklist_path()}.\n"
                     f"Run mimic_analysis.py first: it scans labevents once and writes the "
                     f"blacklist that fixes the feature set for every cohort.")


set_cohort_folder(args.cohort_folder)
cohorts = list_cohorts()

# scan labevents for all cohorts once -> get continuous mappings
run_scan_step(cohorts, args.purge)

# process cohort-wise
for index, cohort in enumerate(cohorts, start=1):
    # skip cohort if last file (classification heatmap) already exists for this cohort
    if figure_path(cohort, "classification_heatmap").exists():
        print(f"[{index}/{len(cohorts)}] Skipping {cohort}: already processed.")
        continue
    print(f"[{index}/{len(cohorts)}] Processing {cohort}.")
    # merge multi ranges
    run_merge_step(cohort, args.all_mappings)
    # create discretised and binarised mappings
    run_mapping_step(cohort, args.all_mappings)
    # separate the cohort into 5 stratified, grouped folds (train/test)
    run_fold_step(cohort)
    # train classifiers with 5-fold cross validation
    train(cohort, args.all_classifiers)
    # do cohort-wise feature importance stability analysis
    cohort_feature_importance_analysis(cohort)
    # cohort-wise fairness analysis
    cohort_fairness_analysis(cohort)
    # per-sample classification heatmap across representations
    sample_classification_analysis(cohort)



analysed, dropped = filter_cohorts_by_roc(cohorts)
print(f"Cross-cohort analyses over {len(analysed)} of {len(cohorts)} cohorts. "
      f"Excluded ({len(dropped)}): {', '.join(dropped) if dropped else 'none'}")

# cross-cohort feature importance analysis
feature_importance_analysis(analysed)

# cross-cohort performance analysis
performance_analysis(analysed)

# cross-cohort fairness analysis
fairness_analysis(analysed)


