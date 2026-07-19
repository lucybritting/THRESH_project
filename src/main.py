import argparse
from utils.pipeline import run_scan_step, run_merge_step, run_mapping_step, run_fold_step, train, list_cohorts, \
    cohort_feature_importance_analysis, feature_importance_analysis

# TODO: note somewhere whether it was done with top100features or all features?

parser = argparse.ArgumentParser()
#parser.add_argument("--cohorts", nargs='+', required=True)
parser.add_argument("--range_merge", nargs='+', default=['widest'], choices=['widest', 'average', 'narrowest'])
parser.add_argument("--purge", action='store_true')  # set to delete all cohort outputs and rebuild them from scratch
parser.add_argument("--balanced_rf_only", action='store_true')  # set to train only balanced_rf, skipping rf and catboost
#parser.add_argument("--top100labs", action='store_true')  # set if want to use only the top 100 labs, not all labs
#parser.add_argument("--classifiers", nargs='+', default='all', choices={'rf', 'balanced_rf', 'catboost'})
args = parser.parse_args()


cohorts = list_cohorts()

# scan labevents for all cohorts once -> get continuous mappings
run_scan_step(cohorts, args.purge)

# process cohort-wise
for cohort in cohorts:
    # merge multi ranges
    run_merge_step(cohort)
    # create discretised and binarised mappings
    run_mapping_step(cohort)
    # separate the cohort into 5 stratified, grouped folds (train/test)
    run_fold_step(cohort)
    # train classifiers with 5-fold cross validation
    train(cohort, args.balanced_rf_only)
    # do cohort-wise feature importance stability analysis
    cohort_feature_importance_analysis(cohort)

# cross-cohort feature importance analysis
feature_importance_analysis(cohorts)


