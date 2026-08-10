import argparse
from utils.pipeline import run_scan_step, run_merge_step, run_mapping_step, run_fold_step, train, list_cohorts, \
    cohort_feature_importance_analysis, feature_importance_analysis, performance_analysis, cohort_fairness_analysis, \
    sample_classification_analysis, fairness_analysis
from utils.dataloader import set_cohort_folder



parser = argparse.ArgumentParser()
parser.add_argument("--cohort_folder", choices=["test", "all"], required=True,
                    help="which cohort folder to run: all -> data/representative_cohorts, test -> data/cohorts")
parser.add_argument("--purge", action='store_true')  # set to delete all cohort outputs and rebuild them from scratch
parser.add_argument("--balanced_rf_only", action='store_true')  # set to train only balanced_rf, skipping rf and catboost
args = parser.parse_args()


set_cohort_folder(args.cohort_folder)  # pick data/cohorts (all) or data/representative_cohorts (test)
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
    # cohort-wise fairness analysis
    cohort_fairness_analysis(cohort)
    # per-sample classification heatmap across representations
    sample_classification_analysis(cohort)


# cross-cohort feature importance analysis
feature_importance_analysis(cohorts)

# cross-cohort performance analysis
performance_analysis(cohorts)

# cross-cohort fairness analysis
fairness_analysis(cohorts)


