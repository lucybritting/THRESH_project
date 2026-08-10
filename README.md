# THRESH_project
Analyse whether clinical intervals/missingness  give better predictability/robustness/fairness for random forests than continuous lab values.

## Data
MIMIC IV hosp

## Project structure


## Setup
environment
config.py

## Workflow
1. Set global variables in `config.py`
2. Run `MIMIC_analysis.py`: 
3. Run `main.py`: python main.py --cohort_folder {test|all}
    --cohort_folder (required): `all` runs every cohort in `data/representative_cohorts`, `test` runs the cohorts in `data/cohorts`.
    --purge to delete all cohort outputs and rebuild them from scratch
    --balanced_rf_only to only train on balanced random forest. catboost and random forest are skipped.


## Outputs
Per cohort:
### figures
`<cohort>_rank_shift.png`
Slope chart of feature importance rank shift between representations.
continuous - discretised
discretised - binarised
continuous - binarised
Takes top 20 labs from each representation. 
Ranked by the average importance across the folds. 
1 is most important. 
Ranks past 50 are compressed but ranking is maintained. 
Items that are never used in a representation are labelled with rank "unused".

`<cohort>_rank_shift_disc_imp.png`
Same as above for representations discretised - discretised + imputetd

`<cohort>_rank_bump.png`
Bump chart of feature importance ranks across folds within representations. 
4 Panels, one for each representation in STABILITY_FEATURE_SETS (cont, disc, disc + imp, bin)
Displays the top 20 labs per representation, ranked by average across folds. 

`<cohort>_fairness.png`
Performance of the model (BEST_CLASSIFIER) per demographic group (M, F, AGE_BANDS).
Gives Performance measures auc_roc, avg_prec, f1, mcc and balanced_acc. Combines sample count per group. 
Averages are averaged over the groups, whole cohort measures are excluded. 

`<cohort>_fairness_criteria.png`
Gap on fairness critera (FAIRNESS_CRITERIA) for age bands (gap between min and max) and gender. Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are removed from gap calculation.

`<cohort>_equalised_odds.png`
Gap of equalised odds between representations disc and disc + imputed to pooled cohort for each demographic group. 
Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are removed from gap calculation.

### folds
### mapping
`<cohort>_continuous.csv`
Rows: hadm_ids
Columns: itemids
Values: Mean of lab value for this (hadm_id, itemid) pair, Nan if a lab was never measured for this hadm_id

`<cohort>_binary.csv`
Rows: hadm_ids
Columns: itemids
Values: 0 if value was never measured for (hadm_id, itemid) pair. 1 if a value was measured. 

`<cohort>_discrete_average.csv`
Discretised values based on range merge strategy average for multi ranges. 
Rows: hadm_ids
Columns: itemids
Values: 
- -1 if continuous value below threshold
- 0 if continuous value within range
- 1 if continuous value above threshold
- NaN if value not measured or measured but no usable reference range

`<cohort>_discrete_narrowest.csv`
Discretised values based on range merge strategy narrowest for multi ranges. 
Rows: hadm_ids
Columns: itemids
Values: 
- -1 if continuous value below threshold
- 0 if continuous value within range
- 1 if continuous value above threshold
- NaN if value not measured or measured but no usable reference range

`<cohort>_discrete_widest.csv`
Discretised values based on range merge strategy widest for multi ranges. 
Rows: hadm_ids
Columns: itemids
Values: 
- -1 if continuous value below threshold
- 0 if continuous value within range
- 1 if continuous value above threshold
- NaN if value not measured or measured but no usable reference range

### metrics
`<cohort>_feature_importance.csv`
Rows: one row per (feature set x fold x feature) (feature is itemid)
Columns: cohort, feature_set, fold, feature, itemid, representation, importance
Recorded only for the feature sets in STABILITY_FEATURE_SETS (cont, disc, disc_imp, bin)

`<cohort>_performance_summary.csv`
Rows: one row per (model, measure) (models: random forest, balanced random forest, catboost; measure: auc_roc, avg_prec, f1, mcc, balanced_acc)
Columns: model, measure, mean and std for all feature sets (cont, disc, disc_imp, bin, disc_bin, cont_bin)

`<cohort>_feature_stability.csv`
Gives kendalls W over folds for each representation. only for BEST_CLASSIFIER.
Rows: One row per representation (cont, disc, disc_imp, bin)
Columns: cohort, representation, n_folds, n_features, kendalls w
n_features gives how many features are entered the ranking, globally unused ones have been dropped. 

`<cohort>_feature_shift.csv`
Per-labitem mean and std of importance across the folds per representation (cont, bin, disc, disc_imp). 
Unused labs have mean=0 and std=0.
Only for BEST-CLASSIFIER. 
Rows: one row per lab
Columns: itemid, mean_<rep>, std_<rep> 

`<cohort>_fairness.csv`
Per-demographic group model performance for every representation (STABILITY_FEATURE_SETS) and dempgraphic group (male, female,  AGE_BANDS)
Only calculated for BEST_CLASSIFIER.
Rows: one row per (feature_set, attribute, group) (attribute = all, gender, age band)
Columns: cohort, feature set, classifier, attribute, group, n, n_positive, prevalence, selection_rate, tpr, fpr, ppv, auc_roc, balanced_acc, avg_prec, f1, mcc

`<cohort>_fairness_criteria.csv`
Difference in fairness criteria between demographic groups. 
Using FAIRNESS_CRITERIA (statistical parity, predictive parity, equal opportunity, predictive equality, equalised odds)
Only calculated for BEST_CLASSIFIER.
Rows: one row per (feature set, attribute, criterion)
columns: cohort, feature set, classifier, attribute, criterion, quantity, difference, ratio, min_group, max_group, n_groups (number of groups taken into account), n_groups_total (number of groups before filtering)

### ranges
`<cohort>_ranges.csv`
Rows: One row per distinct reference range for a (hadm_id, itemid) pair
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

`<cohort>_ranges_average.csv`
Rows: One row per (hadm_id, itemid) pair. One range per row. Merged by average merge strategy: lower threshold averaged, upper threshold averaged
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

`<cohort>_ranges_narrowest.csv`
Rows: One row per (hadm_id, itemid) pair. One range per row. Merged by narroweset merge strategy: greatest lower threshold, smallest upper threshold
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

`<cohort>_ranges_widest.csv`
Rows: One row per (hadm_id, itemid) pair. One range per row. Merged by widest merge strategy: smallest lower threshold, greatest upper threshold
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

### test_predictions
`<cohort>_test_predictions_fold_<n>.csv`
Rows: one row per hadm_id and feature set
Columns: cohort, fold, feature_set, classifier, hadm_id, age, gender, true label, predicted probability, predicted label
Recorded for all feature sets. 

## Reports
Outputs over all cohorts

`rank_difference_cont_disc_heatmap.png`
`rank_difference_cont_bin_heatmap.png`
Heatmap of the difference in rank between continuous representation and discretised/binarised representation. Show the mean top 100 labs across cohorts from continuous representation. 
Labs sorted by average importance (continuous representation): leftmost: most important
Subtracts disc/bin rank from cont rank. So if a labitem becomes more important in disc/bin, it has a positive difference. 

`kendalls_w_heatmap.png`
Shows kendalls W (fold agreement on feature importance) for all cohorts and the representation continuous, discretised and binarised. 

`fairness_eo_age_band.png`
`fairness_eo_gender.png`
Shows equalised odds gap by representation across cohorts for age and gender. Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are excluded. Only shown for BEST_CLASSIFIER.
For age bands it shows the gap between best and worst case. 

## Documentation of the scripts and their functions
TODO: dependencies

### config.py
Set configurations

`TOP100LABS`: if FALSE, all itemids that is found in D_LABITEMS is used. If TRUE, curated list of sll_mimictop100_features_hadm.pkl is read and used. In both cases the blacklisted itemids are subtracted. 
`CHUNK_SIZE`: Set chunk size for labevents scan. 
`BEST_MERGE_STRATEGY`: Only the discretised file for this range merge strategy is used in classifier training.
`ALL_MERGE_STRATEGIES`: List of the names of the three range merge strategies. 
`N_FOLDS`: Number of folds the cohort is split into. 
`BEST_CLASSIFIER`: Classifier for which feature importance is calculated
`STABILITY_FEATURE_SETS`: feature sets for which feature importance is calculated. 
`REPRESENTATION_LABELS`: Display names for the four core representations (`cont` → continuous, `disc` → discretised, `disc_imp` → discretised + imputed, `bin` → binarised). Used for plot headers and column/row labels across the feature-importance, fairness and performance plots. 
`BEST_KNN`: k for nearest neighbour search for imputation. Used in the pipeline. 
`KNN_KS`: list of k for KNN neighbourhood for imputation. Only used in testing script for which the best k is
`AGE_BANDS`: Age groups for fairness analysis
`AGE_BAND_LABELS`: Labels for the age groups
`FAIRNESS_CRITERIA`: statistical parity (selection rate), predictive parity (ppv), equal opportunity (tpr), predictive equality (fpr), equalised odds.
`FAIRNESS_MIN_GROUP_SAMPLES`: min number of samples which a group must have to participate in gap calculation of fairness criteria
`FAIRNESS_MIN_LABEL_SAMPLES`: min number of samples a group must have to participate in gap calculation of fairness criteria
`CLASSIFICATION_HEATMAP_REPS`: Order and display names of all six feature representations (adds `cont_bin` and `disc_bin` to the four in `REPRESENTATION_LABELS`) for the rows of the sample classification heatmap; the dict order sets the row order.

### MIMIC_analysis.py
Lorem ipsum

### dataloader.py
In- and output functions

#### scan_labevents()
Scans labevents file for set of hadm_ids. 
Scans chunk-wise. Chunk size is set in `config.py`. 
Reads the columns `hadm_id, itemid, ref_range_lower, ref_range_upper, value` from labevents file. 
Keeps only rows for which admission is in the set of hadm_ids from the cohorts and itemid is in the set of itemids (if TOP100LABS. Blacklisted are always subtracted).
Returns dicts with ranges per (hadm_id, itemid) pair, values per (hadm_id, itemid) pair and itemid set.

### range_merge.py
Defines the three range merge strategies for multi ranges. Provides function `make_merge_strategies()` that creates a list of fresh instances of the three merge strategies. 
`MergeWarningLog`: 
dataclass to log if a merged range produces invalid range (upper < lower). Saved as csv if not empty.
`RangeMergeStrategy`:
Abstract base class. Each instance carries its own MergeWarningLog. 
`WidestRange`:
Class that takes the minimum of the lower thresholds and maximum of upper thresholds. gives the most permissive normal band. 
`NarrowestRange`:
Class that takes the maximum of the lower thresholds and the minimum of the upper thresholds. Gives the tightest normal band.
`AverageRange`: 
Class that takes the mean of the lower thresholds and the mean of the upper thresholds. 

### plots.py
TODO

### imputation_test.py
TODO

### knn_k_test.py
TODO

### merge_strategy_test.py
TODO

### main.py
Main file to run the pipeline. Implements the function from `pipeline.py`.
In each step the information is written to csv file. In the analysis steps, these csv files are read in. 

Flag `--cohort_folder` (required): `all` runs every cohort in `data/representative_cohorts/`, `test` runs the cohorts in `data/cohorts/`.
Flag `-purge`: delete all cohort outputs and rebuild them from scratch.
Flag `--balanced_rf_only`: Train only balanced random forest. Random forest and catboost are skipped. 

Lists all cohorts from the selected cohort folder (`data/representative_cohorts/` for `all`, `data/cohorts/` for `test`).
Scans labevents file once for all cohorts.
Then processes each cohort:
- merge multi ranges
- create discretised and binarised mappings
- create folds
- trains random forest, balanced random forest and catboost or balanced random forest only. Imputes before training.
- feature importance analysis
- fairness analysis
- per-sample classification heatmap
After cohort-wise processing, does the following:
- cross-cohort feature importance analysis
- cross-cohort performance analysis
- cross-cohort fairness analysis

### pipeline.py
Implements the pipeline functions. 

#### Dependencies
Lorem ipsum

#### run_scan_step()
Scans the labevents file for the union of all hadm_ids from all cohorts in folder data/cohorts. Skips cohorts for which a file <cohort>_ranges.csv already exists (except if --purge is set, then all cohort files have been deleted previously).
Reads in `itemid_blacklist.txt` from output/mimic_analysis and excludes blacklisted itemids 
Calls `scan_labevents()` from `dataloader.py`. 
Averages the values per (hadm_id, itemid) pair. 
**Outputs:**
- `<cohort>_continuous.csv`
- `<cohort>_ranges.csv`

#### run_merge_step()
Merge multi ranges based on three merge strategies.
Merge strategies: 
- widest: smallest lower threshold and greatest upper threshold
- average: average lower threshold and average upper threshold
- narrowest: greatest lower threshold and smallest upper threshold
Loads `<cohort>_ranges.csv` and builds three strategy objects. each strategy object has its MergeWarningLog, which captures if an lower threshold is greater than the upper threshold. In this case its added to the log and range set to `None`. If log not empty, its saved as csv. 
Skips if csv files with the different merged ranges already exist.
Non-existing ranges have thresholds `None`. 
**Output:** 
- `<cohort>_ranges_average.csv`
- `<cohort>_ranges_narrowest.csv`
- `<cohort>_ranges_widest.csv`

#### run_mapping_step()
Turns continuous values into categorical representations. It creates discretised representation based on the reference ranges and binary representation based on whether a value was measured or not. 
If a discretised file already exists, it skips calulation. 
Reads in `<cohort>_continuous.csv`.
**Binary mapping:**
Values set to 1, if value is not None, else 0. Encodes presence/absence of a value. 
**Discretised mapping:** 
Creates mapping for each merge strategy independently. 
-1 if continuous value below range, 0 if within range, 1 if above range, NaN of no value or no usable range. 
Reads in the files `<cohort>_ranges_<mergeStrategy>.csv`.
Calls helper `_discretise()`.
**Output:** 
- `<cohort>_binary.csv`
- `<cohort>_discrete_average.csv`
- `<cohort>_discrete_narrowest.csv`
- `<cohort>_discrete_widest.csv`

#### run_fold_step()
Splits the cohort into `N_FOLDS` folds, each with train and test set (80/20). Splitting is seeded with `FOLD_SEED`, hence split is randomised but deterministic. Checks if folds files already exist. If yes, skips this step. 
Stratifies the split by label: each fold preserves the overall label distribution (highly imbalanced cohorts).
Split is grouped by `subject_id` to avoid leakage of one patient with multiple hadm_ids. The same subject_id never appears in both train and test. 
**Output:** 
One pkl file for each fold. Each containing a (train, test) pair of [subject_id, hadm_id] arrays.

#### train()
Runs 5-fold cross validation, training every classifier (random forest, balanced random forest, catboost) on every representation. Only uses discretised file from `BEST_MERGE_STRATEGY`.
If flag `--balanced_rf_only` is set, then training only on balanced random forest. It creates new representations by concatenating and imputation. 
Uses the following feature sets: 
- continuous values
- discretised values (discretised by BEST_MERGE_STRATEGY)
- discretised + imputed
- binary values
- continuous - binary concatenated
- discretised (imputed) - binary concatenated

For each fold: 
- imputes ranges for discretised representation with helper `_impute_disc()`, based only on train set (recomputed for every fold to avoid leakage.)
- trains every classifier (except if balanced_rf_only is set) on every feature set. Builds a fresh model instance on each call. 
- records feature importance only for BEST_CLASSIFIER and the feature sets in STABILITY_FEATURE_SETS. (Uses helper `_split_feature_name`)
- predicts on test set
- calculates performance measures (auc_roc, avg_prec, f1, mcc, balanced_acc)
- saves predictions for BEST_CLASSIFIER only (hard labels and probabilities)

**Output:** 
- `<cohort>_test_predictions_fold_<n>.csv`
- `<cohort>_feature_importance.csv`
- `<cohort>_performance_summary.csv`

#### _impute_disc()
Builds disc_imp representation for one fold. Imputes ranges from the training set for values that are measured but have no reference range given. 
Calls `_knn_impute_ranges()` as helper. 

#### _knn_impute_ranges()
Performs the range imputation step for each measured (hadm_id, itemid) pair with missing range. 
Only takes ranges from training admissions to avoid leakage. 
Matching rule: same itemid, same gender, closest age, averaged over k training donors. 
For each (itemid, gender) pair find the k nearest neighbours by age and average their reference range thresholds. 

#### cohort_feature_importance_analysis()
Analysis of how much a labitem's importance depends on the representation (cont, disc, disc_imp, bin) and how stable the importance is across the folds. 
Takes file `<cohort>_feature_importance.csv`
**Kendall's W stability:** 
Compute Kendall's W for stability of feature importance across folds within representations. Use helper `_kendalls_w()` (implementation of kendall's w formula with tie correction)
**Mean/std of feature importance:**
Merges the folds to get mean and std importance per labitem per representation.
**Rank shift slope chart:** 
Creates slope charts of rank shifts between representations. 
In one figure: cont - disc, disc - bin, cont - bin
In separate figure: disc - disc_imp
**Rank bump chart for feature importance across folds:**
Within one representation, how stable is each lab's importance rank across the 5 folds.
one figure with 4 panels. One panel for each representation in STABILITY_FEATURE_SETS (continuois, discretised, discretised + imputed, binarised)s
Displays the top 20 ranks per representation, averaged across the folds. 

**Output:** 
- `<cohort>_feature_stability.csv` (Kendalls W across folds per representation)
- `<cohort>_feature_shift.csv` (Mean/std of feature importance per representation)
- `<cohort>_rank_shift.png`
- `<cohort>_rank_shift_disc_imp.png`
- `<cohort>_rank_bump.png`

#### cohort_fairness_analysis()
Compares the model performances per demographic group and computes fairness gaps between groups. 
Takes the pooled predictions on the test sets from each fold. Takes all test predictions concatenated
Creates per-group performance table. Uses helper `_group_performance()` to get model performance per demographic group.
Bias metric gap: calculate gap between bias metrics across the groups. Drop groups with <20 samples and <5 samples per label. 
Uses helper `_group_supported()` to exclude groups with too little support. 
**Output:**
- `<cohort>_fairness.csv`
- `<cohort>_fairness_criteria.csv`
- `<cohort>_fairness.png`
- `<cohort>_fairness_criteria.png`
- `<cohort>_equalised_odds.png`

#### _group_performance()
Helper used by `<cohort>_fairness_analysis()`.
Extracates predictions and ground truth for demographic group from whole cohort. Calculates performance measures only for this group. 
Measures that need positives and negatives are NaN for a single-class group (e.g. F1)

#### _group_supported()
Helper used by `<cohort>_fairness_analysis()`.
Filters groups by whether they have enough support to enter the fairness criteria gap calculation. To participate, a group needs >= 20 samples and >=5 samples per class (FAIRNESS_MIN_LABEL_SAMPLES, FAIRNESS_MIN_GROUP_SAMPLES)

#### _bias_metric_gap()
Helper used by `<cohort>_fairness_analysis()`.
Calculates fairness-gap per fairness criterion across demographic groups. It compares best and worst group. 

#### sample_classification_analysis():
Creates per-sample classification heatmap for BEST_CLASSIFIER. Samples sorted first by true label, then by mean predicted probability across cohorts. 
Takes CLASSIFICATION_HEATMAP_REPS as order and display name for representations. 

#### feature_importance_analysis()
Analyses the feature importance change across cohorts. Gives fold feature importance stability over the folds for each cohort and the representations continuous, discretised and binarised. 
Also analyses rank shift between continuous representation and binarised/discretised representation. 
Loads each cohort's `<cohort>_feature_shift.csv` and `<cohort>_feature_stability.csv` files. 
**Output:**
- `rank_difference_cont_disc_heatmap.png`
- `rank_difference_cont_bin_heatmap.png`
- `kendalls_w_heatmap.png`

#### performance_analysis()
Takes the files `<cohort>_performance_summary.csv` of all cohorts an creates heatmaps by cohort and representation (cont, disc, disc + imp, bin) for the performance measures AUC-ROC, F1, MCC. For BEST_CLASSIFIER only.
**Output:**
- `performance_f1_heatmap.png`
- `performance_mcc_heatmap.png`
- `performance_roc_heatmap.png`

#### fairness_analysis()
Prints equalised odd gaps by representation across cohorts. For gender and age band (max,min).
Reads the files `<cohort>_fairness_criteria.csv` for each cohort. Uses helper `_group_supported()` to exclude demographic groups with insufficient support (FAIRNESS_MIN_GROUP_SAMPLES, FAIRNESS_MIN_LABEL_SAMPLES)

**Output:**
- `fairness_eo_age_band.png`
- `fairness_eo_gender.png`