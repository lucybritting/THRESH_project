# THRESH_project
Analyse whether clinical intervals/missingness  give better predictability/robustness/fairness for random forests than continuous lab values.

## Data
MIMIC IV hosp
- admissions
- d_labitems
- labevents
- patients
- cohorts extracted from MIMIC IV

## Project structure

```
src/
├─ main.py                the pipeline: one step per stage, in order
├─ mimic_analysis.py      run once before main.py, writes itemid_blacklist.txt
├─ cohort_analysis.py     standalone report on what the cohorts contain
├─ config/                paths.py (machine) · params.py (results) · labels.py (display)
├─ data_io/               paths.py · store.py (load_*/save_*) · labevents.py (the raw scan)
├─ prep/                  scan · ranges · mapping · imputation · folds
├─ model/                 classifiers.py (the factories) · train.py
├─ analysis/              importance · performance · fairness · classification
│                         + grouping.py (ICD chapters) · selection.py (the ROC gate)
├─ plots/                 theme.py (the shared look) · plots.py (the charts)
└─ studies/               parameter comparisons behind two constants in config/params.py

data/
├─ raw_files/             labevents, d_labitems, admissions, patients
├─ cohorts_100/           the 100 representative cohorts        (--cohort_folder all)
├─ cohorts_50/            one cohort per code1, a subset of the above   (subset)
├─ cohorts_test/          3 cohorts for a quick run                     (test)
└─ top_features/          the curated top-100 lab list, used only if TOP100LABS

output/
├─ mimic_analysis/        itemid_blacklist.txt and the labevents statistics
├─ <cohort>/              mapping · ranges · folds · metrics · test_predictions · figures
└─ reports/               cross-cohort: performance · fairness · importance
                          + cohort_analysis · test_knn_k · test_range_merge
```


## Setup

```
conda env create -f environment.yaml
conda activate thresh_env
```

Then place the MIMIC-IV hosp files in `data/raw_files/` and the cohort CSVs in one of the
three cohort folders above, and set the global variables (if necessary) in `config/` — at minimum check
`DATA_DIR`/`OUTPUT_DIR` in `config/paths.py` and the choices in `config/params.py`.

## Workflow

> **Cohort files must be named `cohort_<code1>-<code2>.csv.gz`** (e.g. `cohort_F17-C22.csv.gz`). This is a hard requirement, not a convention — see [Cohort naming](#cohort-naming) below. A cohort named anything else runs all the way through training and then crashes in the cross-cohort step.

1. Optional: Set global variables in `config/`
2. Run `mimic_analysis.py`: scans the whole labevents file once and writes `output/mimic_analysis/`, including `itemid_blacklist.txt`, which fixes the feature set for every cohort. Only needs re-running if a blacklist criterion changes — and if it does, step 3 needs `--purge`.
3. Run `main.py`: python main.py --cohort_folder {test|all|subset}
    --cohort_folder (required): `all` runs every cohort in `data/cohorts_100`, `subset` runs `data/cohorts_50` (one cohort per code1), `test` runs the cohorts in `data/cohorts_test`.
    --purge to delete all cohort outputs and rebuild them from scratch
    --all_classifiers to train random forest and catboost as well. Without it only the balanced random forest is trained, which is the only classifier the analyses read.
    --all_mappings to build ranges and discretised mappings for all three merge strategies instead of only the best one.
4. Optionally run `cohort_analysis.py --cohort_folder {test|all|subset}`: describes what the cohorts contain — lab coverage and label distribution — from the files step 3 wrote. Reads only; builds nothing.

The two scripts under `studies/` are not part of this workflow. They were run once to choose `BEST_KNN` and `BEST_MERGE_STRATEGY`, and are re-run only to revisit those choices.

## Cohort naming

The pipeline only runs to completion for cohorts named **`cohort_<code1>-<code2>.csv.gz`** — an ICD-10 code for the disease the cohort is built around, a hyphen, and an ICD-10 code for the target. `cohort_F17-C22.csv.gz`, `cohort_H81-E55.csv.gz`.

The cross-cohort analyses read the codes straight out of the filename, because the cohort size, prevalence and codes are not recorded anywhere else — `_disease_group()` takes the code before the hyphen, `_target_group()` the one after, both in `analysis/grouping.py`. All three group by code1 (importance and fairness for their disease-grouped figures, performance for its disease box plots); only performance also groups by code2.

Only the *start* of each hyphen-segment is parsed — a letter followed by digits — so anything after the code is free text and a mnemonic can ride along: `cohort_C80-D70_NF` parses exactly as `cohort_C80-D70` would. A label placed *before* the code does not work: that segment matches no chapter, and the cohort drops into the "Unmapped ICD chapter" group of every disease-grouped figure rather than raising.

A name with no hyphen at all raises `IndexError` in `performance_analysis()`, which is the last step of a run — after the scan, all folds and all training. Every file in the three cohort folders conforms; this is what a freely named new cohort will hit.


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
7 panels, one for each representation in `CLASSIFICATION_HEATMAP_REPS` — the four single ones and the three concatenated ones. Unlike the rank shift chart above, which is restricted to `STABILITY_FEATURE_SETS`, this one covers every trained feature set.
Displays the top 20 labs per representation, ranked by mean rank across the folds, coloured in blocks of 5 with the top 5 in bold.
In a concatenated representation a lab appears once per encoding block, and its importances are **summed** over those blocks before ranking, so each lab is one line per panel however many ways it is encoded. 

`<cohort>_fairness.png`
Performance of the model (BEST_CLASSIFIER) per demographic group (M, F, AGE_BANDS).
Gives Performance measures auc_roc, avg_prec, f1, mcc and balanced_acc. Combines sample count per group. 
Averages are averaged over the groups, whole cohort measures are excluded. 

`<cohort>_fairness_criteria.png`
Gap on fairness critera (FAIRNESS_CRITERIA) for age bands (gap between min and max) and gender. Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are removed from gap calculation.

`<cohort>_equalised_odds.png`
Per demographic group, the worse of the tpr and fpr distance from the pooled cohort rate, so 0 means the group matches the cohort. One dot per representation: continuous, discretised and binarised. `disc_imp` is dropped here although the two figures above keep it, so the panel compares the three base representations only.
Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are removed from gap calculation, and named in the note under the figure. If every group is under-supported the figure is skipped and a line is printed instead.

`<cohort>_classification_heatmap.png`
One column per admission, one row per representation in `CLASSIFICATION_HEATMAP_REPS`, cell colour is the predicted class. The ground truth is a separate row on top, ruled off from the predictions.
Admissions are sorted by true label and then by mean predicted probability across the representations, so the negatives run left to right in rising confidence, then the positives; a vertical rule marks where the true label flips. Reading down a column says which representations got that admission right, and the block structure says whether they fail on the same admissions or on different ones — which no aggregate measure shows.
All seven representations, for BEST_CLASSIFIER only, over the held-out predictions of all 5 folds pooled. A representation missing from a cohort's predictions is skipped rather than drawn empty.
This is the **last** file written for a cohort, which is what `main.py` checks to decide whether a cohort has already been processed.

### folds
`fold_<i>.pkl`
One file per fold, `N_FOLDS` of them, each a pickled `(train, test)` tuple. Each half is an
`(n, 2)` integer array of `[subject_id, hadm_id]` — the ids, not the features, so the same
split can be applied to every representation.
An 80/20 split: each fold holds out one of the `N_FOLDS` parts as test and trains on the
other four. Stratified by label and grouped by `subject_id`, so a patient with several
admissions never lands on both sides. Seeded with `FOLD_SEED`, so the split is random but
reproducible.
Across the five folds the test halves partition the cohort exactly: every admission is held
out once and only once, which is what lets the fairness analysis pool all five folds' test
predictions and get each admission back a single time.

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
Discretised values based on range merge strategy average for multi ranges. Only written if `main.py` ran with `--all_mappings`. 
Rows: hadm_ids
Columns: itemids
Values: 
- -1 if continuous value below threshold
- 0 if continuous value within range
- 1 if continuous value above threshold
- NaN if value not measured or measured but no usable reference range

`<cohort>_discrete_narrowest.csv`
Discretised values based on range merge strategy narrowest for multi ranges. Only written if `main.py` ran with `--all_mappings`. 
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
Rows: one row per (feature_set x fold x feature) (feature is itemid)
Columns: cohort, feature_set, fold, feature, itemid, representation, importance
Recorded for every trained feature set, all seven of them, and for BEST_CLASSIFIER only. `cohort_feature_importance_analysis()` then reads back only the ones in STABILITY_FEATURE_SETS (cont, disc, disc_imp, bin); the concatenated feature sets are on disk for anyone who wants them.

`<cohort>_performance_summary.csv`
Rows: one row per (model, measure) (models: random forest, balanced random forest, catboost; measure: auc_roc, avg_prec, f1, mcc, balanced_acc)
Columns: model, measure, then `mean_<feature_set>` and `std_<feature_set>` for each of the seven feature sets (cont, disc, disc_imp, bin, cont_bin, disc_bin, cont_disc_bin) — 16 columns in all
A cohort is skipped once processed, so adding a feature set to existing cohorts needs `--purge` (or deleting their outputs); otherwise their summary files keep the columns they were written with.

`<cohort>_feature_stability.csv`
Gives kendalls W over folds for each representation. only for BEST_CLASSIFIER.
Rows: One row per representation (cont, disc, disc_imp, bin)
Columns: cohort, representation, n_folds, n_features, kendalls_w
n_features gives how many features are entered the ranking, globally unused ones have been dropped. 

`<cohort>_feature_shift.csv`
Per-labitem mean and std of importance across the folds per representation (cont, bin, disc, disc_imp). 
Only for BEST-CLASSIFIER. 
Rows: one row per lab that is used at all — a lab whose importance is 0.0 in every fold of every one of those four representations is dropped, not written as a row of zeros, so this file is usually much shorter than the cohort's feature count.
A lab that survives that filter but is unused in *one* representation does get mean=0 and std=0 in that representation's two columns.
Columns: itemid, mean_<rep>, std_<rep> 

`<cohort>_fairness.csv`
Per-demographic group model performance for every trained representation (all seven) and demographic group (male, female, AGE_BANDS). The per-cohort fairness figures show only the four non-concatenated ones, but the csv carries them all.
Only calculated for BEST_CLASSIFIER.
Rows: one row per (feature_set, attribute, group) (attribute = all, gender, age band)
Columns: cohort, feature_set, classifier, attribute, group, n, n_positive, prevalence, selection_rate, tpr, fpr, ppv, auc_roc, balanced_acc, avg_prec, f1, mcc

`<cohort>_fairness_criteria.csv`
Difference in fairness criteria between demographic groups. 
Using FAIRNESS_CRITERIA (statistical parity, predictive parity, equal opportunity, predictive equality, equalised odds)
Only calculated for BEST_CLASSIFIER.
Rows: one row per (feature_set, attribute, criterion)
Columns: cohort, feature_set, classifier, attribute, criterion, quantity, difference, ratio, min_group, max_group, n_groups (number of groups taken into account), n_groups_total (number of groups before filtering)

### ranges
`<cohort>_ranges.csv`
Rows: One row per distinct reference range for a (hadm_id, itemid) pair
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

`<cohort>_ranges_average.csv`
Rows: One row per (hadm_id, itemid) pair. One range per row. Merged by average merge strategy: lower threshold averaged, upper threshold averaged. Only written if `main.py` ran with `--all_mappings`.
columns: hadm_id, itemid, ref_range_lower, ref_range_upper

`<cohort>_ranges_narrowest.csv`
Rows: One row per (hadm_id, itemid) pair. One range per row. Merged by narroweset merge strategy: greatest lower threshold, smallest upper threshold. Only written if `main.py` ran with `--all_mappings`.
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
Outputs over all cohorts.

Cohorts are filtered before the cross-cohort analyses: only cohorts whose fold-averaged AUC-ROC on the `ROC_FILTER_REP` representation reaches `MIN_COHORT_ROC` are included (see `filter_cohorts_by_roc()`). Cohorts below the threshold have not learned anything, so their ranks, gaps and scores would only add noise to the aggregates. The excluded cohorts are printed to the console at the start of the cross-cohort step but are **not** noted on the figures themselves, so record the threshold used when reporting these plots. The per-cohort outputs above are produced for every cohort regardless of the filter.

`rank_difference_cont_disc_heatmap.png`
`rank_difference_cont_bin_heatmap.png`
Heatmap of the difference in rank between continuous representation and discretised/binarised representation. Rows are cohorts, columns are labs, the bottom row is the average across cohorts.
Subtracts disc/bin rank from cont rank. So if a labitem becomes more important in disc/bin, it has a positive difference.
**Columns:** the union of the top `RANK_DIFF_TOP_N` labs by mean rank in each of `RANK_DIFF_REPRESENTATIONS` (cont, disc, bin), minus any lab that carries no reference range in any of the cohorts plotted. Both figures get the same columns, so they can be read against each other lab by lab.
**Colour scale:** diverging red/white/blue, symmetric about zero, with a non-linear ramp — `EMPHASIS_FRACTION` (85%) of each half is spent on shifts up to a knee, taken per figure as the `EMPHASIS_QUANTILE`-th percentile (90th) of its own absolute shifts. Ticked at 0, ±knee and ±limit. Both constants and `_emphasised_diverging_norm()` are in `plots/plots.py`. The scale is set by each figure's own extreme cell, so shades are not comparable between figures or runs.

`rank_difference_cont_disc_disease_heatmap.png`
`rank_difference_cont_bin_disease_heatmap.png`
The same two grids with the cohort rows collapsed to one row per ICD chapter of code1, the cohort count in each row label. Same columns and same colour rules as the per-cohort versions. The bottom row is the average across cohorts, so it is cohort-weighted and matches the per-cohort figure rather than being the mean of the group rows.

`kendalls_w_heatmap.png`
Shows kendalls W (fold agreement on feature importance) for all cohorts and the representations continuous, discretised and binarised (`STABILITY_FEATURE_SETS` minus `CROSS_COHORT_EXCLUDED_REPS`). Bottom row is the average across cohorts.

`kendalls_w_boxplot.png`
The same Kendall's W values as a distribution rather than a grid: one box per representation over all included cohorts, every cohort a dot. Answers "is one representation ranked more stably than another across cohorts" directly, where the heatmap answers it cohort by cohort.

`performance_roc_vs_cohort_properties.png`
One dot per cohort per representation against a cohort property, with a least squares fit per representation: cohort size (log scale) in the first panel, share of positive labels in the second. The dashed rule marks `MIN_COHORT_ROC`; nothing is plotted below it, since the filter removed those cohorts. A third panel plots the two properties against each other, size against prevalence, one neutral dot per cohort and no representations, to show whether the two x axes are themselves related.

`performance_roc_boxplot.png`
Distribution of the per-cohort AUC-ROC, one box per representation over all included cohorts. The box spans the interquartile range, the line in it is the median and the whiskers reach the minimum and maximum, so nothing is trimmed or marked an outlier. Every cohort is also drawn as a dot, jittered horizontally so equal values do not hide behind one another (the jitter is seeded, so reruns look identical).
Rules extend the `ROC_FILTER_REP` (continuous) median and quartiles across the whole axis, so every other representation can be read against that spread by eye. Six representations get a box: everything trained except `CROSS_COHORT_EXCLUDED_REPS` (discretised + imputed).

`performance_roc_boxplot_disease.png`
`performance_roc_boxplot_target.png`
`performance_roc_boxplot_size.png`
`performance_roc_boxplot_prevalence.png`
The same boxes repeated per group of the four stratifications: groups along the x-axis, one box per representation within each group, cohort count under each group label. The concatenated representations (`CONCATENATED_REPS`) are dropped from these four, leaving continuous, discretised and binarised.
The four groupings are the ICD chapter of code1 (disease), the ICD chapter of code2 (target), the cohort-size bin (`COHORT_SIZE_BINS`) and the label-distribution bin (`PREVALENCE_BINS`). Both ICD groupings match the **whole** code and not only its letter, because two letters split across chapters: D00-D49 belongs to Neoplasms with C while D50-D89 is its own chapter, and H00-H59 (eye) and H60-H95 (ear) are separate chapters. So `cohort_H43-H54` is eye and `cohort_H81-E55` is ear. Chapters are defined in `ICD_CHAPTERS` and the group order follows that list; a code matching no chapter (U, Z, or a malformed code) falls into `?`/"Unmapped ICD chapter" rather than silently disappearing. Size and prevalence are read from the cohort label column, not from the cohort name.

Only AUC-ROC is plotted across cohorts. F1, MCC, average precision and balanced accuracy are computed for every fold and written to every `<cohort>_performance_summary.csv`; read them from there.

`fairness_eo_age_band.png`
`fairness_eo_gender.png`
Shows equalised odds gap by representation across cohorts for age and gender. Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are excluded, and named in the note under the figure. Only shown for BEST_CLASSIFIER.
For age bands it shows the gap between best and worst case.

`fairness_eo_boxplot.png`
The same equalised-odds gaps as a distribution: one box per representation, repeated for gender and for age band, every (cohort, attribute) a dot. Lower is fairer, 0 is equal across groups. Three representations — continuous, discretised, binarised — since `CROSS_COHORT_EXCLUDED_REPS` and `CONCATENATED_REPS` are both dropped.
Where the two dumbbell figures above show which cohort moved, this shows whether the representations differ in the aggregate at all.

`fairness_eo_boxplot_disease_gender.png`
`fairness_eo_boxplot_disease_age_band.png`
The same boxes cut by the ICD chapter of code1, one figure per attribute: does discretising cost fairness more in some disease areas than others.

`fairness_eo_boxplot_group_size.png`
One dot per (cohort, demographic group) rather than per cohort: the worse of the tpr and fpr distance from that cohort's own pooled rate, boxed by how many admissions the group holds (`FAIRNESS_GROUP_SIZE_BINS`).
No group-size floor is applied here, unlike every other fairness figure: how the gap behaves as the group shrinks is the point of the figure.

## Documentation of the scripts and their functions

Three scripts are meant to be run, all at the top of `src/`: `mimic_analysis.py` (once, to fix
the feature set), `main.py` (the pipeline), and `cohort_analysis.py` (a report on the inputs).
Everything else is library code, grouped by pipeline stage. Each entry below names the module
its function lives in.

### config/
Set configurations. Split into three modules by what a change to one means, and re-exported
from the package, so `from config import BEST_KNN` works regardless of which module holds it.

`config/paths.py`: where the data lives and how it is read — `PROJECT_ROOT`, `DATA_DIR`,
`OUTPUT_DIR`, the four raw file paths, `COHORT_FOLDERS`, `CHUNK_SIZE`. Describes this machine,
not the experiment.
`config/params.py`: every choice that changes a result — thresholds, folds, `BEST_*`, the ICD
chapters, and the bin edges with their labels. A run is only reproducible against the version
of this file it ran with.
`config/labels.py`: display names that reach a reader and that no result depends on. Bin
labels are the exception and stay in `params.py`, next to the edges `pd.cut` pairs them with.

`TOP100LABS`: if FALSE, all itemids that is found in D_LABITEMS is used. If TRUE, the curated list in `data/top_features/all_mimic_top100_features_hadm.pkl` is read and used instead. In both cases the blacklisted itemids are subtracted. 
`CHUNK_SIZE`: Set chunk size for labevents scan. 
`MIN_RANGE_COVERAGE_PCT`: a lab is blacklisted when a reference range is recorded for no more than this share of the admissions that actually have a value for it. Read by `mimic_analysis.py` as the `low_range_coverage` criterion, not by the pipeline. The exact value is not sensitive: the observed coverages leave an empty band between 5.73% and 37.99%, so anything from 6 to 37 removes the same four labs. Changing it changes the feature set, so re-run `main.py --purge` after. 
`BEST_MERGE_STRATEGY`: Only the discretised file for this range merge strategy is used in classifier training.
`ALL_MERGE_STRATEGIES`: List of the names of the three range merge strategies. 
`N_FOLDS`: Number of folds the cohort is split into. 
`FOLD_SEED`: seed for the fold split, so the shuffle is random but reproducible. A run only reproduces against the same seed *and* the same library versions. 
`METRICS`: the performance measures computed for every (fold, feature set, classifier) and written to `<cohort>_performance_summary.csv`. Only AUC-ROC is plotted across cohorts; the rest are read from the summary files. 
`BEST_CLASSIFIER`: Classifier for which feature importance is calculated
`ROC_FILTER_REP`: Representation whose AUC-ROC decides whether a cohort enters the cross-cohort analyses. 
`MIN_COHORT_ROC`: Minimum fold-averaged AUC-ROC on `ROC_FILTER_REP` for a cohort to enter the cross-cohort analyses. Cohorts below it are dropped by `filter_cohorts_by_roc()`. Does not affect the per-cohort outputs. 
`ICD_CHAPTERS`: ICD-10 chapters used to stratify the cohorts by disease. Entries are `(letter, lowest number, highest number, group key, group name)`; the first match wins and the list order is the row order of the plot. Matched on the whole code because D and H each split across two chapters (D00-D49 with the neoplasms, H00-H59 eye vs H60-H95 ear). Note M is the musculoskeletal chapter and N the genitourinary one. 
`UNMAPPED_ICD_CHAPTER`: Fallback group for a code that matches no chapter (U, Z, or a malformed code), so such a cohort still appears on the plot instead of disappearing. 
`COHORT_SIZE_BINS` / `COHORT_SIZE_LABELS`: Bin edges and labels for stratifying the cohorts by number of admissions in the performance report. 
`PREVALENCE_BINS` / `PREVALENCE_LABELS`: Bin edges and labels for stratifying the cohorts by share of positive labels. Both are fixed edges rather than quantiles, so that the bins stay comparable when the cohort set changes (a different `MIN_COHORT_ROC`, or more cohorts). Edges are left-open / right-closed, as `pd.cut` reads them. 
`RANGE_FEMALE_ONLY_PCT` / `RANGE_MALE_ONLY_PCT` / `RANGE_AGE_SPREAD_YEARS`: thresholds for deciding whether a lab item's several reference ranges are explained by the sex or age of the admissions behind them, used by `mimic_range_structure.png`. Sex needs one group at or above `RANGE_FEMALE_ONLY_PCT` female and one at or below `RANGE_MALE_ONLY_PCT`, i.e. one range per sex; both are insensitive, since no group falls between 66.7% and 95% or between 5% and 19%. Age needs the mean age to differ by `RANGE_AGE_SPREAD_YEARS` across the groups, and unlike the sex cut-offs this one is a judgement call. There is no minimum group size: every range group is compared however few admissions carry it. See the note under `mimic_range_structure.png` for what that buys and what it costs. 
`STABILITY_FEATURE_SETS`: the feature sets the importance analyses read and plot. Importance itself is recorded for every trained feature set; this is the subset that reaches the stability, rank shift and rank bump figures. 
`CROSS_COHORT_EXCLUDED_REPS`: representations left out of the cross-cohort figures. They are still trained, still written to every per-cohort output and still in the summary csv files; this only governs the aggregate figures. Holds `disc_imp`. 
`CONCATENATED_REPS`: the representations built by concatenating the feature blocks of the single ones. Dropped from the *grouped* figures only: a grouped plot already spends its width on the groups, and the question it answers is about the single representations. The ungrouped figures keep them. 
`FAIRNESS_GROUP_SIZE_BINS` / `FAIRNESS_GROUP_SIZE_LABELS`: bin edges and labels for `fairness_eo_boxplot_group_size.png`, which boxes the per-group equalised-odds gaps by how many admissions the group holds. Fixed edges rather than quantiles, for the same reason as `COHORT_SIZE_BINS`. 
`RANK_DIFF_REPRESENTATIONS` / `RANK_DIFF_TOP_N`: which representations contribute columns to the rank difference heatmaps, and how many labs each contributes. The columns are the **union** of the top `RANK_DIFF_TOP_N` of each, so a lab that only becomes important once discretised or binarised is not missed by a continuous-only selection. Raising `RANK_DIFF_TOP_N` widens both figures; the union is always smaller than `RANK_DIFF_TOP_N` × the number of representations, because the representations agree on the strongest labs. 

`REPRESENTATION_LABELS`: Display names for all seven representations — the four core ones (`cont` → continuous, `disc` → discretised, `disc_imp` → discretised + imputed, `bin` → binarised) and the three concatenated ones (`cont_bin` → cont + bin, `disc_bin` → disc + bin, `cont_disc_bin` → cont + disc + bin). Used for plot headers and column/row labels across the feature-importance, fairness and performance plots. Abbreviated where `CLASSIFICATION_HEATMAP_REPS` spells the same names out. 
`METRIC_LABELS`: display names for the metrics in `METRICS`, used for plot titles, panel headers and axis labels. 
`CLASSIFIER_LABELS`: display names for the three classifiers, used in the subtitle of every figure that is specific to one of them (which is nearly all of them, since the analyses read `BEST_CLASSIFIER` only). 
`FAIRNESS_CRITERIA_LABELS` / `FAIRNESS_ATTRIBUTE_LABELS`: display names for the keys of `FAIRNESS_CRITERIA` and for the two demographic attributes, so the figures read "equalised odds" and "Age band" rather than the raw keys. `FAIRNESS_ATTRIBUTE_LABELS` also fixes the group order on the cross-cohort figures, gender before age band. 
`BEST_KNN`: k for nearest neighbour search for imputation. Used in the pipeline. 
`KNN_KS`: list of k for KNN neighbourhood for imputation. Only used in testing script for which the best k is
`AGE_BANDS`: Age groups for fairness analysis
`AGE_BAND_LABELS`: Labels for the age groups
`FAIRNESS_CRITERIA`: statistical parity (selection rate), predictive parity (ppv), equal opportunity (tpr), predictive equality (fpr), equalised odds.
`FAIRNESS_MIN_GROUP_SAMPLES`: min number of samples which a group must have to participate in gap calculation of fairness criteria
`FAIRNESS_MIN_LABEL_SAMPLES`: min number of samples a group must have to participate in gap calculation of fairness criteria
`CLASSIFICATION_HEATMAP_REPS`: Order and display names of all seven feature representations (adds `cont_bin`, `disc_bin` and `cont_disc_bin` to the four core ones) for the rows of the sample classification heatmap; the dict order sets the row order. A representation missing from a cohort's predictions is skipped rather than drawn as an empty row.

### mimic_analysis.py
Runs once, before `main.py`, over the **whole** labevents file (not filtered to any cohort). Collects per-itemid statistics into an `ItemStats` object — the distinct reference ranges and which admissions carry them, the units, how many values parse as numbers vs as strings, and which admissions had a value at all — and writes nine files to `output/mimic_analysis/` (six csv files, the blacklist itself, and two figures). The important product is `itemid_blacklist.txt`, which fixes the feature set for every cohort downstream: `load_all_itemids()` is d_labitems minus the blacklist, so anything blacklisted here never becomes a column.

**Blacklist criteria.** An itemid is blacklisted when it cannot carry a usable discretised feature, for any of four reasons. They are mutually exclusive in the output, in this order of precedence:

| reason | meaning | n |
|---|---|---|
| `too_few_values` | fewer than 10 distinct numeric values: categorical or near-constant, nothing to threshold | 449 |
| `no_range` | measured, but no reference range anywhere in labevents, so it can never be discretised | 195 |
| `low_range_coverage` | a range exists, but on no more than `MIN_RANGE_COVERAGE_PCT` of the admissions that have a value, so the discretised column is empty for nearly all of them while the continuous one stays dense | 4 |
| `never_measured` | not in labevents at all, so it would be an all-NaN column in every cohort | 754 |

`no_range` is the degenerate case of `low_range_coverage` (0% coverage) and is kept apart from it because "no range at all" and "a range too sparse to use" are different situations worth counting separately.

That is 1402 of the 1650 itemids in d_labitems, leaving **248 features**. `never_measured` is the set difference `d_labitems − stats`, since `stats` only gets an entry for an itemid that actually appears in a labevents row.

Changing any criterion changes the feature matrix, so every cohort output built on the old blacklist is invalid. Re-run `main.py` with `--purge`.

**Outputs** (all in `output/mimic_analysis/`):

`itemid_blacklist.txt`
One itemid per line, sorted. Read by `load_blacklist()`.

`mimic_blacklist_analysis.csv`
Readable version of the same list.
Rows: one per blacklisted itemid
Columns: itemid, label, reason, n_numeric_values, n_unique_numeric_values, n_ranges, range_coverage_pct
`range_coverage_pct` is the share of an item's measured admissions that also carry a range — the number the `low_range_coverage` decision rests on. It is carried on every row that has one, not only on the rows dropped for it, so a lab that only just cleared the threshold can be found too. It is empty for `never_measured` rows and for the 79 `too_few_values` items that appear in labevents only with an empty `value`, where there are no measured admissions to take a share of.
The three count columns are empty for `never_measured` rows (no `ItemStats` entry, so nothing was counted) and filled for the others, where `0` means counted and zero. The empty cells make the columns nullable, so pandas reads them back as float and the counts display as `0.0`.

`mimic_blacklist_composition.png`
What the blacklist removes, one row per reason. Left panel: lab items. Right panel: the measurements those items account for, split into values that parse as numbers and values that do not. Direct labels carry the absolute count and, on the right, the share of all recorded lab values.
The two panels rank the reasons differently — by items removed and by measurements removed:

| reason | items | measurements | numeric | text |
|---|---|---|---|---|
| `low_range_coverage` | 4 | 4.72M | 99.7% | 0.3% |
| `too_few_values` | 449 | 2.59M | 12.0% | 88.0% |
| `no_range` | 195 | 2.12M | 96.9% | 3.1% |
| `never_measured` | 754 | 0 | — | — |

**What "recorded lab values" counts.** Not the row count of labevents: a measurement is one row carrying both a `hadm_id` and a non-NaN `value`, which is only about half the file. See `mimic_labevents_coverage.csv` for the exact split.
The four reasons are mutually exclusive and applied in the order shown, so every row but the first is a marginal count and the four bars sum to 1402.
Built by `run_blacklist_composition_plot()`, which reads `mimic_blacklist_analysis.csv`, `mimic_value_analysis.csv` and `mimic_range_analysis.csv` rather than the in-memory stats, so it can be re-run on an existing `output/mimic_analysis/` without rescanning labevents.

`mimic_range_analysis.csv`
Rows: one per (itemid, distinct reference range); itemids with no range at all get a single row with `n_ranges = 0`
Columns: itemid, label, n_ranges, ref_range_lower, ref_range_upper, units, n_hadm_with_range, n_hadm_no_range, n_hadm_no_value
`n_hadm_no_range` counts admissions that had a value but no range, after subtracting those that also appear with a range elsewhere.

`mimic_value_analysis.csv`
Rows: one per itemid seen in labevents
Columns: itemid, label, n_numeric_values, n_string_values, pct_numeric, n_hadm_no_value, string_values
`string_values` is the pipe-joined set of distinct normalised non-numeric values, which is what makes the `too_few_values` decisions auditable.

`mimic_labevents_coverage.csv`
How much of labevents every other count in this folder is actually over. Counted during the scan, before either filter is applied.
Rows: one per line of the filter (`rows_total`, `rows_with_hadm_id`, `rows_with_hadm_id_and_value`, `rows_without_hadm_id`, `rows_without_hadm_id_but_with_value`, `rows_with_hadm_id_but_no_value`, `distinct_hadm_ids`)
Columns: metric, description, n_rows, pct_of_labevents
`rows_with_hadm_id_and_value` is the denominator everywhere else — just over half the file. The bigger of the two filters is `hadm_id`, not the missing value: rows with no admission to attach to are dropped by `dropna(subset=["hadm_id"])` at the top of the scan loop, and most of them do carry a usable value. That is the right denominator here, since a lab result with no admission can never become a feature, but it is not "all lab values in MIMIC". `pct_of_labevents` is blank for `distinct_hadm_ids`, where a share of rows would be meaningless.

`mimic_range_structure.png`
Restricted to the 248 lab items that survive the blacklist, not all 896 seen in labevents.
Left panel: how many distinct reference ranges each feature carries. There is no "none" row — an item with no reference range anywhere is blacklisted by construction, so every surviving feature has at least one. Right panel: for the multi-range features, what separates their ranges — sex, age, both or neither.
An item counts as sex-stratified when it has one range group at or above `RANGE_FEMALE_ONLY_PCT` (95%) female **and** one at or below `RANGE_MALE_ONLY_PCT` (5%): one range per sex, rather than two groups merely far apart. Age-stratified means a spread of at least `RANGE_AGE_SPREAD_YEARS` in mean age across the groups. Every range group is compared, whatever its size, so an age verdict should be read next to `smallest_group_n_hadm` in the summary csv.
Only itemids with more than one range are classified: a single-range item has no split to detect.
Built by `run_range_structure_plot()`, which reads `mimic_range_analysis.csv`, `mimic_range_stratification.csv` and `mimic_blacklist_analysis.csv` (for the feature restriction) rather than the in-memory stats, so it can be re-run without rescanning labevents. It also writes `mimic_range_stratification_summary.csv` below, from the same computation that fills the right panel.

`mimic_range_stratification.csv`
Only for itemids with more than one distinct reference range. Gives mean age, std age and percent female per range group, to check whether MIMIC's ranges are stratified by age or gender (which is the assumption behind the KNN range imputation in `_knn_impute_ranges()`).
Rows: one per (itemid, distinct range)
Columns: itemid, label, units, ref_range_lower, ref_range_upper, n_hadm_with_range, mean_age, std_age, pct_female

`mimic_range_stratification_summary.csv`
The verdict per lab item, collapsing the file above to one row each, and the table behind the right panel of `mimic_range_structure.png`. Written by `_classify_range_stratification()`, which the figure also reads, so the two cannot drift apart when a threshold changes.
Rows: one per **non-blacklisted** itemid with more than one distinct reference range (76). Blacklisted items are excluded here too, so this file and the figure cover the same set.
Columns: itemid, label, classification, stratified_by_sex, stratified_by_age, n_groups, n_hadm, smallest_group_n_hadm, min_pct_female, max_pct_female, min_mean_age, max_mean_age, age_spread_years
`classification` is one of `sex only` (15), `age only` (20), `both` (4) or `neither` (37). Every row gets a verdict — there is no "unjudged" category, since an item is only in this file if it has at least two range groups to compare.
The columns carry the numbers each verdict rests on, so it can be checked by eye: every `stratified_by_sex` row reads 0.0 / 100.0 for min and max percent female, bar Creatinine Whole Blood (99.8), Testosterone (0.5), Calculated Free Testosterone (1.1) and SHBG (0.9). Sorted with the explained items first, then by `n_hadm`, so the largest-evidence items lead.
`smallest_group_n_hadm` is the size of the thinnest range group behind a verdict; sort by it ascending to find the ones resting on few admissions.

### data_io/
In- and output functions. Split into three modules, and re-exported from the package, so
`from data_io import load_cohort` works regardless of which module holds it.

`data_io/paths.py`: every path the pipeline reads or writes, plus directory creation and
`--purge` removal. It also owns the cohort-folder state — `set_cohort_folder()` picks the
folder from `COHORT_FOLDERS`, and `cohort_path()` and `list_cohorts()` read it.
`data_io/store.py`: the `load_*` / `save_*` wrappers over the pipeline's own outputs.
`data_io/labevents.py`: `scan_labevents()`, the one streamed pass over the raw MIMIC table.

Named `data_io` and not `io`: the standard library's `io` is already imported before any
project code runs, so a package called `io` here would never be importable.

#### scan_labevents()
Scans labevents file for set of hadm_ids. 
Scans chunk-wise. Chunk size is set in `config/paths.py`. 
Reads the columns `hadm_id, itemid, ref_range_lower, ref_range_upper, value` from labevents file. 
Keeps only rows for which admission is in the set of hadm_ids from the cohorts and itemid is in the set of itemids (if TOP100LABS. Blacklisted are always subtracted).
Returns dicts with ranges per (hadm_id, itemid) pair, values per (hadm_id, itemid) pair and itemid set.

#### load_labs_with_reference_range()
Returns the itemids that carry a reference range in a cohort: at least one admission with a lower or an upper threshold in `<cohort>_ranges_<strategy>.csv`, defaulting to `BEST_MERGE_STRATEGY`. Used by `feature_importance_analysis()` to drop rangeless labs from the rank difference heatmaps. Reads the merged ranges file rather than the discretised mapping, so it does not depend on the mapping step having run.
Note this is a per-cohort, per-run check, and is a weaker condition than the `no_range` blacklist criterion in `mimic_analysis.py`: a lab can have a range somewhere in labevents (so it survives the blacklist) and still have none in the admissions of a particular cohort.

### prep/
Turning a cohort into the matrices the classifiers train on. Nothing is re-exported from the
package: import the submodule you need, e.g. `from prep.folds import run_fold_step`.

#### prep/ranges.py
Defines the three range merge strategies for multi ranges. Provides function `make_merge_strategies()` that creates a list of fresh instances of the three merge strategies, and `MERGE_STRATEGIES_LIST`, their names, for the callers that select a strategy by name. 
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

### plots/
`plots/theme.py`: the look every chart shares — the colour constants (`REPRESENTATION_COLOUR`,
`SERIES`, the ink/grid/axis greys, `SEQUENTIAL_BLUE`), the label shortening in `_shorten()`,
and the header block placed at fixed inch offsets by `_place_header()`.
`plots/plots.py`: the chart functions the analyses call.
Re-exported from the package, so `from plots import plot_fairness` works either way.

Every function takes the data already reduced, plus a title, subtitle and output path — the
analyses decide what to draw, these decide how it looks.

| function | draws | called by |
|---|---|---|
| `plot_rank_shift_slope_chart()` | a slope chart per panel on a shared rank axis: how a lab's importance rank moves between two representations | importance |
| `plot_rank_bump_chart()` | a bump chart per panel: a lab's rank across the folds, for stability | importance |
| `plot_rank_difference_heatmap()` | rows are cohorts, columns are labs, cell colour is the signed rank shift | importance |
| `plot_value_heatmap()` | a value per (row, column), on a sequential scale — currently only Kendall's W | importance |
| `plot_representation_boxplot()` | one box per representation, repeated per group, every cohort a dot | importance, performance, fairness, both studies |
| `plot_performance_scatter()` | one dot per (cohort, representation) against a cohort property, with a fit line | performance |
| `plot_fairness()` | one panel per measure: rows are demographic groups, dots are representations | fairness |
| `plot_fairness_eo_dumbbell()` | the equalised-odds gap per group as a two-ended dumbbell | fairness |
| `plot_classification_heatmap()` | one row per representation, one column per admission, cell is the predicted class | classification |
| `plot_blacklist_composition()` | two panels over the exclusion reasons: items removed and measurements lost | mimic_analysis |
| `plot_count_panels()` | side-by-side panels of labelled counts as horizontal bars | mimic_analysis |

### cohort_analysis.py
Standalone script, run on its own after `main.py`'s scan and merge steps. Describes the
cohorts themselves rather than the models trained on them:

```
python cohort_analysis.py --cohort_folder {test|all|subset}
```

Per lab and per cohort it counts the (admission, lab) pairs that hold a measured value, how
many of those carry a usable reference range, and the label distribution of each cohort.
Writes `lab_coverage_per_cohort.csv`, `lab_coverage.csv` and `label_distribution.csv` to
`output/reports/cohort_analysis/`, and prints a per-cohort summary. Builds nothing itself:
it exits naming every cohort that has no continuous mapping or no `BEST_MERGE_STRATEGY`
merged ranges.

### studies/
Parameter comparisons, not tests: each one runs the pipeline under several settings and writes
the evidence for a choice recorded in `config/params.py`. Run on their own, never by `main.py`.

#### studies/knn_k_test.py
The evidence behind `BEST_KNN`. For every cohort and fold it imputes the missing reference
ranges once per k in `KNN_KS`, builds a `disc_imp` frame from each, and trains
`BEST_CLASSIFIER` on all of them plus the plain discretised frame as the unimputed reference —
so the distance from that reference is what imputing buys. The merge strategy is held at
`BEST_MERGE_STRATEGY` throughout.

```
python studies/knn_k_test.py --cohort_folder {test|all|subset}
```

Writes to `output/reports/test_knn_k/`: `knn_k_results.csv` (one row per cohort,
representation and fold), `knn_k_comparison.csv` (the summary), `knn_k_change.csv` (how far
apart two k values put the discretised data, per k pair), a boxplot per metric, and the
difference and change boxplots. Runs the merge, mapping and fold steps itself, each of which
skips whatever already exists.

#### studies/merge_strategy_test.py
The evidence behind `BEST_MERGE_STRATEGY`. A single (admission, lab) pair can carry several
different reference ranges and discretisation needs exactly one; this trains `BEST_CLASSIFIER`
on the discretised frame produced by each of the three strategies, with the continuous frame
as the reference.

```
python studies/merge_strategy_test.py --cohort_folder {test|all|subset}
```

Writes to `output/reports/test_range_merge/`: `merge_strategy_results.csv`,
`merge_strategy_comparison.csv`, `merge_strategy_disc_change.csv` (how far apart two
strategies put the discretised data), a boxplot per metric, and the difference and
disc-change boxplots. Requests all three merge strategies itself, so `main.py --all_mappings`
is not a prerequisite.

### main.py
Main file to run the pipeline. Calls the steps in order: `prep/` to build the representations,
`model/train.py` to train, then the `analysis/` steps.
In each step the information is written to csv file. In the analysis steps, these csv files are read in. 

Flag `--cohort_folder` (required): `all` runs every cohort in `data/cohorts_100/`, `subset` runs `data/cohorts_50/` (one cohort per code1), `test` runs the cohorts in `data/cohorts_test/`.
Flag `--purge`: delete all cohort outputs and rebuild them from scratch.
Flag `--all_classifiers`: Also train random forest and catboost. Without it only `BEST_CLASSIFIER` (balanced random forest) is trained — the per-cohort fairness, importance and classification analyses read that one only, so the other two add training time and rows to `<cohort>_performance_summary.csv` that nothing downstream consumes. 
Flag `--all_mappings`: Build merged ranges and discretised mappings for all three merge strategies. Without it only `BEST_MERGE_STRATEGY` is built, which is the only one training uses. `merge_strategy_test.py` needs all three and requests them itself, so this flag is only needed to have them on disk beforehand.

Lists all cohorts from the selected cohort folder (`data/cohorts_100/` for `all`, `data/cohorts_test/` for `test`).
Scans labevents file once for all cohorts.
Then processes each cohort:
- merge multi ranges
- create discretised and binarised mappings
- create folds
- trains `BEST_CLASSIFIER`, or all three classifiers with `--all_classifiers`. Imputes before training, per fold.
- feature importance analysis
- fairness analysis
- per-sample classification heatmap
After cohort-wise processing, does the following:
- filters the cohorts by `MIN_COHORT_ROC` and prints which ones were excluded
- cross-cohort feature importance analysis
- cross-cohort performance analysis
- cross-cohort fairness analysis

All three cross-cohort analyses run on the filtered cohort list only. The per-cohort loop above runs on every cohort.

### Pipeline functions
Split across `prep/`, `model/` and `analysis/`; each entry below names the module it lives in.

#### What each step needs on disk
The stages communicate through files, not through memory, so each one can be re-run on its own
as long as its inputs exist. A step is skipped when its own output is already there, which is
what makes a re-run cheap and a `--purge` necessary after a change to an earlier stage.

| step | needs | writes | skipped when |
|---|---|---|---|
| `run_scan_step()` | `itemid_blacklist.txt`, the cohort CSV, `labevents.csv.gz` | `<cohort>_continuous.csv`, `<cohort>_ranges.csv` | the ranges file exists |
| `run_merge_step()` | `<cohort>_ranges.csv` | `<cohort>_ranges_<strategy>.csv` | that strategy's file exists |
| `run_mapping_step()` | continuous mapping + merged ranges | `<cohort>_binary.csv`, `<cohort>_discrete_<strategy>.csv` | each output exists |
| `run_fold_step()` | the cohort CSV | `folds/fold_<i>.pkl` | all `N_FOLDS` files exist |
| `train()` | mappings, merged ranges, folds, cohort CSV | performance summary, feature importance, test predictions | never — it retrains |
| per-cohort analyses | what `train()` wrote | metrics CSVs and `figures/` | never |
| cross-cohort analyses | the per-cohort metrics of every analysed cohort | `output/reports/` | never |

The per-cohort loop in `main.py` has one more guard in front of all of this: a cohort whose
classification heatmap already exists is skipped whole.

#### run_scan_step()
Lives in `prep/scan.py`.
Scans the labevents file once for the union of the hadm_ids of the cohorts that still need ranges, in whichever folder `--cohort_folder` selected. Cohorts for which a `<cohort>_ranges.csv` already exists are skipped and contribute no hadm_ids, so a re-run scans only what is missing (except with `--purge`, which deleted every cohort output first, so all of them are missing).
Reads in `itemid_blacklist.txt` from output/mimic_analysis and excludes blacklisted itemids 
Calls `scan_labevents()` from `data_io/labevents.py`. 
Averages the values per (hadm_id, itemid) pair. 
**Outputs:**
- `<cohort>_continuous.csv`
- `<cohort>_ranges.csv`

#### run_merge_step()
Lives in `prep/ranges.py`, below the strategies it drives. Called by `main.py` and by the two
parameter studies.
Merge multi ranges based on three merge strategies.
Merge strategies: 
- widest: smallest lower threshold and greatest upper threshold
- average: average lower threshold and average upper threshold
- narrowest: greatest lower threshold and smallest upper threshold
Loads `<cohort>_ranges.csv` and builds the strategy objects. each strategy object has its MergeWarningLog, which captures if an lower threshold is greater than the upper threshold. In this case its added to the log and range set to `None`. If log not empty, its saved as csv. 
Skips if csv files with the different merged ranges already exist.
Non-existing ranges have thresholds `None`. 
Argument `all_mappings` (from the `--all_mappings` flag of `main.py`): if False, only `BEST_MERGE_STRATEGY` is merged and only that one output file is written. The other two are only needed by `merge_strategy_test.py`, which passes `all_mappings=True` itself.
**Output:** 
- `<cohort>_ranges_widest.csv`
- `<cohort>_ranges_average.csv` (only with `all_mappings`)
- `<cohort>_ranges_narrowest.csv` (only with `all_mappings`)

#### run_mapping_step()
Lives in `prep/mapping.py`, together with its helper `_discretise()`. Called by `main.py` and
by the two parameter studies.
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
Argument `all_mappings` works as in `run_merge_step()`: without it only `BEST_MERGE_STRATEGY` is discretised. The binary mapping does not depend on ranges and is always written.
**Output:** 
- `<cohort>_binary.csv`
- `<cohort>_discrete_widest.csv`
- `<cohort>_discrete_average.csv` (only with `all_mappings`)
- `<cohort>_discrete_narrowest.csv` (only with `all_mappings`)

#### run_fold_step()
Lives in `prep/folds.py`, called by `main.py` and by the two parameter studies.
Splits the cohort into `N_FOLDS` folds, each with train and test set (80/20). Splitting is seeded with `FOLD_SEED`, hence split is randomised but deterministic. Checks if folds files already exist. If yes, skips this step. 
Stratifies the split by label: each fold preserves the overall label distribution (highly imbalanced cohorts).
Split is grouped by `subject_id` to avoid leakage of one patient with multiple hadm_ids. The same subject_id never appears in both train and test. 
**Output:** 
One pkl file for each fold. Each containing a (train, test) pair of [subject_id, hadm_id] arrays.

#### train()
Lives in `model/train.py`, with its helper `_split_feature_name()`. The classifier factories are in `model/classifiers.py`.
Runs 5-fold cross validation, training every classifier (random forest, balanced random forest, catboost) on every representation. Only uses discretised file from `BEST_MERGE_STRATEGY`.
Without `--all_classifiers` only `BEST_CLASSIFIER` is trained. It creates new representations by concatenating and imputation. 
Uses the following feature sets: 
- continuous values
- discretised values (discretised by BEST_MERGE_STRATEGY)
- discretised + imputed
- binary values
- continuous - binary concatenated
- discretised - binary concatenated (`disc_bin`)
- continuous - discretised - binary concatenated (`cont_disc_bin`)

The two concatenations carrying a discretised block use the plain discretised frame, not the imputed one, so neither depends on the fold and all three concatenations are built once per cohort, outside the cross-validation loop. Column names do not collide: continuous and binary columns are prefixed, the discretised ones are bare itemids. `disc_imp` is the only representation rebuilt per fold, since its imputation may only see that fold's training admissions.

For each fold: 
- imputes ranges for discretised representation with helper `_impute_disc()`, based only on train set (recomputed for every fold to avoid leakage.)
- trains `BEST_CLASSIFIER` on every feature set, or all three if `--all_classifiers` is set. Builds a fresh model instance on each call. 
- records feature importance for BEST_CLASSIFIER only, but for every feature set. In a concatenated feature set a lab appears once per encoding block, so `representation` says which block a row belongs to and `feature_set` which model it was measured in; only together are they unique. (Uses helper `_split_feature_name`)
- predicts on test set
- calculates performance measures (auc_roc, avg_prec, f1, mcc, balanced_acc)
- saves predictions for BEST_CLASSIFIER only (hard labels and probabilities)

**Output:** 
- `<cohort>_test_predictions_fold_<n>.csv`
- `<cohort>_feature_importance.csv`
- `<cohort>_performance_summary.csv`

#### _impute_disc()
Lives in `prep/imputation.py`, with `_knn_impute_ranges()`.
Builds disc_imp representation for one fold. Imputes ranges from the training set for values that are measured but have no reference range given. 
Calls `_knn_impute_ranges()` as helper. 

#### _knn_impute_ranges()
Performs the range imputation step for each measured (hadm_id, itemid) pair with missing range. 
Only takes ranges from training admissions to avoid leakage. 
Matching rule: same itemid, same gender, closest age, averaged over k training donors. 
For each (itemid, gender) pair find the k nearest neighbours by age and average their reference range thresholds. Fewer than k donors in a group means every donor is used.
Three things are left alone rather than forced:
- A pair with **no donor of the same (itemid, gender)** keeps its missing range, so imputation raises the ceiling on what can be discretised without guaranteeing it is reached. In practice it nearly always is: across six spot-checked cohorts only one had any measured cell left unfilled, 2 of its 12,640.
- Donors and targets **missing age or gender** are dropped, since the matching rule has nothing to match on without them.
- The two bounds are averaged **separately and NaN-skipping**, so a one-sided range stays one-sided: if every donor has only a lower bound, the imputed range has only a lower bound. 

#### cohort_feature_importance_analysis()
Lives in `analysis/importance.py`, with `_kendalls_w()`.
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
One figure with 7 panels, one per representation in `CLASSIFICATION_HEATMAP_REPS`. This part reads the *unfiltered* importance table, not the `STABILITY_FEATURE_SETS` subset the rest of the function uses, and sums a lab's importance over its encoding blocks in the concatenated representations.
Displays the top 20 labs per representation, ranked by mean rank across the folds. 

**Output:** 
- `<cohort>_feature_stability.csv` (Kendalls W across folds per representation)
- `<cohort>_feature_shift.csv` (Mean/std of feature importance per representation)
- `<cohort>_rank_shift.png`
- `<cohort>_rank_shift_disc_imp.png`
- `<cohort>_rank_bump.png`

#### cohort_fairness_analysis()
Lives in `analysis/fairness.py`, with its helpers `_group_performance()`, `_group_supported()` and `_bias_metric_gaps()`.
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

#### _bias_metric_gaps()
Lives in `analysis/fairness.py`. Helper used by `cohort_fairness_analysis()`.
Calculates fairness-gap per fairness criterion across demographic groups. It compares best and worst group. 

#### sample_classification_analysis():
Lives in `analysis/classification.py`.
Creates per-sample classification heatmap for BEST_CLASSIFIER, from the held-out predictions of all 5 folds pooled. Samples sorted first by true label, then by mean predicted probability across the representations.
Takes CLASSIFICATION_HEATMAP_REPS as order and display name for representations; one missing from a cohort's predictions is skipped rather than drawn as an empty row.
Called last in `main.py`'s per-cohort loop, and its figure is what that loop checks to skip an already-processed cohort.
**Output:**
- `<cohort>_classification_heatmap.png`

#### filter_cohorts_by_roc()
Lives in `analysis/selection.py`.
Splits the cohort list into the cohorts good enough for the cross-cohort analyses and the ones dropped. A cohort is kept when its fold-averaged AUC-ROC on `ROC_FILTER_REP` under BEST_CLASSIFIER reaches `MIN_COHORT_ROC`.
Reads `<cohort>_performance_summary.csv`, so it has to run after `train()`. A cohort with no performance summary (never trained, or a run that died mid-cohort) is dropped rather than raising, so a partial batch can still be aggregated. A NaN AUC fails the comparison and is dropped as well.
Prints one line per dropped cohort: the AUC it failed on, or, for a cohort with no summary at all, that there was nothing to read. Returns `(kept, dropped)`.
Called once in `main.py`; only `kept` is passed on to the three cross-cohort analyses.

#### _icd_chapter()
Helper used by `_disease_group()`.
Returns `(group key, group name)` for an ICD code, matched against `ICD_CHAPTERS` on both the letter and the number, since D and H each split across two chapters. `C83` gives `("C, D00-D49", "Neoplasms")`, `H81` gives `("H60-H95", "Ear and mastoid process")`. Falls back to `UNMAPPED_ICD_CHAPTER` for a code that matches nothing.

#### _disease_group() / _target_group()
Helpers used by `performance_analysis()`.
Both return an ICD chapter key via `_icd_chapter()`, using the whole code: `_disease_group()` for code1, `_target_group()` for code2. Reading code2 the same way as code1 means a D or H target is split into its two chapters the way code1 is, and that a chapter key means the same thing on either side of the name.
Both parse the cohort *filename*, so both require the `cohort_<code1>-<code2>` schema — see [Cohort naming](#cohort-naming). An unmappable code degrades to `?` in either. A name with no hyphen at all still makes `_target_group()` raise `IndexError` on `split("-")[1]`, where `_disease_group()` takes `[0]` and returns `?`, so the two do not fail alike on that one input.

#### _group_mean()
Lives in `analysis/grouping.py`. Used by `feature_importance_analysis()` for the disease-grouped rank difference heatmap.
Collapses a per-cohort table to one row per group. Takes either a function mapping cohort name to group key, or a Series of keys indexed by cohort — an ordered categorical Series keeps the rows in bin order rather than sorting the labels alphabetically. Empty categorical bins are dropped rather than drawn as blank rows (`observed=True`).
Row labels carry the number of cohorts in the group, and optionally a second line naming what the group key stands for (`descriptions`, used for the ICD chapter names).

#### feature_importance_analysis()
Lives in `analysis/importance.py`.
Analyses the feature importance change across cohorts. Gives fold feature importance stability over the folds for each cohort and the representations continuous, discretised and binarised. 
Also analyses rank shift between continuous representation and binarised/discretised representation. 
Loads each cohort's `<cohort>_feature_shift.csv` and `<cohort>_feature_stability.csv` files. 
Note that the labs shown in the rank difference heatmaps are selected by their mean rank **across the cohorts passed in**, and the rangeless-lab filter is likewise evaluated over that cohort set only. Changing the cohort set (a different `MIN_COHORT_ROC`, or more cohorts) therefore changes which labs form the columns, not just which cohorts form the rows, so two runs cannot be compared cell by cell.
The number of columns dropped for having no reference range is printed to the console alongside the number kept.
Ranks are computed per cohort over that cohort's own feature set, so the size of the ranking universe differs between cohorts and a rank shift of a given size does not mean the same thing in every row.
**Output:**
- `rank_difference_cont_disc_heatmap.png`
- `rank_difference_cont_bin_heatmap.png`
- `rank_difference_cont_disc_disease_heatmap.png`
- `rank_difference_cont_bin_disease_heatmap.png`
- `kendalls_w_heatmap.png`
- `kendalls_w_boxplot.png`

#### performance_analysis()
Lives in `analysis/performance.py`. The cohort grouping helpers it shares with the other two analyses are in `analysis/grouping.py`.
Takes the files `<cohort>_performance_summary.csv` of the filtered cohorts and plots AUC-ROC by representation, for BEST_CLASSIFIER only. `CROSS_COHORT_EXCLUDED_REPS` (the discretised + imputed representation) is trained and written to the summary csv but left out of these figures.
AUC-ROC is the only measure plotted; the others stay in the summary csv files.
Additionally reads each cohort's label column (`load_cohort()`) to get the cohort size and the share of positive labels, which are not derivable from the cohort name.
Produces one ungrouped box plot over all included cohorts, the same boxes repeated per group of the four stratifications (disease chapter, target chapter, cohort size bin, label distribution bin), and an unbinned scatter of AUC-ROC against size and prevalence. The grouped box plots additionally leave out `CONCATENATED_REPS`, so they carry three representations against the ungrouped plot's six.
**Output:**
- `performance_roc_vs_cohort_properties.png`
- `performance_roc_boxplot.png`
- `performance_roc_boxplot_disease.png`
- `performance_roc_boxplot_target.png`
- `performance_roc_boxplot_size.png`
- `performance_roc_boxplot_prevalence.png`

#### fairness_analysis()
Lives in `analysis/fairness.py`.
Plots the equalised-odds gap by representation across cohorts, for gender and age band, as dumbbells against the continuous representation and as box plots.
Reads two files per cohort: `<cohort>_fairness_criteria.csv` for the gaps themselves, and `<cohort>_fairness.csv` for the per-group counts and rates, which are needed for the exclusion note on the dumbbell figures and for the per-group gaps behind `fairness_eo_boxplot_group_size.png`. Uses helper `_group_supported()` to name the demographic groups with insufficient support (FAIRNESS_MIN_GROUP_SAMPLES, FAIRNESS_MIN_LABEL_SAMPLES); those groups were already excluded from the gaps when `cohort_fairness_analysis()` computed them.
The figures cover continuous, discretised and binarised — `CROSS_COHORT_EXCLUDED_REPS` and `CONCATENATED_REPS` are both dropped. Each figure is skipped with a printed line rather than raising if there is nothing to plot.

**Output:**
- `fairness_eo_age_band.png`
- `fairness_eo_gender.png`
- `fairness_eo_boxplot.png`
- `fairness_eo_boxplot_disease_gender.png`
- `fairness_eo_boxplot_disease_age_band.png`
- `fairness_eo_boxplot_group_size.png`