# THRESH_project
Analyse whether clinical intervals/missingness  give better predictability/robustness/fairness for random forests than continuous lab values.

## Data
MIMIC IV hosp

## Project structure


## Setup
environment
config/

## Workflow

> **Cohort files must be named `cohort_<code1>-<code2>.csv.gz`** (e.g. `cohort_F17-C22.csv.gz`). This is a hard requirement, not a convention — see [Cohort naming](#cohort-naming) below. A cohort named anything else runs all the way through training and then crashes in the cross-cohort step.

1. Set global variables in `config/`
2. Run `mimic_analysis.py`: scans the whole labevents file once and writes `output/mimic_analysis/`, including `itemid_blacklist.txt`, which fixes the feature set for every cohort. Only needs re-running if a blacklist criterion changes — and if it does, step 3 needs `--purge`.
3. Run `main.py`: python main.py --cohort_folder {test|all|subset}
    --cohort_folder (required): `all` runs every cohort in `data/cohorts_100`, `subset` runs `data/cohorts_50` (one cohort per code1), `test` runs the cohorts in `data/cohorts_test`.
    --purge to delete all cohort outputs and rebuild them from scratch
    --balanced_rf_only to only train on balanced random forest. catboost and random forest are skipped.
    --all_mappings to build ranges and discretised mappings for all three merge strategies instead of only the best one.

## Cohort naming

The pipeline only runs to completion for cohorts named **`cohort_<code1>-<code2>.csv.gz`** — an ICD-10 code for the disease the cohort is built around, a hyphen, and an ICD-10 code for the target. `cohort_F17-C22.csv.gz`, `cohort_H81-E55.csv.gz`.

The cross-cohort performance analysis reads the two codes straight out of the filename, because the cohort size, prevalence and codes are not recorded anywhere else. `_disease_group()` takes everything before the hyphen and `_target_group()` takes the first letter of everything after it:

```python
def _target_group(cohort: str) -> str:
    return cohort.removeprefix("cohort_").split("-")[1][0]   # analysis/grouping.py
```

A name with no hyphen makes `split("-")` return a single element, and `[1]` raises `IndexError`. The two chemotherapy cohorts shipped in `data/cohorts_test` used to do exactly this, under their original names, and have been renamed to conform:

| cohort | `_disease_group()` | `_target_group()` |
|---|---|---|
| `cohort_F17-C22` | `F` | `C` |
| `mimic_cohort_aplasia_45_days` (was) | `?` | **IndexError** |
| `cohort_C80-D61_aplasia` (now) | `C, D00-D49` | `D50-D89` |
| `mimic_cohort_NF_30_days` (was) | `?` | **IndexError** |
| `cohort_C80-D70_NF` (now) | `C, D00-D49` | `D50-D89` |

Only the *start* of each hyphen-segment is parsed — a letter followed by digits — so anything after the code is free text. That is what lets the mnemonic ride along: `cohort_C80-D70_NF` parses exactly as `cohort_C80-D70` would. A label placed *before* the code (`cohort_NF_C80-D70`) does not work; it falls back to `?`.

Two things make this worse than an ordinary input check:

- **It fails late.** `_target_group` is handed to `groupby()` as a callable, so pandas only applies it when the target-stratified heatmap is built — in `performance_analysis()`, after the labevents scan, all five folds, all training and the whole feature-importance analysis have finished. You lose the run at the last step, not the first.
- **The two helpers disagree.** `_disease_group()` routes through `_icd_chapter()`, which falls back to `UNMAPPED_ICD_CHAPTER` (`?`) for a code it cannot match, so a non-conforming cohort merely lands in an "Unmapped ICD chapter" row. `_target_group()` indexes the split result directly and has no fallback. So the disease stratification tolerates these names and the target stratification does not.

Every file in all three folders now follows the schema, so no folder crashes at the last step. The failure mode below is still worth knowing about, because it is what any newly added cohort will hit if it is named freely.

**Not yet fixed.** Giving `_target_group()` the same `?` fallback is a one-line change, but it is worth deciding first whether a cohort like aplasia *should* appear in the ICD stratifications at all — a `?` row is honest but carries no information — or whether such cohorts should be dropped from those two figures and kept in every other one.


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
Rows: one row per (feature set x fold x feature) (feature is itemid)
Columns: cohort, feature_set, fold, feature, itemid, representation, importance
Recorded only for the feature sets in STABILITY_FEATURE_SETS (cont, disc, disc_imp, bin)

`<cohort>_performance_summary.csv`
Rows: one row per (model, measure) (models: random forest, balanced random forest, catboost; measure: auc_roc, avg_prec, f1, mcc, balanced_acc)
Columns: model, measure, mean and std for all feature sets (cont, disc, disc_imp, bin, cont_bin, disc_bin, cont_disc_bin)
Summary files written before a feature set was added do not have its columns. `main.py` skips cohorts that are already processed, so add the feature set with `--purge` (or by deleting the cohort outputs) if existing cohorts should get it too.

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
**Columns:** the union of the top `RANK_DIFF_TOP_N` labs by mean rank in each of `RANK_DIFF_REPRESENTATIONS` (cont, disc, bin), minus any lab that carries no reference range in any of the cohorts plotted. Taking the union rather than the continuous top N keeps a lab that only becomes important once discretised or binarised; dropping the rangeless labs removes ones that are NaN in every discretised cell and so can only ever lose rank, which is an artefact of the representation rather than a finding. Both figures get the same columns, so they can be read against each other lab by lab. With the top 50 per representation this gives 71 columns.
**Colour scale:** diverging red/white/blue, symmetric about zero, but the ramp is not linear. Shifts up to `EMPHASIS_SHIFT` (100 ranks) get `EMPHASIS_FRACTION` (70%) of each half of the colour range and everything above shares the rest, because most cells sit well inside ±100 while a few labs move by several hundred and on a linear ramp that tail washes the rest of the grid out to white. The colourbar is ticked at 0, ±100 and ±limit, so the uneven spacing is visible rather than hidden. Implemented by `_emphasised_diverging_norm()` in `plots/plots.py`.
The scale is set by each figure's own most extreme cell, so shades are not comparable between the two figures or between runs.

`kendalls_w_heatmap.png`
Shows kendalls W (fold agreement on feature importance) for all cohorts and the representation continuous, discretised and binarised. 

`performance_roc_heatmap.png`
`performance_f1_heatmap.png`
`performance_mcc_heatmap.png`
Heatmap of the performance measure by cohort and representation (continuous, discretised, binarised, cont + disc + bin). Averaged over the 5 folds. Bottom row is the average across cohorts. Only for BEST_CLASSIFIER.
The discretised + imputed, continuous + binarised and discretised + binarised representations are not shown: they are still trained and still written to `<cohort>_performance_summary.csv`, only left out of these plots.
A cohort whose summary predates a representation has no column for it and shows a blank cell rather than raising, so a mixed set of old and retrained cohorts produces a partly empty column instead of an error.
The colour scale spans the observed range of each figure, so shades are not comparable between figures or between runs.

`performance_roc_disease_group_heatmap.png`
Same AUC-ROC numbers collapsed to one row per disease group, where the group is the ICD-10 chapter of code1 in the cohort name `cohort_<code1>-<code2>`. The **whole** code is matched, not only its letter, because two letters split across chapters: D00-D49 belongs to Neoplasms with C while D50-D89 is its own chapter, and H00-H59 (eye) and H60-H95 (ear) are separate chapters. So `cohort_H43-H54` is eye and `cohort_H81-E55` is ear. Chapters are defined in `ICD_CHAPTERS`, and the row order follows that list.
Each row label names the chapter on a second line. A code matching no chapter (U, Z, or a malformed code) falls into `?`/"Unmapped ICD chapter" rather than silently disappearing.
Row labels carry the number of cohorts in the group (`F (n=4)`); groups are very uneven and a group of one cohort is a single cohort, not a group effect.

`performance_roc_target_group_heatmap.png`
Same, for the target: one row per **first letter** of code2. This one is still letter-only, so a D or H target is not split into its two chapters the way code1 is, and rows carry no chapter name.
Bottom row is the average across cohorts, so it is cohort-weighted and matches `performance_roc_heatmap.png` rather than being the mean of the group rows.

`performance_roc_size_group_heatmap.png`
`performance_roc_prevalence_group_heatmap.png`
Same AUC-ROC numbers collapsed to one row per cohort-size bin (`COHORT_SIZE_BINS`) and one row per label-distribution bin (`PREVALENCE_BINS`). Size and prevalence are read from the cohort label column, not from the cohort name.
Row labels carry the number of cohorts in the bin (`<500 (cohorts=3)`), since `n` would read as the cohort size here.
Bottom row is the average across cohorts, as above.

`performance_roc_vs_cohort_properties.png`
The same two properties unbinned: one dot per cohort per representation, size (log scale) in the left panel and share of positive labels in the right panel, with a least squares fit per representation. Keeps the per-cohort spread that the binned heatmaps average away, so a gradient can be told apart from a threshold and a one-cohort bin cannot masquerade as a trend.
The dashed rule marks `MIN_COHORT_ROC`. Nothing is plotted below it, because the filter removed those cohorts — the truncation is a property of the filter, not of the data.
Note that the filter removes small and weak cohorts preferentially, which flattens any size effect visible here relative to the unfiltered cohort set.

`performance_roc_difference_heatmap.png`
`performance_roc_difference_disease_heatmap.png`
`performance_roc_difference_target_heatmap.png`
`performance_roc_difference_size_heatmap.png`
`performance_roc_difference_prevalence_heatmap.png`
The same AUC-ROC numbers as a difference to the `ROC_FILTER_REP` (continuous) representation: each cell is that representation's AUC-ROC minus the continuous one, so the continuous column itself is dropped. The first is per cohort, the other four per group of the stratifications.
Diverging red/white/blue scale on a range symmetric about zero: white is "same as continuous", blue is better, red is worse. The scale spans each figure's own extreme, so shades are not comparable between these figures.
Because the mean of the differences equals the difference of the means, the grouped versions are exactly the plain stratified heatmaps subtracted column-wise.

`performance_roc_boxplot.png`
Distribution of the per-cohort AUC-ROC, one box per representation over all included cohorts. The box spans the interquartile range, the line in it is the median and the whiskers reach the minimum and maximum, so nothing is trimmed or marked an outlier. Every cohort is also drawn as a dot, jittered horizontally so equal values do not hide behind one another (the jitter is seeded, so reruns look identical).

`performance_roc_boxplot_disease.png`
`performance_roc_boxplot_target.png`
`performance_roc_boxplot_size.png`
`performance_roc_boxplot_prevalence.png`
The same boxes repeated per group of the four stratifications: groups along the x-axis, one box per representation within each group, cohort count under each group label.
These show the spread that the stratified heatmaps reduce to a single mean per cell. Read the small groups with care: with one cohort a "box" is a single point and the quartiles mean nothing, which is exactly what the heatmap cell hides. The disease and target stratifications have many such groups.

Only the AUC-ROC measure gets the four stratified heatmaps, the five difference heatmaps, the scatter and the five box plots; F1 and MCC get the per-cohort heatmap only.

`fairness_eo_age_band.png`
`fairness_eo_gender.png`
Shows equalised odds gap by representation across cohorts for age and gender. Groups with fewer than FAIRNESS_MIN_GROUP_SAMPLES or FAIRNESS_MIN_LABEL_SAMPLES are excluded. Only shown for BEST_CLASSIFIER.
For age bands it shows the gap between best and worst case. 

## Documentation of the scripts and their functions
TODO: dependencies

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

`TOP100LABS`: if FALSE, all itemids that is found in D_LABITEMS is used. If TRUE, curated list of sll_mimictop100_features_hadm.pkl is read and used. In both cases the blacklisted itemids are subtracted. 
`CHUNK_SIZE`: Set chunk size for labevents scan. 
`BEST_MERGE_STRATEGY`: Only the discretised file for this range merge strategy is used in classifier training.
`ALL_MERGE_STRATEGIES`: List of the names of the three range merge strategies. 
`N_FOLDS`: Number of folds the cohort is split into. 
`BEST_CLASSIFIER`: Classifier for which feature importance is calculated
`ROC_FILTER_REP`: Representation whose AUC-ROC decides whether a cohort enters the cross-cohort analyses. 
`MIN_COHORT_ROC`: Minimum fold-averaged AUC-ROC on `ROC_FILTER_REP` for a cohort to enter the cross-cohort analyses. Cohorts below it are dropped by `filter_cohorts_by_roc()`. Does not affect the per-cohort outputs. 
`ICD_CHAPTERS`: ICD-10 chapters used to stratify the cohorts by disease. Entries are `(letter, lowest number, highest number, group key, group name)`; the first match wins and the list order is the row order of the plot. Matched on the whole code because D and H each split across two chapters (D00-D49 with the neoplasms, H00-H59 eye vs H60-H95 ear). Note M is the musculoskeletal chapter and N the genitourinary one. 
`UNMAPPED_ICD_CHAPTER`: Fallback group for a code that matches no chapter (U, Z, or a malformed code), so such a cohort still appears on the plot instead of disappearing. 
`COHORT_SIZE_BINS` / `COHORT_SIZE_LABELS`: Bin edges and labels for stratifying the cohorts by number of admissions in the performance report. 
`PREVALENCE_BINS` / `PREVALENCE_LABELS`: Bin edges and labels for stratifying the cohorts by share of positive labels. Both are fixed edges rather than quantiles, so that the bins stay comparable when the cohort set changes (a different `MIN_COHORT_ROC`, or more cohorts). Edges are left-open / right-closed, as `pd.cut` reads them. 
`RANGE_FEMALE_ONLY_PCT` / `RANGE_MALE_ONLY_PCT` / `RANGE_AGE_SPREAD_YEARS`: thresholds for deciding whether a lab item's several reference ranges are explained by the sex or age of the admissions behind them, used by `mimic_range_structure.png`. Sex needs one group at or above `RANGE_FEMALE_ONLY_PCT` female and one at or below `RANGE_MALE_ONLY_PCT`, i.e. one range per sex; both are insensitive, since no group falls between 66.7% and 95% or between 5% and 19%. Age needs the mean age to differ by `RANGE_AGE_SPREAD_YEARS` across the groups, and unlike the sex cut-offs this one is a judgement call. There is no minimum group size: every range group is compared however few admissions carry it. See the note under `mimic_range_structure.png` for what that buys and what it costs. 
`STABILITY_FEATURE_SETS`: feature sets for which feature importance is calculated. 
`RANK_DIFF_REPRESENTATIONS` / `RANK_DIFF_TOP_N`: which representations contribute columns to the rank difference heatmaps, and how many labs each contributes. The columns are the **union** of the top `RANK_DIFF_TOP_N` of each, so a lab that only becomes important once discretised or binarised is not missed by a continuous-only selection. Raising `RANK_DIFF_TOP_N` widens both figures; the union is always smaller than `RANK_DIFF_TOP_N` × the number of representations, because the representations agree on the strongest labs. 

`REPRESENTATION_LABELS`: Display names for the four core representations (`cont` → continuous, `disc` → discretised, `disc_imp` → discretised + imputed, `bin` → binarised) plus the concatenated `cont_disc_bin` → cont + disc + bin. Used for plot headers and column/row labels across the feature-importance, fairness and performance plots. Kept short because these are column headers on the performance heatmaps. 
`BEST_KNN`: k for nearest neighbour search for imputation. Used in the pipeline. 
`KNN_KS`: list of k for KNN neighbourhood for imputation. Only used in testing script for which the best k is
`AGE_BANDS`: Age groups for fairness analysis
`AGE_BAND_LABELS`: Labels for the age groups
`FAIRNESS_CRITERIA`: statistical parity (selection rate), predictive parity (ppv), equal opportunity (tpr), predictive equality (fpr), equalised odds.
`FAIRNESS_MIN_GROUP_SAMPLES`: min number of samples which a group must have to participate in gap calculation of fairness criteria
`FAIRNESS_MIN_LABEL_SAMPLES`: min number of samples a group must have to participate in gap calculation of fairness criteria
`CLASSIFICATION_HEATMAP_REPS`: Order and display names of all seven feature representations (adds `cont_bin`, `disc_bin` and `cont_disc_bin` to the four core ones) for the rows of the sample classification heatmap; the dict order sets the row order. A representation missing from a cohort's predictions is skipped rather than drawn as an empty row, so a cohort trained before a representation was added simply has one row fewer.

### mimic_analysis.py
Runs once, before `main.py`, over the **whole** labevents file (not filtered to any cohort). Collects per-itemid statistics into an `ItemStats` object — the distinct reference ranges and which admissions carry them, the units, how many values parse as numbers vs as strings, and which admissions had a value at all — and writes four files to `output/mimic_analysis/`. The important product is `itemid_blacklist.txt`, which fixes the feature set for every cohort downstream: `load_all_itemids()` is d_labitems minus the blacklist, so anything blacklisted here never becomes a column.

**Blacklist criteria.** An itemid is blacklisted when it cannot carry a usable discretised feature, for any of three reasons. They are mutually exclusive in the output, in this order of precedence:

| reason | meaning | n |
|---|---|---|
| `too_few_values` | fewer than 10 distinct numeric values: categorical or near-constant, nothing to threshold | 449 |
| `no_range` | measured, but no reference range anywhere in labevents, so it can never be discretised | 195 |
| `never_measured` | not in labevents at all, so it would be an all-NaN column in every cohort | 754 |

That is 1398 of the 1650 itemids in d_labitems, leaving **252 features**. `never_measured` is the set difference `d_labitems − stats`, since `stats` only gets an entry for an itemid that actually appears in a labevents row.

The `no_range` and `never_measured` criteria were added after the feature matrix was found to be 1201 columns wide while only 220–419 labs were measured per cohort. Those permanently empty columns were not harmless: they picked up nonzero random forest importance in the continuous and discretised representations and so occupied real positions in the importance ranking, which inflated the ranking universe to between 275 and 1131 features depending on the cohort and made rank shifts of over 1000 places possible in the rank difference heatmaps. Removing them is what keeps that ranking bounded and comparable across cohorts.

Changing any criterion changes the feature matrix, so every cohort output built on the old blacklist is invalid. Re-run `main.py` with `--purge`.

**Outputs** (all in `output/mimic_analysis/`):

`itemid_blacklist.txt`
One itemid per line, sorted. Read by `load_blacklist()`.

`mimic_blacklist_analysis.csv`
Readable version of the same list.
Rows: one per blacklisted itemid
Columns: itemid, label, reason, n_numeric_values, n_unique_numeric_values, n_ranges
The three count columns are empty for `never_measured` rows and filled for the other two. This is deliberate: those itemids have no `ItemStats` entry at all, so there was nothing to count, which is not the same fact as having been scanned and found to have none. A `0` means "counted, and the answer is zero" — 329 of the `too_few_values` rows are genuinely 0, items whose values are all text (`INTUBATED.` / `NOT INTUBATED.` for 50812, `NEG` / `POS` for 50857) or all `___`, MIMIC's de-identification placeholder. Because the empty cells make the columns nullable, pandas reads them back as float, so the counts display as `0.0` rather than `0`.
Four itemids (51771, 51955, 52374, 53168) have no label in d_labitems; all four are `never_measured`.

`mimic_blacklist_composition.png`
What the blacklist removes, one row per reason. Left panel: lab items. Right panel: the measurements those items account for, split into values that parse as numbers and values that do not. Direct labels carry the absolute count and, on the right, the share of all recorded lab values.
The two panels deliberately disagree, which is the point of the figure. `never_measured` is by far the largest reason by item count (754) and removes exactly zero measurements — those columns were pure padding. `no_range` removes only 195 items but takes 2.12M measurements with it, 97% of them numeric, so it is the one criterion that discards real data: those labs stay usable as continuous features in principle, they simply cannot be discretised, and the study needs all three representations over the same feature set. `too_few_values` removes the most measurements (2.59M) but 88% of them are text (`INTUBATED.`, `NEG`/`POS`, or the `___` de-identification placeholder), so little numeric information is lost.
**What "recorded lab values" counts.** Not the row count of labevents: a measurement is one row carrying both a `hadm_id` and a non-NaN `value`, which is only about half the file. See `mimic_labevents_coverage.csv` for the exact split.
**Why the `no_range` row reads 195 and not 562.** The three reasons are mutually exclusive, applied in the order shown, so every row but the first is a *marginal* count — what that criterion removes that nothing above it already removed. 562 lab items carry no reference range anywhere in labevents, but 367 of them also have fewer than 10 distinct numeric values and are counted in the first row. Hence the row label "(not already counted above)", and hence the three bars summing to the blacklist total (449 + 195 + 754 = 1398) rather than overshooting it. The figure's note states the reconciliation.
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
**Restricted to the 252 lab items that survive the blacklist and become features**, not all 896 seen in labevents. Range structure only matters where a range is going to be used to discretise something, and restricting it makes this figure reconcile with `mimic_blacklist_composition.png`, whose subtitle reports the same 252.
Left panel: how many distinct reference ranges each feature carries — one (176), two (53), three (16), four or more (7). There is deliberately **no "none" row**: an item with no reference range anywhere is blacklisted by construction, under either `no_range` or `too_few_values`, so every surviving feature has at least one range. Right panel: for the multi-range features, what separates their ranges.
76 features carry more than one range, and every one is judged: **15 split by sex only, 20 by age only, 4 by both and 37 by neither** — 19 sex-stratified and 24 age-stratified in total. The headline is that MIMIC's reference ranges are stratified by sex: Hemoglobin, Hematocrit, Creatinine, Red Blood Cells, Alkaline Phosphatase and Ferritin all split cleanly into a 100%-female group and a 0%-female group.
The restriction drops 8 multi-range items, all of them `too_few_values`: % Hemoglobin A1c, Estradiol, Ethanol, Luteinizing Hormone, NTproBNP, Prolactin, Factor XII and eAG.
An item counts as sex-stratified only when it has one range group at or above `RANGE_FEMALE_ONLY_PCT` (95%) female **and** one at or below `RANGE_MALE_ONLY_PCT` (5%) — one range per sex. The rule asks for the groups themselves to be single-sex rather than merely far apart, so an item whose groups sat at 70% and 15% female would not be swept in by the distance between them alone.
Neither sex cut-off is a judgement call: across the 76 features, no range group falls between 66.7% and 95% female, or between 5% and 19%, so the flagged items clear both bounds with room to spare and any cut-off inside those gaps gives the same answer. (An earlier version of this rule used the spread `max − min ≥ 50pp`; it selected the same items but stated the weaker condition.)

**No minimum group size, and why the two counts differ in trust.** Every range group is compared, however few admissions carry it. There used to be a 20-admission floor; removing it changed both counts, in opposite directions and for the same reason.
It is what lets the sex test see the hormone assays. Calculated Free Testosterone, Sex Hormone Binding Globulin, DHEA-Sulfate and Testosterone Free are clean 100/0 splits whose female side is only 1–16 admissions; the floor had been discarding them. A ten-admission group that is 100% female is strong evidence — thirteen consecutive females by chance is about one in eight thousand — so the sex count is safe at any group size.
A mean age is not, and **20 of the 24 age-flagged features rest on a range group of fewer than 20 admissions**. Nine of them are the CBC differential — Eosinophils, Lymphocytes, Monocytes, Neutrophils and the Absolute counts — every one flagged off the *same* 17-admission group at mean age 19.2 sitting against 84,000–100,000 admissions at mean age 59.2. Worse: Bilirubin Total, Protein Total, DHEA-Sulfate and Testosterone Free each hinge on a group of **one** admission, whose "mean age" is simply that person's age, and PT on a group of two against 273,306.
Only four age flags rest on a group of 20 or more, and they are the ones worth taking seriously: Uric Acid (smallest group 9,467), Magnesium (1,695), Glucose (97) and Calcium Total (80). Read `smallest_group_n_hadm` in the summary csv before quoting the age number.
This bears on `_knn_impute_ranges()`, which matches a donor on same gender and closest age: the gender match is doing nearly all the work, and the age match has little behind it once the thin groups are discounted.

**Scope of the right panel, and a checked blind spot.** The panel measures *sex-stratified ranges* — one item, several ranges, split by sex — because that is the case `_knn_impute_ranges()` has to resolve. It cannot see a *sex-specific lab*: one range, given only to one sex, which has no split to detect and nothing to impute. Those items have `n_ranges == 1`, so `run_range_stratification_analysis()` skips them at `if len(s.ranges) <= 1: continue` and they sit unexamined in the left panel's "1 (single range)" bar.
This was checked directly against labevents rather than assumed. Of the 226 single-range items with at least 20 admissions carrying a range, exactly one is male-only — Prostate Specific Antigen (50974), 2410 admissions, 0.5% female — and one is female-only — Cancer Antigen 27.29 (50898), 98.0% female — with CA-125 (93.0%) and Human Chorionic Gonadotropin (90.2%) just below the cut. All four are blacklisted under `too_few_values` (PSA has 6 distinct numeric values in the whole database; the other three have none, their results being recorded as text), so none of them is even in this figure's 252 features and the blind spot has no downstream effect. The remaining 0%-female entries all have fewer than five admissions and are noise.
The same check on the multi-range side found nothing: no item is single-sex overall, the most skewed being WBC Count at 28.7% female, with the large majority between 40 and 60%. So no item is misclassified as "neither" through having all its range groups in one sex.
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
**`smallest_group_n_hadm` is the column to check before trusting an age flag.** It is the size of the thinnest range group behind the verdict, and for 20 of the 24 age-flagged features it is under 20 — as low as 1 for Bilirubin Total, Protein Total, DHEA-Sulfate and Testosterone Free. Sort by it ascending to put the fragile rows first. The sex verdicts need no such caveat: a small group that is 100% female is still convincing.

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
TODO

#### studies/merge_strategy_test.py
TODO

### main.py
Main file to run the pipeline. Calls the steps in order: `prep/` to build the representations,
`model/train.py` to train, then the `analysis/` steps.
In each step the information is written to csv file. In the analysis steps, these csv files are read in. 

Flag `--cohort_folder` (required): `all` runs every cohort in `data/cohorts_100/`, `subset` runs `data/cohorts_50/` (one cohort per code1), `test` runs the cohorts in `data/cohorts_test/`.
Flag `-purge`: delete all cohort outputs and rebuild them from scratch.
Flag `--balanced_rf_only`: Train only balanced random forest. Random forest and catboost are skipped. 
Flag `--all_mappings`: Build merged ranges and discretised mappings for all three merge strategies. Without it only `BEST_MERGE_STRATEGY` is built, which is the only one training uses. `merge_strategy_test.py` needs all three and requests them itself, so this flag is only needed to have them on disk beforehand.

Lists all cohorts from the selected cohort folder (`data/cohorts_100/` for `all`, `data/cohorts_test/` for `test`).
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
- filters the cohorts by `MIN_COHORT_ROC` and prints which ones were excluded
- cross-cohort feature importance analysis
- cross-cohort performance analysis
- cross-cohort fairness analysis

All three cross-cohort analyses run on the filtered cohort list only. The per-cohort loop above runs on every cohort.

### Pipeline functions
Split across `prep/`, `model/` and `analysis/`; each entry below names the module it lives in.

#### Dependencies
Lorem ipsum

#### run_scan_step()
Lives in `prep/scan.py`.
Scans the labevents file for the union of all hadm_ids from all cohorts in folder data/cohorts_test. Skips cohorts for which a file <cohort>_ranges.csv already exists (except if --purge is set, then all cohort files have been deleted previously).
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
If flag `--balanced_rf_only` is set, then training only on balanced random forest. It creates new representations by concatenating and imputation. 
Uses the following feature sets: 
- continuous values
- discretised values (discretised by BEST_MERGE_STRATEGY)
- discretised + imputed
- binary values
- continuous - binary concatenated
- discretised (imputed) - binary concatenated
- continuous - discretised - binary concatenated (`cont_disc_bin`). Uses the plain discretised frame, not the imputed one, so unlike `disc_bin` it does not depend on the fold and is concatenated once per cohort. Column names do not collide: continuous and binary columns are prefixed, the discretised ones are bare itemids.

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
Lives in `prep/imputation.py`, with `_knn_impute_ranges()`.
Builds disc_imp representation for one fold. Imputes ranges from the training set for values that are measured but have no reference range given. 
Calls `_knn_impute_ranges()` as helper. 

#### _knn_impute_ranges()
Performs the range imputation step for each measured (hadm_id, itemid) pair with missing range. 
Only takes ranges from training admissions to avoid leakage. 
Matching rule: same itemid, same gender, closest age, averaged over k training donors. 
For each (itemid, gender) pair find the k nearest neighbours by age and average their reference range thresholds. 

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
one figure with 4 panels. One panel for each representation in STABILITY_FEATURE_SETS (continuois, discretised, discretised + imputed, binarised)s
Displays the top 20 ranks per representation, averaged across the folds. 

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

#### _bias_metric_gap()
Helper used by `<cohort>_fairness_analysis()`.
Calculates fairness-gap per fairness criterion across demographic groups. It compares best and worst group. 

#### sample_classification_analysis():
Lives in `analysis/classification.py`.
Creates per-sample classification heatmap for BEST_CLASSIFIER. Samples sorted first by true label, then by mean predicted probability across cohorts. 
Takes CLASSIFICATION_HEATMAP_REPS as order and display name for representations. 

#### filter_cohorts_by_roc()
Lives in `analysis/selection.py`.
Splits the cohort list into the cohorts good enough for the cross-cohort analyses and the ones dropped. A cohort is kept when its fold-averaged AUC-ROC on `ROC_FILTER_REP` under BEST_CLASSIFIER reaches `MIN_COHORT_ROC`.
Reads `<cohort>_performance_summary.csv`, so it has to run after `train()`. A cohort with no performance summary (never trained, or a run that died mid-cohort) is dropped rather than raising, so a partial batch can still be aggregated. A NaN AUC fails the comparison and is dropped as well.
Prints one line per dropped cohort with its AUC. Returns `(kept, dropped)`.
Called once in `main.py`; only `kept` is passed on to the three cross-cohort analyses.

#### _icd_chapter()
Helper used by `_disease_group()`.
Returns `(group key, group name)` for an ICD code, matched against `ICD_CHAPTERS` on both the letter and the number, since D and H each split across two chapters. `C83` gives `("C, D00-D49", "Neoplasms")`, `H81` gives `("H60-H95", "Ear and mastoid process")`. Falls back to `UNMAPPED_ICD_CHAPTER` for a code that matches nothing.

#### _disease_group() / _target_group()
Helpers used by `performance_analysis()`.
`_disease_group()` returns the ICD chapter of code1 via `_icd_chapter()`, using the whole code. `_target_group()` returns only the first letter of code2.
Both parse the cohort *filename*, so both require the `cohort_<code1>-<code2>` schema — see [Cohort naming](#cohort-naming). `_target_group()` raises `IndexError` on a name without a hyphen, and because it is passed to `groupby()` as a callable it does so only when the target heatmap is built, at the very end of a run. `_disease_group()` degrades to `?` instead of raising, so the two do not fail alike.

#### _stratified_heatmap()
Helper used by `performance_analysis()`.
Collapses a per-cohort performance table to one row per group and plots it. Takes either a function mapping cohort name to group key, or a Series of keys indexed by cohort — an ordered categorical Series keeps the rows in bin order rather than sorting the labels alphabetically. Empty categorical bins are dropped rather than drawn as blank rows.
Row labels carry the number of cohorts in the group, and optionally a second line naming what the group key stands for (`descriptions`, used for the ICD chapter names). The bottom `average` row is the mean across cohorts, not the mean of the group rows, so it is cohort-weighted and matches the per-cohort heatmap.

#### feature_importance_analysis()
Lives in `analysis/importance.py`.
Analyses the feature importance change across cohorts. Gives fold feature importance stability over the folds for each cohort and the representations continuous, discretised and binarised. 
Also analyses rank shift between continuous representation and binarised/discretised representation. 
Loads each cohort's `<cohort>_feature_shift.csv` and `<cohort>_feature_stability.csv` files. 
Note that the labs shown in the rank difference heatmaps are selected by their mean rank **across the cohorts passed in**, and the rangeless-lab filter is likewise evaluated over that cohort set only. Changing the cohort set (a different `MIN_COHORT_ROC`, or more cohorts) therefore changes which labs form the columns, not just which cohorts form the rows, so two runs cannot be compared cell by cell.
The number of columns dropped for having no reference range is printed to the console alongside the number kept.
Ranks are computed per cohort over that cohort's own feature set, so the size of the ranking universe differs between cohorts and a rank shift of a given size does not mean the same thing in every row. Keeping the feature matrix free of permanently empty columns (see the blacklist criteria under `mimic_analysis.py`) is what keeps that spread small.
**Output:**
- `rank_difference_cont_disc_heatmap.png`
- `rank_difference_cont_bin_heatmap.png`
- `kendalls_w_heatmap.png`

#### performance_analysis()
Lives in `analysis/performance.py`. The cohort grouping helpers it shares with the other two analyses are in `analysis/grouping.py`.
Takes the files `<cohort>_performance_summary.csv` of the filtered cohorts and creates heatmaps by cohort and representation (cont, disc, bin) for the performance measures AUC-ROC, F1, MCC. For BEST_CLASSIFIER only. The discretised + imputed representation is trained and written to the summary csv but left out of these plots.
Additionally reads each cohort's label column (`load_cohort()`) to get the cohort size and the share of positive labels, which are not derivable from the cohort name.
For AUC-ROC only, it also produces four stratified views via `_stratified_heatmap()` — by disease group, target group, cohort size bin and label distribution bin — and an unbinned scatter of AUC-ROC against size and prevalence. The stratified views are built before the average row is appended, so that row is not folded into a group.
**Output:**
- `performance_f1_heatmap.png`
- `performance_mcc_heatmap.png`
- `performance_roc_heatmap.png`
- `performance_roc_disease_group_heatmap.png`
- `performance_roc_target_group_heatmap.png`
- `performance_roc_size_group_heatmap.png`
- `performance_roc_prevalence_group_heatmap.png`
- `performance_roc_vs_cohort_properties.png`
- `performance_roc_difference_heatmap.png`
- `performance_roc_difference_disease_heatmap.png`
- `performance_roc_difference_target_heatmap.png`
- `performance_roc_difference_size_heatmap.png`
- `performance_roc_difference_prevalence_heatmap.png`
- `performance_roc_boxplot.png`
- `performance_roc_boxplot_disease.png`
- `performance_roc_boxplot_target.png`
- `performance_roc_boxplot_size.png`
- `performance_roc_boxplot_prevalence.png`

#### fairness_analysis()
Lives in `analysis/fairness.py`.
Prints equalised odd gaps by representation across cohorts. For gender and age band (max,min).
Reads the files `<cohort>_fairness_criteria.csv` for each cohort. Uses helper `_group_supported()` to exclude demographic groups with insufficient support (FAIRNESS_MIN_GROUP_SAMPLES, FAIRNESS_MIN_LABEL_SAMPLES)

**Output:**
- `fairness_eo_age_band.png`
- `fairness_eo_gender.png`