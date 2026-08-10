# Range Analysis and Mapping

Documentation for `reference_range_mapping.py` and `test_range_mapping.py`.

---

## reference_range_mapping.py

Scans the raw MIMIC labevents file to extract reference ranges per (hadm_id, itemid) pair, produces range analysis tables, and discretises lab values against those ranges.

### Input files

| File | Description |
|---|---|
| `../../data/raw_files/labevents.csv.gz` | Raw MIMIC labevents, scanned in chunks |
| `../../data/raw_files/d_labitems.csv.gz` | Lab item labels, used to annotate output tables |
| `data/preprocessed_files/aggregated/{cohort}_agg.csv` | Aggregated lab values per admission, one row per hadm_id |
| `data/cohorts/{cohort}.csv.gz` | Cohort file, used to read discharge times for time-window filtering |

### Output files

Range analysis tables are written to `output/range_analysis/`, discretisation outputs to `output/range_mapping/`.

| File | Location | Description |
|---|---|---|
| `cohort_{cohort}_summary.csv` | `range_analysis/` | Per-itemid counts of admissions with no, single, or multiple ranges |
| `cohort_{cohort}_single_range.csv` | `range_analysis/` | Items with exactly one range across all admissions |
| `cohort_{cohort}_multi_range.csv` | `range_analysis/` | Items with multiple ranges, listing each range and how many admissions had it |
| `cohort_{cohort}_no_range.csv` | `range_analysis/` | Items for which no reference range was found |
| `cohort_{cohort}_hadm_multi_range.csv` | `range_analysis/` | All individual ranges for (hadm_id, itemid) pairs with more than one range |
| `{cohort}_discretised_{suffix}.csv` | `range_mapping/` | Discretised lab values in the same wide structure as the aggregated file |
| `{cohort}_detail_{suffix}.csv` | `range_mapping/` | One row per (hadm_id, itemid): value, discretised value, range used, merge flag |

`{suffix}` is `merged` or `unmerged` depending on the `--range_merge` flag.

### Arguments

| Argument | Type | Description |
|---|---|---|
| `--cohort` | `aplasia` \| `NF` | Required. Selects the cohort to process |
| `--task` | `analysis` \| `discretise` | Required. One or both tasks to run (space-separated) |
| `--range_merge` | `yes` \| `no` \| `both` | Required when `--task` includes `discretise`. Not needed for analysis only. |

### Task: analysis

Builds the range analysis tables. For each lab item in the cohort, counts how many admissions have no range, a single consistent range, or multiple different ranges within the time window. Produces the five `cohort_{cohort}_*.csv` files listed above.

```bash
python src/reference_range_mapping.py --cohort aplasia --task analysis
```

### Task: discretise

Discretises the aggregated lab values against the reference ranges extracted from labevents. For each (hadm_id, itemid) pair with a value, maps to:

| Code | Meaning |
|---|---|
| `-1` | Value below reference range |
| `0` | Value within reference range |
| `1` | Value above reference range |
| `None` | No reference range found |
| `-99` | Multiple ranges exist and unmerged mode is selected |

When `merge` is selected and multiple ranges exist, they are merged into a single range by taking the lowest `ref_range_lower` and the highest `ref_range_upper` across all ranges. Passing `both` produces both output files in one run.

```bash
# Unmerged only
python src/reference_range_mapping.py --cohort aplasia --task discretise --range_merge no

# Merged only
python src/reference_range_mapping.py --cohort aplasia --task discretise --range_merge yes

# Both outputs in one run
python src/reference_range_mapping.py --cohort aplasia --task discretise --range_merge both
```

Both tasks can be combined in one run (`--range_merge` required when discretise is included):

```bash
python src/reference_range_mapping.py --cohort aplasia --task analysis discretise --range_merge both
```

### Notes

- Reference ranges are only collected from labevents within the 14 days before discharge (`WINDOW_DAYS = 14` in the script).
- The labevents file is read in chunks (`CHUNK_SIZE`). A larger chunk size reduces per-chunk Python overhead at the cost of higher peak memory usage.

---

## test_range_mapping.py

Tests and inspection tool for verifying the output of the discretisation pipeline. Always runs the two automated tests; optionally inspects a specific or random (hadm_id, itemid) pair.

### Input files

| File | Location | Description |
|---|---|---|
| `{cohort}_detail_merged.csv` | `range_mapping/` | Merged discretisation detail output |
| `{cohort}_detail_unmerged.csv` | `range_mapping/` | Unmerged discretisation detail output |
| `cohort_{cohort}_hadm_multi_range.csv` | `range_analysis/` | All individual ranges for multi-range pairs |
| `cohort_{cohort}_multi_range.csv` | `range_analysis/` | Per-itemid summary for items with multiple ranges |
| `cohort_{cohort}_single_range.csv` | `range_analysis/` | Per-itemid summary for items with a single range |
| `cohort_{cohort}_no_range.csv` | `range_analysis/` | Items with no reference range |
| `data/cohorts/{cohort}.csv.gz` | `../../data/cohorts` | Cohort file, used to verify hadm_id membership |

### Arguments

| Argument | Type | Description |
|---|---|---|
| `--cohort` | `aplasia` \| `NF` | Required. Selects which cohort's files to load |
| `--inspect` | flag | Run the pair inspection in addition to the tests |
| `--hadm_id` | int | Admission ID to inspect. Must be combined with `--itemid` |
| `--itemid` | int | Lab item ID to inspect. Must be combined with `--hadm_id` |

### Tests

All tests always run.

**Test 1** — verifies that `detail_merged` and `detail_unmerged` are identical for all rows where `merged=False`. Since merging only affects multi-range rows, all other rows should be unchanged between the two files.

**Test 2a** — verifies that every itemid with `merged=True` in `detail_merged` also appears in `multi_range.csv`. Merging can only occur for items that have multiple ranges, so they must be listed there.

**Test 2b** — verifies that every (hadm_id, itemid) pair with `merged=True` in `detail_merged` appears in `hadm_multi_range.csv`. That file lists all individual ranges per multi-range pair, so any merged pair must be present.

**Test 3** — verifies correctness of the `-1 / 0 / 1` mapping for all rows with a valid discretised code. Reconstructs the expected code from value and range bounds and checks it matches. Applies to both single-range rows and merged multi-range rows.

**Test 4a/4b** — verifies that all rows in the merged and unmerged detail files whose itemid appears in `no_range.csv` have `discretised_value = None`. Items without a range cannot be discretised.

**Test 5** — verifies that all (hadm_id, itemid) pairs appearing in `hadm_multi_range.csv` have `discretised_value = -99` in the unmerged detail file. Multi-range pairs must map to `-99` when not merging.

**Test 6** — verifies that the merged range for each `merged=True` row is the minimum `ref_range_lower` and maximum `ref_range_upper` of all individual ranges in `hadm_multi_range.csv` for that pair.

**Test 7a** — verifies that no itemid appears in more than one of `no_range`, `single_range`, or `multi_range`. Each item must belong to exactly one classification.

**Test 7b** — verifies that every itemid present in the detail file appears in at least one classification file. No item should be unclassified.

**Test 8a/8b** — verifies that all hadm_ids in the merged and unmerged detail files belong to the cohort. No unknown admissions should appear in the output.

### Not covered

The correctness of the 14-day time-window filter applied during range extraction cannot be verified without re-scanning the raw labevents file. This must be validated manually if needed.

### Inspection

Activated with `--inspect`. If `--hadm_id` and `--itemid` are not provided, a random pair is sampled from `detail_merged`. Prints:

- A side-by-side table of value, discretised code, and range bounds from the merged and unmerged files
- Which range classification file the itemid appears in (`no_range`, `single_range`, or `multi_range`), with range details
- Whether the (hadm_id, itemid) pair appears in `hadm_multi_range.csv` and all its individual ranges

### Usage

```bash
# Run tests only
python src/test_range_mapping.py --cohort aplasia

# Run tests + inspect a random pair
python src/test_range_mapping.py --cohort aplasia --inspect

# Run tests + inspect a specific pair
python src/test_range_mapping.py --cohort aplasia --inspect --hadm_id 21094628 --itemid 50868
```