"""
Check which lab items in a cohort have reference ranges recorded in labevents.

USAGE
-----
    python check_reference_ranges.py --cohort aplasia
    python check_reference_ranges.py --cohort NF

INPUT
-----
--cohort aplasia | NF
    Selects which cohort to analyse. Determines the following file paths:

    aplasia:
        data/cohorts/mimic_cohort_aplasia_45_days.csv.gz
        data/preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv
    NF:
        data/cohorts/mimic_cohort_NF_30_days.csv.gz
        data/preprocessed_files/aggregated/mimic_cohort_NF_30_days_agg.csv

    Cohort file (.csv.gz):
        Must contain a column 'hadm_id'. Used to restrict the labevents scan
        to admissions belonging to the chosen cohort.

    Aggregated file (_agg.csv):
        Wide-format table with one row per admission and one column per lab
        itemid. A non-NaN cell means that admission has a measurement for that
        item. Used to determine which itemids are relevant and which admissions
        have measurements for each item.

    data/raw_files/labevents.csv.gz (always):
        MIMIC labevents table. Scanned in 500k-row chunks for the columns
        hadm_id, itemid, ref_range_lower, ref_range_upper. Only rows whose
        hadm_id belongs to the chosen cohort are considered.

WHAT IT DOES
------------
For each lab itemid present in the aggregated file, the script determines:
  1. Which cohort admissions have a measurement for that item (from the
     aggregated file, non-NaN cells).
  2. Of those, which have at least one labevents row with a non-null reference
     range (ref_range_lower or ref_range_upper) for the same item.
  3. How many distinct (ref_range_lower, ref_range_upper) pairs exist in
     labevents for that item across the cohort.

OUTPUT
------
output/data_analysis/mimic_cohort_<name>_range_per_item.csv

    One row per lab itemid with the following columns:

    itemid                  — lab item identifier
    hadm_ids_with_range     — number of cohort admissions that have a
                              measurement AND at least one labevents row with
                              a reference range for this item
    hadm_ids_missing_range  — number of cohort admissions that have a
                              measurement but NO reference range in labevents
                              for this item
    n_distinct_ranges       — number of distinct (range_lower, range_upper)
                              pairs observed in labevents for this item within
                              the cohort (0 means no range exists at all)
"""

import argparse
import pandas as pd
from pathlib import Path

DATA_DIR       = Path(__file__).parent.parent / "data"
OUTPUT_DIR     = Path(__file__).parent.parent / "output" / "data_analysis"
LABEVENTS_PATH = DATA_DIR / "raw_files/labevents.csv.gz"

COHORTS = {
    "aplasia": {
        "cohort":     DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
        "aggregated": DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv",
        "output":     OUTPUT_DIR / "mimic_cohort_aplasia_45_days_range_per_item.csv",
    },
    "NF": {
        "cohort":     DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
        "aggregated": DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_NF_30_days_agg.csv",
        "output":     OUTPUT_DIR / "mimic_cohort_NF_30_days_range_per_item.csv",
    },
}

parser = argparse.ArgumentParser(description="Check labevents reference ranges for a cohort.")
parser.add_argument("--cohort", choices=list(COHORTS), required=True, help="Cohort to analyse")
args = parser.parse_args()

paths = COHORTS[args.cohort]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load cohort hadm_ids ────────────────────────────────────────────────────
cohort_hadm_ids = set(pd.read_csv(paths["cohort"], usecols=["hadm_id"])["hadm_id"])
print(f"Cohort: {args.cohort}  ({len(cohort_hadm_ids)} admissions)")

# ── 2. Load aggregated file to get itemids and which hadm_ids have measurements ─
agg = pd.read_csv(paths["aggregated"])
lab_cols = [c for c in agg.columns if c != "hadm_id"]
itemid_set = {int(c) for c in lab_cols}
print(f"Aggregated shape: {agg.shape}  ({len(lab_cols)} lab items)")

agg_indexed = agg.set_index("hadm_id")
hadm_ids_with_measurement: dict[int, set[int]] = {
    int(col): set(agg_indexed.index[agg_indexed[col].notna()])
    for col in lab_cols
}

# ── 3. Scan labevents for ranges (cohort hadm_ids only) ───────────────────────
print("Scanning labevents for reference ranges (this may take a moment)...")

hadm_ids_with_range_in_labevents: dict[int, set[int]] = {iid: set() for iid in itemid_set}
distinct_ranges: dict[int, set[tuple]] = {iid: set() for iid in itemid_set}

for chunk in pd.read_csv(
    LABEVENTS_PATH,
    usecols=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper"],
    chunksize=500_000,
):
    sub = chunk[
        chunk["itemid"].isin(itemid_set) &
        chunk["hadm_id"].isin(cohort_hadm_ids)
    ]
    has_range = sub["ref_range_lower"].notna() | sub["ref_range_upper"].notna()
    sub = sub[has_range].dropna(subset=["hadm_id"])

    for itemid, group in sub.groupby("itemid", sort=False):
        iid = int(itemid)
        hadm_ids_with_range_in_labevents[iid].update(group["hadm_id"].astype(int))
        distinct_ranges[iid].update(zip(group["ref_range_lower"], group["ref_range_upper"]))

# ── 4. Build per-itemid table ──────────────────────────────────────────────────
rows = []
for col in lab_cols:
    iid = int(col)
    with_meas    = hadm_ids_with_measurement[iid]
    with_range   = hadm_ids_with_range_in_labevents[iid] & with_meas
    missing_range = with_meas - with_range
    n_distinct   = len(distinct_ranges[iid])
    rows.append({
        "itemid":                  iid,
        "hadm_ids_with_range":     len(with_range),
        "hadm_ids_missing_range":  len(missing_range),
        "n_distinct_ranges":       n_distinct,
    })

result = pd.DataFrame(rows)
result.to_csv(paths["output"], index=False)
print(f"\nSaved {paths['output']}")
print(result.to_string(index=False))