"""
Identify lab items for which no reference range exists in labevents for the
chosen cohort.

USAGE
-----
    python extract_missing_range_items.py --cohort aplasia
    python extract_missing_range_items.py --cohort NF

INPUT
-----
--cohort aplasia | NF
    Selects which cohort to analyse. Determines the range_per_item file to read:

    aplasia: output/data_analysis/mimic_cohort_aplasia_45_days_range_per_item.csv
    NF:      output/data_analysis/mimic_cohort_NF_30_days_range_per_item.csv

    range_per_item file:
        Output of check_reference_ranges.py. Must contain columns 'itemid' and
        'hadm_ids_with_range'. Items where hadm_ids_with_range == 0 have no
        reference range recorded anywhere in labevents for the cohort.

    data/raw_files/d_labitems.csv.gz (always):
        MIMIC lab item dictionary. Used to look up the label, fluid, and
        category for each itemid with no range.

WHAT IT DOES
------------
Filters the range_per_item file to itemids where hadm_ids_with_range == 0,
meaning labevents contains no reference range for that item across the entire
cohort. It then joins these itemids against d_labitems to retrieve their
human-readable label, fluid type, and category.

OUTPUT
------
output/data_analysis/cohort_<name>_no_range_items.csv

    One row per lab item with no reference range, with columns:

    itemid    — lab item identifier
    label     — human-readable name of the lab item
    fluid     — sample fluid type (e.g. Blood, Urine)
    category  — lab category (e.g. Chemistry, Hematology, Blood Gas)
"""

import argparse
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "data_analysis"

COHORTS = {
    "aplasia": OUTPUT_DIR / "mimic_cohort_aplasia_45_days_range_per_item.csv",
    "NF":      OUTPUT_DIR / "mimic_cohort_NF_30_days_range_per_item.csv",
}

D_LABITEMS_FILE = DATA_DIR / "raw_files/d_labitems.csv.gz"

parser = argparse.ArgumentParser(description="Extract lab items with no reference range in labevents.")
parser.add_argument("--cohort", choices=list(COHORTS), required=True, help="Cohort to analyse")
args = parser.parse_args()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

per_item = pd.read_csv(COHORTS[args.cohort])
no_range_ids = per_item.loc[per_item["hadm_ids_with_range"] == 0, "itemid"]

d_labitems = pd.read_csv(D_LABITEMS_FILE, usecols=["itemid", "label", "fluid", "category"])
result = (
    d_labitems[d_labitems["itemid"].isin(no_range_ids)]
    .reset_index(drop=True)
)

output_file = OUTPUT_DIR / f"cohort_{args.cohort}_no_range_items.csv"
result.to_csv(output_file, index=False)

print(f"Items with no range in labevents: {len(result)}")
print(f"Written to: {output_file}")
print(result.to_string(index=False))