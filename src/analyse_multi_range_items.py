"""
For itemids with more than one distinct reference range, produces a table with
one row per range showing patient demographics for the chosen cohort.

Usage:
    python analyse_multi_range_items.py --cohort aplasia
    python analyse_multi_range_items.py --cohort NF
"""

import argparse
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "data_analysis"

COHORTS = {
    "aplasia": {
        "per_item": OUTPUT_DIR / "mimic_cohort_aplasia_45_days_range_per_item.csv",
        "cohort":   DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
    },
    "NF": {
        "per_item": OUTPUT_DIR / "mimic_cohort_NF_30_days_range_per_item.csv",
        "cohort":   DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
    },
}

LABEVENTS_FILE = DATA_DIR / "raw_files/labevents.csv.gz"
D_LABITEMS     = DATA_DIR / "raw_files/d_labitems.csv.gz"

parser = argparse.ArgumentParser(description="Analyse itemids with multiple reference ranges.")
parser.add_argument("--cohort", choices=list(COHORTS), required=True, help="Cohort to analyse")
args = parser.parse_args()

paths = COHORTS[args.cohort]

# ── 1. Itemids with >1 distinct range ─────────────────────────────────────────
per_item = pd.read_csv(paths["per_item"])
multi = per_item[per_item["n_distinct_ranges"] > 1]
multi_range_ids = set(multi["itemid"].astype(int))

print(f"Cohort: {args.cohort}")
print(f"Itemids with >1 distinct range: {len(multi_range_ids)}")
if not multi_range_ids:
    print("None found — nothing to analyse.")
    raise SystemExit(0)

# ── 2. Item labels ─────────────────────────────────────────────────────────────
item_labels = (
    pd.read_csv(D_LABITEMS, usecols=["itemid", "label"])
    .set_index("itemid")["label"]
    .to_dict()
)

# ── 3. Cohort demographics ─────────────────────────────────────────────────────
cohort = pd.read_csv(paths["cohort"], usecols=["hadm_id", "age", "gender"])
cohort_hadm_ids = set(cohort["hadm_id"])

# ── 4. Scan labevents for (hadm_id, itemid, range_lower, range_upper) ──────────
print("Scanning labevents for reference ranges (this may take a moment)...")
records = []
for chunk in pd.read_csv(
    LABEVENTS_FILE,
    usecols=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper"],
    chunksize=500_000,
):
    sub = chunk[chunk["itemid"].isin(multi_range_ids) & chunk["hadm_id"].isin(cohort_hadm_ids)]
    has_range = sub["ref_range_lower"].notna() | sub["ref_range_upper"].notna()
    records.append(sub[has_range])

obs = (
    pd.concat(records, ignore_index=True)
    .drop_duplicates()
    .dropna(subset=["hadm_id"])
)
obs["hadm_id"] = obs["hadm_id"].astype(int)
obs = obs.merge(cohort, on="hadm_id", how="left")
print(f"  {len(obs)} distinct (hadm_id, itemid, range) records with demographics\n")

# ── 5. Build one row per (itemid, range) ──────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for itemid in sorted(multi_range_ids):
    label = item_labels.get(itemid, "unknown")

    item_df = obs[obs["itemid"] == itemid].drop_duplicates(
        subset=["hadm_id", "ref_range_lower", "ref_range_upper"]
    )

    if item_df.empty:
        continue

    grouped = (
        item_df
        .groupby(["ref_range_lower", "ref_range_upper"], dropna=False)
        .agg(
            n_hadm=("hadm_id", "nunique"),
            genders=("gender", lambda s: "|".join(sorted(s.dropna().unique().tolist()))),
            n_F=("gender", lambda s: (s == "F").sum()),
            n_M=("gender", lambda s: (s == "M").sum()),
            age_mean=("age", "mean"),
            age_std=("age", "std"),
            age_min=("age", "min"),
            age_max=("age", "max"),
        )
        .reset_index()
        .sort_values("ref_range_lower")
    )

    for _, row in grouped.iterrows():
        rows.append({
            "itemid":     itemid,
            "label":      label,
            "range_lower": row["ref_range_lower"],
            "range_upper": row["ref_range_upper"],
            "n_hadm":     int(row["n_hadm"]),
            "genders":    row["genders"],
            "n_F":        int(row["n_F"]),
            "n_M":        int(row["n_M"]),
            "age_mean":   round(row["age_mean"], 2) if pd.notna(row["age_mean"]) else None,
            "age_std":    round(row["age_std"], 2)  if pd.notna(row["age_std"])  else None,
            "age_min":    int(row["age_min"])        if pd.notna(row["age_min"])  else None,
            "age_max":    int(row["age_max"])        if pd.notna(row["age_max"])  else None,
        })

# ── 6. Write main output ──────────────────────────────────────────────────────
result = pd.DataFrame(rows)
output_path = OUTPUT_DIR / f"cohort_{args.cohort}_multi_range_analysis.csv"
result.to_csv(output_path, index=False)

print(result.to_string(index=False))
print(f"\nSaved {output_path}")

# ── 7. Gender-stratified subset ───────────────────────────────────────────────
# An itemid is gender-stratified if every range is used by exactly one gender
# and different ranges are used by different genders.
gender_strat_items = []
for itemid, group in result.groupby("itemid", sort=False):
    all_single_gender = ((group["n_F"] == 0) | (group["n_M"] == 0)).all()
    multiple_genders  = group["genders"].nunique() > 1
    if all_single_gender and multiple_genders:
        gender_strat_items.append(group)

if gender_strat_items:
    gender_strat = pd.concat(gender_strat_items, ignore_index=True)
else:
    gender_strat = result.iloc[0:0].copy()

gender_strat_path = OUTPUT_DIR / f"cohort_{args.cohort}_gender_stratified_ranges.csv"
gender_strat.to_csv(gender_strat_path, index=False)

print(f"\nGender-stratified itemids: {gender_strat['itemid'].nunique()}")
print(gender_strat.to_string(index=False))
print(f"\nSaved {gender_strat_path}")