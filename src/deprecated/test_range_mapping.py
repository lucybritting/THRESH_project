import argparse
from pathlib import Path

import pandas as pd

DATA_DIR     = Path(__file__).parent.parent / "data"
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "range_analysis"
MAPPING_DIR  = Path(__file__).parent.parent / "output" / "range_mapping"

COHORT_FILES = {
    "aplasia": DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
    "NF":      DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
}


def load_mapping_files(cohort):
    merged   = pd.read_csv(MAPPING_DIR / f"{cohort}_detail_merged.csv")
    unmerged = pd.read_csv(MAPPING_DIR / f"{cohort}_detail_unmerged.csv")
    return merged, unmerged


def load_analysis_files(cohort):
    hadm_multi = pd.read_csv(ANALYSIS_DIR / f"cohort_{cohort}_hadm_multi_range.csv")
    multi      = pd.read_csv(ANALYSIS_DIR / f"cohort_{cohort}_multi_range.csv")
    single     = pd.read_csv(ANALYSIS_DIR / f"cohort_{cohort}_single_range.csv")
    no_range   = pd.read_csv(ANALYSIS_DIR / f"cohort_{cohort}_no_range.csv")
    return hadm_multi, multi, single, no_range


# ---- Tests ------------------------------------------------------------------

def test_files_differ_only_in_merged_rows(merged, unmerged):
    key_cols     = ["hadm_id", "itemid"]
    compare_cols = ["value", "discretised_value", "ref_range_lower", "ref_range_upper"]

    m = merged.set_index(key_cols).sort_index()
    u = unmerged.set_index(key_cols).sort_index()

    non_merged_idx = m[m["merged"] == False].index
    identical = m.loc[non_merged_idx, compare_cols].equals(u.loc[non_merged_idx, compare_cols])

    print(f"Test 1  — non-merged rows identical in both files:                    "
          f"{'PASS' if identical else 'FAIL'}")


def test_merged_rows_in_range_files(merged, multi, hadm_multi):
    merged_rows = merged[merged["merged"] == True]

    merged_itemids  = set(merged_rows["itemid"])
    multi_itemids   = set(multi["itemid"])
    missing_itemids = merged_itemids - multi_itemids
    print(f"Test 2a — all merged itemids appear in multi_range.csv:               "
          f"{'PASS' if not missing_itemids else f'FAIL  (missing: {missing_itemids})'}")

    merged_pairs     = set(zip(merged_rows["hadm_id"], merged_rows["itemid"]))
    hadm_multi_pairs = set(zip(hadm_multi["hadm_id"], hadm_multi["itemid"]))
    missing_pairs    = merged_pairs - hadm_multi_pairs
    print(f"Test 2b — all merged (hadm_id, itemid) pairs in hadm_multi_range:     "
          f"{'PASS' if not missing_pairs else f'FAIL  ({len(missing_pairs)} pairs missing)'}")


def test_discretisation_correctness(merged):
    valid = merged[
        merged["discretised_value"].notna() &
        (merged["discretised_value"] != -99)
    ].copy()

    lower = valid["ref_range_lower"]
    upper = valid["ref_range_upper"]
    value = valid["value"]

    # reconstruct expected codes matching priority of _discretise: lower check first
    expected = pd.Series(0, index=valid.index)
    expected[upper.notna() & (value > upper)] = 1
    expected[lower.notna() & (value < lower)] = -1

    errors = (valid["discretised_value"].astype(float) != expected.astype(float)).sum()
    print(f"Test 3  — discretisation correctness (-1 / 0 / 1):                    "
          f"{'PASS' if errors == 0 else f'FAIL  ({errors} rows incorrectly discretised)'}")


def test_no_range_maps_to_none(merged, unmerged, no_range):
    no_range_itemids = set(no_range["itemid"])
    for label, detail in [("merged", merged), ("unmerged", unmerged)]:
        sub        = detail[detail["itemid"].isin(no_range_itemids)]
        n_non_null = sub["discretised_value"].notna().sum()
        print(f"Test 4{'a' if label == 'merged' else 'b'} — no-range items map to None ({label}):              "
              f"{'PASS' if n_non_null == 0 else f'FAIL  ({n_non_null} rows have non-None value)'}")


def test_multi_range_unmerged_maps_to_minus99(unmerged, hadm_multi):
    multi_pair_df = hadm_multi[["hadm_id", "itemid"]].drop_duplicates()
    sub           = unmerged.merge(multi_pair_df, on=["hadm_id", "itemid"], how="inner")
    n_not_99      = (sub["discretised_value"] != -99).sum()
    print(f"Test 5  — multi-range pairs have -99 in unmerged:                     "
          f"{'PASS' if n_not_99 == 0 else f'FAIL  ({n_not_99} rows not -99)'}")


def test_merge_logic_correctness(merged, hadm_multi):
    merged_rows = merged[merged["merged"] == True][
        ["hadm_id", "itemid", "ref_range_lower", "ref_range_upper"]
    ]

    expected = (
        hadm_multi.groupby(["hadm_id", "itemid"])
        .agg(expected_lower=("ref_range_lower", "min"),
             expected_upper=("ref_range_upper", "max"))
        .reset_index()
    )

    check       = merged_rows.merge(expected, on=["hadm_id", "itemid"], how="left")
    lower_match = check["ref_range_lower"].equals(check["expected_lower"])
    upper_match = check["ref_range_upper"].equals(check["expected_upper"])

    print(f"Test 6  — merged range is min lower / max upper of individual ranges:  "
          f"{'PASS' if lower_match and upper_match else 'FAIL'}")


def test_range_classification_completeness(merged, multi, single, no_range):
    all_itemids      = set(merged["itemid"])
    multi_itemids    = set(multi["itemid"])
    single_itemids   = set(single["itemid"])
    no_range_itemids = set(no_range["itemid"])

    overlaps = (
        len(multi_itemids & single_itemids) +
        len(multi_itemids & no_range_itemids) +
        len(single_itemids & no_range_itemids)
    )
    print(f"Test 7a — no itemid appears in more than one classification file:      "
          f"{'PASS' if overlaps == 0 else f'FAIL  ({overlaps} overlapping itemids)'}")

    unclassified = all_itemids - (multi_itemids | single_itemids | no_range_itemids)
    print(f"Test 7b — all itemids in detail file are classified:                  "
          f"{'PASS' if not unclassified else f'FAIL  ({len(unclassified)} itemids not classified)'}")


def test_hadm_ids_in_cohort(merged, unmerged, cohort):
    cohort_hadm_ids = set(
        pd.read_csv(COHORT_FILES[cohort], usecols=["hadm_id"])["hadm_id"]
    )
    for label, detail in [("merged", merged), ("unmerged", unmerged)]:
        unknown = set(detail["hadm_id"]) - cohort_hadm_ids
        print(f"Test 8{'a' if label == 'merged' else 'b'} — all hadm_ids in {label} belong to cohort:           "
              f"{'PASS' if not unknown else f'FAIL  ({len(unknown)} unknown hadm_ids)'}")


# ---- Inspection -------------------------------------------------------------

def inspect_pair(cohort, merged, unmerged, hadm_multi, multi, single, no_range,
                 hadm_id=None, itemid=None):
    if hadm_id is None or itemid is None:
        row     = merged.sample(1).iloc[0]
        hadm_id = int(row["hadm_id"])
        itemid  = int(row["itemid"])
        print("(randomly selected)")

    print(f"\n=== hadm_id={hadm_id}  itemid={itemid}  cohort={cohort} ===\n")

    m_row = merged[(merged["hadm_id"]     == hadm_id) & (merged["itemid"]   == itemid)]
    u_row = unmerged[(unmerged["hadm_id"] == hadm_id) & (unmerged["itemid"] == itemid)]

    if m_row.empty and u_row.empty:
        print("Not found in either detail file.")
        return

    def cell(df_row, col):
        return str(df_row.iloc[0][col]) if not df_row.empty else "N/A"

    print(f"{'':25} {'merged':>12}  {'unmerged':>12}")
    for col in ["value", "discretised_value", "ref_range_lower", "ref_range_upper"]:
        print(f"{col:25} {cell(m_row, col):>12}  {cell(u_row, col):>12}")

    print()
    if itemid in no_range["itemid"].values:
        label = no_range.loc[no_range["itemid"] == itemid, "label"].iloc[0]
        print(f"itemid {itemid} ({label})  →  no_range.csv")
    elif itemid in single["itemid"].values:
        sub   = single[single["itemid"] == itemid]
        label = sub["label"].iloc[0]
        print(f"itemid {itemid} ({label})  →  single_range.csv")
        print(sub[["ref_range_lower", "ref_range_upper", "valueuom"]].to_string(index=False))
    elif itemid in multi["itemid"].values:
        sub   = multi[multi["itemid"] == itemid]
        label = sub["label"].iloc[0]
        print(f"itemid {itemid} ({label})  →  multi_range.csv")
        print(sub[["ref_range_lower", "ref_range_upper", "n_hadm_id", "valueuom"]].to_string(index=False))
    else:
        print(f"itemid {itemid} not found in any range classification file.")

    print()
    pair = hadm_multi[(hadm_multi["hadm_id"] == hadm_id) & (hadm_multi["itemid"] == itemid)]
    if pair.empty:
        print(f"(hadm_id={hadm_id}, itemid={itemid}) not found in hadm_multi_range.csv")
    else:
        print(f"(hadm_id={hadm_id}, itemid={itemid}) found in hadm_multi_range.csv:")
        print(pair[["ref_range_lower", "ref_range_upper", "valueuom"]].to_string(index=False))


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=list(COHORT_FILES), required=True)
    parser.add_argument("--hadm_id", type=int, default=None)
    parser.add_argument("--itemid",  type=int, default=None)
    parser.add_argument("--inspect", action="store_true", default=False,
                        help="Inspect a specific or random (hadm_id, itemid) pair")
    args = parser.parse_args()

    merged, unmerged                    = load_mapping_files(args.cohort)
    hadm_multi, multi, single, no_range = load_analysis_files(args.cohort)

    print(f"\n--- Tests for cohort: {args.cohort} ---\n")
    test_files_differ_only_in_merged_rows(merged, unmerged)
    test_merged_rows_in_range_files(merged, multi, hadm_multi)
    test_discretisation_correctness(merged)
    test_no_range_maps_to_none(merged, unmerged, no_range)
    test_multi_range_unmerged_maps_to_minus99(unmerged, hadm_multi)
    test_merge_logic_correctness(merged, hadm_multi)
    test_range_classification_completeness(merged, multi, single, no_range)
    test_hadm_ids_in_cohort(merged, unmerged, args.cohort)

    if args.inspect:
        inspect_pair(args.cohort, merged, unmerged, hadm_multi, multi, single, no_range,
                     args.hadm_id, args.itemid)