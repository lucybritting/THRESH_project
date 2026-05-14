import argparse
import pandas as pd
from pathlib import Path

# ---- Directories and file paths ----------------
DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

LABEVENTS_PATH  = DATA_DIR / "raw_files/labevents.csv.gz"
D_LABITEMS_PATH = DATA_DIR / "raw_files/d_labitems.csv.gz"

COHORTS = {
    "aplasia": {
        "agg":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_aplasia_45_days.csv.gz",
    },
    "NF": {
        "agg":    DATA_DIR / "preprocessed_files/aggregated/mimic_cohort_NF_30_days_agg.csv",
        "cohort": DATA_DIR / "cohorts/mimic_cohort_NF_30_days.csv.gz",
    },
}

# ---- Runtime Params -----------------------
WINDOW_DAYS = 14
CHUNK_SIZE  = 100000000


def load_cohort_files(args):
    """
    Gets sets of hadm_ids and itemids from aggregated file.
    Reads discharge time from cohort file and calculates window start time, based on global variable WINDOW_DAYS.

    :param cohort: args.cohort
    :return:
        hadm_ids_with_value: dict. itemid -> set of hadm_ids with value
        hadm_id_set: set[int]
        itemid_set: set[int]
        window_start_map: dict (hadm_id -> start_time)
        dischtime_map: dict (hadm_id -> dischtime)

    """
    # ---- Load aggregated file -------------------------
    agg_path = COHORTS[args.cohort]["agg"]
    agg = pd.read_csv(agg_path)

    # get set of itemids and admission ids
    lab_cols = [c for c in agg.columns if c != "hadm_id"]
    itemid_set = {int(c) for c in lab_cols}
    hadm_id_set = set(agg["hadm_id"])


    # re-indexing and building lookup: itemid -> set of hadm_ids with value
    agg_indexed = agg.set_index("hadm_id")  # make hadm_id the row index (faster lookup in DataFrame)
    # create dict: keys = itemid, values = set of admission ids where value for itemid is not NaN
    # for each item (column): finds row where value is not NaN, extracts the hadm_id and stores them as set
    hadm_ids_with_value: dict[int, set[int]] = {
        int(col): set(agg_indexed.index[agg_indexed[col].notna()])
        for col in lab_cols
    }
    print(f"Aggregated: {len(hadm_id_set)} admissions, {len(lab_cols)} lab items")


    # ---- Load cohort for discharge-time windows ---------------------
    # load only hadm_id and dischtime from cohort file
    cohort = pd.read_csv(COHORTS[args.cohort]["cohort"], usecols=["hadm_id", "dischtime"])
    # parse string to numeric datetime object
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    # window_start = dischtime - 14 days
    cohort["window_start"] = cohort["dischtime"] - pd.Timedelta(days=WINDOW_DAYS)
    # create lookup dicts. Set hadm_id as row index.
    dischtime_map = cohort.set_index("hadm_id")["dischtime"].to_dict()  # hadm_id -> dischtime
    window_start_map = cohort.set_index("hadm_id")["window_start"].to_dict()  # hadm_id -> window_start

    print(f"Cohort: {len(dischtime_map)} admissions with dischtime")

    return hadm_ids_with_value, hadm_id_set, itemid_set, window_start_map, dischtime_map

def extract_range(hadm_id_set, itemid_set, window_start, window_end):
    """
    Scans labevents file for labevents for given hadm_ids and itemids within time window.
    Uses global variable CHUNK_SIZE
    :param hadm_id_set: set of admission ids in the cohort
    :param itemid_set: set of itemids
    :param window_start: dict. hadm_id -> window_start
    :param window_end: dict. hadm_id -> dischtime
    :return:
        per_hadm_item_ranges: dict. (hadm_id, itemid)-->{(lower, upper),...}
        range_valueuom: dict. (itemid, lower, upper) --> valueuom
    """
    # ---- Scan labevents --------------------------------------------------
    # collect different ranges for hadm_id and itemid
    # (hadm_id, itemid) -> set of (ref_range_lower, ref_range_upper) tuples
    per_hadm_item_ranges: dict[tuple, set] = {}
    # collects different units for a range
    # (itemid, ref_range_lower, ref_range_upper) -> valueuom (= unit)
    range_valueuom: dict[tuple, set] = {}

    print("Scanning labevents (this may take a moment, because file is large)...")

    # chunk reading the labevents file
    for chunk in pd.read_csv(
            LABEVENTS_PATH,
            usecols=["hadm_id", "itemid", "charttime", "ref_range_lower", "ref_range_upper", "valueuom"],
            chunksize=CHUNK_SIZE,
    ):
        # filter each chunk for relevant rows: hadm_id belongs to cohort AND itemid is from the top100 (a column in the aggregated file)
        sub = chunk[
            chunk["hadm_id"].isin(hadm_id_set) &
            chunk["itemid"].isin(itemid_set)
            ]

        # skip the sub chunk if empty
        if sub.empty:
            continue

        sub = sub.copy()
        # convert datatypes to int and pandas datetime object
        sub["hadm_id"] = sub["hadm_id"].astype(int)
        sub["charttime"] = pd.to_datetime(sub["charttime"])

        # to every lab row add window_start and dischtime (new columns) by looking up the dicts
        # suffix "_dt" stands for datetime object
        sub["window_start"] = sub["hadm_id"].map(window_start)
        sub["dischtime_dt"] = sub["hadm_id"].map(window_end)

        # drop rows that have no window_start or dischtime
        sub = sub.dropna(subset=["window_start", "dischtime_dt"])
        # filter for labevents which are only within the time windows of each admission.
        sub = sub[
            (sub["charttime"] >= sub["window_start"]) &
            (sub["charttime"] <= sub["dischtime_dt"])
            ]

        # keep only rows with a reference range
        sub = sub[sub["ref_range_lower"].notna() | sub["ref_range_upper"].notna()]

        # skip empty sub chunk
        if sub.empty:
            continue

        # drop duplicate rows (same admission, item, range and unit). Its enough to keep them recorded once
        sub = sub.drop_duplicates(
            subset=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper", "valueuom"]
        )

        # iterate over remaining rows to fill the two accumulator dicts
        for row in sub.itertuples(index=False):  # yields each row as tuple
            # get admission and item id
            hadm_id = int(row.hadm_id)
            iid = int(row.itemid)
            # convert NaN to None (NaN is float, NaN != NaN). Would give single item in the set for each NaN.
            lower = None if pd.isna(row.ref_range_lower) else float(row.ref_range_lower)
            upper = None if pd.isna(row.ref_range_upper) else float(row.ref_range_upper)
            range_key = (lower, upper)

            # creates empty set for (hadm_id, iid), if it hasnt been recorded yet, and adds the range key (lower, upper)
            per_hadm_item_ranges.setdefault((hadm_id, iid), set()).add(range_key)

            unit = row.valueuom
            # same for units. record them, only if they are not NaN
            if not pd.isna(unit):
                range_valueuom.setdefault((iid, lower, upper), set()).add(str(unit))

    print("Scan complete.")
    return per_hadm_item_ranges, range_valueuom


def overview_tables(itemids, hadm_ids, hadms_with_value, hadm_item_ranges, valueuom_ranges, args):
    """

    Reads D_LABITEMS file to get labels for lab items
    :param: cohort: args.cohort
    :param itemids: set of item ids in the cohort
    :param hadm_ids: set of admission ids in the cohort
    :param hadms_with_value: dict. itemid -> set of hadm_ids that have value for this item
    :param hadm_item_ranges: per_hadm_item_ranges: dict. (hadm_id, itemid)-->{(lower, upper),...}
    :param valueuom_ranges: range_valueuom: dict. (itemid, lower, upper) --> valueuom
    :return:
    """
    # check if destination folder exists. If not create one
    path = Path(OUTPUT_DIR / "range_analysis")
    path.mkdir(parents=True, exist_ok=True)

    # itemid -> label
    # create dict: key = itemid, value = label
    item_labels: dict[int, str] = (
        pd.read_csv(D_LABITEMS_PATH, usecols=["itemid", "label"])
        .set_index("itemid")["label"]
        .to_dict()
    )

    # Helper to build output paths
    def out(name: str) -> Path:
        return path / f"cohort_{args.cohort}_{name}.csv"

    # ---- Build per-itemid summary ---------------------------------
    # for each itemid: how many admissions have values, how many have ranges (unique, multiple, None)
    summary_rows = []

    # iterate over all itemids
    for item in itemids: # item is int, itemids is set
        with_value = hadms_with_value[item]  # admission ids which have a value for this item
        n_hadm_ids = len(with_value)

        # init counter
        all_ranges = set()
        n_unique = 0
        n_no_range = 0
        n_multi = 0

        # iterate over all admissions which have a value for this item
        for hadm_id in with_value:
            ranges = hadm_item_ranges.get((hadm_id, item),
                                              set())  # gives set of ranges for this item and hadm_id or empty set
            all_ranges.update(ranges)  # add to global range set
            # update counter
            n = len(ranges)
            if n == 0:
                n_no_range += 1
            elif n == 1:
                n_unique += 1
            else:
                n_multi += 1


        summary_rows.append({
            "itemid": item,
            "label": item_labels.get(item),
            "n_hadm_ids": n_hadm_ids,
            "n_hadm_unique_range": n_unique,
            "n_hadm_no_range": n_no_range,
            "n_hadm_multi_range": n_multi,
            "n_ranges": len(all_ranges),
        })

    # convert list of dicts to DataFrame, sort values by itemid (better readability), and add the label.
    summary_df = pd.DataFrame(summary_rows).sort_values("itemid").reset_index(drop=True)
    summary_df.to_csv(out("summary"), index=False)
    print(f"Saved cohort_{args.cohort}_summary.csv  ({len(summary_df)} rows)")

    # ---- Build tables for one and multi range per itemid ----------------------------

    # get sets and init lists
    # get rows with ranges > 1, select their itemid column, convert to set.
    multi_itemids = set(summary_df.loc[summary_df["n_ranges"] > 1, "itemid"])
    multi_range_rows = []
    # select set of itemids with a unique range
    single_itemids = set(summary_df.loc[summary_df["n_ranges"] == 1, "itemid"])
    single_range_rows = []

    # iterate over all itemids
    for item in itemids:

        if item in multi_itemids:
            with_value = hadms_with_value[item]  # admission ids with this value

            # invert the dict: range -> set of hadm_ids that had  this range
            # Reason: we need to know how many admissions have this range
            range_hadm: dict[tuple, set] = {}
            for hadm_id in with_value:
                for r in hadm_item_ranges.get((hadm_id, item), set()):
                    range_hadm.setdefault(r, set()).add(hadm_id)

            # iterate through the new dict and het values for the output table
            for (lower, upper), hadm_set in range_hadm.items():
                units = valueuom_ranges.get((item, lower, upper), set())
                multi_range_rows.append({
                    "itemid": item,
                    "label": item_labels.get(item),
                    "n_hadm_id": len(hadm_set),
                    "valueuom": " | ".join(sorted(units)) if units else None,
                    "ref_range_lower": lower,
                    "ref_range_upper": upper,
                })

        if item in single_itemids:
            # get hadm_ids that have a value for this itemid
            with_value = hadms_with_value[item]
            # invert dict: range -> set of hadm_ids
            range_hadm: dict[tuple, set] = {}
            for hadm_id in with_value:
                for r in hadm_item_ranges.get((hadm_id, item), set()):
                    range_hadm.setdefault(r, set()).add(hadm_id)

            # extract single entry from range_hadm dict
            (lower, upper), hadm_set = next(iter(range_hadm.items()))
            units = valueuom_ranges.get((item, lower, upper), set())
            single_range_rows.append({
                "itemid": item,
                "label": item_labels.get(item),
                "n_hadm_id": len(hadm_set),
                "valueuom": " | ".join(sorted(units)) if units else None,
                "ref_range_lower": lower,
                "ref_range_upper": upper,
            })

    # ---- Per-(hadm_id, itemid) multi-range table ---------------------------------
    hadm_multi_rows = []

    for (hadm_id, item), ranges in hadm_item_ranges.items():
        # skip the non-multirange tuples
        if len(ranges) <= 1:
            continue
        # iterate over ranges in consisten numerical order. Sorted in priority order:
        # 1. x[0] is None: Puts entries where lower is None last
        # 2. x[0]: sorts by the lower bound numerically
        # 3. x[1] is None: puts entries where upper is None last
        # 4. x[1]: Sorts by the upper bound numerically
        # None handling necessary, because float < None when sorting cannot be evaluated
        for (lower, upper) in sorted(ranges, key=lambda x: (x[0] is None, x[0], x[1] is None, x[1])):
            units = valueuom_ranges.get((item, lower, upper), set())
            hadm_multi_rows.append({
                "hadm_id": hadm_id,
                "itemid": item,
                "label": item_labels.get(item),
                "ref_range_lower": lower,
                "ref_range_upper": upper,
                "valueuom": " | ".join(sorted(units)) if units else None,
            })



    # convert lists of dicts to dataframes and write to csv

    multi_df = (
        pd.DataFrame(multi_range_rows, columns=["itemid", "label", "n_hadm_id", "valueuom", "ref_range_lower", "ref_range_upper"])
        .sort_values(["itemid", "ref_range_lower", "ref_range_upper"])
        .reset_index(drop=True)
    )
    multi_df.to_csv(out("multi_range"), index=False)
    print(f"Saved cohort_{args.cohort}_multi_range.csv  ({len(multi_df)} rows)")

    single_df = (
        pd.DataFrame(single_range_rows, columns=["itemid", "label", "n_hadm_id", "valueuom", "ref_range_lower", "ref_range_upper"])
        .sort_values("itemid")
        .reset_index(drop=True)
    )
    single_df.to_csv(out("single_range"), index=False)
    print(f"Saved cohort_{args.cohort}_single_range.csv  ({len(single_df)} rows)")

    no_range_df = (
        summary_df.loc[summary_df["n_ranges"] == 0, ["itemid", "label", "n_hadm_ids"]]
        .sort_values("itemid")
        .reset_index(drop=True)
    )
    no_range_df.to_csv(out("no_range"), index=False)
    print(f"Saved cohort_{args.cohort}_no_range.csv  ({len(no_range_df)} rows)")

    hadm_multi_df = (
        pd.DataFrame(hadm_multi_rows)
        .sort_values(["hadm_id", "itemid", "ref_range_lower", "ref_range_upper"])
        .reset_index(drop=True)
    )
    hadm_multi_df.to_csv(out("hadm_multi_range"), index=False)
    print(f"Saved cohort_{args.cohort}_hadm_multi_range.csv  ({len(hadm_multi_df)} rows)")

# discretiser helper
def _discretiser(value, lower, upper):
    if lower is not None and value < lower:
        return -1
    if upper is not None and value > upper:
        return 1
    return 0

def discretise_lab_values(args, per_hadm_item_ranges):
    # check if destination folder exists. If not, create one
    path = Path(OUTPUT_DIR/"range_mapping")
    path.mkdir(parents=True, exist_ok=True)

    # ---- Load aggregated file -------------------------
    agg_path = COHORTS[args.cohort]["agg"]
    agg = pd.read_csv(agg_path)
    lab_cols = [c for c in agg.columns if c != "hadm_id"]

    # unpivot agg table -> long format (columns: hadm_id, itemid, value)
    agg_long = agg.melt(id_vars="hadm_id", value_vars=lab_cols, var_name="itemid", value_name="value")
    agg_long["itemid"] = agg_long["itemid"].astype(int)
    agg_long = agg_long.dropna(subset=["value"])

    # iterate over agg_long to discretise
    rows_merged = []
    rows_unmerged = []

    modes = ["yes", "no"] if "both" in args.range_merge else args.range_merge

    for row in agg_long.itertuples(index=False):
        hadm_id = row[0]
        itemid = row[1]
        value = row[2]

        # get ranges for this hadm, itemid pair from dict
        ranges = {r for r in per_hadm_item_ranges.get((hadm_id, itemid), set())}
        n_ranges = len(ranges)

        for mode in modes:
            merged = False
            if n_ranges == 0:
                disc_val = None
                lower, upper = None, None
            elif n_ranges == 1:
                lower, upper = next(iter(ranges)) # gets next (and only) tuple from set ranges
                disc_val = _discretiser(value, lower, upper)
            else:
                if mode == "yes":
                    # get lowest and highest threshold
                    lowers = [r[0] for r in ranges if r[0] is not None]
                    uppers = [r[1] for r in ranges if r[1] is not None]
                    lower = min(lowers) if lowers else None
                    upper = max(uppers) if uppers else None
                    disc_val = _discretiser(value, lower, upper)
                    merged = True
                else:
                    disc_val = -99
                    lower, upper = None, None

            if mode == "yes":
                rows_merged.append({
                    "hadm_id": hadm_id,
                    "itemid": itemid,
                    "value": value,
                    "discretised_value": disc_val,
                    "ref_range_lower": lower,
                    "ref_range_upper": upper,
                    "merged": merged,
                })
            else:
                rows_unmerged.append({
                    "hadm_id": hadm_id,
                    "itemid": itemid,
                    "value": value,
                    "discretised_value": disc_val,
                    "ref_range_lower": lower,
                    "ref_range_upper": upper,
                    "merged": merged,
                })

    modes = ["yes", "no"] if "both" in args.range_merge else args.range_merge
    for mode in modes:
        if "yes" in mode:
            detailed_df = pd.DataFrame(rows_merged)
            suffix = "merged"
        else:
            detailed_df = pd.DataFrame(rows_unmerged)
            suffix = "unmerged"

        # reconstruct wide table, same style as aggregated table
        disc_wide = (
            detailed_df.pivot(index="hadm_id", columns="itemid", values="discretised_value")
            .rename(columns=str)
            .reset_index()
            .reindex(columns=agg.columns)
        )
        disc_wide = agg[["hadm_id"]].merge(disc_wide, on="hadm_id", how="left")

        disc_wide.to_csv(path / f"{args.cohort}_discretised_{suffix}.csv", index=False)
        detailed_df.to_csv(path / f"{args.cohort}_detail_{suffix}.csv", index=False)
        print(f"Saved {args.cohort}_discretised_{suffix}.csv  ({len(disc_wide)} rows)")
        print(f"Saved {args.cohort}_detail_{suffix}.csv       ({len(detailed_df)} rows)")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=list(COHORTS), required=True)
    parser.add_argument("--task", nargs='+', choices=['analysis', 'discretise'], required=True)
    parser.add_argument("--range_merge", nargs='+', choices=['yes', 'no', 'both'])
    args = parser.parse_args()

    if "discretise" in args.task and not args.range_merge:
        parser.error("--range_merge is required when --task includes discretise")

    #AGG_PATH = COHORTS[args.cohort]["agg"]
    #COHORT_PATH = COHORTS[args.cohort]["cohort"]

    # load cohort files
    hadm_ids_with_value, hadm_id_set, itemid_set, window_start_map, dischtime_map = load_cohort_files(args)
    # extract ranges
    per_hadm_item_ranges, range_valueuom = extract_range(hadm_id_set, itemid_set, window_start_map, dischtime_map)

    if "analysis" in args.task:
        overview_tables(itemid_set, hadm_id_set, hadm_ids_with_value, per_hadm_item_ranges, range_valueuom, args)

    if "discretise" in args.task:
        discretise_lab_values(args, per_hadm_item_ranges)