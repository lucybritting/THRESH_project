"""
Scans labevents file in chunks
"""

import pandas as pd

from config.paths import LABEVENTS_PATH, CHUNK_SIZE
from .store import load_all_itemids, load_top100_itemids


# --------------- SCAN LABEVENTS -------------------------
def scan_labevents(hadm_id_set: set[int], top100labs: bool) -> tuple[dict, dict, set]:
    # load either top 100 itemids or all itemids
    itemid_set = load_top100_itemids() if top100labs else load_all_itemids()

    # collect different ranges for pairs (hadm_id, itemid): {(hadm_id, itemid) -> set of (ref_range_lower, ref_range_upper) tuples}
    per_hadm_item_ranges: dict[tuple, set] = {}
    # collect values for (hadm_id, itemid) pair: (hadm_id, itemid) -> list of float values
    per_hadm_item_values: dict[tuple, list] = {}

    chunk_counter = 0
    print("Scanning labevents (this may take a moment, because file is large)...")

    # chunk reading the labevents file
    for chunk in pd.read_csv(
            LABEVENTS_PATH,
            usecols=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper", "value"],
            chunksize=CHUNK_SIZE,
    ):
        # filter each chunk for relevant rows: hadm_id belongs to cohort AND itemid is from the
        sub = chunk[
            chunk["hadm_id"].isin(hadm_id_set) &
            chunk["itemid"].isin(itemid_set)
            ]

        # skip the sub chunk if empty
        if sub.empty:
            continue

        sub = sub.copy()
        # convert datatypes to int
        sub["hadm_id"] = sub["hadm_id"].astype(int)
        sub["itemid"] = sub["itemid"].astype(int)

        # iterate over rows to fill the accumulator dicts
        for row in sub.itertuples(index=False):  # yields each row as tuple
            # get admission and item id
            hadm_id = row.hadm_id
            itemid = row.itemid
            # convert NaN to None (NaN is float, NaN != NaN). Would give single item in the set for each NaN.
            lower = None if pd.isna(row.ref_range_lower) else float(row.ref_range_lower)
            upper = None if pd.isna(row.ref_range_upper) else float(row.ref_range_upper)
            range_key = (lower, upper)

            # creates empty set for (hadm_id, itemid), if it hasnt been recorded yet, and adds the range key (lower, upper)
            per_hadm_item_ranges.setdefault((hadm_id, itemid), set()).add(range_key)

            # NaN values are treated as no measurement and skipped
            if not pd.isna(row.value):
                try:
                    value = float(row.value)
                    per_hadm_item_values.setdefault((hadm_id, itemid), []).append(value)
                except (ValueError, TypeError):
                    pass  # non-numeric string values are ignored (blacklist excludes purely non-numeric itemids)

        chunk_counter += 1
        print(f"Chunk {chunk_counter} processed...")

    print("Scan complete.")
    return per_hadm_item_ranges, per_hadm_item_values, itemid_set


