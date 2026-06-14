from dataclasses import dataclass, field

from utils.dataloader import *


# collects all the statistics for a labitem during the labevents scan
@dataclass
class ItemStats:
    ranges: dict = field(default_factory=dict)  # (lower, upper) -> set[hadm_id]
    range_units: dict = field(default_factory=dict)  # (lower, upper) -> set[str]
    hadm_no_range: set = field(default_factory=set)  # hadm_ids with a value but no range
    hadm_with_value: set = field(default_factory=set)  # hadm_ids that had at least one non-NaN value
    numeric_count: int = 0
    string_count: int = 0
    string_values: set = field(default_factory=set)  # unique normalised non-numeric string values


def run_range_stratification_analysis(stats: dict, item_labels: dict, analysis_dir: Path) -> None:
    """
    For each itemid with more than one distinct reference range, compute age and gender
    statistics per range group to check if ranges are stratified by age or gender.
    Saves results to mimic_range_stratification.csv.
    """
    # build hadm_id -> (anchor_age, gender) from admissions + patients
    admissions = pd.read_csv(ADMISSIONS_PATH, usecols=["hadm_id", "subject_id"])
    patients = pd.read_csv(PATIENTS_PATH, usecols=["subject_id", "gender", "anchor_age"])
    hadm_info = admissions.merge(patients, on="subject_id").set_index("hadm_id")
    hadm_age = hadm_info["anchor_age"].to_dict()  # hadm_id -> age
    hadm_gender = hadm_info["gender"].to_dict()  # hadm_id -> "M" or "F"

    rows = []
    for itemid, s in sorted(stats.items()):
        if len(s.ranges) <= 1:
            continue  # only analyse itemids with multiple ranges

        for (lower, upper), hadm_ids in sorted(
                s.ranges.items(),
                key=lambda x: (x[0][0] is None, x[0][0], x[0][1] is None, x[0][1])
        ):
            ages = [hadm_age[h] for h in hadm_ids if h in hadm_age]
            genders = [hadm_gender[h] for h in hadm_ids if h in hadm_gender]
            n = len(ages)
            mean_age = round(sum(ages) / n, 1) if n > 0 else None
            std_age = round((sum((a - mean_age) ** 2 for a in ages) / n) ** 0.5, 1) if n > 1 else None
            pct_female = round(genders.count("F") / len(genders) * 100, 1) if genders else None

            rows.append({
                "itemid": itemid,
                "label": item_labels.get(itemid),
                "units": " | ".join(sorted(s.range_units.get((lower, upper), set()))) or None,
                "ref_range_lower": lower,
                "ref_range_upper": upper,
                "n_hadm_with_range": len(hadm_ids),
                "mean_age": mean_age,
                "std_age": std_age,
                "pct_female": pct_female,
            })

    df = pd.DataFrame(rows)
    df.to_csv(analysis_dir / "mimic_range_stratification.csv", index=False)
    print(f"Saved mimic_range_stratification.csv ({df['itemid'].nunique()} itemids with multiple ranges)")


def run_mimic_analysis():
    analysis_dir = create_analysis_dir()

    # itemid -> ItemStats
    stats: dict[int, ItemStats] = {}
    all_hadm_ids: set = set()  # all hadm_ids seen in labevents

    chunk_counter = 0
    print("Scanning labevents...")

    # ---------------------- Scan labevents -------------------------------------------
    # scan labevents chunk-wise
    for chunk in pd.read_csv(
            LABEVENTS_PATH,
            usecols=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper", "value", "valueuom"],
            chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.dropna(
            subset=["hadm_id"])  # drop rows without a hospital admission. These will never appear in a cohort.
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        all_hadm_ids.update(chunk["hadm_id"])

        # iterate over the rows of labevents
        for row in chunk.itertuples(index=False):
            hadm_id = row.hadm_id
            itemid = row.itemid
            lower = None if pd.isna(row.ref_range_lower) else float(row.ref_range_lower)
            upper = None if pd.isna(row.ref_range_upper) else float(row.ref_range_upper)

            s = stats.setdefault(itemid, ItemStats())  # s is the ItemStats instance for this itemid

            # classify value as numeric or non-numeric string (NaN rows are ignored)
            if not pd.isna(row.value):
                s.hadm_with_value.add(hadm_id)
                try:
                    float(row.value)
                    s.numeric_count += 1
                except (ValueError, TypeError):
                    s.string_count += 1
                    s.string_values.add(row.value.strip().upper())

            # accumulate range info
            has_range = lower is not None or upper is not None
            if has_range:
                range_key = (lower, upper)
                s.ranges.setdefault(range_key, set()).add(hadm_id)  # add hadm_id to the range dict
                if not pd.isna(row.valueuom):
                    s.range_units.setdefault(range_key, set()).add(str(row.valueuom))  # add unit to the range
            elif not pd.isna(row.value):
                s.hadm_no_range.add(hadm_id)  # add hadm_id to no_range list
                # Note: hadm_no_range may overlap with hadm_ids in ranges (same admission
                # had some rows with range and some without). This is corrected when building
                # n_hadm_no_range below by subtracting those that also appear in ranges.

        chunk_counter += 1
        print(f"Chunk {chunk_counter} processed.")

    print("Scan complete.")

    # ---------------- Process item stats objects into output rows -----------------------------------

    # retrieve item labels from d_labitems: itemid -> label
    item_labels = (
        pd.read_csv(D_LABITEMS_PATH, usecols=["itemid", "label"])
        .set_index("itemid")["label"].to_dict()
    )
    # build rows for range analysis output
    range_rows = []
    # build rows for value analysis output
    value_rows = []

    # iterate over all itemids to create output rows
    for itemid in sorted(stats.keys()):
        s = stats[itemid]
        n_ranges = len(s.ranges)
        total = s.numeric_count + s.string_count  # total number of labevent entries for this item where value is not NaN
        pct_numeric = round(s.numeric_count / total * 100,
                            1) if total > 0 else None  # percentage of non-missing values that are numeric

        # hadm_ids that never had a range for this item
        hadm_with_any_range = set().union(
            *s.ranges.values()) if s.ranges else set()  # union of all hadm_id sets across all ranges for this item -> every hadm_id that have any range for this item.
        n_hadm_no_range = len(
            s.hadm_no_range - hadm_with_any_range)  # hadm_ids that have a value but no range (subtracted those that also appear with any range)
        n_hadm_no_value = len(
            all_hadm_ids - s.hadm_with_value)  # hadm_ids with no non-NaN value (NaN rows + never measured)

        value_rows.append({
            "itemid": itemid,
            "label": item_labels.get(itemid),
            "n_numeric_values": s.numeric_count,
            "n_string_values": s.string_count,
            "pct_numeric": pct_numeric,
            "n_hadm_no_value": n_hadm_no_value,
            "string_values": " | ".join(sorted(s.string_values)) or None,
        })

        if s.ranges:
            # iterate over the ranges sorted in ascending order (None last)
            for (lower, upper) in sorted(
                    s.ranges.keys(),
                    key=lambda x: (x[0] is None, x[0], x[1] is None, x[1])
            ):
                range_rows.append({
                    "itemid": itemid,
                    "label": item_labels.get(itemid),
                    "n_ranges": n_ranges,
                    "ref_range_lower": lower,
                    "ref_range_upper": upper,
                    "units": " | ".join(sorted(s.range_units.get((lower, upper), set()))) or None,
                    "n_hadm_with_range": len(s.ranges[(lower, upper)]),
                    "n_hadm_no_range": n_hadm_no_range,
                    "n_hadm_no_value": n_hadm_no_value,
                })
        else:
            range_rows.append({
                "itemid": itemid,
                "label": item_labels.get(itemid),
                "n_ranges": 0,
                "ref_range_lower": None,
                "ref_range_upper": None,
                "n_hadm_with_range": None,
                "units": None,
                "n_hadm_no_range": n_hadm_no_range,
                "n_hadm_no_value": n_hadm_no_value,
            })

    # ------------- Save the results -----------------------------------------

    range_df = pd.DataFrame(range_rows)
    range_df.to_csv(analysis_dir / "mimic_range_analysis.csv", index=False)
    print("Saved mimic_range_analysis.csv")

    value_df = pd.DataFrame(value_rows)
    value_df.to_csv(analysis_dir / "mimic_value_analysis.csv", index=False)
    print("Saved mimic_value_analysis.csv")

    run_range_stratification_analysis(stats, item_labels, analysis_dir)

    # create blacklist with itemids that only have non-numeric values
    blacklist = {itemid for itemid, s in stats.items() if s.numeric_count == 0}
    with open(analysis_dir / "itemid_blacklist.txt", "w") as f:
        f.write("\n".join(str(iid) for iid in sorted(blacklist)))
    print(f"Saved itemid_blacklist.txt ({len(blacklist)} blacklisted itemids)")


if __name__ == "__main__":
    run_mimic_analysis()
