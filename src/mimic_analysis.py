from dataclasses import dataclass, field

from config import *
from data_io import *
from plots import plot_blacklist_composition, plot_count_panels

# display names for the blacklist reasons, in the row order of the composition plot.
# Ordered by how many measurements the reason removes, so the panel that carries the
# surprise (never_measured is the largest reason and costs nothing) reads last.
# The reasons are mutually exclusive, earlier ones winning, so every row after the first
# is a marginal count. Saying so in the label matters: 562 lab items have no reference
# range anywhere, but 367 of them are already gone for having no numeric values, so the
# no_range row reads 195. The figure's note reconciles the two numbers.
BLACKLIST_REASON_LABELS = {
    "too_few_values": "fewer than 10 distinct\nnumeric values",
    "no_range": "no reference range anywhere\n(not already counted above)",
    "low_range_coverage": "a range, but on too few\nof the measured admissions",
    "never_measured": "never seen\nin labevents",
}


# collects all the statistics for a labitem during the labevents scan
@dataclass
class ItemStats:
    ranges: dict = field(default_factory=dict)  # (lower, upper) -> set[hadm_id]
    range_units: dict = field(default_factory=dict)  # (lower, upper) -> set[str]
    hadm_no_range: set = field(default_factory=set)  # hadm_ids with a value but no range
    hadm_with_value: set = field(default_factory=set)  # hadm_ids that had at least one non-NaN value
    numeric_count: int = 0
    numeric_values: set = field(default_factory=set) # unique float values
    string_count: int = 0
    string_values: set = field(default_factory=set)  # unique normalised non-numeric string values


def _range_coverage_pct(s: ItemStats) -> float | None:
    """
    Share of the admissions that have a value for this lab which also have a reference
    range, as a percentage. None when the lab was never measured, where the question does
    not arise. Intersected with hadm_with_value rather than counted straight off the range
    dict, because a labevents row can carry a range and a NaN value, which would otherwise
    let coverage exceed 100%.
    """
    if not s.hadm_with_value:
        return None
    ranged = set().union(*s.ranges.values()) if s.ranges else set()
    return len(ranged & s.hadm_with_value) / len(s.hadm_with_value) * 100


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


def run_blacklist_composition_plot(analysis_dir: Path) -> None:
    """
    Plots what the blacklist removes, per reason: lab items on the left, the measurements
    that go with them on the right. Reads the csv files rather than the in-memory stats, so
    it can be re-run on an existing output folder without rescanning labevents.
    """
    blacklist_df = pd.read_csv(analysis_dir / "mimic_blacklist_analysis.csv")
    value_df = pd.read_csv(analysis_dir / "mimic_value_analysis.csv").set_index("itemid")
    counts = value_df[["n_numeric_values", "n_string_values"]]
    total_measurements = int(counts.to_numpy().sum())

    # how many items have no reference range in total, versus how many this figure charges
    # to no_range. They differ because the reasons are mutually exclusive: an item with no
    # range AND no numeric values is counted once, under the earlier reason.
    range_df = pd.read_csv(analysis_dir / "mimic_range_analysis.csv")
    n_ranges = range_df.groupby("itemid")["n_ranges"].first()
    n_no_range_total = int((n_ranges == 0).sum())
    n_no_range_charged = int((blacklist_df["reason"] == "no_range").sum())
    n_no_range_overlap = n_no_range_total - n_no_range_charged

    rows = []
    for reason, label in BLACKLIST_REASON_LABELS.items():
        itemids = blacklist_df.loc[blacklist_df["reason"] == reason, "itemid"]
        # never_measured itemids have no row in the value analysis at all, so reindex and
        # fill: their measurement count is a true zero, not a missing number
        removed = counts.reindex(itemids).fillna(0).sum()
        rows.append((label, len(itemids),
                     int(removed["n_numeric_values"]), int(removed["n_string_values"])))

    n_blacklisted = len(blacklist_df)
    n_all_items = len(pd.read_csv(D_LABITEMS_PATH, usecols=["itemid"]))
    removed_measurements = sum(r[2] + r[3] for r in rows)

    plot_blacklist_composition(
        rows,
        title="Labs removed",
        subtitle=f"{n_blacklisted} of {n_all_items} lab items in d_labitems are excluded, leaving "
                 f"{n_all_items - n_blacklisted} as features. The excluded items carry "
                 f"{removed_measurements:,} recorded lab values "
                 f"({removed_measurements / total_measurements * 100:.1f}% of {total_measurements:,}); "
                 f"the {n_all_items - n_blacklisted} retained features carry "
                 f"{total_measurements - removed_measurements:,}.",
        path=analysis_dir / "mimic_blacklist_composition.png",
        total_measurements=total_measurements,
        note=f"A measurement is one labevents row with both admission id and a non-NaN value. Rows with no "
             f"admission id are dropped before counting anything. The total here is therefore not the row count "
             f"of labevents. The reasons for exclusion are mutually exclusive and applied in the order shown. "
             f"{n_no_range_total} lab items have no reference range anywhere but {n_no_range_overlap} of them are "
             f"already excluded because of fewer than 10 distinct values.")


def _classify_range_stratification(strat_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per multi-range lab item: whether its several reference ranges are explained by
    the sex or the age of the admissions behind them, with the numbers the verdict rests on.
    The figure and the summary csv both come from here, so they cannot disagree.

    Every range group is compared, whatever its size. That is deliberate for sex, where a
    ten-admission group that is 100% female is real evidence, but it leaves the age verdict
    at the mercy of tiny groups: `smallest_group_n_hadm` is carried in the output so an age
    flag can be read against the group that produced it.
    """
    labels = strat_df.drop_duplicates("itemid").set_index("itemid")["label"]
    by_item = strat_df.groupby("itemid")
    stats = pd.DataFrame({
        "n_groups": by_item.size(),
        "n_hadm": by_item["n_hadm_with_range"].sum(),
        "smallest_group_n_hadm": by_item["n_hadm_with_range"].min(),
        "min_pct_female": by_item["pct_female"].min(),
        "max_pct_female": by_item["pct_female"].max(),
        "min_mean_age": by_item["mean_age"].min(),
        "max_mean_age": by_item["mean_age"].max(),
    })

    stats["age_spread_years"] = (stats["max_mean_age"] - stats["min_mean_age"]).round(1)
    # stratified by sex means the item has one essentially all-female range group and one
    # essentially all-male one, not merely two groups some distance apart
    by_sex = ((stats["max_pct_female"] >= RANGE_FEMALE_ONLY_PCT)
              & (stats["min_pct_female"] <= RANGE_MALE_ONLY_PCT))
    by_age = stats["age_spread_years"] >= RANGE_AGE_SPREAD_YEARS

    classification = pd.Series("neither", index=stats.index)
    classification[by_sex & ~by_age] = "sex only"
    classification[~by_sex & by_age] = "age only"
    classification[by_sex & by_age] = "both"

    out = stats.assign(label=labels.reindex(stats.index),
                       stratified_by_sex=by_sex,
                       stratified_by_age=by_age,
                       classification=classification)
    out.index.name = "itemid"
    columns = ["label", "classification", "stratified_by_sex", "stratified_by_age",
               "n_groups", "n_hadm", "smallest_group_n_hadm",
               "min_pct_female", "max_pct_female", "min_mean_age", "max_mean_age",
               "age_spread_years"]
    # the explained items first, then by how much data stands behind the verdict
    order = {"sex only": 0, "both": 1, "age only": 2, "neither": 3}
    return (out[columns]
            .assign(_order=classification.map(order))
            .sort_values(["_order", "n_hadm"], ascending=[True, False])
            .drop(columns="_order")
            .reset_index())


def run_range_structure_plot(analysis_dir: Path) -> None:
    """
    Plots how many reference ranges a lab item carries, and for the items carrying several,
    whether the split is explained by the sex or the age of the admissions behind each
    range. Reads the csv files rather than the in-memory stats, so it can be re-run on an
    existing output folder without rescanning labevents.

    Restricted to the lab items that survive the blacklist, i.e. the ones that actually
    become features. Reference range structure only matters where a range is going to be
    used to discretise something.
    """
    range_df = pd.read_csv(analysis_dir / "mimic_range_analysis.csv")
    strat_df = pd.read_csv(analysis_dir / "mimic_range_stratification.csv")
    blacklist = set(pd.read_csv(analysis_dir / "mimic_blacklist_analysis.csv")["itemid"])

    # ----- panel A: distinct ranges per feature -------------------------------------
    n_ranges = range_df.groupby("itemid")["n_ranges"].first()
    n_ranges = n_ranges[~n_ranges.index.isin(blacklist)]
    # no "none" row: an item with no range anywhere is blacklisted by construction, so
    # every surviving feature carries at least one
    counts = {label: int((n_ranges == n).sum()) for label, n in
              (("1 (single range)", 1), ("2", 2), ("3", 3))}
    counts["4 or more"] = int((n_ranges >= 4).sum())
    # the multi-range items are the ones panel B drills into, so they carry the accent
    ranges_rows = [(label, value, label != "1 (single range)")
                   for label, value in counts.items()]
    n_features = len(n_ranges)

    # ----- panel B: what separates the ranges of a multi-range feature ---------------
    strat_df = strat_df[~strat_df["itemid"].isin(blacklist)]
    verdicts = _classify_range_stratification(strat_df)
    verdicts.to_csv(analysis_dir / "mimic_range_stratification_summary.csv", index=False)
    print(f"Saved mimic_range_stratification_summary.csv ({len(verdicts)} multi-range itemids).")

    by_sex, by_age = verdicts["stratified_by_sex"], verdicts["stratified_by_age"]
    strat_rows = [
        ("sex only", int((by_sex & ~by_age).sum()), True),
        ("age only", int((~by_sex & by_age).sum()), True),
        ("both", int((by_sex & by_age).sum()), True),
        ("neither", int((~by_sex & ~by_age).sum()), False),
    ]

    plot_count_panels(
        panels=[("Distinct reference ranges per lab item", "lab items", ranges_rows),
                ("Stratification of multi-range lab items", "lab items", strat_rows)],
        title="Reference range structure",
        subtitle=f"Over the {n_features} lab items that are used as features.",
        path=analysis_dir / "mimic_range_structure.png",
        emphasis_label="carries several ranges / the split is explained",
        quiet_label="a single range / unexplained",
        note=f"Restricted to the {n_features} lab items that are not excluded. "
             f"'Explained by sex' means the item has one range "
             f"group that is at least {RANGE_FEMALE_ONLY_PCT}% female and another that is at most "
             f"{RANGE_MALE_ONLY_PCT}%; 'by age', that mean age differs by at least "
             f"{RANGE_AGE_SPREAD_YEARS} years across the groups. ")


def run_mimic_analysis():
    analysis_dir = create_analysis_dir()

    # itemid -> ItemStats
    stats: dict[int, ItemStats] = {}
    all_hadm_ids: set = set()  # all hadm_ids seen in labevents

    # what the two filters below cost, counted before either is applied. Every other count
    # in this module is over the rows that survive both, so without this there is no way to
    # say what the denominator actually is -- and it is only about half the file.
    coverage = dict.fromkeys(
        ["rows_total", "rows_with_hadm_id", "rows_with_hadm_id_and_value",
         "rows_without_hadm_id", "rows_without_hadm_id_but_with_value",
         "rows_with_hadm_id_but_no_value"], 0)

    chunk_counter = 0
    print("Scanning labevents...")

    # ---------------------- Scan labevents -------------------------------------------
    # scan labevents chunk-wise
    for chunk in pd.read_csv(
            LABEVENTS_PATH,
            usecols=["hadm_id", "itemid", "ref_range_lower", "ref_range_upper", "value", "valueuom"],
            chunksize=CHUNK_SIZE,
    ):
        has_hadm_id, has_value = chunk["hadm_id"].notna(), chunk["value"].notna()
        coverage["rows_total"] += len(chunk)
        coverage["rows_with_hadm_id"] += int(has_hadm_id.sum())
        coverage["rows_with_hadm_id_and_value"] += int((has_hadm_id & has_value).sum())
        coverage["rows_without_hadm_id"] += int((~has_hadm_id).sum())
        coverage["rows_without_hadm_id_but_with_value"] += int((~has_hadm_id & has_value).sum())
        coverage["rows_with_hadm_id_but_no_value"] += int((has_hadm_id & ~has_value).sum())

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
                    v = float(row.value)
                    s.numeric_values.add(v)
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

    # ---------------- Labevents coverage -----------------------------------------------
    # written before the per-itemid tables, so a run that dies later still leaves the
    # denominator on disk. One row per line of the filter, long rather than wide, since
    # this is a report to read rather than a table to join against.
    total = coverage["rows_total"]
    coverage_rows = [
        ("rows_total", "every row in labevents", total),
        ("rows_with_hadm_id", "tied to a hospital admission",
         coverage["rows_with_hadm_id"]),
        ("rows_with_hadm_id_and_value", "tied to an admission AND carrying a value — "
         "the denominator of every other count in this folder",
         coverage["rows_with_hadm_id_and_value"]),
        ("rows_without_hadm_id", "dropped: no admission to attach to",
         coverage["rows_without_hadm_id"]),
        ("rows_without_hadm_id_but_with_value", "of those dropped rows, how many held a "
         "usable value", coverage["rows_without_hadm_id_but_with_value"]),
        ("rows_with_hadm_id_but_no_value", "dropped: admission known but value empty",
         coverage["rows_with_hadm_id_but_no_value"]),
        ("distinct_hadm_ids", "admissions appearing anywhere in labevents",
         len(all_hadm_ids)),
    ]
    coverage_df = pd.DataFrame(coverage_rows, columns=["metric", "description", "n_rows"])
    # the share is meaningless for the admission count, hence None rather than 0
    coverage_df["pct_of_labevents"] = [
        round(n / total * 100, 2) if key != "distinct_hadm_ids" else None
        for key, _, n in coverage_rows]
    coverage_df.to_csv(analysis_dir / "mimic_labevents_coverage.csv", index=False)
    print(f"Saved mimic_labevents_coverage.csv: {coverage['rows_with_hadm_id_and_value']:,} of "
          f"{total:,} labevents rows carry both a hadm_id and a value and are counted "
          f"everywhere else; {coverage['rows_without_hadm_id']:,} have no hadm_id "
          f"({coverage['rows_without_hadm_id_but_with_value']:,} of them with a value) and "
          f"{coverage['rows_with_hadm_id_but_no_value']:,} have an admission but no value.")

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

    # ------------- Blacklist ------------------------------------------------
    # Everything the pipeline cannot turn into a discretised feature is blacklisted here,
    # so the feature matrix holds labs that can actually carry a signal. Without this the
    # matrix is d_labitems minus the blacklist, which leaves hundreds of columns that are
    # NaN in every admission of every cohort but still take up feature importance ranks.
    #   too_few_values:      categorical or near-constant, nothing to threshold
    #   no_range:            measured, but no reference range anywhere in labevents, so it
    #                        can never be discretised
    #   low_range_coverage:  a range exists but for no more than MIN_RANGE_COVERAGE_PCT of
    #                        the admissions that have a value, so the discretised column is
    #                        empty for nearly all of them while the continuous one is dense
    #   never_measured:      not in labevents at all, an all-NaN column by construction
    # The reasons are mutually exclusive, earlier ones winning. no_range is the degenerate
    # case of low_range_coverage (0%) and is kept separate because "no range at all" and
    # "a range too sparse to use" are different situations worth counting apart.
    coverage = {itemid: _range_coverage_pct(s) for itemid, s in stats.items()}
    too_few_values = {itemid for itemid, s in stats.items() if len(s.numeric_values) < 10}
    no_range = {itemid for itemid, s in stats.items() if not s.ranges} - too_few_values
    low_range_coverage = {itemid for itemid, pct in coverage.items()
                          if pct is not None and pct <= MIN_RANGE_COVERAGE_PCT
                          } - too_few_values - no_range
    never_measured = set(item_labels) - set(stats)
    blacklist = too_few_values | no_range | low_range_coverage | never_measured

    with open(analysis_dir / "itemid_blacklist.txt", "w") as f:
        f.write("\n".join(str(iid) for iid in sorted(blacklist)))
    print(f"Saved itemid_blacklist.txt ({len(blacklist)} blacklisted itemids: "
          f"{len(too_few_values)} with < 10 unique numeric values, {len(no_range)} with no "
          f"reference range anywhere, {len(low_range_coverage)} with a range on no more than "
          f"{MIN_RANGE_COVERAGE_PCT}% of measured admissions, "
          f"{len(never_measured)} never seen in labevents). "
          f"{len(item_labels) - len(blacklist)} itemids remain as features.")

    # readable analysis of the blacklisted itemids, carrying why each one was dropped
    reasons = {**{i: "never_measured" for i in never_measured},
               **{i: "low_range_coverage" for i in low_range_coverage},
               **{i: "no_range" for i in no_range},
               **{i: "too_few_values" for i in too_few_values}}
    blacklist_rows = [{
        "itemid": itemid,
        "label": item_labels.get(itemid),
        "reason": reasons[itemid],
        # None rather than 0 for the never measured ones: they have no stats entry at all,
        # which is not the same as having been seen with zero numeric values
        "n_numeric_values": stats[itemid].numeric_count if itemid in stats else None,
        "n_unique_numeric_values": len(stats[itemid].numeric_values) if itemid in stats else None,
        "n_ranges": len(stats[itemid].ranges) if itemid in stats else None,
        # the number the low_range_coverage decision rests on, carried for every row so a
        # lab that only just cleared the threshold can be found too
        "range_coverage_pct": (round(coverage[itemid], 2)
                               if coverage.get(itemid) is not None else None),
    } for itemid in sorted(blacklist)]
    blacklist_df = pd.DataFrame(blacklist_rows)
    blacklist_df.to_csv(analysis_dir / "mimic_blacklist_analysis.csv", index=False)
    print("Saved mimic_blacklist_analysis.csv")

    run_blacklist_composition_plot(analysis_dir)
    run_range_structure_plot(analysis_dir)


if __name__ == "__main__":
    run_mimic_analysis()
