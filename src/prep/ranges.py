"""The reference range merge strategies, and the step that applies them."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from config.params import BEST_MERGE_STRATEGY
from data_io.paths import merged_ranges_path, merge_warnings_path
from data_io.store import load_ranges, save_merged_ranges


@dataclass
class MergeWarningLog:
    """
    Class to accumulate the merge warnings.
    Merge warnings occur, if upper threshold smaller or equal to lower threshold.
    Saves the merge warnings to a csv file.
    """
    records: list[dict] = field(default_factory=list)

    def add(self, hadm_id, itemid, lower, upper):
        self.records.append({
            "hadm_id": hadm_id,
            "itemid": itemid,
            "lower": lower,
            "upper": upper
        })

    def save(self, path):
        if self.records:
            pd.DataFrame(self.records).to_csv(path, index=False)
            print(f"{len(self.records)} merge warnings.")
        else:
            print("0 warnings")


class RangeMergeStrategy(ABC):
    """
    Base class for all merge strategies.
    """

    def __init__(self, warning_log: MergeWarningLog | None = None):
        self.warning_log = warning_log

    @property  # makes it accessible as an attribute (without parenthesis)
    @abstractmethod  # marks as has to be overwritten
    def name(self) -> str:
        ...

    @abstractmethod
    def merge(self, lowers: list, uppers: list, hadm_id=None, itemid=None) -> tuple:
        ...


class WidestRange(RangeMergeStrategy):
    """
    Merge strategy that takes the smallest lower range and the largest upper range.
    """

    @property
    def name(self) -> str:
        return "widest"

    def merge(self, lowers: list, uppers: list, hadm_id=None, itemid=None) -> tuple:
        lower = min(lowers) if lowers else None
        upper = max(uppers) if uppers else None
        if lower is not None and upper is not None and upper < lower:
            if self.warning_log is not None:
                self.warning_log.add(hadm_id, itemid, lower, upper)
            return None, None
        return lower, upper


class NarrowestRange(RangeMergeStrategy):
    """
    Merge strategy that takes the largest lower range and the smallest upper range.
    """

    @property
    def name(self) -> str:
        return "narrowest"

    def merge(self, lowers: list, uppers: list, hadm_id=None, itemid=None) -> tuple:
        lower = max(lowers) if lowers else None
        upper = min(uppers) if uppers else None
        if lower is not None and upper is not None and upper < lower:
            if self.warning_log is not None:
                self.warning_log.add(hadm_id, itemid, lower, upper)
            return None, None
        return lower, upper


class AverageRange(RangeMergeStrategy):
    """
    Merge strategy that takes the average lower range and the average upper range.
    If a threshold does not exist, returns None for it.
    If upper > lower, returns None for it and adds to Merge warning log.
    """

    @property
    def name(self) -> str:
        return "average"

    def merge(self, lowers: list, uppers: list, hadm_id=None, itemid=None) -> tuple:
        lower = sum(lowers) / len(lowers) if lowers else None
        upper = sum(uppers) / len(uppers) if uppers else None
        if lower is not None and upper is not None and upper < lower:
            if self.warning_log is not None:
                self.warning_log.add(hadm_id, itemid, lower, upper)
            return None, None
        return lower, upper


def make_merge_strategies() -> list:
    return [
        WidestRange(warning_log=MergeWarningLog()),
        AverageRange(warning_log=MergeWarningLog()),
        NarrowestRange(warning_log=MergeWarningLog()),
    ]


MERGE_STRATEGIES_LIST = [s.name for s in make_merge_strategies()]


def run_merge_step(cohort: str, all_mappings: bool = False) -> None:
    # load the ranges file. ranges is dict: (hadm_id, itemid) -> set of tuples {(lower, upper), ...}
    ranges = load_ranges(cohort)
    # by default only the strategy the study reports on; --all_mappings builds the others too,
    # which the merge strategy comparison in src/studies needs
    strategies = [s for s in make_merge_strategies()
                  if all_mappings or s.name == BEST_MERGE_STRATEGY]
    strategies_to_run = [s for s in strategies
                         # only select merges for which no files exist
                         if not merged_ranges_path(cohort, s.name).exists()]

    existing = [s.name for s in strategies
                if merged_ranges_path(cohort, s.name).exists()]
    if existing:
        print(f"Cohort {cohort}: merged range files already exist for {', '.join(existing)}.")

    if strategies_to_run:
        merged = {s.name: {} for s in
                  strategies_to_run}  # dict to save the merged ranges. string strategy_name -> dict of ranges

        # go over all (hadm_id, itemid) pairs in this cohort
        for (hadm_id, itemid), range_set in ranges.items():  # iterate over the ranges dict for this cohort
            # for each (hadm_id, itemid) pair create lists for lower and upper thresholds
            lowers = [r[0] for r in range_set if r[0] is not None]
            uppers = [r[1] for r in range_set if r[1] is not None]
            # for each strategy get lower and upper threshold and add to dict
            for strategy in strategies_to_run:
                lower, upper = strategy.merge(lowers, uppers, hadm_id=hadm_id, itemid=itemid)
                merged[strategy.name][(hadm_id, itemid)] = (lower, upper)

        # save the merged ranges files
        for strategy in strategies_to_run:
            save_merged_ranges(cohort, merged[strategy.name], strategy.name)
            if strategy.warning_log is not None:
                print(f"Saved warnings for cohort {cohort} and strategy {strategy.name}: ")
                strategy.warning_log.save(merge_warnings_path(cohort, strategy.name))
