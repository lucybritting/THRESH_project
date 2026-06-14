from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


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
