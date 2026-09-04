"""How the cohorts are grouped for the cross-cohort figures: ICD chapter, and group means."""

import re

import pandas as pd

from config.params import ICD_CHAPTERS, UNMAPPED_ICD_CHAPTER


def _icd_chapter(code: str) -> tuple[str, str]:
    """
    (group key, group name) of an ICD code, matched on both its letter and its number
    """
    match = re.match(r"([A-Za-z])(\d+)", code)
    if match:
        letter, number = match.group(1).upper(), int(match.group(2))
        for chapter_letter, low, high, key, name in ICD_CHAPTERS:
            if chapter_letter == letter and low <= number <= high:
                return key, name
    return UNMAPPED_ICD_CHAPTER


# chapter key -> chapter name, and the key order the plot rows follow.
ICD_GROUP_NAMES = {key: name for *_, key, name in ICD_CHAPTERS} | {UNMAPPED_ICD_CHAPTER[0]: UNMAPPED_ICD_CHAPTER[1]}
ICD_GROUP_ORDER = list(ICD_GROUP_NAMES)


def _disease_group(cohort: str) -> str:
    """
    ICD chapter of the cohort's disease, from the whole of first code in the cohort name
    (cohort_<code1>-<code2>)
    """
    return _icd_chapter(cohort.removeprefix("cohort_").split("-")[0])[0]


def _target_group(cohort: str) -> str:
    """
    ICD chapter of the cohort's target, from the whole of second code in the cohort name
    """
    return _icd_chapter(cohort.removeprefix("cohort_").split("-")[1])[0]


def _group_mean(values_df: pd.DataFrame, group_of, count_label: str = "n",
                descriptions: dict | None = None) -> pd.DataFrame:
    """
    Collapse a per-cohort table to one row per group, labelling each row with the number of
    cohorts behind it. Shared by the plain and the difference heatmaps.
    """
    # observed=True: a categorical with an empty bin would otherwise add an all-NaN row
    grouped = values_df.groupby(group_of, observed=True)
    group_df = grouped.mean()
    n_cohorts = grouped.size()
    # groups differ a lot in size, so carry the cohort count in the row label, and spell
    # out on a second line what the key means where the key alone is not self-explanatory
    group_df.index = [f"{key} ({count_label}={n_cohorts[key]})"
                      + (f"\n{descriptions[key]}" if descriptions and key in descriptions else "")
                      for key in group_df.index]
    return group_df
