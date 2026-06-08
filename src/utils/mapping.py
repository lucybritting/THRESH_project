import argparse
from dataclasses import dataclass

import pandas as pd
from pathlib import Path

# ---- Directories and file paths
DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

LABEVENTS_PATH  = DATA_DIR / "raw_files/labevents.csv.gz"
D_LABITEMS_PATH = DATA_DIR / "raw_files/d_labitems.csv.gz"

# ---- Runtime Params -----------------------
WINDOW_DAYS = 14
CHUNK_SIZE  = 100000000

# ---- Data loading -------------------------
# TODO: no dischtime filter here.
@dataclass
class CohortData:
    hadm_id_set: set[int]
    itemid_set: set[int]
    hadm_ids_with_value: dict[int, set[int]]


def load_aggregated_file(cohort):
    # TODO: adjust the file paths if necessary
    """
    Loads the aggregated file. It contains the continuous lab values
    :param cohort: string. Name of the cohort.
    :return: pandas dataframe with aggregated values.
    """
    # TODO: remove this if no longer necessary
    if cohort == "aplasia":
        cohort = "aplasia_45_days"
    elif cohort == "NF":
        cohort = "NF_30_days"
    path = DATA_DIR / f"preprocessed_files/aggregated/mimic_cohort_{cohort}_agg.csv"
    df = pd.read_csv(path)
    # TODO: return df or process to lists/dicts here?
    # get set of itemids and admission ids
    lab_cols = [c for c in df.columns if c != "hadm_id"]
    itemid_set = {int(c) for c in lab_cols}
    hadm_id_set = set(df["hadm_id"])

    # re-indexing and building lookup: itemid -> set of hadm_ids with value
    agg_indexed = df.set_index("hadm_id")  # make hadm_id the row index (faster lookup in DataFrame)
    # create dict: keys = itemid, values = set of admission ids where value for itemid is not NaN
    # for each item (column): finds row where value is not NaN, extracts the hadm_id and stores them as set
    hadm_ids_with_value: dict[int, set[int]] = {
        int(col): set(agg_indexed.index[agg_indexed[col].notna()])
        for col in lab_cols
    }
    print(f"Aggregated file for cohort {cohort}: {len(hadm_id_set)} admissions, {len(lab_cols)} lab items")
    return hadm_id_set, itemid_set, hadm_ids_with_value

def load_cohort_file(cohort):
    # TODO: adjust the file paths if necessary
    # TODO: read also age, gender and label from the cohort file.
    # TODO: currently this function is not necessary since we don't have windows
    """
    Loads the cohort file. Contains: ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'los', 'dod', 'age', 'gender', 'race', 'insurance', 'label']
    :param cohort: name of the cohort. String
    :return:
    """
    # TODO: remove this if no longer necessary
    if cohort == "aplasia":
        cohort = "aplasia_45_days"
    elif cohort == "NF":
        cohort = "NF_30_days"
    path = DATA_DIR / f"cohorts/mimic_cohort_{cohort}.csv.gz"
    df = pd.read_csv(path, usecols=["hadm_id", "dischtime"])
    return None





def load_data(cohorts):
    """
    Loads all the files needed.
    # TODO: adopt to cohort file template
    :param cohorts: list of cohort names
    :return:
    """
    # Load all cohorts
    cohort_data = {cohort:load_aggregated_file(cohort) for cohort in cohorts}

    # build union sets for the single labevents scan
    all_hadm_ids = set().union(*[d.hadm_id_set for d in cohort_data.values()])
    all_item_ids = set().union(*[d.itemid_set for d in cohort_data.values()])

