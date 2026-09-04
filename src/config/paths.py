"""Where the data lives and how it is read."""

from pathlib import Path

# src/config/paths.py -> src/config -> src -> the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Directories and file paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

LABEVENTS_PATH = DATA_DIR / "raw_files/labevents.csv.gz"
D_LABITEMS_PATH = DATA_DIR / "raw_files/d_labitems.csv.gz"
ADMISSIONS_PATH = DATA_DIR / "raw_files/admissions.csv.gz"
PATIENTS_PATH = DATA_DIR / "raw_files/patients.csv.gz"

# cohort CSVs are read from one of these sub-folders of data/
COHORT_FOLDERS = {
    "all": "cohorts_100",    # the 100 representative cohorts
    "subset": "cohorts_50",  # one cohort per code1
    "test": "cohorts_test",  # small test set
}

# rows per chunk when streaming labevents, which is far too large to hold in memory
CHUNK_SIZE = 5000000
