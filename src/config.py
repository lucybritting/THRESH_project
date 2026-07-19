from pathlib import Path

# ---- Directories and file paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

LABEVENTS_PATH = DATA_DIR / "raw_files/labevents.csv.gz"
D_LABITEMS_PATH = DATA_DIR / "raw_files/d_labitems.csv.gz"
ADMISSIONS_PATH = DATA_DIR / "raw_files/admissions.csv.gz"
PATIENTS_PATH = DATA_DIR / "raw_files/patients.csv.gz"

CHUNK_SIZE = 5000000

# set true if only use the top100labs. If false, all labs are considered
TOP100LABS = False

BEST_MERGE_STRATEGY = "widest"

ALL_MERGE_STRATEGIES = ["widest", "average", "narrowest"]

METRICS = ["auc_roc", "avg_prec", "f1", "mcc", "balanced_acc"]

N_FOLDS = 5

KNN_KS = [1, 5, 10]  # neighbour counts to try for range imputation
# set to the int which is the best k for knn imputation of ranges
BEST_KNN = 1


