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

# human-readable titles for the metrics, used for plot panel/column headers
METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "avg_prec": "Average precision",
    "f1": "F1",
    "mcc": "MCC",
    "balanced_acc": "Balanced accuracy",
}

N_FOLDS = 5
FOLD_SEED = 42

KNN_KS = [1, 5, 10]  # neighbour counts to try for range imputation
# set to the int which is the best k for knn imputation of ranges
BEST_KNN = 1

BEST_CLASSIFIER = "balanced_rf"

# full display names for the classifiers, used in plot subheaders
CLASSIFIER_LABELS = {"rf": "RF", "balanced_rf": "Balanced RF",
                     "catboost": "Catboost"}

# age bands for the fairness analysis (lower bound given and included). The open
# upper edge (inf) gives the "80+" band, so there is one label per interval.
AGE_BANDS = [18, 40, 65, 80, float("inf")]
AGE_BAND_LABELS = ["18-39", "40-64", "65-79", "80+"]

# each criterion is satisfied when its quantity is equal across the groups of an
# attribute. Equalised odds is the pair (tpr, fpr), so it is the worse of the two.
FAIRNESS_CRITERIA = {
    "statistical_parity": ["selection_rate"],  # independence of the prediction
    "predictive_parity": ["ppv"],              # equal precision
    "equal_opportunity": ["tpr"],              # equal recall on the positives
    "predictive_equality": ["fpr"],            # equal false alarm rate
    "equalised_odds": ["tpr", "fpr"],          # both error rates at once
}

# human-readable labels for plot headers/rows
FAIRNESS_CRITERIA_LABELS = {
    "statistical_parity": "statistical parity",
    "predictive_parity": "predictive parity",
    "equal_opportunity": "equal opportunity",
    "predictive_equality": "predictive equality",
    "equalised_odds": "equalised odds",
}
FAIRNESS_ATTRIBUTE_LABELS = {"age_band": "Age band", "gender": "Gender"}

# a demographic group only enters the fairness-criteria gaps with enough support
FAIRNESS_MIN_GROUP_SAMPLES = 20
FAIRNESS_MIN_LABEL_SAMPLES = 5



# List of representations that are used for feature importance and fairness analyses
STABILITY_FEATURE_SETS = ["cont", "disc", "disc_imp", "bin"]

# human-readable display names for the representations, used for plot headers/labels
REPRESENTATION_LABELS = {
    "cont": "continuous", "disc": "discretised",
    "disc_imp": "discretised + imputed", "bin": "binarised"}

# order and display names of the representations for the classification heatmap
CLASSIFICATION_HEATMAP_REPS = {
    "cont": "continuous", "disc": "discretised", "disc_imp": "discretised + imputed",
    "bin": "binarised", "cont_bin": "continuous + binarised",
    "disc_bin": "discretised + binarised"}