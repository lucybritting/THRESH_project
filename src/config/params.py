"""Every choice that changes a result."""

# set true to use only the curated top100 lab list, false to use every itemid in
# d_labitems. Either way the blacklist is subtracted.
TOP100LABS = False

# a lab is blacklisted when a reference range is recorded for no more than this share of
# the admissions that actually have a value for it.
MIN_RANGE_COVERAGE_PCT = 10

# reference range merging
BEST_MERGE_STRATEGY = "widest"

ALL_MERGE_STRATEGIES = ["widest", "average", "narrowest"]

# Thresholds for deciding whether a lab item's several reference ranges are explained by
# the sex or the age of the admissions carrying them.
RANGE_FEMALE_ONLY_PCT = 95     # a group at or above this share of female admissions
RANGE_MALE_ONLY_PCT = 5        # a group at or below it
RANGE_AGE_SPREAD_YEARS = 10    # spread in mean age across an item's range groups

# params for cross validation
N_FOLDS = 5
FOLD_SEED = 42

# params for imputation
KNN_KS = [1, 5, 10]  # neighbour counts to try for range imputation
BEST_KNN = 1

BEST_CLASSIFIER = "balanced_rf"

# performance metrics
METRICS = ["auc_roc", "avg_prec", "f1", "mcc", "balanced_acc"]


# a cohort only enters the cross-cohort analyses when its reference representation
# reaches this fold-averaged AUC-ROC
ROC_FILTER_REP = "cont"
MIN_COHORT_ROC = 0.55

# List of representations that are used for feature importance and fairness analyses
STABILITY_FEATURE_SETS = ["cont", "disc", "disc_imp", "bin"]

# Representations left out of the cross-cohort overviews.
CROSS_COHORT_EXCLUDED_REPS = ["disc_imp"]

# the representations built by concatenating the feature blocks of the single ones.
CONCATENATED_REPS = ["cont_bin", "disc_bin", "cont_disc_bin"]

# columns of the rank difference heatmaps: the union of the top N labs of each of these
# representations
RANK_DIFF_REPRESENTATIONS = ["cont", "disc", "bin"]
RANK_DIFF_TOP_N = 50


# ICD-10 chapters, for stratifying the cohorts by the disease
# Entries are (letter, lowest number, highest number, group key, group name)
ICD_CHAPTERS = [
    ("A",  0, 99, "A-B",        "Infectious and parasitic diseases"),
    ("B",  0, 99, "A-B",        "Infectious and parasitic diseases"),
    ("C",  0, 99, "C, D00-D49", "Neoplasms"),
    ("D",  0, 49, "C, D00-D49", "Neoplasms"),
    ("D", 50, 89, "D50-D89",    "Blood, blood-forming organs, immune mechanism"),
    ("E",  0, 99, "E",          "Endocrine, nutritional, metabolic diseases"),
    ("F",  0, 99, "F",          "Mental, behavioural, neurodevelopmental disorders"),
    ("G",  0, 99, "G",          "Nervous system"),
    ("H",  0, 59, "H00-H59",    "Eye and adnexa"),
    ("H", 60, 95, "H60-H95",    "Ear and mastoid process"),
    ("I",  0, 99, "I",          "Circulatory system"),
    ("J",  0, 99, "J",          "Respiratory system"),
    ("K",  0, 99, "K",          "Digestive system"),
    ("L",  0, 99, "L",          "Skin and subcutaneous tissue"),
    ("M",  0, 99, "M",          "Musculoskeletal system and connective tissue"),
    ("N",  0, 99, "N",          "Genitourinary system"),
    ("O",  0, 99, "O",          "Pregnancy, childbirth, puerperium"),
    ("P",  0, 99, "P",          "Conditions originating in the perinatal period"),
    ("Q",  0, 99, "Q",          "Congenital malformations, deformations, chromosomal abnormalities"),
    ("R",  0, 99, "R",          "Unclassified"),
    ("S",  0, 99, "S-T",        "Injury, poisoning, other external causes"),
    ("T",  0, 99, "S-T",        "Injury, poisoning, other external causes"),
    ("V",  0, 99, "V-Y",        "External causes of morbidity"),
    ("W",  0, 99, "V-Y",        "External causes of morbidity"),
    ("X",  0, 99, "V-Y",        "External causes of morbidity"),
    ("Y",  0, 99, "V-Y",        "External causes of morbidity"),
]
# a code that matches no chapter above
UNMAPPED_ICD_CHAPTER = ("?", "Unmapped ICD chapter")

# bins for stratifying the cohorts by size and by label distribution in the performance
# report
COHORT_SIZE_BINS = [0, 500, 2000, 10000, float("inf")]
COHORT_SIZE_LABELS = ["<500", "500-2k", "2k-10k", ">10k"]
PREVALENCE_BINS = [0.0, 0.05, 0.20, 1.0]
PREVALENCE_LABELS = ["<5%", "5-20%", ">20%"]

# age bands for the fairness analysis (lower bound given and included).
AGE_BANDS = [18, 40, 65, 80, float("inf")]
AGE_BAND_LABELS = ["18-39", "40-64", "65-79", "80+"]


FAIRNESS_CRITERIA = {
    "statistical_parity": ["selection_rate"],
    "predictive_parity": ["ppv"],
    "equal_opportunity": ["tpr"],
    "predictive_equality": ["fpr"],
    "equalised_odds": ["tpr", "fpr"],
}

# a demographic group only enters the fairness-criteria gaps with enough support
FAIRNESS_MIN_GROUP_SAMPLES = 20
FAIRNESS_MIN_LABEL_SAMPLES = 5

# bins for the per-group equalised-odds gaps by how many admissions the group holds.
FAIRNESS_GROUP_SIZE_BINS = [0, 100, 500, 2000, float("inf")]
FAIRNESS_GROUP_SIZE_LABELS = ["<100", "100-500", "500-2k", ">2k"]
