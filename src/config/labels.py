"""Display names, used only for plot headers and labels."""

# titles for the metrics, used for plot panel/column headers
METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "avg_prec": "Average precision",
    "f1": "F1",
    "mcc": "MCC",
    "balanced_acc": "Balanced accuracy",
}

# full display names for the classifiers, used in plot subheaders
CLASSIFIER_LABELS = {"rf": "RF", "balanced_rf": "Balanced RF",
                     "catboost": "Catboost"}

# labels for plot headers/rows
FAIRNESS_CRITERIA_LABELS = {
    "statistical_parity": "statistical parity",
    "predictive_parity": "predictive parity",
    "equal_opportunity": "equal opportunity",
    "predictive_equality": "predictive equality",
    "equalised_odds": "equalised odds",
}
FAIRNESS_ATTRIBUTE_LABELS = {"age_band": "Age band", "gender": "Gender"}

# display names for the representations, used for plot headers/labels
REPRESENTATION_LABELS = {
    "cont": "continuous", "disc": "discretised",
    "disc_imp": "discretised + imputed", "bin": "binarised",
    "cont_bin": "cont + bin", "disc_bin": "disc + bin",
    "cont_disc_bin": "cont + disc + bin"}

# order and display names of the representations for the classification heatmap
CLASSIFICATION_HEATMAP_REPS = {
    "cont": "continuous", "disc": "discretised", "disc_imp": "discretised + imputed",
    "bin": "binarised", "cont_bin": "continuous + binarised",
    "disc_bin": "discretised + binarised",
    "cont_disc_bin": "continuous + discretised + binarised"}
