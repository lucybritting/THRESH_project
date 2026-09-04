"""Per-sample classification heatmap: which representations get which admission right."""

import numpy as np

from config.params import BEST_CLASSIFIER
from config.labels import CLASSIFICATION_HEATMAP_REPS, CLASSIFIER_LABELS
from data_io.paths import figure_path
from data_io.store import load_test_predictions
from plots import plot_classification_heatmap


def sample_classification_analysis(cohort: str) -> None:
    """
    Per-sample classification heatmap for BEST_CLASSIFIER.
    """
    predictions_df = load_test_predictions(cohort)  # all folds concatenated
    predictions_df = predictions_df[predictions_df["classifier"] == BEST_CLASSIFIER] # take only BEST_CLASSIFIER

    rep_order = [rep for rep in CLASSIFICATION_HEATMAP_REPS
                 if rep in predictions_df["feature_set"].unique()] # sort representations. Robust if one representation does not exist.

    # order samples by ground truth, then by mean prediction probabiltiy
    per_sample = predictions_df.groupby("hadm_id").agg(y_true=("y_true", "first"),
                                                       mean_prob=("y_prob", "mean"))
    sample_order = per_sample.sort_values(["y_true", "mean_prob"]).index

    # rows = representations, columns = samples, values = predicted class
    pred = predictions_df.pivot(index="feature_set", columns="hadm_id", values="y_pred")
    pred = pred.reindex(index=rep_order, columns=sample_order)

    truth_row = per_sample.loc[sample_order, "y_true"].to_numpy()[None, :]  # ground truth on top
    data = np.vstack([truth_row, pred.to_numpy()])
    row_labels = ["ground truth"] + [CLASSIFICATION_HEATMAP_REPS[rep] for rep in rep_order]

    # x-position where the sorted true label flips from 0 to 1
    n_negative = int((per_sample.loc[sample_order, "y_true"] == 0).sum())
    split_x = n_negative - 0.5 if 0 < n_negative < len(sample_order) else None

    plot_classification_heatmap(
        data, row_labels, n_ground_truth_rows=1, split_x=split_x,
        title=f"Per-sample classification by representation — {cohort}",
        subtitle=f"Held-out predictions of all 5 folds pooled. Classifier: {CLASSIFIER_LABELS[BEST_CLASSIFIER]}",
        xlabel=f"{data.shape[1]:,} samples",
        path=figure_path(cohort, "classification_heatmap"))
