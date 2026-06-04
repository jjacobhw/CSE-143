"""Metrics, plots, and reporting helpers for the experiments."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

# Fixed label order so the confusion matrix is always laid out the same way.
LABELS = ["ham", "spam"]


def calculate_metrics(y_true, y_pred) -> dict:
    """Return the five metrics we care about, treating "spam" as positive.

    F2 is like F1 but weights recall more heavily (beta=2).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, pos_label="spam", zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, pos_label="spam", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """Draw a labeled confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()

    # Save to results/ if a path was given (so we can put it in the report).
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def save_metrics_summary(results_list, save_path):
    """Save a list of result dicts (one per experiment) to a CSV file."""
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv(save_path, index=False)
    return summary_df


def print_classification_report(y_true, y_pred):
    """Print scikit-learn's per-class precision/recall/F1 report."""
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
