# Functions for calculating metrics, drawing the confusion matrix, and saving results

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

# Fixed label order so the confusion matrix always looks the same
LABELS = ["ham", "spam"]


# Returns the five metrics we care about, with spam as the positive class
# F2 is like F1 but weights recall more (beta=2)
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, pos_label="spam", zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, pos_label="spam", zero_division=0),
    }


# Draws the confusion matrix as a heatmap and saves it if we pass a path
def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
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

    # Save to results/ if we got a path, so we can use it in the report
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# Saves a list of result dicts (one per experiment) to a CSV file
def save_metrics_summary(results_list, save_path):
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv(save_path, index=False)
    return summary_df


# Prints scikit-learn's per-class precision/recall/F1 report.
def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
