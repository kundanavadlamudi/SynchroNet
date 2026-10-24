"""Evaluate trained bot-classification models with real metrics only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "results" / "reports"
GRAPH_DIR = BASE_DIR / "results" / "graphs"
COMPARISON_FILE = REPORT_DIR / "ml_model_comparison.csv"
SUMMARY_FILE = REPORT_DIR / "ml_evaluation_summary.json"

CLASS_NAMES = {0: "Genuine", 1: "Bot"}


def _to_builtin(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_predictions(
    y_true,
    y_pred,
    y_proba=None,
    split_name: str = "validation",
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]

    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="binary", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="binary", zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred, labels=labels, average="binary", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["Genuine", "Bot"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["roc_auc_note"] = (
            "ROC-AUC requires both classes in y_true and prediction "
            "probabilities."
        )

    return metrics


def plot_confusion_matrix(
    matrix,
    output_file: Path,
    title: str,
) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(array, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Genuine", "Bot"])
    ax.set_yticks([0, 1], ["Genuine", "Bot"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for (row, column), value in np.ndenumerate(array):
        ax.text(column, row, int(value), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_file, dpi=120)
    plt.close(fig)
    return output_file


def save_evaluation_outputs(
    comparison_rows: list[dict],
    summary: dict,
    comparison_file: Path | None = None,
    summary_file: Path | None = None,
) -> tuple[Path, Path]:
    comparison_path = Path(comparison_file or COMPARISON_FILE)
    summary_path = Path(summary_file or SUMMARY_FILE)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(comparison_rows).to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_to_builtin),
        encoding="utf-8",
    )
    return comparison_path, summary_path


def comparison_row(model_name: str, metrics: dict) -> dict:
    return {
        "model": model_name,
        "split": metrics.get("split"),
        "n_samples": metrics.get("n_samples"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
    }
