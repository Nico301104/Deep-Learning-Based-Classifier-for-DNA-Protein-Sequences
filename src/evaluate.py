"""
Evaluare: metrici (accuracy, precision, recall, F1, AUC-ROC, matrice de
confuzie) si generarea graficelor salvate in /results.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend non-interactiv: sigur pentru rulare fara GUI/headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config, data
from .model import DNA_CNN
from .train import TrainHistory, get_device


# --------------------------------------------------------------------------
# Predictii
# --------------------------------------------------------------------------


@torch.no_grad()
def predict_cnn_probabilities(
    model: DNA_CNN, sequences: list[str], device: torch.device | None = None
) -> np.ndarray:
    """Intoarce probabilitatile prezise (dupa sigmoid) pentru clasa
    pozitiva (promotor), pentru o lista de secvente."""
    if device is None:
        device = get_device()
    model.eval()
    X = data.one_hot_encode_batch(sequences, max_length=config.MAX_SEQ_LENGTH)
    X_tensor = torch.from_numpy(X).to(device)
    logits = model(X_tensor)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    return probs


# --------------------------------------------------------------------------
# Metrici
# --------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # AUC-ROC necesita ambele clase prezente in y_true
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


# --------------------------------------------------------------------------
# Grafice
# --------------------------------------------------------------------------


def plot_loss_curve(history: TrainHistory, save_path=config.LOSS_CURVE_PATH) -> None:
    epochs = range(1, len(history.train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.train_losses, label="Train loss", marker="o", markersize=3)
    plt.plot(epochs, history.val_losses, label="Validation loss", marker="o", markersize=3)
    plt.axvline(
        history.best_epoch,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"Best epoch ({history.best_epoch})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCE)")
    plt.title("CNN 1D - Loss curve (train vs. validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path=config.ROC_CURVE_PATH,
    label: str = "CNN 1D",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    save_path=config.CONFUSION_MATRIX_PATH,
    class_names: tuple[str, str] = ("Non-promotor", "Promotor"),
    title: str = "Confusion Matrix - Test set",
) -> None:
    cm = np.asarray(cm)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Persistare metrici
# --------------------------------------------------------------------------


def save_metrics(all_metrics: dict[str, Any], json_path=config.METRICS_JSON_PATH) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)


def format_metrics_summary(all_metrics: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("REZUMAT METRICI - Clasificator secvente ADN")
    lines.append("=" * 60)
    for model_name, metrics in all_metrics.items():
        if model_name == "dataset_source":
            continue
        lines.append(f"\n[{model_name}]")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            value = metrics.get(key)
            if value is not None:
                lines.append(f"  {key:12s}: {value:.4f}")
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            lines.append(f"  confusion_matrix: {cm}")
    lines.append("=" * 60)
    return "\n".join(lines)
