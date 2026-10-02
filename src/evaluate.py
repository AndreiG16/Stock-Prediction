"""
Evaluation metrics and visualizations for the stock movement classifier.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from pathlib import Path

PLOTS_DIR   = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
LABEL_NAMES = ["Down", "Flat", "Up"]
COLORS      = {"Down": "#e74c3c", "Flat": "#95a5a6", "Up": "#2ecc71"}


def print_report(y_true, y_pred) -> float:
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"   Macro F1: {f1:.4f}")
    return f1


def plot_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Saved: {path.name}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    top    = importance_df.head(top_n)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — XGBoost Stock Classifier", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Saved: {path.name}")


def plot_label_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (y, title) in zip(axes, [(y_train, "Train Set"), (y_test, "Test Set")]):
        counts = y.value_counts().sort_index()
        labels = [LABEL_NAMES[i] for i in counts.index]
        clrs   = [COLORS[l] for l in labels]
        bars   = ax.bar(labels, counts.values, color=clrs, edgecolor="white", width=0.5)
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{count:,}", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"Label Distribution — {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "label_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Saved: {path.name}")


def plot_confidence_distribution(proba: np.ndarray, y_true):
    """Show that high-confidence predictions are more accurate."""
    max_proba = proba.max(axis=1)
    preds     = proba.argmax(axis=1)
    correct   = (preds == np.array(y_true))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_proba[correct],  bins=30, alpha=0.65, color="#2ecc71",
            label=f"Correct  (n={correct.sum():,})", density=True)
    ax.hist(max_proba[~correct], bins=30, alpha=0.65, color="#e74c3c",
            label=f"Incorrect (n={(~correct).sum():,})", density=True)
    ax.set_xlabel("Max Predicted Probability (model confidence)", fontsize=11)
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "confidence_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Saved: {path.name}")
