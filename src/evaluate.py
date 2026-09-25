"""
Produces evaluation plots from the trained scorecard's test predictions:
  - ROC curve
  - Score distribution by outcome (good vs bad)

Run after train.py. Saves PNGs to models/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    df = pd.read_csv(os.path.join(MODELS_DIR, "test_scored_sample.csv"))

    fpr, tpr, _ = roc_curve(df["default"], -df["score"])  # lower score = higher risk
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Scorecard on Held-Out Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df.loc[df["default"] == 0, "score"], bins=30, alpha=0.6, label="Good (no default)", color="#16a34a")
    ax.hist(df.loc[df["default"] == 1, "score"], bins=30, alpha=0.6, label="Bad (default)", color="#dc2626")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "score_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"Saved roc_curve.png and score_distribution.png to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
