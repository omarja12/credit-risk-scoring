"""
End-to-end training pipeline:
  1. Load (or generate) applicant data
  2. Train/test split
  3. Fit WOE binning on train only (avoid leakage)
  4. Transform both splits to WOE space
  5. Fit logistic regression on WOE features
  6. Build a points-based scorecard from the fitted model
  7. Evaluate (AUC, Gini, KS) and save all artifacts to models/
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from woe_binning import fit_all, transform  # noqa: E402
from scorecard import build_scorecard, score_from_woe, ScorecardConfig, grade_from_score  # noqa: E402

NUMERIC_COLS = [
    "age", "annual_income", "employment_length_years", "loan_amount",
    "loan_term_months", "dti", "num_credit_lines", "credit_utilization",
    "num_late_payments_2y", "months_since_last_delinquency",
    "credit_history_years", "has_bankruptcy_10y",
]
# Discrete features get explicit, business-meaningful bins instead of quantiles.
CUSTOM_EDGES = {
    "num_late_payments_2y": [-np.inf, 0, 1, np.inf],   # 0 / 1 / 2+
    "has_bankruptcy_10y": [-np.inf, 0, np.inf],        # no / yes
}
CATEGORICAL_COLS = ["home_ownership", "loan_purpose"]
TARGET = "default"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "applicants.csv")
    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df[TARGET], random_state=42
    )

    # 1) Fit WOE bins on train, drop features with very low information value
    fitted_bins = fit_all(train_df, TARGET, NUMERIC_COLS, CATEGORICAL_COLS, CUSTOM_EDGES)
    iv_report = {k: v.iv for k, v in fitted_bins.items()}
    kept_features = [k for k, iv in iv_report.items() if iv >= 0.02]
    fitted_bins = {k: v for k, v in fitted_bins.items() if k in kept_features}

    X_train = transform(train_df, fitted_bins)
    X_test = transform(test_df, fitted_bins)
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    # 2) Fit logistic regression on WOE-transformed features
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    test_ks = ks_statistic(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}  (Gini = {2*test_auc-1:.4f})")
    print(f"Test KS:   {test_ks:.4f}")
    print("\nInformation Value by feature (higher = more predictive):")
    for k, v in sorted(iv_report.items(), key=lambda kv: -kv[1]):
        flag = "" if v >= 0.02 else "  (dropped, IV < 0.02)"
        print(f"  {k:32s} IV={v:.4f}{flag}")

    # 3) Build points-based scorecard
    coef_dict = {feat: c for feat, c in zip(X_train.columns, model.coef_[0])}
    config = ScorecardConfig()
    scorecard = build_scorecard(coef_dict, model.intercept_[0], config)

    # sanity check: score the test set via the scorecard and compare AUC
    test_scores = []
    for _, row in X_test.iterrows():
        woe_values = row.to_dict()
        result = score_from_woe(woe_values, scorecard)
        test_scores.append(result["score"])
    scorecard_auc = roc_auc_score(y_test, [-s for s in test_scores])  # lower score = higher risk
    print(f"\nScorecard-derived AUC (should match model AUC): {scorecard_auc:.4f}")

    # 4) Save all artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_model.joblib"))
    joblib.dump(fitted_bins, os.path.join(MODELS_DIR, "woe_bins.joblib"))
    with open(os.path.join(MODELS_DIR, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=2, default=float)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "train_auc": train_auc, "test_auc": test_auc, "test_ks": test_ks,
            "test_gini": 2 * test_auc - 1,
            "information_values": iv_report,
            "kept_features": kept_features,
            "n_train": len(train_df), "n_test": len(test_df),
            "default_rate": float(df[TARGET].mean()),
        }, f, indent=2)

    test_df = test_df.copy()
    test_df["score"] = test_scores
    test_df["grade"] = test_df["score"].apply(lambda s: grade_from_score(s, scorecard))
    test_df.to_csv(os.path.join(MODELS_DIR, "test_scored_sample.csv"), index=False)

    print(f"\nSaved model, WOE bins, scorecard, and metrics to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
