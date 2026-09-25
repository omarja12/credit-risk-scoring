"""
Exports the trained scorecard to site/scorecard.json for the web app.

The web app never re-implements the model: it only looks up points in the
table written here, so the browser and Python always agree. The parity test
in tests/test_web_parity.py checks exactly that.

Run after train.py.
"""
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from scorecard import GRADE_BANDS, APPROVE_MAX_PD, REFER_MAX_PD, pd_to_score  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "site", "scorecard.json")

LABELS = {
    "credit_utilization": "Credit card utilization",
    "dti": "Debt-to-income ratio",
    "num_late_payments_2y": "Late payments, last 2 years",
    "months_since_last_delinquency": "Time since last delinquency",
    "has_bankruptcy_10y": "Bankruptcy, last 10 years",
    "annual_income": "Annual income",
    "employment_length_years": "Time with current employer",
    "loan_purpose": "Loan purpose",
}
PURPOSES = {
    "debt_consolidation": "Debt consolidation", "credit_card": "Credit card refinancing",
    "home_improvement": "Home improvement", "car": "Car", "medical": "Medical",
    "small_business": "Small business",
}


def _unit(feat, v):
    if feat in ("dti", "credit_utilization"):
        return f"{v:.0f}%"
    if feat == "annual_income":
        return f"${round(v, -3):,.0f}"
    if feat == "employment_length_years":
        v = round(v)
        return f"{v:.0f} year" + ("" if v == 1 else "s")
    if feat == "months_since_last_delinquency":
        return f"{v:.0f} months"
    return f"{v:g}"


def describe_bin(feat, lo, hi):
    if feat == "num_late_payments_2y":
        return {0: "None", 1: "One"}.get(int(hi), "Two or more") if not math.isinf(hi) else "Two or more"
    if feat == "has_bankruptcy_10y":
        return "No" if not math.isinf(hi) else "Yes"
    if feat == "months_since_last_delinquency":
        if math.isinf(hi):
            return f"Over {lo:.0f} months, or never"
        if math.isinf(lo):
            return f"Within {hi:.0f} months"
        return f"{lo:.0f} to {hi:.0f} months"
    if math.isinf(lo):
        return f"{_unit(feat, hi)} or less"
    if math.isinf(hi):
        return f"Over {_unit(feat, lo)}"
    if feat == "employment_length_years":
        return f"{round(lo)} to {_unit(feat, hi)}"
    return f"{_unit(feat, lo)} to {_unit(feat, hi)}"


def main():
    bins = joblib.load(os.path.join(MODELS, "woe_bins.joblib"))
    with open(os.path.join(MODELS, "scorecard.json")) as f:
        sc = json.load(f)
    with open(os.path.join(MODELS, "metrics.json")) as f:
        metrics = json.load(f)

    features = []
    for name, fb in sorted(bins.items(), key=lambda kv: -kv[1].iv):
        v = sc["variables"][f"{name}_woe"]
        pts = lambda woe: v["base_points"] + v["points_per_woe"] * woe  # noqa: E731
        entry = {"key": name, "label": LABELS.get(name, name), "iv": round(fb.iv, 4)}
        if fb.is_numeric:
            entry["type"] = "numeric"
            entry["edges"] = [None if math.isinf(e) else float(e) for e in fb.edges]
            entry["bins"] = []
            for i in range(len(fb.edges) - 1):
                lo, hi = fb.edges[i], fb.edges[i + 1]
                label = f"({lo:.2f}, {hi:.2f}]"
                woe = fb.bins[label].woe if label in fb.bins else 0.0
                entry["bins"].append({"points": pts(woe), "text": describe_bin(name, lo, hi)})
        else:
            entry["type"] = "categorical"
            cats = sorted(fb.bins, key=lambda c: -fb.bins[c].woe)
            entry["categories"] = cats
            entry["bins"] = [{"points": pts(fb.bins[c].woe), "text": PURPOSES.get(c, c)} for c in cats]
        entry["best"] = int(np.argmax([b["points"] for b in entry["bins"]]))
        features.append(entry)

    grades = []
    for upper, grade, label in GRADE_BANDS:
        grades.append({"grade": grade, "label": label, "max_pd": upper,
                       "min_score": None if upper >= 1 else None})
    # score boundaries between grades (higher score = lower PD)
    for g in grades:
        g["min_score"] = pd_to_score(g["max_pd"], sc) if g["max_pd"] < 1 else sc["meta"]["min_score"]

    scored = pd.read_csv(os.path.join(MODELS, "test_scored_sample.csv"))
    meta = sc["meta"]
    scored["pd"] = 1 / (1 + np.exp((scored["score"] - meta["offset"]) / meta["factor"]))
    calib = []
    for g in grades:
        sub = scored[scored["grade"] == g["grade"]]
        calib.append({"grade": g["grade"], "share": len(sub) / len(scored),
                      "predicted": float(sub["pd"].mean()), "actual": float(sub["default"].mean())})

    fpr, tpr, _ = roc_curve(scored["default"], -scored["score"])
    idx = np.unique(np.linspace(0, len(fpr) - 1, 80).astype(int))
    roc = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]

    out = {
        "meta": {"offset": meta["offset"], "factor": meta["factor"],
                 "score_min": meta["min_score"], "score_max": meta["max_score"],
                 "approve_max_pd": APPROVE_MAX_PD, "refer_max_pd": REFER_MAX_PD},
        "features": features,
        "grades": grades,
        "metrics": {k: metrics[k] for k in ("test_auc", "test_gini", "test_ks", "default_rate",
                                            "n_train", "n_test")},
        "calibration": calib,
        "roc": roc,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUT} ({len(features)} features)")


if __name__ == "__main__":
    main()
