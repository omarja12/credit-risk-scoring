"""
Converts a WOE-transformed logistic regression into a points-based
scorecard, using the standard "PDO" (Points to Double the Odds)
methodology used across the credit industry:

    score = Offset + Factor * ln(odds)
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)

Each variable's contribution to the score is:

    points_i = -(woe_i * coef_i + intercept / n_vars) * Factor + Offset / n_vars

This gives a fully interpretable score where every feature bin maps to
a fixed number of points, exactly like real-world FICO-style scorecards.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ScorecardConfig:
    base_score: int = 600          # score at base_odds
    base_odds: float = 20.0        # good:bad odds of 20:1 at base_score
    pdo: int = 50                  # points to double the odds
    min_score: int = 300
    max_score: int = 850

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


def build_scorecard(coef: dict, intercept: float, config: ScorecardConfig) -> dict:
    """
    coef: {feature_name_woe: coefficient} from fitted LogisticRegression
    Returns a dict of feature -> (multiplier, offset_share) to compute points,
    plus metadata, so points_i = -(woe * coef_i) * factor + offset/n
    """
    n_vars = len(coef)
    factor = config.factor
    offset = config.offset

    scorecard = {
        "meta": {
            "factor": factor,
            "offset": offset,
            "intercept": intercept,
            "n_vars": n_vars,
            "min_score": config.min_score,
            "max_score": config.max_score,
        },
        "variables": {}
    }
    for feat, c in coef.items():
        scorecard["variables"][feat] = {
            "coef": c,
            "points_per_woe": -c * factor,
            "base_points": -(intercept / n_vars) * factor + offset / n_vars,
        }
    return scorecard


def score_from_woe(woe_values: dict, scorecard: dict) -> dict:
    """
    woe_values: {feature_name_woe: woe_value_for_this_applicant}
    Returns total score (clipped to range) + per-feature point breakdown.
    """
    breakdown = {}
    total = 0.0
    for feat, v in scorecard["variables"].items():
        woe = woe_values.get(feat, 0.0)
        pts = v["base_points"] + v["points_per_woe"] * woe
        breakdown[feat] = round(pts, 1)
        total += pts

    cfg = scorecard["meta"]
    total_clipped = float(np.clip(total, cfg["min_score"], cfg["max_score"]))
    return {"score": round(total_clipped, 0), "raw_score": round(total, 1), "breakdown": breakdown}


def score_to_pd(score: float, scorecard: dict) -> float:
    """Invert score -> probability of default, for consistency checks."""
    meta = scorecard["meta"]
    log_odds = (score - meta["offset"]) / meta["factor"]
    odds = np.exp(log_odds)
    pd_ = 1 / (1 + odds)
    return float(np.clip(pd_, 1e-6, 1 - 1e-6))


# Risk grades are defined on probability of default, not on raw score cut-offs,
# so the grade and the PD shown to the user can never contradict each other.
GRADE_BANDS = [
    (0.010, "A", "Very low risk"),
    (0.025, "B", "Low risk"),
    (0.050, "C", "Moderate risk"),
    (0.100, "D", "Elevated risk"),
    (1.000, "E", "High risk"),
]

# Illustrative credit policy: approve below 5% PD, manual review 5-10%, decline above.
APPROVE_MAX_PD = 0.05
REFER_MAX_PD = 0.10


def grade_from_pd(pd_: float) -> tuple:
    for upper, grade, label in GRADE_BANDS:
        if pd_ < upper:
            return grade, label
    return GRADE_BANDS[-1][1], GRADE_BANDS[-1][2]


def decision_from_pd(pd_: float) -> str:
    if pd_ < APPROVE_MAX_PD:
        return "Approve"
    if pd_ < REFER_MAX_PD:
        return "Refer"
    return "Decline"


def pd_to_score(pd_: float, scorecard: dict) -> float:
    meta = scorecard["meta"]
    odds = (1 - pd_) / pd_
    return meta["offset"] + meta["factor"] * np.log(odds)


def grade_from_score(score: float, scorecard: dict) -> str:
    return grade_from_pd(score_to_pd(score, scorecard))[0]


def max_points_by_feature(fitted_bins: dict, scorecard: dict) -> dict:
    """Best achievable points per feature (its safest bin). Used for reason codes."""
    out = {}
    for name, fb in fitted_bins.items():
        v = scorecard["variables"][f"{name}_woe"]
        woes = [b.woe for b in fb.bins.values()]
        out[name] = max(v["base_points"] + v["points_per_woe"] * w for w in woes)
    return out
