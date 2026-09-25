import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from woe_binning import fit_numeric_binning, fit_categorical_binning, transform
from scorecard import (build_scorecard, score_from_woe, score_to_pd, pd_to_score,
                       grade_from_pd, decision_from_pd, ScorecardConfig)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 2000
    dti = rng.uniform(0, 60, n)
    y = (rng.random(n) < (dti / 120)).astype(int)  # higher dti -> higher default prob
    return pd.DataFrame({"dti": dti, "default": y})


def test_numeric_binning_monotonic_direction(toy_data):
    fb = fit_numeric_binning(toy_data["dti"], toy_data["default"], "dti")
    # higher dti bins should generally have lower WOE (more bad / fewer good)
    labels = list(fb.bins.keys())
    woes = [fb.bins[l].woe for l in labels]
    assert woes[0] > woes[-1]  # first (low dti) bin has higher WOE than last (high dti) bin


def test_woe_transform_shape(toy_data):
    fb = fit_numeric_binning(toy_data["dti"], toy_data["default"], "dti")
    fitted = {"dti": fb}
    out = transform(toy_data, fitted)
    assert "dti_woe" in out.columns
    assert len(out) == len(toy_data)
    assert out["dti_woe"].notna().all()


def test_categorical_binning_basic():
    x = pd.Series(["A", "A", "B", "B", "B"])
    y = pd.Series([0, 1, 0, 0, 0])
    fb = fit_categorical_binning(x, y, "cat")
    assert set(fb.bins.keys()) == {"A", "B"}
    assert fb.iv >= 0


def test_scorecard_roundtrip_pd_consistency():
    coef = {"a_woe": 0.8, "b_woe": -0.5}
    intercept = -2.0
    config = ScorecardConfig()
    sc = build_scorecard(coef, intercept, config)

    woe_values = {"a_woe": 0.3, "b_woe": -0.2}
    result = score_from_woe(woe_values, sc)
    assert config.min_score <= result["score"] <= config.max_score

    pd_est = score_to_pd(result["raw_score"], sc)
    assert 0 < pd_est < 1


def test_grades_follow_pd_bands():
    assert grade_from_pd(0.005)[0] == "A"
    assert grade_from_pd(0.02)[0] == "B"
    assert grade_from_pd(0.04)[0] == "C"
    assert grade_from_pd(0.08)[0] == "D"
    assert grade_from_pd(0.30)[0] == "E"


def test_decisions_follow_policy():
    assert decision_from_pd(0.02) == "Approve"
    assert decision_from_pd(0.07) == "Refer"
    assert decision_from_pd(0.20) == "Decline"


def test_score_pd_roundtrip():
    sc = build_scorecard({"a_woe": -1.0}, -3.0, ScorecardConfig())
    for p in (0.01, 0.05, 0.2):
        assert abs(score_to_pd(pd_to_score(p, sc), sc) - p) < 1e-9


def test_higher_score_means_lower_pd():
    # In a real fitted scorecard, coefficients on WOE features are negative:
    # a higher WOE (safer bin, more "good" than "bad") should raise the score
    # and lower the estimated probability of default.
    coef = {"a_woe": -1.0}
    config = ScorecardConfig()
    sc = build_scorecard(coef, -2.0, config)
    low_risk = score_from_woe({"a_woe": 2.0}, sc)
    high_risk = score_from_woe({"a_woe": -2.0}, sc)
    assert low_risk["score"] > high_risk["score"]
    assert score_to_pd(low_risk["raw_score"], sc) < score_to_pd(high_risk["raw_score"], sc)


def test_bins_are_monotonic_after_merging():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.uniform(0, 100, n)
    # noisy but increasing risk, which tends to produce a non-monotonic raw bin
    y = (rng.random(n) < 0.02 + 0.0006 * x + rng.normal(0, 0.01, n).clip(-0.02, 0.02)).astype(int)
    fb = fit_numeric_binning(pd.Series(x), pd.Series(y), "x", max_bins=10)
    rates = [b.bad_rate for b in fb.bins.values()]
    assert all(b >= a for a, b in zip(rates, rates[1:]))
