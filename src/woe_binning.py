"""
Weight of Evidence (WOE) binning.

This is the standard technique behind most retail credit scorecards:
each raw feature is bucketed into bins, and each bin is replaced by its
WOE value, which measures how much that bin shifts the odds of default
relative to the overall population:

    WOE_bin = ln( %good_in_bin / %bad_in_bin )

Binning first (rather than feeding raw features into the model) gives:
  - monotonic, business-explainable relationships per variable
  - robustness to outliers
  - a natural path to a points-based scorecard (see scorecard.py)
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing so log-odds never blow up on 0 counts


@dataclass
class BinInfo:
    label: str
    woe: float
    count: int
    bad_rate: float


@dataclass
class FeatureBinning:
    name: str
    is_numeric: bool
    edges: Optional[list] = None          # numeric: bin edges
    bins: dict = field(default_factory=dict)  # bin label -> BinInfo
    iv: float = 0.0                        # Information Value for this feature

    def _numeric_bin_label(self, value) -> str:
        edges = self.edges
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (value > lo or i == 0) and value <= hi:
                return f"({lo:.2f}, {hi:.2f}]"
        return f"({edges[-2]:.2f}, {edges[-1]:.2f}]"

    def transform_value(self, value) -> float:
        if self.is_numeric:
            if pd.isna(value):
                label = "MISSING"
            else:
                label = self._numeric_bin_label(float(value))
        else:
            label = str(value) if str(value) in self.bins else "OTHER"
        info = self.bins.get(label) or self.bins.get("OTHER") or self.bins.get("MISSING")
        return info.woe if info else 0.0

    def transform_series(self, s: pd.Series) -> pd.Series:
        return s.apply(self.transform_value)


def _woe_and_iv(good_count, bad_count, total_good, total_bad):
    good_dist = max(good_count, EPS) / max(total_good, 1)
    bad_dist = max(bad_count, EPS) / max(total_bad, 1)
    woe = np.log(good_dist / bad_dist)
    iv_component = (good_dist - bad_dist) * woe
    return woe, iv_component


def _merge_to_monotonic(x: pd.Series, y: pd.Series, edges: list) -> list:
    """
    Merge adjacent bins until the bad rate moves in one direction only.

    Direction comes from the rank correlation between the feature and the
    target. Keeping bins monotonic means points never go up and then down
    as a feature increases, which is what reviewers and regulators expect.
    """
    direction = np.sign(x.corr(y, method="spearman")) or 1.0
    edges = list(edges)
    while len(edges) > 3:
        binned = pd.cut(x, bins=edges)
        rates = y.groupby(binned, observed=False).mean().to_numpy()
        diffs = np.diff(rates) * direction
        bad = np.where(diffs < 0)[0]
        if len(bad) == 0:
            break
        i = bad[0]
        del edges[i + 1]  # merge bin i with bin i+1
    return edges


def fit_numeric_binning(x: pd.Series, y: pd.Series, name: str, max_bins: int = 6,
                        edges: Optional[list] = None, monotonic: bool = True) -> FeatureBinning:
    total_bad = y.sum()
    total_good = len(y) - total_bad

    custom = edges is not None
    if edges is None:
        quantiles = np.linspace(0, 1, max_bins + 1)
        edges = sorted(set(np.quantile(x.dropna(), quantiles)))
        if len(edges) < 3:
            edges = [x.min() - 1, x.median(), x.max() + 1]
    edges = list(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    if monotonic and not custom:
        edges = _merge_to_monotonic(x, y, edges)

    fb = FeatureBinning(name=name, is_numeric=True, edges=edges)
    binned = pd.cut(x, bins=edges)

    iv_total = 0.0
    for interval in binned.cat.categories:
        mask = binned == interval
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bad_count = y[mask].sum()
        good_count = n_bin - bad_count
        woe, iv_c = _woe_and_iv(good_count, bad_count, total_good, total_bad)
        iv_total += iv_c
        lo, hi = interval.left, interval.right
        label = f"({lo:.2f}, {hi:.2f}]"
        fb.bins[label] = BinInfo(label=label, woe=round(woe, 4), count=int(n_bin),
                                   bad_rate=round(bad_count / n_bin, 4))

    if x.isna().any():
        mask = x.isna()
        bad_count = y[mask].sum()
        good_count = mask.sum() - bad_count
        woe, iv_c = _woe_and_iv(good_count, bad_count, total_good, total_bad)
        iv_total += iv_c
        fb.bins["MISSING"] = BinInfo("MISSING", round(woe, 4), int(mask.sum()),
                                       round(bad_count / max(mask.sum(), 1), 4))

    fb.iv = round(iv_total, 4)
    return fb


def fit_categorical_binning(x: pd.Series, y: pd.Series, name: str) -> FeatureBinning:
    total_bad = y.sum()
    total_good = len(y) - total_bad

    fb = FeatureBinning(name=name, is_numeric=False)
    iv_total = 0.0
    for category in x.astype(str).unique():
        mask = x.astype(str) == category
        n_bin = mask.sum()
        bad_count = y[mask].sum()
        good_count = n_bin - bad_count
        woe, iv_c = _woe_and_iv(good_count, bad_count, total_good, total_bad)
        iv_total += iv_c
        fb.bins[category] = BinInfo(category, round(woe, 4), int(n_bin),
                                      round(bad_count / n_bin, 4))
    fb.iv = round(iv_total, 4)
    return fb


def fit_all(df: pd.DataFrame, target: str, numeric_cols: list, categorical_cols: list,
            custom_edges: Optional[dict] = None) -> dict:
    custom_edges = custom_edges or {}
    y = df[target]
    fitted = {}
    for col in numeric_cols:
        fitted[col] = fit_numeric_binning(df[col], y, col, edges=custom_edges.get(col))
    for col in categorical_cols:
        fitted[col] = fit_categorical_binning(df[col], y, col)
    return fitted


def transform(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, fb in fitted.items():
        out[f"{col}_woe"] = fb.transform_series(df[col])
    return out
