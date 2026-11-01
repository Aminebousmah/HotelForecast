"""Pure accuracy metrics. All inputs are array-likes of identical length."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(actual: ArrayLike, forecast: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {f.shape}")
    return a, f


def wape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Weighted absolute percentage error: sum|a - f| / sum(a)."""
    a, f = _pair(actual, forecast)
    denominator = np.abs(a).sum()
    return float(np.abs(a - f).sum() / denominator) if denominator else float("nan")


def mape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Mean absolute percentage error, ignoring zero actuals."""
    a, f = _pair(actual, forecast)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask]))) if mask.any() else float("nan")


def smape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Symmetric MAPE, bounded in [0, 2]."""
    a, f = _pair(actual, forecast)
    denominator = np.abs(a) + np.abs(f)
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(a[mask] - f[mask]) / denominator[mask]))


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Relative bias: positive means over-forecasting."""
    a, f = _pair(actual, forecast)
    denominator = a.sum()
    return float((f - a).sum() / denominator) if denominator else float("nan")


def coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of actuals falling inside [lower, upper]."""
    a, lo = _pair(actual, lower)
    a, hi = _pair(actual, upper)
    return float(np.mean((a >= lo) & (a <= hi)))
