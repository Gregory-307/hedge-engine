from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator

from .config import Settings

settings = Settings()

_CURVE_CACHE: dict[str, "PchipInterpolator"] = {}


def _load_knots(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse YAML knots file returning (scores, hedge) arrays."""
    with file_path.open("r", encoding="utf-8") as fh:
        data: List[dict[str, float]] = yaml.safe_load(fh)
    scores = np.array([row["score"] for row in data], dtype=float)
    hedge = np.array([row["hedge"] for row in data], dtype=float)
    return scores, hedge


def _get_spline() -> PchipInterpolator:
    """Return cached monotonic spline evaluator."""
    key = str(settings.spline_knots)
    if key not in _CURVE_CACHE:
        x, y = _load_knots(settings.spline_knots)
        _CURVE_CACHE[key] = PchipInterpolator(x, y, extrapolate=False)
    return _CURVE_CACHE[key]


def liquidity_weight(depth1pct_usd: float) -> float:
    """Weight score by order-book depth (monotonic)."""
    return min(1.0, np.log1p(depth1pct_usd) / np.log1p(10_000_000))


def compute_hedge(
    score: float,
    depth1pct_usd: float,
) -> Tuple[float, float]:
    """Return (hedge_pct, confidence) according to spline sizing logic."""
    weight = liquidity_weight(depth1pct_usd)
    effective_score = score * weight
    spline = _get_spline()

    hedge_pct = float(spline(effective_score))
    hedge_pct = max(0.0, min(settings.max_hedge_pct, hedge_pct))

    confidence = min(weight, 1.0)
    return hedge_pct, confidence 