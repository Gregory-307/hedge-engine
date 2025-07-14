from typing import Tuple


def compute_hedge(score: float) -> Tuple[float, float]:
    """Temporary stub: returns fixed hedge pct and confidence."""
    hedge_pct = max(0.05, min(1.0, score))
    confidence = score
    return hedge_pct, confidence 