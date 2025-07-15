import numpy as np

from hedge_engine.sizer import compute_hedge


def test_boundary_values():
    hedge_low, _ = compute_hedge(score=0.0, depth1pct_usd=5_000_000)
    hedge_high, _ = compute_hedge(score=1.0, depth1pct_usd=5_000_000)
    assert hedge_low >= 0.05
    assert abs(hedge_high - 1.0) < 1e-6


def test_monotonicity():
    scores = np.linspace(0.0, 1.0, 20)
    hedge_pcts = [compute_hedge(s, depth1pct_usd=5_000_000)[0] for s in scores]
    assert all(earlier <= later for earlier, later in zip(hedge_pcts, hedge_pcts[1:])) 