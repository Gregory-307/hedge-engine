from hypothesis import given, strategies as st

from hedge_engine.sizer import compute_hedge


@given(score=st.floats(min_value=0.0, max_value=1.0))
def test_bounds(score: float):
    hedge, _ = compute_hedge(score, depth1pct_usd=5_000_000)
    assert 0.05 <= hedge <= 1.0


@given(s1=st.floats(min_value=0.0, max_value=1.0), s2=st.floats(min_value=0.0, max_value=1.0))
def test_monotone(s1: float, s2: float):
    hedge1, _ = compute_hedge(s1, depth1pct_usd=5_000_000)
    hedge2, _ = compute_hedge(s2, depth1pct_usd=5_000_000)
    if s1 < s2:
        assert hedge1 <= hedge2
    elif s2 < s1:
        assert hedge2 <= hedge1
    else:
        assert hedge1 == hedge2 