import timeit

from hedge_engine.sizer import compute_hedge


def test_compute_hedge_speed():
    # Warm-up call (compile/interpreter caches)
    compute_hedge(0.5, depth1pct_usd=5_000_000)

    repetitions = 10_000
    duration = timeit.timeit(
        stmt="compute_hedge(0.42, depth1pct_usd=5_000_000)",
        globals={"compute_hedge": compute_hedge},
        number=repetitions,
    )
    avg_us = (duration / repetitions) * 1e6
    # Allow generous threshold for CI variability; target is <5 µs median locally.
    assert avg_us < 50, f"compute_hedge too slow: {avg_us:.2f} µs > 50 µs" 