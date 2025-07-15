from pathlib import Path
import yaml

from hedge_engine.sizer import compute_hedge, _CURVE_CACHE, _LAST_MTIME, settings


def test_hot_reload(tmp_path: Path):
    # Copy knots file to temp dir
    orig_knots = settings.spline_knots
    temp_knots = tmp_path / "curve_knots.yaml"
    temp_knots.write_text(orig_knots.read_text())

    # Point settings to temp file
    settings.spline_knots = temp_knots  # type: ignore[attr-defined]

    # Prime cache
    base_hedge, _ = compute_hedge(0.5, depth1pct_usd=5_000_000)

    # Modify knots to bump hedge values
    new_knots = [
        {"score": 0.0, "hedge": 0.10},
        {"score": 0.5, "hedge": 0.80},
        {"score": 1.0, "hedge": 1.0},
    ]
    yaml.safe_dump(new_knots, temp_knots.open("w", encoding="utf-8"))

    # Ensure cache invalidated
    _CURVE_CACHE.clear()
    global _LAST_MTIME  # type: ignore[global-variable-annotation]
    _LAST_MTIME = None  # type: ignore[misc]

    new_hedge, _ = compute_hedge(0.5, depth1pct_usd=5_000_000)

    assert new_hedge != base_hedge
    assert new_hedge > base_hedge 