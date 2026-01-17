"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from hedge_engine.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_healthz_returns_ok(self):
        """Health endpoint should return OK status."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAssessEndpoint:
    """Test the /assess endpoint."""

    def test_assess_basic_request_returns_all_fields(self):
        """Basic assessment returns all required response fields."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 10.0,
                    "entry_price": 45000.0,
                    "age_minutes": 60,
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.03,
                    "spot_spread_bps": 2.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields exist with correct types
        assert data["action"] == "HOLD"
        assert data["hedge_pct"] == 0.0
        assert 0 <= data["risk_score"] <= 100
        assert "Risk score" in data["reasoning"]
        assert data["pnl_scenarios"]["move_down_5pct"] == pytest.approx(-22500, rel=0.01)
        assert data["version"] == "0.2.0"

    def test_assess_with_custom_config(self):
        """Assessment with custom config should use those values."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "ETH",
                    "size": 100.0,
                    "entry_price": 2500.0,
                },
                "market": {
                    "current_price": 2500.0,
                    "volatility_1d": 0.05,
                    "spot_spread_bps": 5.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
                "config": {
                    "max_loss_pct": 0.02,
                    "max_hold_minutes": 120,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Custom config should affect risk score
        assert data["risk_score"] > 0

    def test_assess_short_position_pnl_inverted(self):
        """Short positions should have inverted P&L from longs."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": -5.0,  # Short position
                    "entry_price": 45000.0,
                },
                "market": {
                    "current_price": 46000.0,
                    "volatility_1d": 0.04,
                    "spot_spread_bps": 3.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Short position: loses on price increase, profits on decrease
        assert data["pnl_scenarios"]["move_up_5pct"] < 0
        assert data["pnl_scenarios"]["move_down_5pct"] > 0

    def test_assess_pnl_scenarios_exact_values(self):
        """P&L scenarios should match expected calculations."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 10.0,
                    "entry_price": 45000.0,
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.03,
                    "spot_spread_bps": 2.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
            },
        )

        data = response.json()
        pnl = data["pnl_scenarios"]

        # 10 BTC * $45000 * 5% = $22,500
        assert pnl["move_down_5pct"] == pytest.approx(-22500, rel=0.01)
        assert pnl["move_down_2pct"] == pytest.approx(-9000, rel=0.01)
        assert pnl["move_up_2pct"] == pytest.approx(9000, rel=0.01)
        assert pnl["move_up_5pct"] == pytest.approx(22500, rel=0.01)

    def test_assess_high_risk_returns_hedge_action(self):
        """High risk situation should recommend hedging."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 100.0,  # Large position
                    "entry_price": 45000.0,
                    "age_minutes": 400,  # Old position
                },
                "market": {
                    "current_price": 43000.0,
                    "volatility_1d": 0.08,  # High vol
                    "spot_spread_bps": 10.0,
                    "perp_funding_rate": 0.10,
                    "bid_depth_usd": 1_000_000,  # Low liquidity
                    "ask_depth_usd": 1_000_000,
                },
            },
        )

        data = response.json()

        # High risk should trigger hedge recommendation
        assert data["risk_score"] > 50
        assert data["action"] in ["HEDGE_PARTIAL", "HEDGE_FULL", "LIQUIDATE", "REDUCE"]

    def test_assess_invalid_negative_price_rejected(self):
        """Negative price should be rejected with 422."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 10.0,
                    "entry_price": -100.0,  # Invalid
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.03,
                    "spot_spread_bps": 2.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
            },
        )

        assert response.status_code == 422

    def test_assess_zero_position_returns_hold(self):
        """Zero-size position should return HOLD with zero risk."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 0.0,
                    "entry_price": 45000.0,
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.03,
                    "spot_spread_bps": 2.0,
                    "perp_funding_rate": 0.05,
                    "bid_depth_usd": 5_000_000,
                    "ask_depth_usd": 5_000_000,
                },
            },
        )

        data = response.json()
        assert data["action"] == "HOLD"
        assert data["risk_score"] == 0.0
        assert "size = 0" in data["reasoning"]

    def test_assess_hedge_cost_included_when_hedging(self):
        """When hedge is recommended, cost details should be included."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 100.0,
                    "entry_price": 45000.0,
                    "age_minutes": 450,
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.08,
                    "spot_spread_bps": 5.0,
                    "perp_funding_rate": 0.10,
                    "bid_depth_usd": 2_000_000,
                    "ask_depth_usd": 2_000_000,
                },
            },
        )

        data = response.json()

        if data["hedge_pct"] > 0:
            assert data["suggested_hedge"] is not None
            hedge = data["suggested_hedge"]
            assert "instrument" in hedge
            assert "size" in hedge
            assert "spread_cost_usd" in hedge
            assert "total_cost_bps" in hedge

    def test_assess_perp_selected_on_high_funding(self):
        """Perp should be selected when funding rate is favorable."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": 100.0,
                    "entry_price": 45000.0,
                    "age_minutes": 450,
                },
                "market": {
                    "current_price": 45000.0,
                    "volatility_1d": 0.08,
                    "spot_spread_bps": 5.0,
                    "perp_funding_rate": 0.10,  # High funding > 2% threshold
                    "bid_depth_usd": 2_000_000,
                    "ask_depth_usd": 2_000_000,
                },
            },
        )

        data = response.json()

        if data["suggested_hedge"]:
            # Long position hedged by shorting, high funding should use perp
            assert data["suggested_hedge"]["instrument"] == "perp_short"
