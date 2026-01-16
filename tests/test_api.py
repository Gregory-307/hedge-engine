"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from hedge_engine.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_healthz_returns_ok(self):
        """Health endpoint should return OK."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAssessEndpoint:
    """Test the /assess endpoint."""

    def test_assess_basic_request(self):
        """Basic assessment request should succeed."""
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
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "action" in data
        assert "hedge_pct" in data
        assert "risk_score" in data
        assert "reasoning" in data
        assert "pnl_scenarios" in data
        assert "version" in data

    def test_assess_with_custom_config(self):
        """Assessment with custom config should succeed."""
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
                },
                "config": {
                    "max_loss_pct": 0.02,
                    "max_hold_minutes": 120,
                },
            },
        )

        assert response.status_code == 200

    def test_assess_short_position(self):
        """Short positions should be handled correctly."""
        response = client.post(
            "/assess",
            json={
                "position": {
                    "asset": "BTC",
                    "size": -5.0,  # Short position
                    "entry_price": 45000.0,
                },
                "market": {
                    "current_price": 46000.0,  # Price moved against us
                    "volatility_1d": 0.04,
                    "spot_spread_bps": 3.0,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Short position should show loss on price increase
        assert data["pnl_scenarios"]["move_up_5pct"] < 0

    def test_assess_returns_pnl_scenarios(self):
        """Assessment should return all P&L scenarios."""
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
                },
            },
        )

        data = response.json()
        pnl = data["pnl_scenarios"]

        # All scenario fields should be present
        assert "move_down_5pct" in pnl
        assert "move_down_2pct" in pnl
        assert "move_up_2pct" in pnl
        assert "move_up_5pct" in pnl
        assert "hedged_down_5pct" in pnl
        assert "hedged_up_5pct" in pnl

    def test_assess_high_risk_returns_hedge(self):
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
                    "current_price": 43000.0,  # Price dropped
                    "volatility_1d": 0.08,  # High vol
                    "spot_spread_bps": 10.0,
                    "bid_depth_usd": 1_000_000,  # Low liquidity relative to position
                    "ask_depth_usd": 1_000_000,
                },
            },
        )

        data = response.json()

        # High risk should trigger hedge recommendation
        assert data["risk_score"] > 50
        assert data["action"] in ["HEDGE_PARTIAL", "HEDGE_FULL", "LIQUIDATE", "REDUCE"]

    def test_assess_invalid_price_rejected(self):
        """Invalid price should be rejected."""
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
                },
            },
        )

        assert response.status_code == 422  # Validation error

    def test_assess_includes_version(self):
        """Response should include API version."""
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
                },
            },
        )

        data = response.json()
        assert data["version"] == "0.2.0"
