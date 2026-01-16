"""Tests for the inventory risk assessor."""

import pytest
from hedge_engine.assessor import (
    assess_inventory_risk,
    calculate_pnl_scenarios,
    estimate_hedge_cost,
    calculate_risk_score,
)
from hedge_engine.models import (
    Action,
    HedgeInstrument,
    InventoryPosition,
    MarketConditions,
    RiskConfig,
)


@pytest.fixture
def long_position() -> InventoryPosition:
    """A typical long BTC position."""
    return InventoryPosition(
        asset="BTC",
        size=10.0,  # 10 BTC long
        entry_price=45000.0,
        age_minutes=60,
    )


@pytest.fixture
def short_position() -> InventoryPosition:
    """A typical short BTC position."""
    return InventoryPosition(
        asset="BTC",
        size=-10.0,  # 10 BTC short
        entry_price=45000.0,
        age_minutes=60,
    )


@pytest.fixture
def normal_market() -> MarketConditions:
    """Normal market conditions."""
    return MarketConditions(
        current_price=45000.0,
        volatility_1d=0.03,  # 3% daily vol
        spot_spread_bps=2.0,
        perp_funding_rate=0.05,  # 5% annual
        bid_depth_usd=5_000_000,
        ask_depth_usd=5_000_000,
    )


@pytest.fixture
def high_vol_market() -> MarketConditions:
    """High volatility market."""
    return MarketConditions(
        current_price=44000.0,  # Price moved down
        volatility_1d=0.08,  # 8% daily vol (extreme)
        spot_spread_bps=10.0,  # Wider spreads
        perp_funding_rate=0.10,  # High funding
        bid_depth_usd=2_000_000,  # Less liquidity
        ask_depth_usd=2_000_000,
    )


class TestPnLScenarios:
    """Test P&L calculation under different scenarios."""

    def test_long_position_pnl_down(self, long_position, normal_market):
        """Long position loses on down moves."""
        pnl = calculate_pnl_scenarios(long_position, normal_market)

        # 10 BTC * $45000 * -5% = -$22,500
        assert pnl.move_down_5pct == pytest.approx(-22500, rel=0.01)
        assert pnl.move_down_2pct == pytest.approx(-9000, rel=0.01)

    def test_long_position_pnl_up(self, long_position, normal_market):
        """Long position profits on up moves."""
        pnl = calculate_pnl_scenarios(long_position, normal_market)

        assert pnl.move_up_5pct == pytest.approx(22500, rel=0.01)
        assert pnl.move_up_2pct == pytest.approx(9000, rel=0.01)

    def test_short_position_pnl_down(self, short_position, normal_market):
        """Short position profits on down moves."""
        pnl = calculate_pnl_scenarios(short_position, normal_market)

        # Short profits when price goes down
        assert pnl.move_down_5pct == pytest.approx(22500, rel=0.01)

    def test_short_position_pnl_up(self, short_position, normal_market):
        """Short position loses on up moves."""
        pnl = calculate_pnl_scenarios(short_position, normal_market)

        # Short loses when price goes up
        assert pnl.move_up_5pct == pytest.approx(-22500, rel=0.01)

    def test_hedged_pnl_reduced(self, long_position, normal_market):
        """Hedging reduces P&L exposure."""
        pnl = calculate_pnl_scenarios(long_position, normal_market, hedge_pct=0.5)

        # 50% hedge means 50% of the exposure
        assert pnl.hedged_down_5pct == pytest.approx(-11250, rel=0.01)
        assert pnl.hedged_up_5pct == pytest.approx(11250, rel=0.01)


class TestHedgeCost:
    """Test hedge cost estimation."""

    def test_spread_cost_calculation(self, long_position, normal_market):
        """Spread cost is correctly calculated."""
        cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0)

        # Notional: 10 BTC * $45000 = $450,000
        # Spread cost: $450,000 * 2bps = $90
        assert cost.spread_cost_usd == pytest.approx(90, rel=0.01)

    def test_perp_instrument_selected_high_funding(self, long_position, normal_market):
        """Perp selected when funding is favorable for the hedge."""
        # High positive funding means shorting perp earns funding
        cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0)
        assert cost.instrument == HedgeInstrument.PERP_SHORT

    def test_spot_instrument_selected_low_funding(self, long_position):
        """Spot selected when funding is not favorable."""
        low_funding_market = MarketConditions(
            current_price=45000.0,
            volatility_1d=0.03,
            spot_spread_bps=2.0,
            perp_funding_rate=0.01,  # Low funding
            bid_depth_usd=5_000_000,
            ask_depth_usd=5_000_000,
        )
        cost = estimate_hedge_cost(long_position, low_funding_market, hedge_pct=1.0)
        assert cost.instrument == HedgeInstrument.SPOT_SELL

    def test_partial_hedge_cost_proportional(self, long_position, normal_market):
        """Partial hedge cost is proportional to hedge size."""
        full_cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0)
        half_cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=0.5)

        assert half_cost.size == pytest.approx(full_cost.size * 0.5, rel=0.01)
        assert half_cost.spread_cost_usd == pytest.approx(
            full_cost.spread_cost_usd * 0.5, rel=0.01
        )


class TestRiskScore:
    """Test risk score calculation."""

    def test_low_risk_score_normal_conditions(self, long_position, normal_market):
        """Normal conditions should have low risk score."""
        pnl = calculate_pnl_scenarios(long_position, normal_market)
        config = RiskConfig()
        score = calculate_risk_score(long_position, normal_market, pnl, config)

        # Normal vol, reasonable position age, good liquidity
        assert score < 50

    def test_high_risk_score_extreme_vol(self, long_position, high_vol_market):
        """Extreme volatility should increase risk score."""
        pnl = calculate_pnl_scenarios(long_position, high_vol_market)
        config = RiskConfig()
        score = calculate_risk_score(long_position, high_vol_market, pnl, config)

        # High vol should push score up
        assert score > 40

    def test_risk_score_increases_with_age(self, normal_market):
        """Older positions have higher risk scores."""
        young_position = InventoryPosition(
            asset="BTC", size=10.0, entry_price=45000.0, age_minutes=30
        )
        old_position = InventoryPosition(
            asset="BTC", size=10.0, entry_price=45000.0, age_minutes=400
        )

        config = RiskConfig()
        young_pnl = calculate_pnl_scenarios(young_position, normal_market)
        old_pnl = calculate_pnl_scenarios(old_position, normal_market)

        young_score = calculate_risk_score(young_position, normal_market, young_pnl, config)
        old_score = calculate_risk_score(old_position, normal_market, old_pnl, config)

        assert old_score > young_score

    def test_risk_score_bounded(self, long_position, high_vol_market):
        """Risk score should be bounded 0-100."""
        pnl = calculate_pnl_scenarios(long_position, high_vol_market)
        config = RiskConfig()
        score = calculate_risk_score(long_position, high_vol_market, pnl, config)

        assert 0 <= score <= 100


class TestAssessInventoryRisk:
    """Test the main assessment function."""

    def test_hold_recommendation_low_risk(self, long_position, normal_market):
        """Low risk positions should get HOLD recommendation."""
        result = assess_inventory_risk(long_position, normal_market)

        assert result.action == Action.HOLD
        assert result.hedge_pct == 0.0
        assert result.risk_score < 50

    def test_hedge_recommendation_high_vol(self, long_position, high_vol_market):
        """High volatility should trigger hedge recommendation."""
        # Make position older to increase risk
        long_position.age_minutes = 300

        result = assess_inventory_risk(long_position, high_vol_market)

        # Should recommend some hedging
        assert result.action in [Action.HEDGE_PARTIAL, Action.HEDGE_FULL, Action.REDUCE]
        assert result.hedge_pct > 0

    def test_reasoning_is_provided(self, long_position, normal_market):
        """Assessment should provide reasoning."""
        result = assess_inventory_risk(long_position, normal_market)

        assert len(result.reasoning) > 0
        assert "Risk score" in result.reasoning

    def test_re_evaluate_time_based_on_risk(self, long_position, normal_market, high_vol_market):
        """Higher risk should have shorter re-evaluation time."""
        low_risk_result = assess_inventory_risk(long_position, normal_market)

        long_position.age_minutes = 300
        high_risk_result = assess_inventory_risk(long_position, high_vol_market)

        # High risk should check more frequently
        assert high_risk_result.re_evaluate_minutes <= low_risk_result.re_evaluate_minutes

    def test_pnl_scenarios_included(self, long_position, normal_market):
        """Assessment should include P&L scenarios."""
        result = assess_inventory_risk(long_position, normal_market)

        assert result.pnl_scenarios is not None
        assert result.pnl_scenarios.move_down_5pct != 0
        assert result.pnl_scenarios.move_up_5pct != 0

    def test_custom_config_respected(self, long_position, normal_market):
        """Custom risk config should be respected."""
        strict_config = RiskConfig(
            max_loss_pct=0.01,  # Very strict: 1% max loss
            max_hold_minutes=60,  # Short hold time
        )

        result = assess_inventory_risk(long_position, normal_market, strict_config)

        # Stricter config should increase risk score
        default_result = assess_inventory_risk(long_position, normal_market)
        assert result.risk_score >= default_result.risk_score
