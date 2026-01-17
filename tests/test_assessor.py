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


@pytest.fixture
def default_config() -> RiskConfig:
    """Default risk configuration."""
    return RiskConfig()


class TestPnLScenarios:
    """Test P&L calculation under different scenarios."""

    def test_long_position_pnl_exact_values(self, long_position, normal_market, default_config):
        """Long position P&L matches expected calculation."""
        pnl = calculate_pnl_scenarios(long_position, normal_market, default_config)

        # 10 BTC * $45000 * -5% = -$22,500
        assert pnl.move_down_5pct == pytest.approx(-22500, rel=0.001)
        # 10 BTC * $45000 * -2% = -$9,000
        assert pnl.move_down_2pct == pytest.approx(-9000, rel=0.001)
        # 10 BTC * $45000 * +2% = $9,000
        assert pnl.move_up_2pct == pytest.approx(9000, rel=0.001)
        # 10 BTC * $45000 * +5% = $22,500
        assert pnl.move_up_5pct == pytest.approx(22500, rel=0.001)

    def test_short_position_pnl_exact_values(self, short_position, normal_market, default_config):
        """Short position P&L is inverted from long."""
        pnl = calculate_pnl_scenarios(short_position, normal_market, default_config)

        # Short profits on down moves: -10 BTC * $45000 * -5% = +$22,500
        assert pnl.move_down_5pct == pytest.approx(22500, rel=0.001)
        # Short loses on up moves: -10 BTC * $45000 * +5% = -$22,500
        assert pnl.move_up_5pct == pytest.approx(-22500, rel=0.001)

    def test_hedged_pnl_exact_values(self, long_position, normal_market, default_config):
        """50% hedge reduces P&L by exactly 50%."""
        pnl = calculate_pnl_scenarios(long_position, normal_market, default_config, hedge_pct=0.5)

        # Hedged values are half of unhedged
        assert pnl.hedged_down_5pct == pytest.approx(-11250, rel=0.001)
        assert pnl.hedged_up_5pct == pytest.approx(11250, rel=0.001)

    def test_custom_move_percentages(self, long_position, normal_market):
        """Custom scenario move percentages are respected."""
        config = RiskConfig(large_move_pct=0.10, small_move_pct=0.03)
        pnl = calculate_pnl_scenarios(long_position, normal_market, config)

        # 10 BTC * $45000 * -10% = -$45,000
        assert pnl.move_down_5pct == pytest.approx(-45000, rel=0.001)
        # 10 BTC * $45000 * -3% = -$13,500
        assert pnl.move_down_2pct == pytest.approx(-13500, rel=0.001)


class TestHedgeCost:
    """Test hedge cost estimation."""

    def test_spread_cost_exact_calculation(self, long_position, normal_market, default_config):
        """Spread cost formula: notional * spread_bps / 10000."""
        cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0, config=default_config)

        # Notional: 10 BTC * $45000 = $450,000
        # Spread cost: $450,000 * 2 / 10000 = $90
        assert cost.spread_cost_usd == pytest.approx(90, rel=0.001)

    def test_funding_cost_calculation_long_hedge(self, long_position, normal_market, default_config):
        """Funding cost for hedging a long (shorting perp) is negative when funding > 0."""
        cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0, config=default_config)

        # Daily funding: 5% / 365 = 0.0137%
        # Funding cost: $450,000 * 0.0137% = $61.64/day
        # For long hedge (shorting perp), we RECEIVE funding, so it's negative
        expected_daily = -(450000 * 0.05 / 365)
        assert cost.funding_cost_1d_usd == pytest.approx(expected_daily, rel=0.001)

    def test_custom_days_per_year(self, long_position, normal_market):
        """Custom days_per_year affects funding calculation."""
        config = RiskConfig(days_per_year=360)  # Some markets use 360
        cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0, config=config)

        expected_daily = -(450000 * 0.05 / 360)
        assert cost.funding_cost_1d_usd == pytest.approx(expected_daily, rel=0.001)

    def test_perp_selected_when_funding_exceeds_threshold(self, long_position, default_config):
        """Perp short selected when funding rate > threshold (2%)."""
        high_funding_market = MarketConditions(
            current_price=45000.0,
            volatility_1d=0.03,
            spot_spread_bps=2.0,
            perp_funding_rate=0.05,  # 5% > 2% threshold
            bid_depth_usd=5_000_000,
            ask_depth_usd=5_000_000,
        )
        cost = estimate_hedge_cost(long_position, high_funding_market, hedge_pct=1.0, config=default_config)
        assert cost.instrument == HedgeInstrument.PERP_SHORT

    def test_spot_selected_when_funding_below_threshold(self, long_position, default_config):
        """Spot sell selected when funding rate < threshold (2%)."""
        low_funding_market = MarketConditions(
            current_price=45000.0,
            volatility_1d=0.03,
            spot_spread_bps=2.0,
            perp_funding_rate=0.01,  # 1% < 2% threshold
            bid_depth_usd=5_000_000,
            ask_depth_usd=5_000_000,
        )
        cost = estimate_hedge_cost(long_position, low_funding_market, hedge_pct=1.0, config=default_config)
        assert cost.instrument == HedgeInstrument.SPOT_SELL

    def test_partial_hedge_cost_scales_linearly(self, long_position, normal_market, default_config):
        """50% hedge has 50% of the spread cost."""
        full_cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=1.0, config=default_config)
        half_cost = estimate_hedge_cost(long_position, normal_market, hedge_pct=0.5, config=default_config)

        assert half_cost.size == pytest.approx(full_cost.size * 0.5, rel=0.001)
        assert half_cost.spread_cost_usd == pytest.approx(full_cost.spread_cost_usd * 0.5, rel=0.001)

    def test_beneficial_funding_reduces_total_cost(self, long_position, default_config):
        """When we receive funding, total cost can be less than spread cost."""
        high_funding_market = MarketConditions(
            current_price=45000.0,
            volatility_1d=0.03,
            spot_spread_bps=2.0,
            perp_funding_rate=0.20,  # 20% funding - we receive a lot
            bid_depth_usd=5_000_000,
            ask_depth_usd=5_000_000,
        )
        cost = estimate_hedge_cost(long_position, high_funding_market, hedge_pct=1.0, config=default_config)

        # Spread cost: $90
        # Funding received: $450,000 * 20% / 365 = $246.58/day (negative cost)
        # Total: $90 - $246.58 = -$156.58, but floored at 0
        assert cost.total_cost_usd == 0  # Can't be negative (floored)
        assert cost.funding_cost_1d_usd < 0  # We receive funding


class TestRiskScore:
    """Test risk score calculation with exact expected values."""

    def test_risk_score_calculation_exact(self, long_position, normal_market, default_config):
        """Verify exact risk score formula breakdown."""
        pnl = calculate_pnl_scenarios(long_position, normal_market, default_config)
        score = calculate_risk_score(long_position, normal_market, pnl, default_config)

        # Manual calculation with default config:
        # Loss severity: 5% move = $22,500 loss on $450,000 = 5% loss
        #   vol_adjustment = 0.03 / 0.05 = 0.6
        #   loss_pct = 0.05 >= trigger (0.02), so: 15 + (0.03/0.03)*20 = 35
        #   loss_score = min(35, 35 * 0.6) = 21
        # Age: 60/240 = 0.25 * 10 = 2.5 (under urgent threshold)
        # Volatility: 0.03/0.08 * 25 = 9.375
        # Liquidity: sqrt(450000/5000000) * 20 = sqrt(0.09) * 20 = 6.0
        # Total: 21 + 2.5 + 9.375 + 6.0 = 38.875

        assert score == pytest.approx(38.875, abs=0.01)

    def test_risk_score_with_custom_weights(self, long_position, normal_market):
        """Custom weight configuration changes score components."""
        # Increase loss weight, decrease others
        config = RiskConfig(
            loss_weight=50.0, age_weight=15.0, vol_weight=20.0, liquidity_weight=15.0,
            loss_trigger_points=20.0  # Scale trigger points proportionally
        )
        pnl = calculate_pnl_scenarios(long_position, normal_market, config)
        score = calculate_risk_score(long_position, normal_market, pnl, config)

        # With higher loss weight, score should be different
        # loss_score = min(50, (20 + (0.03/0.03)*30) * 0.6) = min(50, 50*0.6) = 30
        # age = 60/240 * (15/2) = 1.875 -- NO, age_normal_points defaults to 10
        # age = 60/240 * 10 = 2.5, but max is 15 so within range
        # Actually age calc: position under urgent, so age_score = (60/240) * age_normal_points
        # Since age_normal_points defaults to 10: 0.25 * 10 = 2.5
        # vol = 0.03/0.08 * 20 = 7.5
        # liq = sqrt(0.09) * 15 = 4.5
        # Total: 30 + 2.5 + 7.5 + 4.5 = 44.5
        assert score == pytest.approx(44.5, abs=0.1)

    def test_risk_score_increases_with_volatility(self, long_position, default_config):
        """Higher volatility increases risk score."""
        low_vol_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.02,
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )
        high_vol_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.08,
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        low_pnl = calculate_pnl_scenarios(long_position, low_vol_market, default_config)
        high_pnl = calculate_pnl_scenarios(long_position, high_vol_market, default_config)

        low_score = calculate_risk_score(long_position, low_vol_market, low_pnl, default_config)
        high_score = calculate_risk_score(long_position, high_vol_market, high_pnl, default_config)

        assert high_score > low_score
        # Vol affects both loss_score multiplier AND vol_score component
        assert high_score - low_score > 15  # Significant difference due to double-counting

    def test_risk_score_increases_with_age(self, normal_market, default_config):
        """Older positions have higher risk scores."""
        young_position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=30)
        old_position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=400)

        young_pnl = calculate_pnl_scenarios(young_position, normal_market, default_config)
        old_pnl = calculate_pnl_scenarios(old_position, normal_market, default_config)

        young_score = calculate_risk_score(young_position, normal_market, young_pnl, default_config)
        old_score = calculate_risk_score(old_position, normal_market, old_pnl, default_config)

        # Age component: young = 30/240 * 10 = 1.25
        # old = 10 + (160/240)*10 = 10 + 6.67 = 16.67
        # Difference: 16.67 - 1.25 = 15.42
        assert old_score > young_score
        assert old_score - young_score == pytest.approx(15.42, abs=0.1)

    def test_risk_score_bounded_0_100(self, long_position, high_vol_market, default_config):
        """Risk score is always bounded between 0 and 100."""
        pnl = calculate_pnl_scenarios(long_position, high_vol_market, default_config)
        score = calculate_risk_score(long_position, high_vol_market, pnl, default_config)

        assert 0 <= score <= 100

    def test_zero_position_has_zero_risk(self, normal_market, default_config):
        """Position with size=0 should have zero risk score."""
        flat = InventoryPosition(asset="BTC", size=0, entry_price=45000.0)
        pnl = calculate_pnl_scenarios(flat, normal_market, default_config)
        score = calculate_risk_score(flat, normal_market, pnl, default_config)

        assert score == 0.0


class TestAssessInventoryRisk:
    """Test the main assessment function."""

    def test_hold_recommendation_for_low_risk(self, long_position, normal_market):
        """Low risk positions (score < 40) get HOLD recommendation."""
        result = assess_inventory_risk(long_position, normal_market)

        assert result.action == Action.HOLD
        assert result.hedge_pct == 0.0
        assert result.risk_score < 40

    def test_reduce_recommendation_deterministic(self):
        """Position engineered to be in REDUCE range (40-54) gets REDUCE."""
        # Carefully tuned to get score ~45
        position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=180)
        market = MarketConditions(
            current_price=45000.0, volatility_1d=0.04,
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=4_000_000, ask_depth_usd=4_000_000,
        )

        result = assess_inventory_risk(position, market)

        # Verify we hit the REDUCE range
        assert 40 <= result.risk_score < 55, f"Score {result.risk_score} not in REDUCE range"
        assert result.action == Action.REDUCE
        assert result.hedge_pct == 0.25

    def test_hedge_partial_deterministic(self):
        """Position engineered to be in HEDGE_PARTIAL range (55-69)."""
        # Lower vol and better liquidity to get score in 55-69 range
        position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=280)
        market = MarketConditions(
            current_price=45000.0, volatility_1d=0.045,
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=3_500_000, ask_depth_usd=3_500_000,
        )

        result = assess_inventory_risk(position, market)

        assert 55 <= result.risk_score < 70, f"Score {result.risk_score} not in HEDGE_PARTIAL range"
        assert result.action == Action.HEDGE_PARTIAL
        assert result.hedge_pct == 0.5

    def test_hedge_full_deterministic(self):
        """Position engineered to be in HEDGE_FULL range (70-84)."""
        # Tuned params to get score ~77: reduce vol and age, improve liquidity
        position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=380)
        market = MarketConditions(
            current_price=45000.0, volatility_1d=0.055,
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=2_000_000, ask_depth_usd=2_000_000,
        )

        result = assess_inventory_risk(position, market)

        assert 70 <= result.risk_score < 85, f"Score {result.risk_score} not in HEDGE_FULL range"
        assert result.action == Action.HEDGE_FULL
        assert result.hedge_pct == 1.0

    def test_liquidate_deterministic(self):
        """Position engineered to trigger LIQUIDATE (score >= 85)."""
        # Extreme conditions: old position, high vol, low liquidity
        position = InventoryPosition(asset="BTC", size=50.0, entry_price=45000.0, age_minutes=480)
        market = MarketConditions(
            current_price=45000.0, volatility_1d=0.10,  # Extreme vol
            spot_spread_bps=5.0, perp_funding_rate=0.05,
            bid_depth_usd=500_000, ask_depth_usd=500_000,  # Very thin
        )

        result = assess_inventory_risk(position, market)

        assert result.risk_score >= 85, f"Score {result.risk_score} should trigger LIQUIDATE"
        assert result.action == Action.LIQUIDATE
        assert result.hedge_pct == 1.0
        assert "critical" in result.reasoning.lower()

    def test_hedge_cost_too_high_returns_hold(self):
        """When hedge cost exceeds max_hedge_cost_bps, action is HOLD."""
        position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=300)
        # Wide spread makes hedging expensive
        expensive_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.05,
            spot_spread_bps=100.0,  # 100bps spread = very expensive
            perp_funding_rate=0.05,
            bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        result = assess_inventory_risk(position, expensive_market)

        # Should want to hedge based on risk, but cost is prohibitive
        assert result.action == Action.HOLD
        assert result.hedge_pct == 0.0
        assert "cost too high" in result.reasoning.lower()

    def test_reasoning_contains_risk_score(self, long_position, normal_market):
        """Assessment reasoning includes the risk score value."""
        result = assess_inventory_risk(long_position, normal_market)

        assert "Risk score" in result.reasoning
        assert str(round(result.risk_score)) in result.reasoning

    def test_re_evaluate_time_uses_config_thresholds(self, long_position, normal_market, high_vol_market):
        """Re-evaluation times come from config thresholds."""
        config = RiskConfig()
        low_risk_result = assess_inventory_risk(long_position, normal_market, config)

        long_position.age_minutes = 400
        high_risk_result = assess_inventory_risk(long_position, high_vol_market, config)

        # Verify specific config values are used
        assert low_risk_result.re_evaluate_minutes == config.low_risk_reeval_minutes
        if high_risk_result.risk_score >= config.high_risk_reeval_threshold:
            assert high_risk_result.re_evaluate_minutes == config.high_risk_reeval_minutes

    def test_custom_config_changes_thresholds(self, long_position, normal_market):
        """Custom config with lower thresholds triggers action sooner."""
        strict_config = RiskConfig(
            max_loss_pct=0.01, hedge_trigger_loss_pct=0.005,
            max_hold_minutes=60, urgent_hold_minutes=30,
            reduce_threshold=20,
        )

        strict_result = assess_inventory_risk(long_position, normal_market, strict_config)
        default_result = assess_inventory_risk(long_position, normal_market)

        assert strict_result.risk_score >= default_result.risk_score

    def test_zero_size_position_returns_hold(self, normal_market):
        """Zero-size (flat) position returns HOLD with zero risk."""
        flat_position = InventoryPosition(asset="BTC", size=0.0, entry_price=45000.0)

        result = assess_inventory_risk(flat_position, normal_market)

        assert result.action == Action.HOLD
        assert result.risk_score == 0.0
        assert result.hedge_pct == 0.0
        assert result.pnl_scenarios.move_down_5pct == 0
        assert "size = 0" in result.reasoning

    def test_position_notional_calculated_correctly(self, long_position, normal_market):
        """Position notional uses current price, not entry price."""
        result = assess_inventory_risk(long_position, normal_market)

        # Notional = size * current_price = 10 * 45000 = 450000
        assert result.position_notional_usd == pytest.approx(450000, rel=0.001)


class TestHedgeInstrumentSelection:
    """Test that correct hedge instrument is selected based on funding."""

    def test_short_position_perp_long_on_negative_funding(self, short_position, default_config):
        """Short position uses perp long when funding is significantly negative."""
        negative_funding = MarketConditions(
            current_price=45000.0, volatility_1d=0.03, spot_spread_bps=2.0,
            perp_funding_rate=-0.05, bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        cost = estimate_hedge_cost(short_position, negative_funding, 1.0, default_config)
        assert cost.instrument == HedgeInstrument.PERP_LONG

    def test_short_position_spot_buy_on_neutral_funding(self, short_position, default_config):
        """Short position uses spot buy when funding is not significantly negative."""
        neutral_funding = MarketConditions(
            current_price=45000.0, volatility_1d=0.03, spot_spread_bps=2.0,
            perp_funding_rate=-0.01, bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        cost = estimate_hedge_cost(short_position, neutral_funding, 1.0, default_config)
        assert cost.instrument == HedgeInstrument.SPOT_BUY

    def test_custom_funding_threshold(self, long_position):
        """Custom funding threshold changes instrument selection."""
        market = MarketConditions(
            current_price=45000.0, volatility_1d=0.03, spot_spread_bps=2.0,
            perp_funding_rate=0.03, bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        # Default threshold (2%): 3% > 2%, select perp
        default_config = RiskConfig()
        assert estimate_hedge_cost(long_position, market, 1.0, default_config).instrument == HedgeInstrument.PERP_SHORT

        # Higher threshold (5%): 3% < 5%, select spot
        strict_config = RiskConfig(funding_rate_threshold=0.05)
        assert estimate_hedge_cost(long_position, market, 1.0, strict_config).instrument == HedgeInstrument.SPOT_SELL


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_position(self, normal_market, default_config):
        """Very small position still has base risk components."""
        tiny_position = InventoryPosition(asset="BTC", size=0.001, entry_price=45000.0, age_minutes=0)
        pnl = calculate_pnl_scenarios(tiny_position, normal_market, default_config)
        score = calculate_risk_score(tiny_position, normal_market, pnl, default_config)

        # Even tiny positions have: loss_score (21) + vol_score (9.375) + liquidity (~0)
        # age = 0, so age_score = 0
        assert score == pytest.approx(30.4, abs=0.1)

    def test_very_large_position_vs_liquidity(self, default_config):
        """Position larger than book depth has high liquidity risk."""
        large_position = InventoryPosition(asset="BTC", size=100.0, entry_price=45000.0, age_minutes=0)
        thin_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.03, spot_spread_bps=2.0,
            perp_funding_rate=0.05, bid_depth_usd=1_000_000, ask_depth_usd=1_000_000,
        )

        pnl = calculate_pnl_scenarios(large_position, thin_market, default_config)
        score = calculate_risk_score(large_position, thin_market, pnl, default_config)

        # Liquidity component: sqrt(4500000/1000000) * 20 = sqrt(4.5) * 20 = 42.4, capped at 20
        assert score > 40  # Elevated due to liquidity

    def test_position_at_max_age(self, normal_market, default_config):
        """Position at max age has max age score component."""
        old_position = InventoryPosition(asset="BTC", size=10.0, entry_price=45000.0, age_minutes=480)
        pnl = calculate_pnl_scenarios(old_position, normal_market, default_config)
        score = calculate_risk_score(old_position, normal_market, pnl, default_config)

        # Age at max (480): age_score = 10 + (240/240)*10 = 20 (maxed)
        # Total: 21 + 20 + 9.375 + 6 = 56.375
        assert score == pytest.approx(56.4, abs=0.1)

    def test_zero_volatility_uses_max_risk_adjustment(self, long_position, default_config):
        """Zero volatility uses vol_adj_max for loss severity."""
        zero_vol_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.0, spot_spread_bps=2.0,
            perp_funding_rate=0.05, bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        pnl = calculate_pnl_scenarios(long_position, zero_vol_market, default_config)
        score = calculate_risk_score(long_position, zero_vol_market, pnl, default_config)

        # Zero vol: vol_adjustment = 2.0 (max) -> loss_score = min(35, 35*2.0) = 35
        # Vol regime component = 0 (0/0.08 * 25 = 0)
        # Total: 35 + 2.5 + 0 + 6 = 43.5
        assert score == pytest.approx(43.5, abs=0.1)

    def test_custom_vol_bounds(self, long_position):
        """Custom vol_adj_min and vol_adj_max are respected."""
        config = RiskConfig(vol_adj_min=0.5, vol_adj_max=1.5)

        low_vol_market = MarketConditions(
            current_price=45000.0, volatility_1d=0.01,  # Would give 0.2 adjustment
            spot_spread_bps=2.0, perp_funding_rate=0.05,
            bid_depth_usd=5_000_000, ask_depth_usd=5_000_000,
        )

        pnl = calculate_pnl_scenarios(long_position, low_vol_market, config)
        score = calculate_risk_score(long_position, low_vol_market, pnl, config)

        # Vol adjustment should be floored at 0.5 (not 0.2)
        # loss_score = min(35, 35 * 0.5) = 17.5
        # age = 2.5, vol_regime = 0.01/0.08*25 = 3.125, liq = 6
        # Total: 17.5 + 2.5 + 3.125 + 6 = 29.125
        assert score == pytest.approx(29.125, abs=0.1)
