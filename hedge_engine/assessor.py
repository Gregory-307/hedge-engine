"""Core inventory risk assessment logic."""

from __future__ import annotations

from .models import (
    Action,
    HedgeCost,
    HedgeInstrument,
    HedgeRecommendation,
    InventoryPosition,
    MarketConditions,
    PnLScenarios,
    RiskConfig,
)


def calculate_pnl_scenarios(
    position: InventoryPosition,
    market: MarketConditions,
    hedge_pct: float = 0.0,
) -> PnLScenarios:
    """
    Calculate P&L under different market move scenarios.

    Args:
        position: Current inventory position
        market: Current market conditions
        hedge_pct: Fraction of position hedged (0-1)
    """
    size = position.size
    current_price = market.current_price

    # P&L calculation: size already encodes direction
    # Long (size > 0): profits on up moves, loses on down moves
    # Short (size < 0): profits on down moves, loses on up moves
    def pnl_at_move(move_pct: float) -> float:
        price_change = current_price * move_pct
        return size * price_change

    unhedged_down_5 = pnl_at_move(-0.05)
    unhedged_down_2 = pnl_at_move(-0.02)
    unhedged_up_2 = pnl_at_move(0.02)
    unhedged_up_5 = pnl_at_move(0.05)

    # Hedged P&L (hedge reduces exposure by hedge_pct)
    unhedged_fraction = 1.0 - hedge_pct
    hedged_down_5 = unhedged_down_5 * unhedged_fraction
    hedged_down_2 = unhedged_down_2 * unhedged_fraction
    hedged_up_2 = unhedged_up_2 * unhedged_fraction
    hedged_up_5 = unhedged_up_5 * unhedged_fraction

    return PnLScenarios(
        move_down_5pct=unhedged_down_5,
        move_down_2pct=unhedged_down_2,
        move_up_2pct=unhedged_up_2,
        move_up_5pct=unhedged_up_5,
        hedged_down_5pct=hedged_down_5,
        hedged_down_2pct=hedged_down_2,
        hedged_up_2pct=hedged_up_2,
        hedged_up_5pct=hedged_up_5,
    )


def estimate_hedge_cost(
    position: InventoryPosition,
    market: MarketConditions,
    hedge_pct: float,
) -> HedgeCost:
    """
    Estimate cost of hedging a position.

    For a long position: hedge by selling spot or shorting perp
    For a short position: hedge by buying spot or longing perp
    """
    hedge_size = abs(position.size) * hedge_pct
    hedge_notional = hedge_size * market.current_price

    # Spread cost (crossing the bid-ask)
    spread_cost_usd = hedge_notional * (market.spot_spread_bps / 10000)

    # Funding cost for perps (daily)
    # Positive funding = longs pay shorts
    # If we're hedging a long (going short perp), we RECEIVE funding if rate > 0
    # If we're hedging a short (going long perp), we PAY funding if rate > 0
    daily_funding_rate = market.perp_funding_rate / 365
    funding_cost_1d = hedge_notional * daily_funding_rate
    if position.is_long:
        # Shorting perp to hedge long: receive funding if rate > 0
        funding_cost_1d = -funding_cost_1d
    # else: longing perp to hedge short: pay funding if rate > 0

    # Choose instrument
    if position.is_long:
        # Hedge long by shorting
        # Prefer perp if funding is favorable (we receive), else spot
        if market.perp_funding_rate > 0.02:  # High funding, we receive
            instrument = HedgeInstrument.PERP_SHORT
        else:
            instrument = HedgeInstrument.SPOT_SELL
    else:
        # Hedge short by longing
        if market.perp_funding_rate < -0.02:  # Negative funding, shorts receive
            instrument = HedgeInstrument.PERP_LONG
        else:
            instrument = HedgeInstrument.SPOT_BUY

    total_cost = spread_cost_usd + max(0, funding_cost_1d)
    total_cost_bps = (total_cost / hedge_notional) * 10000 if hedge_notional > 0 else 0

    return HedgeCost(
        instrument=instrument,
        size=hedge_size,
        spread_cost_usd=spread_cost_usd,
        funding_cost_1d_usd=funding_cost_1d,
        total_cost_usd=total_cost,
        total_cost_bps=total_cost_bps,
    )


def calculate_risk_score(
    position: InventoryPosition,
    market: MarketConditions,
    pnl_scenarios: PnLScenarios,
    config: RiskConfig,
) -> float:
    """
    Calculate a 0-100 risk score.

    Higher = more urgent to hedge.

    Factors:
    - Potential loss vs threshold
    - Position age
    - Volatility regime
    - Size relative to liquidity
    """
    score = 0.0
    notional = abs(position.size * market.current_price)

    # 1. Loss severity (0-30 points)
    # Volatility-adjusted: in low vol markets, 5% moves are unlikely
    # Scale potential loss by how likely it is given current volatility
    worst_loss = abs(min(pnl_scenarios.move_down_5pct, pnl_scenarios.move_up_5pct))
    loss_pct = worst_loss / notional if notional > 0 else 0
    # Vol adjustment: 5% move is ~1.7 daily sigma at 3% vol, but could be < 1 sigma at 8% vol
    vol_multiple = 0.05 / market.volatility_1d if market.volatility_1d > 0 else 2.0
    vol_adjustment = min(1.0, 1.0 / vol_multiple)  # Low vol = less concern
    loss_score = min(30, (loss_pct / config.max_loss_pct) * 30 * vol_adjustment)
    score += loss_score

    # 2. Position age (0-20 points)
    age_ratio = position.age_minutes / config.max_hold_minutes
    age_score = min(20, age_ratio * 20)
    score += age_score

    # 3. Volatility regime (0-25 points)
    if market.volatility_1d >= config.extreme_vol_threshold:
        vol_score = 25.0
    elif market.volatility_1d >= config.high_vol_threshold:
        vol_score = 15.0
    else:
        vol_score = market.volatility_1d / config.high_vol_threshold * 10
    score += vol_score

    # 4. Size vs liquidity (0-15 points)
    # If position is large relative to available liquidity, more urgent
    relevant_depth = market.bid_depth_usd if position.is_long else market.ask_depth_usd
    if relevant_depth > 0:
        size_ratio = notional / relevant_depth
        liquidity_score = min(15, size_ratio * 15)
    else:
        liquidity_score = 15  # No liquidity = max concern
    score += liquidity_score

    return min(100, max(0, score))


def determine_action(
    risk_score: float,
    position: InventoryPosition,
    market: MarketConditions,
    hedge_cost: HedgeCost,
    config: RiskConfig,
) -> tuple[Action, float, str]:
    """
    Determine recommended action based on risk assessment.

    Returns: (action, hedge_pct, reasoning)
    """
    # Check if hedge cost is prohibitive
    if hedge_cost.total_cost_bps > config.max_hedge_cost_bps:
        return (
            Action.HOLD,
            0.0,
            f"Hedge cost too high ({hedge_cost.total_cost_bps:.1f}bps > {config.max_hedge_cost_bps}bps limit). "
            f"Hold and wait for better conditions.",
        )

    # Decision tree based on risk score
    # Thresholds calibrated so normal market conditions result in HOLD
    if risk_score >= 75:
        # Critical risk
        return (
            Action.LIQUIDATE,
            1.0,
            f"Risk score {risk_score:.0f}/100 (critical). "
            f"Position too risky to hold. Liquidate immediately.",
        )

    if risk_score >= 60:
        # High risk - full hedge
        return (
            Action.HEDGE_FULL,
            1.0,
            f"Risk score {risk_score:.0f}/100 (high). "
            f"Recommend full hedge via {hedge_cost.instrument.value}. "
            f"Cost: ${hedge_cost.total_cost_usd:.2f} ({hedge_cost.total_cost_bps:.1f}bps).",
        )

    if risk_score >= 45:
        # Moderate risk - partial hedge
        hedge_pct = 0.5
        return (
            Action.HEDGE_PARTIAL,
            hedge_pct,
            f"Risk score {risk_score:.0f}/100 (moderate). "
            f"Recommend {hedge_pct*100:.0f}% hedge via {hedge_cost.instrument.value}. "
            f"Reduces downside while keeping upside.",
        )

    if risk_score >= 35:
        # Low-moderate - small hedge or reduce
        hedge_pct = 0.25
        return (
            Action.REDUCE,
            hedge_pct,
            f"Risk score {risk_score:.0f}/100 (low-moderate). "
            f"Consider reducing position by {hedge_pct*100:.0f}% to free up risk budget.",
        )

    # Low risk - hold
    return (
        Action.HOLD,
        0.0,
        f"Risk score {risk_score:.0f}/100 (low). "
        f"Position within acceptable risk parameters. Hold.",
    )


def assess_inventory_risk(
    position: InventoryPosition,
    market: MarketConditions,
    config: RiskConfig | None = None,
) -> HedgeRecommendation:
    """
    Main entry point: Assess inventory risk and provide actionable recommendation.

    Args:
        position: Current inventory position
        market: Current market conditions
        config: Risk configuration (uses defaults if None)

    Returns:
        HedgeRecommendation with action, costs, scenarios, and reasoning
    """
    if config is None:
        config = RiskConfig()

    # Calculate P&L scenarios (unhedged)
    pnl_scenarios = calculate_pnl_scenarios(position, market, hedge_pct=0.0)

    # Estimate cost of a full hedge (we'll adjust based on recommendation)
    full_hedge_cost = estimate_hedge_cost(position, market, hedge_pct=1.0)

    # Calculate risk score
    risk_score = calculate_risk_score(position, market, pnl_scenarios, config)

    # Determine action
    action, hedge_pct, reasoning = determine_action(
        risk_score, position, market, full_hedge_cost, config
    )

    # Calculate actual hedge cost and hedged scenarios
    if hedge_pct > 0:
        actual_hedge_cost = estimate_hedge_cost(position, market, hedge_pct)
        pnl_scenarios = calculate_pnl_scenarios(position, market, hedge_pct)
    else:
        actual_hedge_cost = None

    # Determine re-evaluation time
    if risk_score >= 60:
        re_eval = 15  # High risk: check every 15 min
    elif risk_score >= 40:
        re_eval = 60  # Moderate: hourly
    else:
        re_eval = 240  # Low: every 4 hours

    return HedgeRecommendation(
        action=action,
        hedge_pct=hedge_pct,
        suggested_hedge=actual_hedge_cost,
        pnl_scenarios=pnl_scenarios,
        risk_score=risk_score,
        reasoning=reasoning,
        re_evaluate_minutes=re_eval,
        position_notional_usd=abs(position.size * market.current_price),
        current_unrealized_pnl=position.unrealized_pnl,
    )
