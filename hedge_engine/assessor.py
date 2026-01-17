"""Core inventory risk assessment logic."""

from __future__ import annotations

from loguru import logger

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
    config: RiskConfig,
    hedge_pct: float = 0.0,
) -> PnLScenarios:
    """
    Calculate P&L under different market move scenarios.

    Args:
        position: Current inventory position
        market: Current market conditions
        config: Risk configuration with scenario move percentages
        hedge_pct: Fraction of position hedged (0-1)
    """
    size = position.size
    current_price = market.current_price
    large_move = config.large_move_pct
    small_move = config.small_move_pct

    # P&L calculation: size already encodes direction
    # Long (size > 0): profits on up moves, loses on down moves
    # Short (size < 0): profits on down moves, loses on up moves
    def pnl_at_move(move_pct: float) -> float:
        price_change = current_price * move_pct
        return size * price_change

    unhedged_down_large = pnl_at_move(-large_move)
    unhedged_down_small = pnl_at_move(-small_move)
    unhedged_up_small = pnl_at_move(small_move)
    unhedged_up_large = pnl_at_move(large_move)

    # Hedged P&L (hedge reduces exposure by hedge_pct)
    unhedged_fraction = 1.0 - hedge_pct
    hedged_down_large = unhedged_down_large * unhedged_fraction
    hedged_down_small = unhedged_down_small * unhedged_fraction
    hedged_up_small = unhedged_up_small * unhedged_fraction
    hedged_up_large = unhedged_up_large * unhedged_fraction

    return PnLScenarios(
        move_down_5pct=unhedged_down_large,
        move_down_2pct=unhedged_down_small,
        move_up_2pct=unhedged_up_small,
        move_up_5pct=unhedged_up_large,
        hedged_down_5pct=hedged_down_large,
        hedged_down_2pct=hedged_down_small,
        hedged_up_2pct=hedged_up_small,
        hedged_up_5pct=hedged_up_large,
    )


def estimate_hedge_cost(
    position: InventoryPosition,
    market: MarketConditions,
    hedge_pct: float,
    config: RiskConfig,
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
    daily_funding_rate = market.perp_funding_rate / config.days_per_year
    funding_cost_1d = hedge_notional * daily_funding_rate
    if position.is_long:
        # Shorting perp to hedge long: receive funding if rate > 0
        funding_cost_1d = -funding_cost_1d
    # else: longing perp to hedge short: pay funding if rate > 0

    # Choose instrument based on funding rate threshold
    threshold = config.funding_rate_threshold
    if position.is_long:
        # Hedge long by shorting
        # Prefer perp if funding is favorable (we receive), else spot
        if market.perp_funding_rate > threshold:
            instrument = HedgeInstrument.PERP_SHORT
        else:
            instrument = HedgeInstrument.SPOT_SELL
    else:
        # Hedge short by longing
        if market.perp_funding_rate < -threshold:
            instrument = HedgeInstrument.PERP_LONG
        else:
            instrument = HedgeInstrument.SPOT_BUY

    # Funding can be negative (we receive) which reduces total cost
    # But total cost can't go negative (we can't be paid to hedge)
    total_cost = max(0, spread_cost_usd + funding_cost_1d)
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

    Factors (weights from config, default sum to 100):
    - Loss severity: 0-loss_weight points (potential loss vs thresholds, vol-adjusted)
    - Position age: 0-age_weight points (time held vs limits)
    - Volatility regime: 0-vol_weight points (current market vol)
    - Size vs liquidity: 0-liquidity_weight points (position size relative to depth)

    See RiskConfig docstring for design note on intentional volatility double-counting.
    """
    score = 0.0
    notional = abs(position.size * market.current_price)

    # Handle zero-size position
    if notional == 0:
        return 0.0

    # 1. Loss severity (0 to loss_weight points)
    # Uses hedge_trigger_loss_pct as the "concerning" threshold
    # and max_loss_pct as the "critical" threshold
    worst_loss = abs(min(pnl_scenarios.move_down_5pct, pnl_scenarios.move_up_5pct))
    loss_pct = worst_loss / notional

    # Vol adjustment: scale risk based on current vol vs baseline
    # Low vol → adjustment < 1.0 (reduces loss_score)
    # Baseline vol → adjustment = 1.0
    # High vol → adjustment > 1.0 (increases loss_score)
    if market.volatility_1d > 0:
        vol_adjustment = market.volatility_1d / config.baseline_vol
        vol_adjustment = min(config.vol_adj_max, max(config.vol_adj_min, vol_adjustment))
    else:
        vol_adjustment = config.vol_adj_max  # No data = assume high risk

    # Scale: 0 at no loss, loss_trigger_points at trigger, loss_weight at max
    above_trigger_points = config.loss_weight - config.loss_trigger_points
    if loss_pct <= config.hedge_trigger_loss_pct:
        loss_score = (loss_pct / config.hedge_trigger_loss_pct) * config.loss_trigger_points
    else:
        # Above trigger: scale from loss_trigger_points to loss_weight
        excess = loss_pct - config.hedge_trigger_loss_pct
        remaining = config.max_loss_pct - config.hedge_trigger_loss_pct
        if remaining > 0:
            loss_score = config.loss_trigger_points + (excess / remaining) * above_trigger_points
        else:
            loss_score = config.loss_weight

    loss_score = min(config.loss_weight, loss_score * vol_adjustment)
    score += loss_score

    # 2. Position age (0 to age_weight points)
    # Non-linear: accelerates after urgent_hold_minutes
    above_urgent_points = config.age_weight - config.age_normal_points
    if position.age_minutes <= config.urgent_hold_minutes:
        age_score = (position.age_minutes / config.urgent_hold_minutes) * config.age_normal_points
    else:
        # Urgent phase (accelerated)
        excess = position.age_minutes - config.urgent_hold_minutes
        remaining = config.max_hold_minutes - config.urgent_hold_minutes
        if remaining > 0:
            age_score = config.age_normal_points + (excess / remaining) * above_urgent_points
        else:
            age_score = config.age_weight

    age_score = min(config.age_weight, age_score)
    score += age_score

    # 3. Volatility regime (0 to vol_weight points)
    # Smooth scaling from 0 to extreme threshold
    if market.volatility_1d >= config.extreme_vol_threshold:
        vol_score = config.vol_weight
    else:
        vol_score = (market.volatility_1d / config.extreme_vol_threshold) * config.vol_weight
    score += vol_score

    # 4. Size vs liquidity (0 to liquidity_weight points)
    # For long positions, we care about bid depth (we'd sell into bids)
    # For short positions, we care about ask depth (we'd buy from asks)
    relevant_depth = market.bid_depth_usd if position.is_long else market.ask_depth_usd
    if relevant_depth > 0:
        size_ratio = notional / relevant_depth
        # Square root scaling: small positions low risk, diminishing returns for large
        liquidity_score = min(config.liquidity_weight, (size_ratio ** 0.5) * config.liquidity_weight)
    else:
        liquidity_score = config.liquidity_weight  # No liquidity = max concern
    score += liquidity_score

    return float(min(100, max(0, score)))


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

    # Decision tree based on risk score using configurable thresholds
    if risk_score >= config.liquidate_threshold:
        # Critical risk - liquidate immediately
        return (
            Action.LIQUIDATE,
            1.0,
            f"Risk score {risk_score:.0f}/100 (critical). "
            f"Position too risky to hold. Liquidate immediately.",
        )

    if risk_score >= config.hedge_full_threshold:
        # High risk - full hedge
        return (
            Action.HEDGE_FULL,
            1.0,
            f"Risk score {risk_score:.0f}/100 (high). "
            f"Recommend full hedge via {hedge_cost.instrument.value}. "
            f"Cost: ${hedge_cost.total_cost_usd:.2f} ({hedge_cost.total_cost_bps:.1f}bps).",
        )

    if risk_score >= config.hedge_partial_threshold:
        # Moderate risk - partial hedge
        hedge_pct = config.hedge_partial_pct
        return (
            Action.HEDGE_PARTIAL,
            hedge_pct,
            f"Risk score {risk_score:.0f}/100 (moderate). "
            f"Recommend {hedge_pct*100:.0f}% hedge via {hedge_cost.instrument.value}. "
            f"Reduces downside while keeping upside.",
        )

    if risk_score >= config.reduce_threshold:
        # Low-moderate - small hedge or reduce
        hedge_pct = config.reduce_pct
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

    notional = abs(position.size * market.current_price)

    # Handle zero-size (flat) position
    if position.size == 0:
        logger.info(
            "Flat position assessed",
            extra={"asset": position.asset, "action": "HOLD", "risk_score": 0},
        )
        return HedgeRecommendation(
            action=Action.HOLD,
            hedge_pct=0.0,
            suggested_hedge=None,
            pnl_scenarios=PnLScenarios(
                move_down_5pct=0, move_down_2pct=0,
                move_up_2pct=0, move_up_5pct=0,
            ),
            risk_score=0.0,
            reasoning="No position to assess (size = 0).",
            re_evaluate_minutes=config.low_risk_reeval_minutes,
            position_notional_usd=0.0,
            # unrealized_pnl is passed through for user context, not used in calculations
            current_unrealized_pnl=0.0,
        )

    # Calculate P&L scenarios (unhedged)
    pnl_scenarios = calculate_pnl_scenarios(position, market, config, hedge_pct=0.0)

    # Estimate cost of a full hedge (we'll adjust based on recommendation)
    full_hedge_cost = estimate_hedge_cost(position, market, hedge_pct=1.0, config=config)

    # Calculate risk score
    risk_score = calculate_risk_score(position, market, pnl_scenarios, config)

    # Determine action
    action, hedge_pct, reasoning = determine_action(
        risk_score, position, market, full_hedge_cost, config
    )

    # Log the assessment
    logger.info(
        "Position assessed",
        extra={
            "asset": position.asset,
            "size": position.size,
            "notional_usd": notional,
            "risk_score": round(risk_score, 1),
            "action": action.value,
            "hedge_pct": hedge_pct,
        },
    )

    # Log high-risk situations at warning level
    if action == Action.LIQUIDATE:
        logger.warning(
            "LIQUIDATE recommended",
            extra={"asset": position.asset, "risk_score": risk_score},
        )
    elif action in (Action.HEDGE_FULL, Action.HEDGE_PARTIAL):
        logger.info(
            "Hedge recommended",
            extra={
                "asset": position.asset,
                "hedge_pct": hedge_pct,
                "instrument": full_hedge_cost.instrument.value,
            },
        )

    # Calculate actual hedge cost and hedged scenarios
    if hedge_pct > 0:
        actual_hedge_cost = estimate_hedge_cost(position, market, hedge_pct, config)
        pnl_scenarios = calculate_pnl_scenarios(position, market, config, hedge_pct)
    else:
        actual_hedge_cost = None

    # Determine re-evaluation time based on configurable thresholds
    if risk_score >= config.high_risk_reeval_threshold:
        re_eval = config.high_risk_reeval_minutes
    elif risk_score >= config.moderate_risk_reeval_threshold:
        re_eval = config.moderate_risk_reeval_minutes
    else:
        re_eval = config.low_risk_reeval_minutes

    return HedgeRecommendation(
        action=action,
        hedge_pct=hedge_pct,
        suggested_hedge=actual_hedge_cost,
        pnl_scenarios=pnl_scenarios,
        risk_score=risk_score,
        reasoning=reasoning,
        re_evaluate_minutes=re_eval,
        position_notional_usd=abs(position.size * market.current_price),
        # unrealized_pnl is passed through for user context, not used in calculations
        current_unrealized_pnl=position.unrealized_pnl,
    )
