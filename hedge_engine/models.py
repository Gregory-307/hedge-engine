"""Data models for inventory risk assessment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """Recommended action for a position."""

    HOLD = "HOLD"
    HEDGE_PARTIAL = "HEDGE_PARTIAL"
    HEDGE_FULL = "HEDGE_FULL"
    REDUCE = "REDUCE"
    LIQUIDATE = "LIQUIDATE"


class HedgeInstrument(str, Enum):
    """Available hedging instruments."""

    SPOT_SELL = "spot_sell"
    SPOT_BUY = "spot_buy"
    PERP_SHORT = "perp_short"
    PERP_LONG = "perp_long"


@dataclass
class InventoryPosition:
    """A market maker's inventory position."""

    asset: str  # e.g., "BTC", "ETH"
    size: float  # Quantity (positive = long, negative = short)
    entry_price: float  # Average entry price in USD
    age_minutes: int = 0  # How long position has been held
    unrealized_pnl: float = 0.0  # Current unrealized P&L

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0


@dataclass
class MarketConditions:
    """Current market state for risk assessment."""

    current_price: float  # Current spot price
    volatility_1d: float  # 1-day realized volatility (decimal, e.g., 0.05 = 5%)
    spot_spread_bps: float  # Bid-ask spread in basis points
    perp_funding_rate: float  # Annualized funding rate (decimal, e.g., 0.05 = 5%)
    bid_depth_usd: float  # Liquidity on bid side at 1%
    ask_depth_usd: float  # Liquidity on ask side at 1%


@dataclass
class PnLScenarios:
    """P&L under different market moves."""

    move_down_5pct: float
    move_down_2pct: float
    move_up_2pct: float
    move_up_5pct: float

    # After hedging
    hedged_down_5pct: float = 0.0
    hedged_down_2pct: float = 0.0
    hedged_up_2pct: float = 0.0
    hedged_up_5pct: float = 0.0


@dataclass
class HedgeCost:
    """Estimated cost of hedging."""

    instrument: HedgeInstrument
    size: float  # How much to hedge
    spread_cost_usd: float  # Cost from bid-ask spread
    funding_cost_1d_usd: float  # Daily funding cost (for perps)
    total_cost_usd: float  # Total immediate cost
    total_cost_bps: float  # Cost as basis points of notional


@dataclass
class RiskScoreBreakdown:
    """
    Breakdown of risk score components.

    Shows exactly what's contributing to the total score,
    so traders can understand WHY the score is what it is.
    """

    loss_severity: float  # Points from potential loss (0-35)
    position_age: float  # Points from holding time (0-20)
    volatility_regime: float  # Points from market volatility (0-25)
    size_vs_liquidity: float  # Points from position size vs depth (0-20)
    total: float  # Sum of all components (0-100)

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for API responses."""
        return {
            "loss_severity": round(self.loss_severity, 1),
            "position_age": round(self.position_age, 1),
            "volatility_regime": round(self.volatility_regime, 1),
            "size_vs_liquidity": round(self.size_vs_liquidity, 1),
            "total": round(self.total, 1),
        }


@dataclass
class PositionSummary:
    """
    Quick summary for decision-making.

    The key numbers a trader needs at a glance.
    """

    worst_case_loss_usd: float  # Biggest loss in scenarios
    best_case_gain_usd: float  # Biggest gain in scenarios
    position_side: str  # "LONG" or "SHORT"
    notional_usd: float  # Position value
    age_hours: float  # Position age in human-readable form
    hedge_order: str | None  # Concrete instruction, e.g., "Sell 5.0 BTC via perp_short"


@dataclass
class HedgeRecommendation:
    """Actionable output from risk assessment."""

    action: Action
    hedge_pct: float  # 0.0 to 1.0
    suggested_hedge: HedgeCost | None
    pnl_scenarios: PnLScenarios
    risk_score: float  # 0-100 (for quick reference)
    risk_breakdown: RiskScoreBreakdown  # Component breakdown
    reasoning: str
    re_evaluate_minutes: int  # When to reassess

    # Position context
    position_notional_usd: float
    current_unrealized_pnl: float

    # Quick summary for decision-making
    summary: PositionSummary


@dataclass
class RiskConfig:
    """
    Configurable risk parameters.

    Design Note - Volatility Double-Counting:
    Volatility affects the risk score in TWO ways (intentional):
    1. vol_adjustment multiplies loss_score (low vol = less likely to hit loss scenarios)
    2. vol_score is additive (high vol regime = faster market = needs attention)
    This creates a non-linear compounding effect where extreme vol has outsized impact,
    which matches real-world behavior where vol spikes require urgent action.
    """

    # Scenario move percentages for P&L calculation
    large_move_pct: float = 0.05  # 5% move for stress scenarios
    small_move_pct: float = 0.02  # 2% move for moderate scenarios

    # Loss thresholds
    max_loss_pct: float = 0.05  # Max acceptable loss (5%)
    hedge_trigger_loss_pct: float = 0.02  # Start hedging at 2% potential loss

    # Time thresholds
    max_hold_minutes: int = 480  # 8 hours max hold
    urgent_hold_minutes: int = 240  # 4 hours = urgent

    # Cost thresholds
    max_hedge_cost_bps: float = 50  # Don't hedge if cost > 50bps

    # Volatility thresholds
    baseline_vol: float = 0.05  # 5% daily vol = baseline for vol adjustment
    extreme_vol_threshold: float = 0.08  # 8% daily vol = extreme
    vol_adj_min: float = 0.3  # Floor for volatility adjustment multiplier
    vol_adj_max: float = 2.0  # Cap for volatility adjustment multiplier

    # Action thresholds (risk score -> action)
    liquidate_threshold: int = 85
    hedge_full_threshold: int = 70
    hedge_partial_threshold: int = 55
    reduce_threshold: int = 40

    # Hedge percentages
    hedge_partial_pct: float = 0.5  # 50% hedge for partial
    reduce_pct: float = 0.25  # 25% reduction

    # Instrument selection
    funding_rate_threshold: float = 0.02  # Use perp if funding > 2%

    # Risk score component weights (must sum to 100)
    loss_weight: float = 35.0  # Max points for loss severity
    age_weight: float = 20.0  # Max points for position age
    vol_weight: float = 25.0  # Max points for volatility regime
    liquidity_weight: float = 20.0  # Max points for size vs liquidity

    # Loss score internal splits
    loss_trigger_points: float = 15.0  # Points at hedge_trigger_loss_pct
    # Remaining (loss_weight - loss_trigger_points) allocated above trigger

    # Age score internal splits
    age_normal_points: float = 10.0  # Points for 0 to urgent_hold_minutes
    # Remaining (age_weight - age_normal_points) for urgent to max

    # Re-evaluation thresholds and times
    high_risk_reeval_threshold: float = 60.0  # Score above this = high risk
    moderate_risk_reeval_threshold: float = 40.0  # Score above this = moderate
    high_risk_reeval_minutes: int = 15  # Check every 15 min if high risk
    moderate_risk_reeval_minutes: int = 60  # Hourly if moderate
    low_risk_reeval_minutes: int = 240  # Every 4 hours if low

    # Funding calculation
    days_per_year: int = 365  # For annualized funding rate conversion
