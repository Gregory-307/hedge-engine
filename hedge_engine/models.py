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
    PUT_BUY = "put_buy"
    CALL_BUY = "call_buy"


@dataclass
class InventoryPosition:
    """A market maker's inventory position."""

    asset: str  # e.g., "BTC", "ETH"
    size: float  # Quantity (positive = long, negative = short)
    entry_price: float  # Average entry price in USD
    age_minutes: int = 0  # How long position has been held
    unrealized_pnl: float = 0.0  # Current unrealized P&L

    @property
    def notional_usd(self) -> float:
        """Position notional in USD at entry."""
        return abs(self.size * self.entry_price)

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

    @property
    def mid_spread_cost_bps(self) -> float:
        """Cost to cross the spread (half the spread)."""
        return self.spot_spread_bps / 2


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
class HedgeRecommendation:
    """Actionable output from risk assessment."""

    action: Action
    hedge_pct: float  # 0.0 to 1.0
    suggested_hedge: HedgeCost | None
    pnl_scenarios: PnLScenarios
    risk_score: float  # 0-100 (for quick reference)
    reasoning: str
    re_evaluate_minutes: int  # When to reassess

    # Position context
    position_notional_usd: float
    current_unrealized_pnl: float


@dataclass
class RiskConfig:
    """Configurable risk parameters."""

    # Loss thresholds
    max_loss_pct: float = 0.05  # Max acceptable loss (5%)
    hedge_trigger_loss_pct: float = 0.02  # Start hedging at 2% potential loss

    # Time thresholds
    max_hold_minutes: int = 480  # 8 hours max hold
    urgent_hold_minutes: int = 240  # 4 hours = urgent

    # Cost thresholds
    max_hedge_cost_bps: float = 50  # Don't hedge if cost > 50bps

    # Volatility thresholds
    high_vol_threshold: float = 0.04  # 4% daily vol = high
    extreme_vol_threshold: float = 0.08  # 8% daily vol = extreme
