"""API endpoints for inventory risk assessment."""

from datetime import datetime, timezone

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from . import __version__ as PACKAGE_VERSION
from .assessor import assess_inventory_risk
from .models import (
    InventoryPosition,
    MarketConditions,
    RiskConfig,
)

router = APIRouter()


# ============== Request/Response Models ==============


class PositionRequest(BaseModel):
    """Input: Position data from the market maker."""

    asset: str = Field(..., description="Asset symbol (e.g., 'BTC', 'ETH')")
    size: float = Field(..., description="Position size (positive=long, negative=short)")
    entry_price: float = Field(..., gt=0, description="Average entry price in USD")
    age_minutes: int = Field(0, ge=0, description="How long position has been held")
    unrealized_pnl: float = Field(0.0, description="Current unrealized P&L")


class MarketRequest(BaseModel):
    """Input: Current market conditions."""

    current_price: float = Field(..., gt=0, description="Current spot price")
    volatility_1d: float = Field(
        ..., ge=0, le=1, description="1-day realized volatility (e.g., 0.05 = 5%)"
    )
    spot_spread_bps: float = Field(..., ge=0, description="Bid-ask spread in basis points")
    perp_funding_rate: float = Field(
        0.0, description="Annualized perp funding rate (e.g., 0.05 = 5%)"
    )
    bid_depth_usd: float = Field(
        1_000_000, gt=0, description="Liquidity on bid side at 1%"
    )
    ask_depth_usd: float = Field(
        1_000_000, gt=0, description="Liquidity on ask side at 1%"
    )


class ConfigRequest(BaseModel):
    """Optional: Custom risk configuration."""

    max_loss_pct: float = Field(0.05, gt=0, le=1, description="Max acceptable loss (e.g., 0.05 = 5%)")
    hedge_trigger_loss_pct: float = Field(0.02, gt=0, le=1, description="Start hedging at this loss %")
    max_hold_minutes: int = Field(480, gt=0, description="Max hold time before forced action")
    max_hedge_cost_bps: float = Field(50, gt=0, description="Max acceptable hedge cost in bps")


class AssessRequest(BaseModel):
    """Combined request for risk assessment."""

    position: PositionRequest
    market: MarketRequest
    config: ConfigRequest | None = None


class HedgeCostResponse(BaseModel):
    """Hedge cost details."""

    instrument: str
    size: float
    spread_cost_usd: float
    funding_cost_1d_usd: float
    total_cost_usd: float
    total_cost_bps: float


class PnLScenariosResponse(BaseModel):
    """P&L under different market scenarios."""

    move_down_5pct: float
    move_down_2pct: float
    move_up_2pct: float
    move_up_5pct: float
    hedged_down_5pct: float
    hedged_down_2pct: float
    hedged_up_2pct: float
    hedged_up_5pct: float


class AssessResponse(BaseModel):
    """Output: Risk assessment with actionable recommendation."""

    # Core recommendation
    action: str
    hedge_pct: float
    risk_score: float
    reasoning: str

    # Hedge details (if applicable)
    suggested_hedge: HedgeCostResponse | None

    # Scenarios
    pnl_scenarios: PnLScenariosResponse

    # Context
    position_notional_usd: float
    current_unrealized_pnl: float
    re_evaluate_minutes: int

    # Metadata
    version: str
    ts_ms: int


# ============== Endpoints ==============


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/assess", response_model=AssessResponse, status_code=status.HTTP_200_OK)
async def assess(req: AssessRequest) -> AssessResponse:
    """
    Assess inventory risk and get actionable recommendation.

    Takes position data + market conditions, returns:
    - Recommended action (HOLD, HEDGE, LIQUIDATE, etc.)
    - Hedge details (instrument, size, cost)
    - P&L scenarios (what happens if market moves ±2%, ±5%)
    - Risk score (0-100)
    """
    # Convert request to domain models
    position = InventoryPosition(
        asset=req.position.asset,
        size=req.position.size,
        entry_price=req.position.entry_price,
        age_minutes=req.position.age_minutes,
        unrealized_pnl=req.position.unrealized_pnl,
    )

    market = MarketConditions(
        current_price=req.market.current_price,
        volatility_1d=req.market.volatility_1d,
        spot_spread_bps=req.market.spot_spread_bps,
        perp_funding_rate=req.market.perp_funding_rate,
        bid_depth_usd=req.market.bid_depth_usd,
        ask_depth_usd=req.market.ask_depth_usd,
    )

    config = None
    if req.config:
        config = RiskConfig(
            max_loss_pct=req.config.max_loss_pct,
            hedge_trigger_loss_pct=req.config.hedge_trigger_loss_pct,
            max_hold_minutes=req.config.max_hold_minutes,
            max_hedge_cost_bps=req.config.max_hedge_cost_bps,
        )

    # Run assessment
    result = assess_inventory_risk(position, market, config)

    # Build response
    hedge_response = None
    if result.suggested_hedge:
        hedge_response = HedgeCostResponse(
            instrument=result.suggested_hedge.instrument.value,
            size=result.suggested_hedge.size,
            spread_cost_usd=result.suggested_hedge.spread_cost_usd,
            funding_cost_1d_usd=result.suggested_hedge.funding_cost_1d_usd,
            total_cost_usd=result.suggested_hedge.total_cost_usd,
            total_cost_bps=result.suggested_hedge.total_cost_bps,
        )

    pnl_response = PnLScenariosResponse(
        move_down_5pct=result.pnl_scenarios.move_down_5pct,
        move_down_2pct=result.pnl_scenarios.move_down_2pct,
        move_up_2pct=result.pnl_scenarios.move_up_2pct,
        move_up_5pct=result.pnl_scenarios.move_up_5pct,
        hedged_down_5pct=result.pnl_scenarios.hedged_down_5pct,
        hedged_down_2pct=result.pnl_scenarios.hedged_down_2pct,
        hedged_up_2pct=result.pnl_scenarios.hedged_up_2pct,
        hedged_up_5pct=result.pnl_scenarios.hedged_up_5pct,
    )

    return AssessResponse(
        action=result.action.value,
        hedge_pct=result.hedge_pct,
        risk_score=result.risk_score,
        reasoning=result.reasoning,
        suggested_hedge=hedge_response,
        pnl_scenarios=pnl_response,
        position_notional_usd=result.position_notional_usd,
        current_unrealized_pnl=result.current_unrealized_pnl,
        re_evaluate_minutes=result.re_evaluate_minutes,
        version=PACKAGE_VERSION,
        ts_ms=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
    )
