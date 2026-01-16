from fastapi import APIRouter, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
from .sizer import compute_hedge
from . import __version__ as PACKAGE_VERSION
from .decision_logger import DecisionLogger

router = APIRouter()


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

class HedgeRequest(BaseModel):
    asset: Literal["BTC", "ETH", "LTC", "XRP"]
    amount_usd: float = Field(..., gt=0, le=1e12, description="Position size in USD to hedge")
    override_score: float | None = Field(None, ge=0.0, le=1.0, description="Manual score override (0-1)")

class HedgeResponse(BaseModel):
    hedge_pct: float
    notional_usd: float
    confidence: float
    version: str
    ts_ms: int

@router.post("/hedge", response_model=HedgeResponse, status_code=status.HTTP_200_OK)
async def hedge(req: HedgeRequest, background_tasks: BackgroundTasks) -> HedgeResponse:
    # Integration points: In production, these values come from upstream services:
    # - score: sentiment-pipeline publishes to Redis channel `sentiment:latest:{asset}`
    # - depth_usd: order-book service writes to ClickHouse (depth at ±1% from mid)
    # For standalone testing, use override_score or defaults below.
    score = req.override_score if req.override_score is not None else 0.5
    depth_usd = 5_000_000  # Default depth; override via future Redis integration
    hedge_pct, confidence = compute_hedge(score, depth1pct_usd=depth_usd)

    record = {
        "asset": req.asset,
        "amount_usd": req.amount_usd,
        "hedge_pct": hedge_pct,
        "confidence": confidence,
        "ts_ms": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
    }

    # Fire-and-forget DB logging
    background_tasks.add_task(DecisionLogger.log, record)

    resp = HedgeResponse(
        hedge_pct=hedge_pct,
        notional_usd=req.amount_usd * hedge_pct,
        confidence=confidence,
        version=PACKAGE_VERSION,
        ts_ms=record["ts_ms"],
    )
    return resp 