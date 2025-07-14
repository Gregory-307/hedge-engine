from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Literal
from datetime import datetime, timezone

router = APIRouter()

@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

class HedgeRequest(BaseModel):
    asset: Literal["BTC", "ETH", "LTC", "XRP"]
    amount_usd: float
    override_score: float | None = None

class HedgeResponse(BaseModel):
    hedge_pct: float
    notional_usd: float
    confidence: float
    version: str
    ts_ms: int

@router.post("/hedge", response_model=HedgeResponse, status_code=status.HTTP_200_OK)
async def hedge(req: HedgeRequest) -> HedgeResponse:
    # Placeholder logic: always hedge 10 %
    hedge_pct = 0.10
    resp = HedgeResponse(
        hedge_pct=hedge_pct,
        notional_usd=req.amount_usd * hedge_pct,
        confidence=1.0,
        version="0.1.0",
        ts_ms=int(datetime.now(tz=timezone.utc).timestamp() * 1000),
    )
    return resp 