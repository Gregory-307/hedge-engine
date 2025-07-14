from pathlib import Path
from pydantic import BaseSettings, Field
from typing import Literal

class Settings(BaseSettings):
    env: Literal["dev", "prod"] = "dev"
    redis_url: str = "redis://localhost:6379/0"
    db_dsn: str = "postgresql+psycopg2://hedge:hedge@localhost/hedge"
    max_hedge_pct: float = Field(1.0, ge=0, le=1)
    spline_knots: Path = Path("configs/curve_knots.yaml")
    stale_score_s: int = 3

    class Config:
        env_prefix = "HE_" 