# Hedge-Engine – Detailed Build Plan
*Version 0.1 — 2025-07-14*

> Goal: translate Swing Sustainability Scores + liquidity metrics into concrete hedge-sizing recommendations and expose them via a low-latency REST API / gRPC endpoint. Designed for **≤50 ms** median latency (in-memory calc) and **>99.99 %** uptime.

---
## 0  Overview
The service consumes real-time sentiment scores (pushed to Redis by *sentiment-pipeline*) and order-book liquidity deltas (written to ClickHouse), runs a configurable sizing function, and returns a JSON struct `{hedge_pct, notional_usd, confidence, version}`.

Key constraints:
• **Speed** – synchronous HTTP ≤50 ms; internal calc ≤5 ms.
• **Safety** – circuit-breakers on stale / extreme inputs.
• **Auditability** – every decision logged to Postgres (immutable).
• **KISS / YAGNI** – start with a monotonic spline curve; plug-in DSL later if needed.

---
## 1  Repo Skeleton
| Path | Purpose |
|------|---------|
| `pyproject.toml` | Python 3.11; deps: fastapi, uvicorn[standard], pydantic, redis, sqlalchemy, httpx, numpy, scipy, loguru, ruff, mypy, pytest |
| `hedge_engine/__init__.py` | Expose `__version__`. |
| `hedge_engine/config.py` | Pydantic `Settings` (env-driven). |
| `hedge_engine/sizer.py` | Pure functions turning scores → hedge %. |
| `hedge_engine/validators.py` | Input sanity checks & circuit breakers. |
| `hedge_engine/api.py` | FastAPI router & DI glue. |
| `hedge_engine/main.py` | `get_application()` + Uvicorn entrypoint. |
| `hedge_engine/db.py` | SQLAlchemy engine & models (decision log). |
| `hedge_engine/redis_bridge.py` | Async Redis subscriber / cache. |
| `tests/` | Pytest suites incl. property-based tests for curve monotonicity. |
| `configs/` | YAML: sizing curve knots, risk limits, env toggles. |
| `Dockerfile` | Alpine slim → `uvicorn hedge_engine.main:app`. |
| `.github/workflows/ci.yaml` | Lint → Test → MyPy strict.
| `logs/` | Dev logs committed daily. |

---
## 2  Config & Environment (`config.py`)
```python
class Settings(BaseSettings):
    env: Literal["dev", "prod"] = "dev"
    redis_url: str = "redis://redis:6379/0"
    clickhouse_url: str = "http://clickhouse:8123"
    db_dsn: str = "postgresql+psycopg2://hedge:hedge@db/hedge"
    max_hedge_pct: float = 1.0   # cap
    spline_knots: Path = Path("configs/curve_knots.yaml")
    stale_score_s: int = 3       # reject if older
    log_decisions: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
```
Loaded once on startup; hot-reloaded if file watcher detects `curve_knots.yaml` change.

---
## 3  Input & Output Models
```python
class HedgeRequest(BaseModel):
    asset: Literal["BTC", "ETH", "LTC", "XRP"]
    amount_usd: float                 # exposure needing hedge
    override_score: float | None = None  # manual knob

class HedgeResponse(BaseModel):
    hedge_pct: float                  # 0..1
    notional_usd: float               # amount_usd * hedge_pct
    confidence: float                 # propagated from sentiment conf & liquidity quality
    version: str                      # git-SHA of sizer module
    ts_ms: int
```

---
## 4  Sizing Logic (`sizer.py`)
1. Fetch latest `score` + `liquidity.depth1pct` for asset from Redis/ClickHouse (<=1 ms cached).
2. Validate recency → raise `StaleDataError` if `score_ts > settings.stale_score_s`.
3. Compute **effective score** = score × liquidity_weight(depth).
4. Evaluate **monotonic cubic spline** defined by `spline_knots.yaml`:
```yaml
# example knots
- {score: 0.0, hedge: 0.05}
- {score: 0.3, hedge: 0.25}
- {score: 0.7, hedge: 0.75}
- {score: 1.0, hedge: 1.0}
```
5. Clamp to `0 … settings.max_hedge_pct`.
6. Confidence = min(model_confidence, liquidity_quality).
7. Return tuple `(hedge_pct, confidence)`.

Unit tests cover boundary knots, monotonicity, and caps.

---
## 5  API Design (`api.py`)
| Method | Path | Description |
|--------|------|-------------|
| `POST /hedge` | Body → `HedgeRequest`; returns `HedgeResponse`. |
| `GET /healthz` | Liveness probe. |
| `GET /metrics` | Prometheus exporter via `prometheus_fastapi_instrumentator`. |
| `POST /curve/reload` | Hot-reload knots (auth-guarded). |

Security: JWT auth middleware (stub for now, permissive in *dev*).

---
## 6  Decision Logging (`db.py`)
Table `decisions`:
| Column | Type |
|--------|------|
| id | UUID PK |
| ts | TIMESTAMPTZ |
| asset | TEXT |
| amount_usd | NUMERIC |
| score | REAL |
| liquidity_depth | REAL |
| hedge_pct | REAL |
| confidence | REAL |
| version | TEXT |

Inserted asynchronously via background task to keep request path fast.

---
## 7  Deployment & Docker
```dockerfile
FROM python:3.11-alpine
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[prod]"
EXPOSE 8000
CMD ["uvicorn", "hedge_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
CI pushes image to GHCR; `docker compose up` spins redis + db + hedge-engine.

---
## 8  CI / Quality Gates
* **ruff & black** on every PR.
* **pytest -q** with coverage ≥80 %.
* **mypy --strict** (SQLite stub for Postgres driver).
* **Docker build** & `curl /healthz` smoke test.

---
## 9  Milestones
| ID | Deliverable | Owner | Acceptance |
|----|-------------|-------|------------|
| HE-01 | Repo skeleton + CI green | new dev | `pytest` & `curl /healthz` pass in CI. |
| HE-02 | Sizing spline v0.1 | new dev | Unit tests show monotonic; manual curve YAML reload works. |
| HE-03 | Decision logging | new dev | Row appears in Postgres; latency ≤2 ms. |
| HE-04 | Dockerized API | you | `docker run` → respond ≤100 ms. |
| HE-05 | Circuit breakers & metrics | new dev | 99th-pct latency ≤50 ms under load; stale data errors logged. |
| HE-06 | Canary deploy | you | Grafana shows error rate <0.1 %. |

---
## 10  Open Questions
1. gRPC vs REST for intranet calls?  (REST assumed for now.)
2. Auth method — JWT vs mTLS?  JWT placeholder.
3. Versioning strategy for spline curves — Git tracked YAML vs DB table?
4. Confidence blending formula — linear min vs harmonic mean?
5. Deployment target — EKS vs bare-metal? (affects readiness probes.) 