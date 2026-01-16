# Hedge-Engine

**Low-latency hedge-sizing microservice** — the decision layer of the Crypto Swing Analysis Suite.

[![CI](https://github.com/Gregory-307/hedge-engine/actions/workflows/ci.yaml/badge.svg)](https://github.com/Gregory-307/hedge-engine/actions/workflows/ci.yaml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gregory-307/hedge-engine/blob/main/notebooks/hedge_engine_colab.ipynb)

---

## What It Does

Takes a **sentiment score** (0.0–1.0) and **order-book liquidity depth**, returns a **hedge percentage** via monotonic spline interpolation.

```
Input:  score=0.72, depth=$5M  →  Output: hedge 68% of position
```

**Key specs:**
- Sub-50 µs compute latency (tested)
- Hot-reload sizing curve without restart
- Circuit breaker for DB failures
- Immutable audit trail of all decisions

---

## System Architecture

Hedge-engine is the **final decision layer** in a real-time crypto market intelligence system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CRYPTO SWING ANALYSIS SUITE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │  web-search-sdk │    │  twitter-stack  │    │  Market Data    │        │
│   │  ─────────────  │    │  ────────────   │    │  ───────────    │        │
│   │  • Google News  │    │  • Tweet ingest │    │  • CoinGecko    │        │
│   │  • Wikipedia    │    │  • Account mgmt │    │  • Order books  │        │
│   │  • Paywalls     │    │  • Proxy rotate │    │  • On-chain     │        │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘        │
│            │                      │                      │                  │
│            └──────────────┬───────┴──────────────────────┘                  │
│                           ▼                                                 │
│              ┌────────────────────────┐                                     │
│              │   sentiment-pipeline   │                                     │
│              │   ──────────────────   │                                     │
│              │   • VADER + LLM        │                                     │
│              │   • Technical analysis │                                     │
│              │   • 10+ coefficients   │                                     │
│              │   • 96.2% test pass    │                                     │
│              └───────────┬────────────┘                                     │
│                          │                                                  │
│                          │ Redis pub/sub                                    │
│                          │ sentiment:latest:{asset}                         │
│                          ▼                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                      ★ HEDGE-ENGINE ★                            │     │
│   │   ────────────────────────────────────────────────────────────   │     │
│   │                                                                  │     │
│   │   score (0-1) ──┐                                                │     │
│   │                 ├──► Monotonic Spline ──► hedge_pct (0-100%)    │     │
│   │   depth ($) ────┘    (configurable YAML)                         │     │
│   │                                                                  │     │
│   │   Features:                                                      │     │
│   │   • POST /hedge      - sizing recommendation                     │     │
│   │   • GET /healthz     - liveness probe                            │     │
│   │   • GET /metrics     - Prometheus metrics                        │     │
│   │   • Circuit breaker  - DB failure resilience                     │     │
│   │   • Decision logging - immutable audit trail                     │     │
│   │   • <50µs latency    - real-time capable                         │     │
│   │                                                                  │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                          │                                                  │
│                          ▼                                                  │
│                   Trading Systems                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Related Repositories

| Repository | Purpose | Status |
|------------|---------|--------|
| [web-search-sdk](https://github.com/Gregory-307/web-search-sdk) | Async web scraping (news, Wikipedia, paywalls) | Production |
| [twitter-stack](https://github.com/Gregory-307/twitter-stack) | Twitter scraping with account/proxy management | Production |
| [sentiment-pipeline](https://github.com/Gregory-307/sentiment-pipeline) | Real-time sentiment analysis & coefficient generation | Production |
| **hedge-engine** (this repo) | Score → hedge sizing decision layer | Production |

---

## Features

| Feature | Description |
|---------|-------------|
| **Monotonic Spline Sizing** | Smooth, monotonic curve from configurable YAML knots |
| **Liquidity Weighting** | Confidence scaled by order-book depth |
| **Hot-Reload Curve** | Update sizing parameters without restart |
| **Circuit Breaker** | Stop hammering failed DB after 3 failures, auto-retry after 30s |
| **Async Decision Logging** | Background PostgreSQL writes, fallback queue on failure |
| **Prometheus Metrics** | Circuit state gauge, latency histograms |
| **Sub-50µs Latency** | Spline evaluation is O(1), no I/O on critical path |
| **Dockerized** | docker-compose with API + Redis + Postgres |

---

## Quick Start

### Docker (recommended)
```bash
docker compose up --build
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Local Development
```bash
pip install -e ".[dev]"
uvicorn hedge_engine.main:app --reload
```

### Test the API
```bash
# Health check
curl http://localhost:8000/healthz

# Get hedge recommendation
curl -X POST http://localhost:8000/hedge \
  -H "Content-Type: application/json" \
  -d '{"asset": "BTC", "amount_usd": 100000, "override_score": 0.7}'
```

**Response:**
```json
{
  "hedge_pct": 0.6825,
  "notional_usd": 68250.0,
  "confidence": 0.85,
  "version": "0.1.0",
  "ts_ms": 1705420800000
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Liveness probe |
| `POST` | `/hedge` | Get hedge sizing recommendation |
| `GET` | `/metrics` | Prometheus metrics |

### POST /hedge

**Request:**
```json
{
  "asset": "BTC",           // BTC, ETH, LTC, XRP
  "amount_usd": 100000,     // Position size to hedge
  "override_score": 0.7     // Optional: manual score (0.0-1.0)
}
```

**Response:**
```json
{
  "hedge_pct": 0.6825,      // Recommended hedge percentage
  "notional_usd": 68250.0,  // amount_usd × hedge_pct
  "confidence": 0.85,       // Model confidence
  "version": "0.1.0",       // API version
  "ts_ms": 1705420800000    // Timestamp (ms)
}
```

---

## Sizing Algorithm

The hedge percentage is computed via **monotonic cubic spline interpolation**:

1. **Fetch inputs**: sentiment `score` (0-1) + `liquidity_depth` (USD at ±1%)
2. **Apply liquidity weight**: `effective_score = score × liquidity_weight(depth)`
3. **Evaluate spline**: Defined by configurable knots in `configs/curve_knots.yaml`
4. **Clamp**: Ensure result is within `[0, max_hedge_pct]`

**Default curve knots:**
```yaml
- {score: 0.0, hedge: 0.05}   # Very bullish → minimal hedge
- {score: 0.3, hedge: 0.25}
- {score: 0.7, hedge: 0.75}
- {score: 1.0, hedge: 1.00}   # Very bearish → full hedge
```

The curve is **monotonic** — higher scores always produce equal or higher hedge percentages.

---

## Configuration

Environment variables (prefix `HE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HE_ENV` | `dev` | Environment (dev/prod) |
| `HE_REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `HE_DB_DSN` | `postgresql+asyncpg://...` | Postgres DSN |
| `HE_MAX_HEDGE_PCT` | `1.0` | Maximum hedge percentage |
| `HE_STALE_SCORE_S` | `3` | Reject scores older than N seconds |

---

## Testing

```bash
# Run all tests
pytest -q

# With coverage
pytest --cov=hedge_engine --cov-report=term

# Property-based tests (monotonicity, bounds)
pytest tests/test_sizer_prop.py -v
```

**Test coverage:**
- Unit tests for boundary values and monotonicity
- Property-based tests via Hypothesis
- Performance regression tests (<50µs)
- Circuit breaker state transitions
- Async logging with DB failure handling

---

## Interactive Demo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gregory-307/hedge-engine/blob/main/notebooks/hedge_engine_colab.ipynb)

The notebook walks through:
1. Health check call
2. Hedge sizing with override score
3. Scraping Prometheus metrics
4. Using `compute_hedge()` directly in Python

---

## Project Structure

```
hedge-engine/
├── hedge_engine/
│   ├── main.py              # FastAPI app factory
│   ├── api.py               # REST endpoints
│   ├── sizer.py             # Spline sizing algorithm
│   ├── config.py            # Pydantic settings
│   ├── decision_logger.py   # Async Postgres logging
│   ├── circuit_breaker.py   # Failure resilience
│   └── validators.py        # Input validation
├── configs/
│   └── curve_knots.yaml     # Spline configuration
├── tests/
│   ├── test_sizer.py        # Unit tests
│   ├── test_sizer_prop.py   # Property-based tests
│   ├── test_perf.py         # Latency benchmarks
│   └── ...
├── docker-compose.yml       # Full stack
├── Dockerfile
└── pyproject.toml
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
