# Inventory Risk Engine

**Real-time inventory risk assessment for market makers.**

[![CI](https://github.com/Gregory-307/hedge-engine/actions/workflows/ci.yaml/badge.svg)](https://github.com/Gregory-307/hedge-engine/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

Takes your **position data** and **market conditions**, returns an **actionable recommendation** with P&L scenarios and hedge costs.

```
Position + Market Conditions → Action + Hedge Details + P&L Scenarios
```

**For market makers who need to:**
- Assess inventory risk in real-time
- Get actionable hedge recommendations (not abstract scores)
- See P&L scenarios for different market moves
- Compare hedge instrument costs (spot vs perp)
- Know exactly what to do: HOLD, HEDGE, REDUCE, or LIQUIDATE

---

## Quick Start

### Install
```bash
pip install -e ".[dev]"
```

### Run the API
```bash
uvicorn hedge_engine.main:app --reload
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Assess a Position
```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "position": {
      "asset": "BTC",
      "size": 10.0,
      "entry_price": 45000.0,
      "age_minutes": 60
    },
    "market": {
      "current_price": 44000.0,
      "volatility_1d": 0.05,
      "spot_spread_bps": 3.0,
      "perp_funding_rate": 0.08,
      "bid_depth_usd": 5000000,
      "ask_depth_usd": 5000000
    }
  }'
```

**Response:**
```json
{
  "action": "HEDGE_PARTIAL",
  "hedge_pct": 0.5,
  "risk_score": 47.3,
  "reasoning": "Risk score 47/100 (moderate). Recommend 50% hedge via perp_short. Reduces downside while keeping upside.",
  "suggested_hedge": {
    "instrument": "perp_short",
    "size": 5.0,
    "spread_cost_usd": 66.0,
    "funding_cost_1d_usd": -24.11,
    "total_cost_usd": 66.0,
    "total_cost_bps": 3.0
  },
  "pnl_scenarios": {
    "move_down_5pct": -11000.0,
    "move_down_2pct": -4400.0,
    "move_up_2pct": 4400.0,
    "move_up_5pct": 11000.0,
    "hedged_down_5pct": -5500.0,
    "hedged_up_5pct": 5500.0
  },
  "position_notional_usd": 440000.0,
  "re_evaluate_minutes": 60,
  "version": "0.2.0"
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/healthz` | Liveness probe |
| `POST` | `/assess` | Assess position risk and get recommendation |
| `GET` | `/metrics` | Prometheus metrics |

### POST /assess

**Request:**
```json
{
  "position": {
    "asset": "BTC",
    "size": 10.0,           // Positive = long, negative = short
    "entry_price": 45000.0,
    "age_minutes": 60       // How long position held
  },
  "market": {
    "current_price": 44000.0,
    "volatility_1d": 0.05,       // 5% daily vol
    "spot_spread_bps": 3.0,      // Bid-ask spread
    "perp_funding_rate": 0.08,   // Annual funding rate
    "bid_depth_usd": 5000000,    // Liquidity on bid
    "ask_depth_usd": 5000000     // Liquidity on ask
  },
  "config": {                    // Optional overrides
    "max_loss_pct": 0.05,
    "max_hold_minutes": 480
  }
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `action` | string | `HOLD`, `HEDGE_PARTIAL`, `HEDGE_FULL`, `REDUCE`, `LIQUIDATE` |
| `hedge_pct` | float | Recommended hedge percentage (0.0-1.0) |
| `risk_score` | float | 0-100 risk score |
| `reasoning` | string | Human-readable explanation |
| `suggested_hedge` | object | Instrument, size, and cost breakdown |
| `pnl_scenarios` | object | P&L at ±2% and ±5% market moves |
| `re_evaluate_minutes` | int | When to reassess (higher risk = sooner) |

---

## Risk Assessment Logic

### Risk Score (0-100)

The risk score is calculated from four factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Loss Severity** | 0-30 | Volatility-adjusted potential loss vs your max tolerance |
| **Position Age** | 0-20 | Older positions = higher risk (inventory should turn over) |
| **Volatility Regime** | 0-25 | High vol markets require faster decisions |
| **Size vs Liquidity** | 0-15 | Large position relative to book depth = harder to exit |

### Action Thresholds

| Risk Score | Action | Hedge % |
|------------|--------|---------|
| 75+ | `LIQUIDATE` | 100% |
| 60-74 | `HEDGE_FULL` | 100% |
| 45-59 | `HEDGE_PARTIAL` | 50% |
| 35-44 | `REDUCE` | 25% |
| 0-34 | `HOLD` | 0% |

### Hedge Instrument Selection

For **long positions** (need to reduce delta by selling):
- **Perp short** if funding > 2% (you earn funding)
- **Spot sell** otherwise

For **short positions** (need to reduce delta by buying):
- **Perp long** if funding < -2% (shorts earn funding)
- **Spot buy** otherwise

---

## P&L Scenarios

Every assessment includes P&L projections:

| Scenario | Description |
|----------|-------------|
| `move_down_5pct` | P&L if price drops 5% |
| `move_down_2pct` | P&L if price drops 2% |
| `move_up_2pct` | P&L if price rises 2% |
| `move_up_5pct` | P&L if price rises 5% |
| `hedged_down_5pct` | P&L with hedge if price drops 5% |
| `hedged_up_5pct` | P&L with hedge if price rises 5% |

---

## Configuration

### Risk Config (per-request)

Pass custom risk parameters in the `config` field:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_loss_pct` | 0.05 | Max acceptable loss (5%) |
| `hedge_trigger_loss_pct` | 0.02 | Start hedging at this loss |
| `max_hold_minutes` | 480 | Max hold time (8 hours) |
| `max_hedge_cost_bps` | 50 | Don't hedge if cost > 50bps |

---

## Testing

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=hedge_engine --cov-report=term

# Type checking
mypy hedge_engine/

# Linting
ruff check hedge_engine/
```

---

## Project Structure

```
hedge-engine/
├── hedge_engine/
│   ├── main.py       # FastAPI app factory
│   ├── api.py        # REST endpoints
│   ├── assessor.py   # Core risk assessment logic
│   └── models.py     # Data models
├── tests/
│   ├── test_api.py       # API endpoint tests
│   └── test_assessor.py  # Assessment logic tests
├── pyproject.toml
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
