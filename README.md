# Hedge-Engine

Low-latency hedge-sizing micro-service written in Python 3.10.

---

## Features
* FastAPI HTTP API with `/healthz`, `/hedge`, and `/metrics` endpoints.
* Monotonic spline sizing algorithm with hot-reload from YAML knots.
* Liquidity-weighted hedge sizing, sub-50 µs median latency.
* Async Postgres decision logging with circuit-breaker + fallback queue.
* Real-time Prometheus metrics.
* Dockerfile + docker-compose stack (API + Redis + Postgres).
* CI pipeline: lint, type-check, tests, docker build & smoke test.
* Canary-deploy workflow (staging ➜ production).

## Quick start (local)
```bash
# Build & start full stack
docker compose up --build

# Open interactive docs
open http://localhost:8000/docs
```

## Interactive demo notebook
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gregory-307/hedge-engine/blob/main/notebooks/hedge_engine_demo.py?format=py)

The notebook walks through:
1. Health check call.
2. Hedge sizing request with override score.
3. Scraping Prometheus metrics.
4. Using `compute_hedge` directly in Python.

---

© 2025 Hedge-Engine Dev Team – MIT License 