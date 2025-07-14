# Repo-Specific Planning Hub  
*Version 0.1 – initial scaffold*

> This file tracks granular plans for each repository. Update sections as tasks are scoped or completed. Keep entries concise—link out to detailed issues or logs when needed.

---

## 1  web-search-sdk
**Purpose**  Async search (Google, news) with full-article parsing; feeds sentiment pipeline.

| Area | Status | Next Step |
|------|--------|-----------|
| CLI export (`export_articles`) | Prototype works for Google; news scraping WIP | Extend to NewsAPI & output gzip NDJSON to S3 |
| Proxy rotation | Basic list hard-coded | Move to dynamic pool & health-check |
| Response schema | JSON Pydantic model v0.1 | Align with `sentiment-pipeline` schema once defined |

_Backlog & Questions_: TBD

---

## 2  twitter-sdk
**Purpose**  High-throughput tweet ingestion & virality detection.

| Area | Status | Next Step |
|------|--------|-----------|
| Stream ingest | Sample stream in dev env | Switch to filtered stream (crypto keywords) |
| Bot filtering | Not implemented | Integrate `BotFilter` from sentiment-pipeline |
| Export CLI (`export`) | Missing | Write exporter to gzip NDJSON in S3 layout |
| **LLM pre-clean** | Missing | Tokenise & drop non-English/emoji-only tweets using Llama tokenizer |

_Backlog & Questions_: TBD

---

## 3  sentiment-pipeline (planned)
**Purpose**  Cleaning, feature engineering, and sentiment scoring.

| Area | Status | Next Step |
|------|--------|-----------|
| Repo skeleton | Not created | Init repo with pyproject, logs/ *(see* `docs/sentiment_pipeline_plan.md` *)* |
| Data schema | Draft in MasterPlan | Implement `schema.py` with liquidity & whale fields |
| BotFilter | Not started | Design interface + Botometer integration |
| **Model Zoo** | Not started | Add `llama_local.py`, `gpt4_gateway.py`, `models/router.py` for tiered LLM sentiment |
| **Cost control** | Not started | Implement token quota & Redis cache for Tier-2 LLM calls |

_Backlog & Questions_: TBD

---

## 4  ml-backtester (planned)
**Purpose**  Historical replay & metric calculations.

| Area | Status | Next Step |
|------|--------|-----------|
| Repo skeleton | Not created | Init with data_loader.py & engine stub |
| Liquidity metrics | Not implemented | Script to compute spread Δ & depth1pct |

_Backlog & Questions_: TBD

---

## 5  hedge-engine (planned)
**Purpose**  Map Swing Score + liquidity delta → hedge size; expose REST API.

| Area | Status | Next Step |
|------|--------|-----------|
| API stub | Not created | Flask/FastAPI endpoint returning placeholder JSON |
| Dockerization | Not started | Write Dockerfile for local/CI spin-up |
| **Detailed Plan** | Created | See `docs/hedge_engine_plan.md` |

_Backlog & Questions_: TBD

---

## How to Update This File
1. Edit the relevant section—keep table rows short.  
2. Commit with message `docs: update repo_plans <repo>`. 