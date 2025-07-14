# Research Integration Notes – v2025-07-14

This memo maps the July 2025 literature review to our current architecture and converts insights into concrete backlog items. Keep it living: strike-through or date-stamp tasks as they complete.

---

## 1  Key Takeaways
1. Liquidity/volatility effects are stronger and more consistent than pure return prediction.  
2. Sentiment alpha decays within ~24 h; sub-hour bars needed for influencer spikes.  
3. Bot & pump manipulation are pervasive — filtering is **P0**.  
4. Reddit & Google Trends largely lag price → treat as regime/context, not triggers.  
5. Model complexity has diminishing returns; start lean, benchmark later.  
6. BTC sentiment spills into alts; whale transfers correlate with vol spikes.  
7. Sentiment + search indices can flag bubble / explosivity regimes.

---

## 2  Gap Analysis
| Gap | Impact | Priority | Action |
|-----|--------|----------|--------|
| Liquidity feature undefined | Hedge sizing may ignore best predictor | P0 | Spec & extract `spread_Δ`, `depth1pct`, `whale_tx_ct` |
| Bot filter only stub | Edge may invert due to bot noise | P0 | Implement `BotFilter` + test harness |
| Time-decay absent | Signals overweight stale data | P1 | Add `decay.py` util (λ≈0.7/24 h) |
| Reddit/Trends weight unbounded | Lag signals could dominate | P1 | Cap weight ≤ 10 % or move to regime flag |
| Cross-asset spillover missing | Hedge engine blind to BTC->ETH risk | P1 | Add BTC sentiment feature to ETH scorer; simple VAR study |
| Eval lacks liquidity metrics | No proof hedge improves spread | P1 | Log Δspread, VWAP slippage, realised vol |
| Bubble alert absent | May miss crash early-warnings | P2 | Prototype BSCADF bubble score |
| Alt channels (Telegram/Discord) | Manipulation early signals missing | P2 | Add ingest backlog post-MVP |

---

## 3  Re-prioritised Roadmap (pre-model)
1. **Data Schema Update (P0)** – include liquidity & whale fields; add `timestamp_ms`.  
2. **Bot Filter (P0)** – production-grade; blocklist, Botometer, heuristic fallback.  
3. **Time-Decay Features (P1)** – exponential decay utility; configurable half-life.  
4. **Liquidity Metrics Extractor (P1)** – compute spread/depth from order-book snapshots.  
5. **Cross-Asset VAR Study (P1)** – quantify BTC sentiment spillover to ETH returns.  
6. **Bubble/Vol Alert Module (P2)** – jump/logit or BSCADF on sentiment + search.  
7. **Extended Ingestion Backlog (P2)** – Telegram, Discord, whale-alert, YouTube.

---

## 4  Pending Artefacts
- [x] `docs/research_evidence_matrix.md` (committed)  
- [ ] Update `sentiment-pipeline/README.md` with new P0-P1 tasks  
- [ ] Create GitHub issues / dev logs for each gap-analysis action  
- [ ] Link evidence matrix from `MasterPlan.md`

---

## 5  Questions for Next Sync
1. Stable-coins & L2 tokens: in scope?  
2. Desired intraday bar granularity (5-min vs 1-min)?  
3. Acceptable Botometer API cost ceiling?  
4. Who owns Telegram/Discord ingestion – existing infra or new micro-service? 