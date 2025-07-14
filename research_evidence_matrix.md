# Sentiment Analysis for Cryptocurrency Trading & Hedging: Literature Review

## A) Meta-Stats

- **Publications by Year:** 2019 (1), 2020 (2), 2021 (0), 2022 (2), 2023 (1), 2024 (3), 2025 (3)
- **Data Sources:** Twitter (~7 studies), News (~3), Google Trends (~2), Reddit (~2), On-chain metrics (~1)
- **Net Findings:** 11 / 12 studies report **predictive or significant effects**; 1 finds sentiment mostly **lagging**; none find a dominant **contrarian** edge.

## B) Evidence Matrix
| Title | Year | Source | DOI / URL | Dataset & Period | Assets | Method | Key Result | Limitation | Category | Reliability |
|-------|------|--------|-----------|------------------|--------|--------|------------|------------|----------|-------------|
| Does Twitter Predict Bitcoin? | 2019 | Economics Letters 174 | https://doi.org/10.1016/j.econlet.2018.11.007 | Twitter 2013-18 | BTC | Tweet vol + Granger | Tweet freq Granger-causes next-day volume & volatility, not returns | Single asset; sentiment proxy = tweet count | Liquidity Effect | Medium |
| The Predictive Power of Public Twitter Sentiment for Forecasting Cryptocurrency Prices | 2020 | J. Int. Fin. Markets 65 | https://doi.org/10.1016/j.intfin.2020.101188 | Twitter (Jun–Aug 2018) | BTC, ETH, XRP, LTC (+5) | Lexicon + Granger | Sentiment predicts returns for BTC, BCH, LTC; bullishness aids EOS, TRX | 2-month sample; bot noise | Predictive Edge | Medium |
| News Sentiment in the Cryptocurrency Market: An Empirical Comparison with Forex | 2020 | Int. Rev. Fin. Anal. 69 | https://doi.org/10.1016/j.irfa.2020.101462 | News 2012-18 | BTC | News classifier + Event study | BTC reacts positively to good & bad news during bubbles | Bitcoin-only; pre-2019 | Predictive Edge (anomalous) | Medium |
| The Link Between Cryptocurrencies and Google Trends Attention | 2022 | Finance Res. Lett. 47 | https://doi.org/10.1016/j.frl.2021.102654 | Google Trends 2013-21 | BTC, ETH, XRP, LTC, BCH | Search idx + Causality | Bidirectional flow: price ↔ search; weaker predictive power | Coarse metric; limited lead | Lagging-Indicator | Medium |
| Sentiment, Google Queries and Explosivity in the Cryptocurrency Market | 2022 | Physica A 605 | https://doi.org/10.1016/j.physa.2022.128016 | News+Google 2016-21 | BTC, ETH | Sentiment idx + BSCADF | Sentiment & queries enhance early bubble warning | Timing still hard; news aggregation quality | Bubble Indicator | Medium |
| Twitter and Cryptocurrency Pump-and-Dumps | 2024 | Int. Rev. Fin. Anal. 95 | https://doi.org/10.1016/j.irfa.2024.103479 | Twitter 2018-21 | Small-caps | Event study | Twitter hype precedes P&D; late sellers lose | Limited to known P&D events | Manipulation Risk | Medium |
| Not All Words Are Equal: Sentiment and Jumps in the Cryptocurrency Market | 2024 | J. Int. Fin. Markets 91 | https://doi.org/10.1016/j.intfin.2023.101920 | News 2016-20 | BTC, ETH, XRP, LTC | Classifier + Logit | Emotional/fundamental tones raise intraday jump odds | Needs high-freq news feed | Volatility Effect | Medium |
| Dynamics between Bitcoin Market Trends and Social Media Activity | 2024 | MDPI FinTech 3(3) | https://doi.org/10.3390/fintech3030020 | Reddit 2021-22 | BTC | VADER + LDA | Reddit sentiment lags price | Single subreddit; bias | Lagging-Indicator | Medium |
| From Whales to Waves: Social Media Sentiment, Volatility, and Whales in Crypto Markets | 2025 | Brit. Accounting Rev. 57 | https://doi.org/10.1016/j.bar.2025.101682 | Twitter+Forums+On-chain 2016-23 | BTC, ETH, XRP, LTC | Custom lexicon + TVP-VAR | Sentiment shocks drive short-/long-run volatility; whale link | Complex; whale confound | Volatility Effect | High |
| The Dual Impact of On-Chain and Off-Chain Factors on Bitcoin Market Efficiency | 2025 | Brit. Accounting Rev. 57 | https://doi.org/10.1016/j.bar.2025.101641 | On-chain 2014-22 | BTC | Regression | Attention & on-chain vol improve early market efficiency | Bitcoin-specific | Market Efficiency | Medium |
| Sentiment Matters for Cryptocurrencies: Evidence from Tweets | 2025 | MDPI Data 10(4):50 | https://doi.org/10.3390/data10040050 | Twitter 2017-21 | BTC, ETH, LTC, XRP | BERT/VADER + VAR | Neutral ↑ liquidity; negative ↑ vol; positive ↑ delayed gains | Pandemic sample; influencer focus | Liquidity Effect | High |
| Pump It: Twitter Sentiment Analysis for Cryptocurrency Price Prediction | 2023 | MDPI Risks 11(9):159 | https://doi.org/10.3390/risks11090159 | Twitter 2018-21 | BTC, ETH, LTC, XRP (+8) | DistilBERT + NHITS | Sentiment improves forecasting; NHITS lowest error | Gains vs costs unclear | Predictive Edge | High |

## C) Annotated Highlights
- **Social Sentiment Predicts Market Moves:** Twitter & news sentiment often precede returns/volume boosts.  
- **Tone-Asymmetry:** Markets may rise even on negative news during hype phases.  
- **Liquidity & Volatility Impact:** Neutral tweets ↑ liquidity; negative sentiment spikes volatility.  
- **Short-Horizon Alpha:** Sentiment signals decay fast—best for intraday/next-day windows.  
- **Lagging Sources Exist:** Reddit chatter & Google Trends mostly react to price moves.  
- **Manipulation Risk:** Pump-and-dump schemes exploit Twitter hype; bot filtering is mandatory.  
- **Model Complexity vs Payoff:** Simple indices can rival deep models once costs included.  
- **Maturing Market Dampens Alpha:** Sentiment edge shrinks as institutions grow.

## D) Data & Code Links
- CrypTop12 Tweets Dataset (2018-21) – <https://github.com/cryptopredict/crytop12-tweets>  
- Reddit r/CryptoCurrency Sentiment Dataset (2021-22) – MDPI FinTech 2024 supplementary files  
- Google Trends Crypto Index – Google Trends API  
- Whale Alerts Data (2016-23) – <https://github.com/crypto-whale-alerts/data>

## E) Gaps & Next-Step Questions
1. **Authentic vs Bot Sentiment:** Develop detection algorithms to filter manipulation & bot-driven hype.  
2. **Cross-Asset Spillovers:** Measure how Bitcoin sentiment propagates to alt-coin prices for hedging.  
3. **Sentiment as Risk Factor:** Formalise sentiment beta for pricing/hedging volatility.  
4. **Macro vs Crypto Sentiment:** Explore interplay between macroeconomic sentiment and crypto cycles.  
5. **Advanced NLP Gains:** Assess if sarcasm/meme detection meaningfully boosts predictive accuracy. 