# Dollar Decline Hedge Analyzer 💵📉
**Portfolio Project #10 | Asset Correlation | Python | yfinance | scipy**

**[Live Dashboard →](https://dollar-losing-value-project-tedpoihphzohguurpsc7ak.streamlit.app/)**

## Results at a Glance

| Asset | Outperformance During Weakness | DXY Correlation | Verdict |
|-------|-------------------------------|-----------------|---------|
| Gold | **+0.37%** | -0.11 | ✅ Only true hedge |
| Bitcoin | -0.47% | +0.03 (p=0.786) | ❌ Not a dollar hedge |
| Silver | -2.61% | +0.28 | ❌ Moves with dollar |
| Ethereum | -3.05% | +0.29 | ❌ Risk asset behavior |
| Oil | -35.2% | **+0.74** | ❌ Amplifies dollar moves |

> **Gold was the ONLY asset that directionally hedged dollar weakness.** Bitcoin showed near-zero correlation (p=0.786) — statistically indistinguishable from noise — directly challenging the digital gold narrative with 10 years of empirical data.

---

## Overview

This project builds an empirical framework to identify which asset classes — gold, silver, crude oil, Bitcoin, Ethereum, international equities, and REITs — historically hedge U.S. dollar weakness using 10 years of monthly data (2015-2024).

Rather than relying on financial media narratives, this project tests hedge effectiveness statistically using correlation analysis, regime detection, and rolling window correlation to show whether relationships strengthen or weaken over time.

The core economic question: **When the dollar weakens, which assets actually protect purchasing power — and does the data support the narratives?**

---

## Dataset

**Source:** Yahoo Finance via `yfinance` — 8 assets, monthly frequency, 2015–2024 (120 observations)

| Asset | Ticker | Role in Analysis |
|-------|--------|-----------------|
| U.S. Dollar Index | DX-Y.NYB | Benchmark — measured against all assets |
| Gold ETF | GLD | Classic safe haven and traditional dollar hedge |
| Silver ETF | SLV | Monetary and industrial metal — hedge candidate |
| Crude Oil ETF | USO | Globally priced in USD — theoretical dollar hedge |
| International Equities | EFA | Non-US stocks benefit from dollar weakness in theory |
| REIT | VNQ | Hard asset — inflation and dollar hedge candidate |
| Bitcoin | BTC-USD | Digital gold narrative — tested empirically here |
| Ethereum | ETH-USD | Broader crypto — higher beta than BTC |

---

## Methodology

**1. Dollar Weakness Regime Detection**
DXY periods labeled Weakness when trading below its 12-month moving average. Found 29 weakness months (24.2%) and 91 strength months (75.8%) over the sample.

**2. Regime Performance Analysis**
Average monthly return calculated for each asset during weakness vs strength regimes. T-test applied to test statistical significance of performance differences.

**3. Correlation Analysis**
Pearson correlation computed between monthly DXY returns and each asset's monthly returns. Negative = directional hedge. P-values reported for all assets.

**4. Rolling 24-Month Correlation**
Rolling window shows whether hedge effectiveness is strengthening or weakening over time — critical for understanding current market relevance.

---

## Key Findings

- **Gold is the only empirically validated dollar hedge** — correlation of -0.11, +17.7% cumulative return during weakness months, and effectiveness **strengthening since 2022**
- **Bitcoin's correlation with DXY is +0.025 (p=0.786)** — statistically indistinguishable from zero. The digital gold narrative is not supported by 10 years of data
- **Oil has the strongest correlation at +0.74 — but it's positive** — oil amplifies dollar moves rather than hedging them. Supply shocks (OPEC, COVID, shale) dominated currency effects over this period
- **Silver moves WITH the dollar** (+0.28, statistically significant) due to industrial demand dominating its monetary properties
- **Gold's rolling hedge effectiveness has been strengthening since 2022**, making it more relevant to the current dollar weakness episode than the full-sample average suggests
- Dollar weakness in this sample was violent and fast (-11.89% avg DXY) vs gradual strengthening (+21.23%) — confirming why investors rotate to hedges quickly when weakness hits

---

## Dollar Regime Analysis

| Regime | Months | Avg DXY Return |
|--------|--------|---------------|
| Dollar Weakness (DXY below 12M MA) | 29 months (24.2%) | -11.89% |
| Dollar Strength (DXY above 12M MA) | 91 months (75.8%) | +21.23% |

**Major Weakness Episodes Identified:**
- Jun 2018 — Trade war uncertainty, Fed pivot fears
- Dec 2019 — Pre-COVID risk-on environment
- Mar 2020 — COVID crash, Fed flooded system with liquidity
- Dec 2021 — Inflation era, real rates deeply negative

---

## How to Run

```bash
pip install yfinance pandas numpy matplotlib seaborn scipy
# No external downloads needed — yfinance pulls all data directly
jupyter notebook dollar_decline_hedge_analyzer.ipynb
```

---

## Tools & Libraries

- Python 3.12
- `yfinance` — all asset price data
- `pandas` / `numpy` — data manipulation and regime labeling
- `scipy` — Pearson correlation, t-tests
- `matplotlib` / `seaborn` — 5-panel dashboard visualization

---

## About

Built as part of an Economics portfolio by an economics major exploring machine learning applications in finance and policy. Original research on a live market question — dollar weakness accelerating in 2025-2026.
