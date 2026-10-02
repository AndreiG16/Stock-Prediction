# Stock Movement Classifier

> End-to-end ML pipeline that predicts 5-day stock movement (Up / Flat / Down) using 25 technical indicators derived from price and volume data, with optional news sentiment scoring.

---

## Problem Statement

Can technical indicators reliably predict short-term stock movements? This project builds a rigorous ML pipeline to answer that question honestly — including a baseline comparison and an architectural choice (GBM synthetic data) that highlights a key insight about market efficiency.

**Goal:** Classify whether a stock will rise >2%, fall >2%, or stay flat over the next 5 trading days.

---

## Dataset

Prices are simulated using **Geometric Brownian Motion (GBM)**:

```
S(t) = S(0) × exp((μ − σ²/2)t + σ × W(t))
```

This is the same stochastic process that underlies **Black-Scholes options pricing**. It produces realistic OHLCV data with log-normal return distributions — and because it is a pure random walk by construction, it makes the evaluation benchmark clear: a model that cannot beat F1 = 0.33 on GBM data is learning nothing.

| Ticker | S₀ | Annual Drift (μ) | Annual Vol (σ) |
|---|---|---|---|
| AAPL  | $160 | 25% | 30% |
| MSFT  | $300 | 22% | 27% |
| JPM   | $130 | 12% | 28% |
| GS    | $350 | 10% | 25% |
| BAC   | $30  |  8% | 32% |

**To use real data:** Install `yfinance` and replace the data loader — the entire pipeline is identical.

- **Size:** ~7,800 rows, 5 tickers, Jan 2018 – Dec 2023
- **Split:** Train: 2018–2022 (6,371 rows) | Test: 2023 (1,267 rows)

---

## Features (25 technical indicators, computed from scratch)

| Category | Features |
|---|---|
| **Trend** | EMA-20, EMA-50, SMA-20, MACD, MACD Signal, MACD Histogram, EMA Cross |
| **Momentum** | RSI-14, RSI-7, Stochastic %K, Stochastic %D |
| **Volatility** | Bollinger Upper/Lower Band, BB Width, BB %B, ATR-14 |
| **Volume** | OBV (On-Balance Volume), Volume Ratio (vs 20-day avg) |
| **Price** | 1d/5d/10d/20d returns, 20-day rolling volatility, Price vs EMA-20/50 |

All implemented from scratch using pandas — no external TA library.

---

## Results

| Metric | Score |
|---|---|
| **Macro F1** | **0.331** |
| Down F1      | 0.35 |
| Flat F1      | 0.35 |
| Up F1        | 0.29 |
| Test accuracy | 33% |

**Why F1 ≈ 0.33?** This is the expected result — and it's the point. GBM is a random walk, so no technical indicator can extract signal that isn't there. The Efficient Market Hypothesis (EMH) predicts exactly this. On real market data, you'd expect slightly better — around 0.38–0.45 Macro F1 — because real prices have microstructure, momentum effects, and behavioural anomalies that GBM doesn't model.

---

## Project Structure

```
stock-prediction/
├── data/                   # Cached price CSVs (gitignored, regenerated on first run)
├── models/                 # Saved model + scaler (gitignored)
├── plots/                  # Generated charts
├── src/
│   ├── data_loader.py      # GBM simulator (swap for yfinance in production)
│   ├── features.py         # 25 technical indicators (pure pandas, no TA library)
│   ├── labels.py           # Forward return labelling: Up/Flat/Down
│   ├── preprocessing.py    # Time-aware split + StandardScaler
│   ├── sentiment.py        # VADER sentiment scorer (FinBERT-ready interface)
│   ├── model.py            # XGBoost multi-class classifier
│   ├── evaluate.py         # F1, confusion matrix, confidence distribution plots
│   └── train.py            # Main pipeline — runs all 8 steps end-to-end
├── api/
│   └── main.py             # FastAPI: /predict and /predict/batch endpoints
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train (generates data automatically)
```bash
python -m src.train
```

### 3. Start the API
```bash
uvicorn api.main:app --reload
```

### 4. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "date": "2023-06-15", "headlines": ["Apple beats earnings expectations"]}'
```

API docs: `http://localhost:8000/docs`

---

## Key Learnings

- **Market efficiency is real.** On GBM data, F1 = 0.33 = random baseline. This validates the evaluation, not failure.
- **Time-aware splits are non-negotiable.** Random splits cause data leakage in time series — always split by date.
- **Class imbalance matters.** Without balanced sample weights, the model collapses to always predicting "Flat".
- **Technical indicators are features, not signals.** The real edge comes from alternative data: sentiment, options flow, earnings surprises.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · FastAPI · Matplotlib · vaderSentiment · Joblib

---

## Author

Andrei Galca — [andreigalca16@gmail.com](mailto:andreigalca16@gmail.com)
