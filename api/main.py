"""
Stock Movement Prediction REST API

Endpoints:
    GET  /health          — Health check
    GET  /tickers         — List available tickers
    POST /predict         — Predict 5-day movement for a ticker on a given date
    POST /predict/batch   — Predict movement over a date range

Run with:
    uvicorn api.main:app --reload
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from src.data_loader import load_all_tickers, TICKERS
from src.features import engineer_features
from src.labels import LABEL_MAP, FORWARD_DAYS
from src.preprocessing import FEATURE_COLS
from src.model import load_model
from src.sentiment import score_headlines, sentiment_label

app = FastAPI(
    title="Stock Movement Prediction API",
    description="Predicts 5-day stock movement (Up/Flat/Down) using 25 technical indicators.",
    version="1.0.0",
)

# Load everything at startup
try:
    model        = load_model()
    scaler       = joblib.load(Path(__file__).resolve().parent.parent / "models" / "scaler.pkl")
    df           = load_all_tickers()
    df_featured  = engineer_features(df)
    print(f"✅ API ready. {len(TICKERS)} tickers loaded.")
except Exception as e:
    model = scaler = df_featured = None
    print(f"⚠️  Could not load model: {e}. Run src/train.py first.")


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str
    date: date
    headlines: list[str] = []   # Optional news headlines for sentiment context


class PredictResponse(BaseModel):
    ticker: str
    date: str
    prediction: str
    confidence: float
    probabilities: dict
    news_sentiment: float | None
    news_sentiment_label: str | None
    forward_days: int
    interpretation: str


class BatchPredictRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "tickers": TICKERS}


@app.get("/tickers")
def list_tickers():
    return {"tickers": TICKERS}


@app.post("/predict", response_model=PredictResponse)
def predict_single(request: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run src/train.py first.")

    ticker = request.ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"'{ticker}' not available. Choose from: {TICKERS}")

    target_date = pd.Timestamp(request.date)
    store = df_featured[
        (df_featured["Ticker"] == ticker) &
        (df_featured.index == target_date)
    ]

    if store.empty:
        ticker_df = df_featured[df_featured["Ticker"] == ticker]
        raise HTTPException(
            404,
            f"No data for {ticker} on {request.date}. "
            f"Available: {ticker_df.index.min().date()} to {ticker_df.index.max().date()}"
        )

    available = [f for f in FEATURE_COLS if f in store.columns]
    X_scaled  = scaler.transform(store[available].values)
    proba     = model.predict_proba(X_scaled)[0]
    pred_idx  = int(proba.argmax())
    pred_lbl  = LABEL_MAP[pred_idx]

    # Sentiment
    sentiment_score = None
    sentiment_lbl   = None
    if request.headlines:
        sentiment_score = round(score_headlines(request.headlines), 4)
        sentiment_lbl   = sentiment_label(sentiment_score)

    interpretations = {
        "Up":   f"Model expects {ticker} to rise >2% over the next {FORWARD_DAYS} trading days.",
        "Down": f"Model expects {ticker} to fall >2% over the next {FORWARD_DAYS} trading days.",
        "Flat": f"Model expects {ticker} to stay within ±2% over the next {FORWARD_DAYS} trading days.",
    }

    return PredictResponse(
        ticker=ticker,
        date=str(request.date),
        prediction=pred_lbl,
        confidence=round(float(proba.max()), 4),
        probabilities={
            "Down": round(float(proba[0]), 4),
            "Flat": round(float(proba[1]), 4),
            "Up":   round(float(proba[2]), 4),
        },
        news_sentiment=sentiment_score,
        news_sentiment_label=sentiment_lbl,
        forward_days=FORWARD_DAYS,
        interpretation=interpretations[pred_lbl],
    )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run src/train.py first.")

    ticker = request.ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"'{ticker}' not in {TICKERS}")

    mask = (
        (df_featured["Ticker"] == ticker) &
        (df_featured.index >= pd.Timestamp(request.start_date)) &
        (df_featured.index <= pd.Timestamp(request.end_date))
    )
    subset = df_featured[mask].copy()
    if subset.empty:
        raise HTTPException(404, f"No data for {ticker} in that date range.")

    available = [f for f in FEATURE_COLS if f in subset.columns]
    X_scaled  = scaler.transform(subset[available].values)
    proba     = model.predict_proba(X_scaled)
    preds     = proba.argmax(axis=1)

    results = [
        {
            "date":       str(dt.date()),
            "prediction": LABEL_MAP[int(preds[i])],
            "confidence": round(float(proba[i].max()), 4),
            "prob_down":  round(float(proba[i][0]), 4),
            "prob_flat":  round(float(proba[i][1]), 4),
            "prob_up":    round(float(proba[i][2]), 4),
        }
        for i, dt in enumerate(subset.index)
    ]

    return {
        "ticker":        ticker,
        "start_date":    str(request.start_date),
        "end_date":      str(request.end_date),
        "n_predictions": len(results),
        "predictions":   results,
    }
