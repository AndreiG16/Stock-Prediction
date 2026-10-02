"""
Technical indicator feature engineering — implemented from scratch with pandas.
No external TA library needed; shows understanding of the underlying math.
"""
import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, Bollinger Bands, ATR, OBV and momentum features.
    Expects columns: Open, High, Low, Close, Volume.
    """
    df    = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend: Moving Averages ───────────────────────────────────
    df["ema_20"]  = close.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = close.ewm(span=50, adjust=False).mean()
    df["sma_20"]  = close.rolling(20).mean()

    # ── Trend: MACD (12/26/9) ────────────────────────────────────
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"]   = df["macd"] - df["macd_signal"]

    # ── Momentum: RSI ────────────────────────────────────────────
    def rsi(series, period):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = rsi(close, 14)
    df["rsi_7"]  = rsi(close, 7)

    # ── Momentum: Stochastic Oscillator ─────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Volatility: Bollinger Bands (20, 2σ) ─────────────────────
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── Volatility: ATR (14) ─────────────────────────────────────
    prev_close     = close.shift(1)
    tr             = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]   = tr.rolling(14).mean()

    # ── Volume: OBV ──────────────────────────────────────────────
    direction   = np.sign(close.diff()).fillna(0)
    df["obv"]   = (direction * vol).cumsum()
    df["volume_ratio"] = vol / vol.rolling(20).mean()

    # ── Price-based returns ──────────────────────────────────────
    df["return_1d"]  = close.pct_change(1)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Price relative to moving averages (normalised)
    df["price_vs_ema20"] = close / df["ema_20"] - 1
    df["price_vs_ema50"] = close / df["ema_50"] - 1
    df["ema_cross"]      = df["ema_20"] / df["ema_50"] - 1   # golden/death cross

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators per ticker, then drop NaN rows."""
    frames = []
    for ticker, group in df.groupby("Ticker"):
        enriched = add_technical_indicators(group.copy())
        frames.append(enriched)
    result = pd.concat(frames)
    result = result.dropna()
    print(f"✅ Features engineered: {result.shape[1]} columns, {len(result):,} rows after dropping NaN")
    return result
