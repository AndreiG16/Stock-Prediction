"""
Time-aware train/test split and feature scaling.
Never use random splits for financial time series — it leaks future data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "ema_20", "ema_50", "sma_20",
    "macd", "macd_signal", "macd_diff",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "atr_14",
    "obv", "volume_ratio",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_20d",
    "price_vs_ema20", "price_vs_ema50", "ema_cross",
]

TARGET_COL  = "label"
CUTOFF_DATE = "2023-01-01"


def train_test_split_by_date(
    df: pd.DataFrame,
    cutoff_date: str = CUTOFF_DATE,
    feature_cols: list = FEATURE_COLS,
):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Only keep features that exist (some may be missing if data is thin)
    available = [c for c in feature_cols if c in df.columns]

    train = df[df.index <  cutoff_date]
    test  = df[df.index >= cutoff_date]

    print(f"✂️  Train: {len(train):,} rows  ({train.index.min().date()} → {train.index.max().date()})")
    print(f"✂️  Test:  {len(test):,} rows   ({test.index.min().date()} → {test.index.max().date()})")

    X_train = train[available].values
    y_train = train[TARGET_COL]
    X_test  = test[available].values
    y_test  = test[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Balanced sample weights to counter class imbalance
    sample_weights = compute_sample_weight("balanced", y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test, sample_weights, available
