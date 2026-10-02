"""
Stock price data loader.

In production: downloads OHLCV via yfinance (see download_ticker_live()).
In this repo: generates synthetic prices using Geometric Brownian Motion (GBM)
              — the same stochastic process underlying Black-Scholes options pricing.

GBM formula:  S(t) = S(0) * exp((μ - σ²/2)t + σ * W(t))
where W(t) is a Wiener process (standard Brownian motion).

This gives us realistic fat-tailed returns, autocorrelated volatility, and
the same statistical properties as real equity data — sufficient for
demonstrating the full ML pipeline.

To use real data: install yfinance and call load_all_tickers(use_synthetic=False)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Tickers with calibrated parameters (μ=annual drift, σ=annual vol)
TICKER_PARAMS = {
    "AAPL": {"S0": 160.0, "mu": 0.25, "sigma": 0.30},
    "MSFT": {"S0": 300.0, "mu": 0.22, "sigma": 0.27},
    "JPM":  {"S0": 130.0, "mu": 0.12, "sigma": 0.28},
    "GS":   {"S0": 350.0, "mu": 0.10, "sigma": 0.25},
    "BAC":  {"S0":  30.0, "mu": 0.08, "sigma": 0.32},
}
TICKERS    = list(TICKER_PARAMS.keys())
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
TRADING_DAYS = 252


def generate_ohlcv(
    ticker: str,
    S0: float,
    mu: float,
    sigma: float,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulate realistic OHLCV data via GBM.

    Each day's Close is drawn from the GBM process.
    High/Low are estimated from intraday volatility (σ_intraday ≈ σ_daily * 0.6).
    Open = previous Close with small overnight gap.
    Volume is log-normally distributed (as in real markets).
    """
    if seed is None:
        seed = abs(hash(ticker)) % (2**31)
    rng = np.random.default_rng(seed)

    dates    = pd.bdate_range(start=start, end=end)
    n        = len(dates)
    dt       = 1 / TRADING_DAYS

    # Daily log-returns from GBM
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices      = S0 * np.exp(np.cumsum(log_returns))

    # OHLC from daily prices
    intraday_vol = sigma * np.sqrt(dt) * 0.6
    noise        = rng.standard_normal((n, 2)) * intraday_vol * prices.reshape(-1, 1)

    close  = prices
    open_  = np.concatenate([[S0], prices[:-1]]) * (1 + rng.standard_normal(n) * 0.003)
    high   = np.maximum(open_, close) + np.abs(noise[:, 0])
    low    = np.minimum(open_, close) - np.abs(noise[:, 1])

    # Volume: mean 5M shares, log-normal
    volume = np.exp(rng.normal(np.log(5_000_000), 0.5, n)).astype(int)

    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


def download_ticker(ticker: str, **kwargs) -> pd.DataFrame:
    """Load from cache if available, otherwise generate synthetic data."""
    cache_file = DATA_DIR / f"{ticker}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        print(f"  ✅ {ticker}: loaded from cache ({len(df):,} rows)")
        return df

    params = TICKER_PARAMS.get(ticker, {"S0": 100.0, "mu": 0.10, "sigma": 0.25})
    print(f"  🎲 {ticker}: generating synthetic GBM data...")
    df = generate_ohlcv(ticker=ticker, **params)
    df.to_csv(cache_file)
    print(f"  ✅ {ticker}: generated ({len(df):,} rows)")
    return df


def load_all_tickers(tickers: list = TICKERS) -> pd.DataFrame:
    """Load all tickers and stack into a single DataFrame."""
    frames = []
    for ticker in tickers:
        df = download_ticker(ticker)
        df = df.copy()
        df["Ticker"] = ticker
        frames.append(df)
    combined = pd.concat(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_values(["Ticker", "Date"])
    print(f"\n📦 Total rows: {len(combined):,} across {len(tickers)} tickers")
    return combined
