"""
Create classification labels for stock movement prediction.

Label: 5-day forward return
    Up   (2) : forward return >  +THRESHOLD
    Down (0) : forward return < -THRESHOLD
    Flat (1) : between -THRESHOLD and +THRESHOLD
"""
import pandas as pd

THRESHOLD    = 0.02   # 2% — below this in either direction is considered flat
FORWARD_DAYS = 5

LABEL_MAP   = {0: "Down", 1: "Flat", 2: "Up"}
LABEL_NAMES = ["Down", "Flat", "Up"]


def create_labels(df: pd.DataFrame, threshold: float = THRESHOLD, n_days: int = FORWARD_DAYS) -> pd.DataFrame:
    """
    Compute 5-day forward return per ticker and assign class labels.
    Drops the last n_days rows per ticker (no future return available).
    """
    frames = []
    for ticker, group in df.groupby("Ticker"):
        group = group.copy()
        group["forward_return"] = group["Close"].pct_change(n_days).shift(-n_days)
        group["label"] = 1  # default: Flat
        group.loc[group["forward_return"] >  threshold, "label"] = 2  # Up
        group.loc[group["forward_return"] < -threshold, "label"] = 0  # Down
        frames.append(group)

    result = pd.concat(frames)
    result = result.dropna(subset=["forward_return", "label"])
    result["label"] = result["label"].astype(int)

    label_counts = result["label"].value_counts().sort_index()
    print(f"📊 Label distribution:")
    for label, count in label_counts.items():
        pct = count / len(result) * 100
        print(f"   {LABEL_MAP[label]:>5} ({label}): {count:>5}  ({pct:.1f}%)")

    return result
