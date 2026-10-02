"""
XGBoost multi-class classifier for stock movement prediction.

Classes:
    0 = Down  (5-day forward return < -2%)
    1 = Flat  (5-day forward return in [-2%, +2%])
    2 = Up    (5-day forward return > +2%)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
LABEL_NAMES = ["Down", "Flat", "Up"]


def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    sample_weights: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
) -> xgb.XGBClassifier:
    """Train XGBoost with class-balanced sample weights and early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    # Use 85% for training, 15% as internal validation (for early stopping)
    split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
    w_tr = sample_weights[:split]

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  🌳 Best iteration: {model.best_iteration}")
    return model


def predict(model: xgb.XGBClassifier, X: np.ndarray):
    """Return (predicted class labels, class probabilities)."""
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    return preds, proba


def get_feature_importance(model: xgb.XGBClassifier, feature_cols: list) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def save_model(model: xgb.XGBClassifier, path=None):
    if path is None:
        path = MODELS_DIR / "xgboost_stock_model.pkl"
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")


def load_model(path=None) -> xgb.XGBClassifier:
    if path is None:
        path = MODELS_DIR / "xgboost_stock_model.pkl"
    return joblib.load(path)
