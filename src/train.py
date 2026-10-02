"""
Main training pipeline — Stock Movement Classifier.

Steps:
  1. Download OHLCV data via yfinance (cached to /data)
  2. Engineer 25 technical indicator features
  3. Create Up/Flat/Down labels from 5-day forward returns
  4. Time-aware train/test split (no random splits for time series!)
  5. Train XGBoost classifier with balanced class weights
  6. Evaluate: Macro F1, confusion matrix, feature importance
  7. Save model to /models

Run with:
    python -m src.train
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_tickers
from src.features import engineer_features
from src.labels import create_labels
from src.preprocessing import train_test_split_by_date
from src.model import train_model, predict, get_feature_importance, save_model
from src.evaluate import (
    print_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_label_distribution,
    plot_confidence_distribution,
)


def main():
    print("\n" + "═" * 55)
    print("  Stock Movement Classifier — Training Pipeline")
    print("═" * 55)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n🔄 Step 1: Downloading price data...")
    df = load_all_tickers()

    # ── 2. Feature engineering ────────────────────────────────────
    print("\n🔧 Step 2: Engineering technical indicators...")
    df_featured = engineer_features(df)

    # ── 3. Labels ─────────────────────────────────────────────────
    print("\n🏷️  Step 3: Creating movement labels (±2% threshold, 5-day forward)...")
    df_labeled = create_labels(df_featured)

    # ── 4. Split ──────────────────────────────────────────────────
    print("\n✂️  Step 4: Time-aware train/test split...")
    X_train, X_test, y_train, y_test, sample_weights, feature_cols = train_test_split_by_date(df_labeled)

    # ── 5. Train ──────────────────────────────────────────────────
    print("\n🚀 Step 5: Training XGBoost classifier...")
    model = train_model(X_train, y_train, sample_weights)

    # ── 6. Evaluate ───────────────────────────────────────────────
    print("\n📊 Step 6: Evaluating on test set...")
    preds, proba = predict(model, X_test)
    f1 = print_report(y_test, preds)

    print("\n📌 Top 10 Feature Importances:")
    importance_df = get_feature_importance(model, feature_cols)
    print(importance_df.head(10).to_string(index=False))

    # ── 7. Plots ──────────────────────────────────────────────────
    print("\n📈 Step 7: Generating plots...")
    plot_label_distribution(y_train, y_test)
    plot_confusion_matrix(y_test, preds)
    plot_confidence_distribution(proba, y_test)
    plot_feature_importance(importance_df)

    # ── 8. Save ───────────────────────────────────────────────────
    save_model(model)
    print(f"\n✅ Done. Macro F1: {f1:.4f} | Check /plots for visualizations.")


if __name__ == "__main__":
    main()
