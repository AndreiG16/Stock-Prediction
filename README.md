# Sales Forecasting — Rossmann Store Sales

> End-to-end time series forecasting pipeline comparing Moving Average, Prophet, and XGBoost models on real retail sales data.

---

## Problem Statement

Rossmann operates over 3,000 drug stores across 7 European countries. Store managers need reliable 6-week sales forecasts to plan staffing, inventory, and promotions. Inaccurate forecasts lead to overstocking, understaffing, and lost revenue.

**Goal:** Predict daily sales for each store, accounting for promotions, school holidays, seasonality, and competition.

---

## Dataset

- **Source:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- **Size:** ~1M rows, 1,115 stores, Jan 2013 – Jul 2015
- **Key challenge:** Stores vary significantly by type, location, assortment, and promotion strategy

**Download instructions:**
1. Go to the Kaggle link above
2. Download `train.csv` and `store.csv`
3. Place both files in the `/data` folder

---

## Approach

Three models trained and compared:

| Model | Type | Key Idea |
|---|---|---|
| Moving Average | Statistical Baseline | Average of last 14 days — if ML can't beat this, it's useless |
| Prophet | Additive Decomposition | Automatically handles seasonality and trend changes |
| XGBoost | Gradient Boosting | Lag features + rolling stats, strongest performer |

---

## Results

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Moving Average | 1,444.28 | 1,892.24 | 23.61% |
| Prophet | 680.80 | 759.72 | 15.58% |
| **XGBoost** | **641.03** | **888.57** | **10.16%** |

---

## Project Structure

```
sales-forecasting/
├── data/                   # Place train.csv and store.csv here
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── src/
│   ├── data_loader.py      # Load and merge CSVs
│   ├── preprocessing.py    # Feature engineering and train/test split
│   ├── baseline_model.py   # Moving average baseline
│   ├── prophet_model.py    # Facebook Prophet model
│   ├── xgboost_model.py    # XGBoost with lag features
│   ├── evaluate.py         # MAE, RMSE, MAPE + plots
│   └── train.py            # Main script — runs all models
├── api/
│   └── main.py             # FastAPI forecast endpoint
├── models/                 # Saved model files (git-ignored)
├── plots/                  # Generated forecast charts
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add data
Download `train.csv` and `store.csv` from Kaggle and place in `/data`.

### 3. Train all models
```bash
python -m src.train
```

### 4. Start the API
```bash
uvicorn api.main:app --reload
```

### 5. Make a forecast request
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "start_date": "2015-06-01", "end_date": "2015-06-30"}'
```

API docs available at: `http://localhost:8000/docs`

---

## Key Findings

- **Promotions are the #1 predictor** (30% feature importance) — whether a store runs a promotion dominates every other signal including seasonality and store type
- **XGBoost beats Prophet on average error** (10.2% MAPE vs 15.6%) but Prophet wins on RMSE — XGBoost is more accurate day-to-day, Prophet handles large spikes better
- **Rolling 28-day average** is the second strongest feature — recent sales history is a better predictor than raw lag values
- **Both ML models beat the baseline by a large margin** — Moving Average hits 23.6% MAPE, confirming that the added complexity is justified

---

## Key Learnings

- **Time-aware splits matter:** Random train/test splits cause data leakage in time series — always split by date
- **Lag features are powerful:** Knowing sales from 7/14/28 days ago is the strongest signal for XGBoost
- **Baseline first:** Always benchmark against a simple model — complexity is only justified if it beats the baseline
- **Prophet vs XGBoost:** Prophet is easier to interpret and handles trend changes automatically; XGBoost is more accurate when you invest in feature engineering

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Prophet · FastAPI · Matplotlib · Joblib

---

## Author

Andrei Galca — [andreigalca16@gmail.com](mailto:andreigalca16@gmail.com)
