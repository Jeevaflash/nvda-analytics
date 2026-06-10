"""
NVIDIA Stock Analytics & Prediction Pipeline
Fetches latest data, engineers features, trains model, saves predictions.
Run daily via GitHub Actions.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


def fetch_data():
    """Download NVIDIA stock data from yfinance (full history)."""
    print("Fetching NVIDIA stock data...")
    nvda = yf.download("NVDA", period="max", auto_adjust=True)
    nvda.columns = nvda.columns.get_level_values(0)
    nvda = nvda.reset_index()
    nvda.columns = [c.lower() for c in nvda.columns]
    nvda = nvda.dropna()
    nvda["date"] = pd.to_datetime(nvda["date"])
    nvda = nvda.sort_values("date").reset_index(drop=True)
    print(f"  Loaded {len(nvda)} rows | {nvda['date'].min().date()} → {nvda['date'].max().date()}")
    return nvda


def engineer_features(df):
    """Add technical indicators and lag features."""
    df = df.copy()

    # Moving averages
    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # Price lags
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Daily return & rolling volatility
    df["daily_return"]   = df["close"].pct_change()
    df["volatility_10"]  = df["daily_return"].rolling(10).std()

    # RSI (14-period)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # Bollinger Band width
    mid     = df["close"].rolling(20).mean()
    std20   = df["close"].rolling(20).std()
    df["bb_width"] = (2 * std20) / (mid + 1e-9)

    # Target: next-day close
    df["target"] = df["close"].shift(-1)

    # Only keep last 2 years for stable post-split training
    cutoff = df["date"].max() - timedelta(days=730)
    df = df[df["date"] >= cutoff].copy()

    df = df.dropna().reset_index(drop=True)
    print(f"  Feature-engineered dataset: {len(df)} rows")
    return df


FEATURES = [
    "ma5", "ma10", "ma20", "ma50",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "daily_return", "volatility_10",
    "rsi", "macd", "bb_width",
    "volume"
]


def train_and_predict(df):
    """Train Linear Regression + Random Forest, return metrics and prediction."""
    X = df[FEATURES]
    y = df["target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    def metrics(actual, predicted):
        return {
            "mae":  round(float(mean_absolute_error(actual, predicted)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
            "r2":   round(float(r2_score(actual, predicted)), 4),
        }

    lr_metrics = metrics(y_test, lr_preds)
    rf_metrics = metrics(y_test, rf_preds)

    print(f"  LR  → MAE: {lr_metrics['mae']}  RMSE: {lr_metrics['rmse']}  R²: {lr_metrics['r2']}")
    print(f"  RF  → MAE: {rf_metrics['mae']}  RMSE: {rf_metrics['rmse']}  R²: {rf_metrics['r2']}")

    # Predict next trading day using last available row
    last_row = X.iloc[[-1]]
    lr_next  = float(lr.predict(last_row)[0])
    rf_next  = float(rf.predict(last_row)[0])

    # Build comparison dataframe (last 200 test points)
    comparison = pd.DataFrame({
        "date":      df["date"].iloc[split:].values,
        "actual":    y_test.values,
        "lr_pred":   lr_preds,
        "rf_pred":   rf_preds,
    })

    return {
        "lr_metrics":  lr_metrics,
        "rf_metrics":  rf_metrics,
        "lr_next":     round(lr_next, 2),
        "rf_next":     round(rf_next, 2),
        "comparison":  comparison,
    }


def build_summary(df):
    """Compute KPI summary stats."""
    latest      = df.iloc[-1]
    prev        = df.iloc[-2]
    price_chg   = latest["close"] - prev["close"]
    pct_chg     = (price_chg / prev["close"]) * 100
    ytd_start   = df[df["date"].dt.year == latest["date"].year]["close"].iloc[0]
    ytd_return  = ((latest["close"] - ytd_start) / ytd_start) * 100
    vol_avg30   = df["volume"].rolling(30).mean().iloc[-1]

    return {
        "latest_date":    str(latest["date"].date()),
        "latest_close":   round(float(latest["close"]), 2),
        "open":           round(float(latest["open"]), 2),
        "high":           round(float(latest["high"]), 2),
        "low":            round(float(latest["low"]), 2),
        "volume":         int(latest["volume"]),
        "price_change":   round(float(price_chg), 2),
        "pct_change":     round(float(pct_chg), 2),
        "all_time_high":  round(float(df["close"].max()), 2),
        "all_time_low":   round(float(df["close"].min()), 2),
        "ytd_return":     round(float(ytd_return), 2),
        "vol_avg30":      int(vol_avg30),
        "rsi_latest":     round(float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 0, 1),
    }


def save_outputs(df_full, df_feat, results, summary):
    """Persist everything the Streamlit app needs."""
    # 1. Full price history (for chart)
    price_hist = df_full[["date", "open", "high", "low", "close", "volume"]].copy()
    price_hist["date"] = price_hist["date"].astype(str)
    price_hist.to_csv("data/price_history.csv", index=False)

    # 2. Prediction comparison (for accuracy chart)
    comp = results["comparison"].copy()
    comp["date"] = comp["date"].astype(str)
    comp.to_csv("data/prediction_comparison.csv", index=False)

    # 3. Summary JSON (KPIs + predictions)
    summary["lr_next_price"]   = results["lr_next"]
    summary["rf_next_price"]   = results["rf_next"]
    summary["lr_metrics"]      = results["lr_metrics"]
    summary["rf_metrics"]      = results["rf_metrics"]
    summary["generated_at"]    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    with open("data/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  Saved: data/price_history.csv, data/prediction_comparison.csv, data/summary.json")


def main():
    import os
    os.makedirs("data", exist_ok=True)

    df_full = fetch_data()
    df_feat = engineer_features(df_full)
    results = train_and_predict(df_feat)

    # Attach engineered columns back to full df for summary (need rsi)
    df_summary = engineer_features(df_full)
    summary    = build_summary(df_summary)

    save_outputs(df_full, df_feat, results, summary)

    print("\n✅ Pipeline complete.")
    print(f"   Today's Close : ${summary['latest_close']}")
    print(f"   LR Prediction : ${results['lr_next']}")
    print(f"   RF Prediction : ${results['rf_next']}")


if __name__ == "__main__":
    main()
