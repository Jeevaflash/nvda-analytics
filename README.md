# NVIDIA Stock Analytics & Prediction Dashboard

Live dashboard: [Your Streamlit Community Cloud link here]

## What this does

- Fetches full NVIDIA stock history from Yahoo Finance daily
- Engineers 15 technical indicators (MA, RSI, MACD, Bollinger Bands, lags)
- Trains Linear Regression + Random Forest on the last 2 years of data
- Predicts next trading day's closing price
- Displays results on a minimalist Streamlit dashboard

## Project structure

```
├── fetch_and_predict.py        # data pipeline + ML (run daily)
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── .streamlit/config.toml      # dark theme config
├── .github/workflows/
│   └── update.yml              # GitHub Actions daily automation
└── data/                       # auto-generated, committed by bot
    ├── price_history.csv
    ├── prediction_comparison.csv
    └── summary.json
```

## Deploy steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/nvda-analytics.git
git push -u origin main
```

### 2. Run pipeline once to generate data/
```bash
pip install -r requirements.txt
python fetch_and_predict.py
git add data/
git commit -m "add initial data"
git push
```

### 3. Deploy to Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Connect your GitHub account
3. Select this repo, branch: main, main file: app.py
4. Deploy — you'll get a public link immediately

### 4. GitHub Actions auto-runs every weekday at 11:30 PM IST
No further setup needed. The bot commits new data/ files and
Streamlit Cloud auto-refreshes from the updated repo.
