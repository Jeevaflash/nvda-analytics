"""
NVIDIA Stock Analytics Dashboard
Streamlit app — reads pre-generated data files, no live API calls.
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NVDA Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0f; color: #e2e8f0; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

.kpi-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 0.5rem; }
.kpi-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 0.25rem; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 600; color: #f1f5f9; line-height: 1; }
.kpi-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
.kpi-up   { color: #4ade80; }
.kpi-down { color: #f87171; }

.pred-box { background: #0f1623; border: 1px solid #1d3a5c; border-radius: 10px; padding: 1.8rem 2rem; margin-top: 0.5rem; text-align: center; }
.pred-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #3b82f6; margin-bottom: 0.5rem; }
.pred-value { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #60a5fa; line-height: 1; }
.pred-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.4rem; }

.section-header { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #475569; border-bottom: 1px solid #1e1e2e; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1.5rem; }
.ticker { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; background: #1a1a2e; color: #76c7f0; padding: 0.2rem 0.6rem; border-radius: 4px; }
.top-bar { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1.8rem; }
.page-title { font-size: 1.5rem; font-weight: 600; color: #f1f5f9; margin: 0; }
.timestamp { font-size: 0.72rem; color: #334155; margin-left: auto; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    with open("data/summary.json") as f:
        summary = json.load(f)
    price_hist = pd.read_csv("data/price_history.csv", parse_dates=["date"])
    pred_comp  = pd.read_csv("data/prediction_comparison.csv", parse_dates=["date"])
    return summary, price_hist, pred_comp

try:
    summary, price_hist, pred_comp = load_data()
except FileNotFoundError:
    st.error("Data files not found. Run `python fetch_and_predict.py` first.")
    st.stop()


# ── Helpers ────────────────────────────────────────────────────────────────────
def signed(val):
    return ("▲", "kpi-up") if val >= 0 else ("▼", "kpi-down")

def kpi(label, value, sub="", color_class=""):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color_class}">{value}</div>
      {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────────
chg = summary["price_change"]
pct = summary["pct_change"]
arrow, chg_cls = signed(chg)
chg_str = f"{arrow} ${abs(chg):.2f} ({abs(pct):.2f}%)"

st.markdown(f"""
<div class="top-bar">
  <span class="ticker">NVDA</span>
  <span class="page-title">NVIDIA Stock Analytics</span>
  <span class="timestamp">Updated {summary['generated_at']}</span>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi("Today's Close", f"${summary['latest_close']}", summary['latest_date'])
with c2: kpi("Day Change", chg_str, f"Prev close ${summary['latest_close'] - chg:.2f}", chg_cls)
with c3: kpi("Day Range", f"${summary['low']} – ${summary['high']}")
with c4:
    y_arrow, y_cls = signed(summary['ytd_return'])
    kpi("YTD Return", f"{y_arrow} {abs(summary['ytd_return']):.1f}%", color_class=y_cls)
with c5: kpi("RSI (14)", f"{summary['rsi_latest']}", "Overbought >70 · Oversold <30")
with c6:
    vol_m = summary['volume'] / 1_000_000
    kpi("Volume", f"{vol_m:.1f}M", f"30d avg {summary['vol_avg30']//1_000_000:.1f}M")


# ── PRICE CHART ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price History — Last 2 Years</div>', unsafe_allow_html=True)

two_yr = price_hist[price_hist["date"] >= price_hist["date"].max() - pd.Timedelta(days=730)]
ma50   = two_yr["close"].rolling(50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=two_yr["date"], y=two_yr["close"],
    mode="lines", line=dict(color="#3b82f6", width=1.5),
    fill="tozeroy", fillcolor="rgba(59,130,246,0.07)", name="Close"))
fig.add_trace(go.Scatter(x=two_yr["date"], y=ma50,
    mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"), name="50-Day MA"))
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0), height=320,
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#64748b")),
    xaxis=dict(gridcolor="#1e1e2e"), yaxis=dict(gridcolor="#1e1e2e", tickprefix="$"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)


# ── MODEL ACCURACY + STATS ─────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown('<div class="section-header">Model Accuracy — Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    tail = pred_comp.tail(200)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tail["date"], y=tail["actual"],
        mode="lines", line=dict(color="#e2e8f0", width=1.5), name="Actual"))
    fig2.add_trace(go.Scatter(x=tail["date"], y=tail["rf_pred"],
        mode="lines", line=dict(color="#a78bfa", width=1.2, dash="dot"), name="Random Forest"))
    fig2.add_trace(go.Scatter(x=tail["date"], y=tail["lr_pred"],
        mode="lines", line=dict(color="#34d399", width=1, dash="dash"), name="Linear Regression"))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=240,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#64748b")),
        xaxis=dict(gridcolor="#1e1e2e"), yaxis=dict(gridcolor="#1e1e2e", tickprefix="$"),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)
    lr_m = summary["lr_metrics"]
    rf_m = summary["rf_metrics"]
    st.dataframe(pd.DataFrame({
        "Metric": ["MAE ($)", "RMSE ($)", "R²"],
        "Lin. Reg": [lr_m["mae"], lr_m["rmse"], lr_m["r2"]],
        "Rand. Forest": [rf_m["mae"], rf_m["rmse"], rf_m["r2"]],
    }).set_index("Metric"), use_container_width=True, height=145)
    kpi("All-Time High", f"${summary['all_time_high']}")
    kpi("All-Time Low",  f"${summary['all_time_low']}")


# ── PREDICTION BLOCK ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Next Trading Day Prediction</div>', unsafe_allow_html=True)

rf_price  = summary["rf_next_price"]
lr_price  = summary["lr_next_price"]
consensus = round((rf_price + lr_price) / 2, 2)
con_chg   = consensus - summary["latest_close"]
con_pct   = (con_chg / summary["latest_close"]) * 100
con_arrow, con_cls = signed(con_chg)

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-label">Random Forest</div>
      <div class="pred-value">${rf_price}</div>
      <div class="pred-sub">R² {rf_m['r2']} · MAE ${rf_m['mae']}</div>
    </div>""", unsafe_allow_html=True)
with p2:
    st.markdown(f"""
    <div class="pred-box" style="border-color:#2d4a6e; background:#0d1520;">
      <div class="pred-label" style="color:#60a5fa; font-size:0.7rem;">Consensus Forecast</div>
      <div class="pred-value" style="font-size:3.4rem; color:#93c5fd;">${consensus}</div>
      <div class="pred-sub" style="font-size:0.8rem;">
        <span class="{con_cls}">{con_arrow} ${abs(con_chg):.2f} ({abs(con_pct):.2f}%)</span>
        &nbsp;vs today's close
      </div>
    </div>""", unsafe_allow_html=True)
with p3:
    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-label">Linear Regression</div>
      <div class="pred-value">${lr_price}</div>
      <div class="pred-sub">R² {lr_m['r2']} · MAE ${lr_m['mae']}</div>
    </div>""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e1e2e;
     font-size:0.68rem; color:#334155; text-align:center; letter-spacing:0.05em;">
  Data sourced from Yahoo Finance · Predictions are model outputs, not financial advice ·
  Automated daily via GitHub Actions
</div>
""", unsafe_allow_html=True)
