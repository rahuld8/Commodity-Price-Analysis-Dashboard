# -------------------------------------------------------
# 🌾 Commodity Price Analysis Dashboard (FYTD + Hybrid Forecast)
# Final — ✅ Market & District Filters Added (Dependent on State)
# -------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Commodity Price Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Database Configuration
# -----------------------------
CONNECTION_STRING = "db_connection_string"

TABLE_NAME = "commodity_mandi_price"

# -----------------------------
# Utility Functions
# -----------------------------
@st.cache_data(ttl=900)
def load_data_from_db(conn_str, table_name):
    engine = create_engine(conn_str)
    query = text(f"""
        SELECT * FROM {table_name}
        WHERE arrival_date >= '2025-04-01' AND arrival_date <= '2026-03-31';
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df.columns = [c.lower() for c in df.columns]
    if "arrival_date" in df.columns:
        df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce", dayfirst=True)
        df["month_year"] = df["arrival_date"].dt.to_period("M").dt.to_timestamp()
    for c in ["min_price", "max_price", "avg_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def format_inr(x):
    try:
        if pd.isna(x):
            return "N/A"
        return f"₹{x:,.0f}"
    except Exception:
        return "N/A"

def compute_volatility(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    mean = series.mean()
    return (series.std() / mean * 100) if mean != 0 else np.nan

# -----------------------------
# Load Data (FYTD only)
# -----------------------------
st.markdown("""
<div style="background:transparent;padding-bottom:6px;">
<h1 style='color:#F2F2F2;margin:6px 0;'>📊 Commodity Price Analysis Dashboard</h1>
<p style='color:#AAAAAA;margin:0 0 10px 0;'>FYTD data with Hybrid Forecast & Volatility (Now includes District & Market filters).</p>
</div>
""", unsafe_allow_html=True)

try:
    df = load_data_from_db(CONNECTION_STRING, TABLE_NAME)
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ No data available for FY 2025–26.")
    st.stop()

# -----------------------------
# Sidebar Filters (Dark Theme + Hierarchy + Defaults)
# -----------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .css-1lcbmhc { padding-top: 12px; }
    .sidebar-title { color:#E6EEF8; font-weight:700; margin-bottom:6px; }
    .sidebar-sub { color:#AEB8C6; font-size:12px; margin-bottom:8px; }
    .sidebar-divider { border-top:1px solid rgba(255,255,255,0.1); margin:10px 0 12px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-title'>🎛️ Dashboard Filters</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Filtered by: Commodity → State → District → Market → Date</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

# 🌿 Commodity filter
commodity_opts = sorted(df["commodity_name"].dropna().unique())
commodity_default = ["Soyabean"] if "Soyabean" in commodity_opts else (commodity_opts[:1] if commodity_opts else [])
commodity = st.sidebar.multiselect("🌿 **Select Commodity**", options=commodity_opts, default=commodity_default)

# 🗾 State filter
state_opts = sorted(df["state"].dropna().unique())
state_default = ["Rajasthan"] if "Rajasthan" in state_opts else ([state_opts[0]] if state_opts else [])
state = st.sidebar.multiselect("🗾 **Select State**", options=state_opts, default=state_default)

# 🏙️ District filter (depends on State)
district_opts = sorted(df[df["state"].isin(state)]["district"].dropna().unique())
district_default = ["Kota"] if "Kota" in district_opts else (district_opts[:1] if district_opts else [])
district = st.sidebar.multiselect("🏙️ **Select District**", options=district_opts, default=district_default)

# 🏪 Market filter (depends on District)
market_opts = sorted(df[df["district"].isin(district)]["market"].dropna().unique())
market_default = ["Itawa"] if "Itawa" in market_opts else (market_opts[:1] if market_opts else [])
market = st.sidebar.multiselect("🏪 **Select Market**", options=market_opts, default=market_default)

# 🗓️ Date filter
min_date, max_date = df["arrival_date"].min().date(), df["arrival_date"].max().date()
date_range = st.sidebar.date_input(
    "📅 **Select Date Range**",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# -----------------------------
# Apply Filters to Data
# -----------------------------
mask = (
    (df["arrival_date"].dt.date >= date_range[0]) &
    (df["arrival_date"].dt.date <= date_range[1])
)
if commodity:
    mask &= df["commodity_name"].isin(commodity)
if state:
    mask &= df["state"].isin(state)
if district:
    mask &= df["district"].isin(district)
if market:
    mask &= df["market"].isin(market)

df_f = df[mask].copy()
if df_f.empty:
    st.warning("⚠️ No data found for selected filters — showing all FYTD data.")
    df_f = df.copy()

# -----------------------------
# Styling (Dark theme + KPI card CSS)
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0f1115; }
.kpi-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
.kpi { flex:1 1 23%; min-width:220px; background:linear-gradient(135deg,#22242A,#1B1C20);
    border-radius:12px; padding:18px; color:#FFF; box-shadow:0 6px 20px rgba(0,0,0,.6);
    border:1px solid rgba(255,255,255,.02); height:130px; display:flex; flex-direction:column;
    justify-content:space-between;}
.kpi .label { color:#BFC7D4; font-size:13px; display:flex; align-items:center; gap:8px; }
.kpi .value { color:#FFF; font-size:20px; font-weight:700; }
.kpi .desc { color:#8b93a1; font-size:12px; }
.commodity-title { color:#F2F2F2; font-weight:700; font-size:20px; margin:12px 0 6px; }
@media(max-width:900px){.kpi{flex:1 1 48%;}}
@media(max-width:520px){.kpi{flex:1 1 100%;}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ✅ KPI Section (per commodity)
# -----------------------------
if "commodity_name" in df_f.columns:
    selected_comms = sorted(df_f["commodity_name"].unique())
    for comm in selected_comms:
        df_c = df_f[df_f["commodity_name"] == comm]
        avg_price = df_c["avg_price"].mean()
        min_price = df_c["min_price"].min()
        max_price = df_c["max_price"].max()
        volatility = compute_volatility(df_c["avg_price"])
        st.markdown(f"<div class='commodity-title'>🌾 {comm}</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-row">
        <div class="kpi"><div class="label">💰 Average Price</div>
        <div class="value">{format_inr(avg_price)}</div><div class="desc">Per Quintal</div></div>
        <div class="kpi"><div class="label">📉 Lowest Price</div>
        <div class="value">{format_inr(min_price)}</div><div class="desc">Recorded min</div></div>
        <div class="kpi"><div class="label">📈 Highest Price</div>
        <div class="value">{format_inr(max_price)}</div><div class="desc">Recorded max</div></div>
        <div class="kpi"><div class="label">⚠️ Volatility</div>
        <div class="value">{f'{volatility:.2f}%' if pd.notna(volatility) else 'N/A'}</div><div class="desc">Std dev / mean</div></div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# 📈 Monthly Trend
# -----------------------------
st.markdown("### 📈 Monthly Price Trend")
df_trend = (
    df_f.groupby(["commodity_name", pd.Grouper(key="arrival_date", freq="M")])
    .agg(avg_price=("avg_price", "mean"))
    .reset_index()
)
fig_trend = px.line(
    df_trend,
    x="arrival_date",
    y="avg_price",
    color="commodity_name",
    markers=True,
    title="Average Price Trend (Monthly)"
)
fig_trend.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Avg Price (₹)",
    hovermode="x unified",
    plot_bgcolor="#111",
    paper_bgcolor="#111",
    font=dict(color="#fff")
)
st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------
# 🔮 Hybrid Forecast (Prophet + ARIMA + Baseline)
# -----------------------------
st.markdown("### 🔮 Hybrid Forecasting (Prophet + ARIMA + Baseline)")

forecast_choice = st.selectbox(
    "Select forecast period",
    ["7 days", "15 days", "1 month (30 days)"],
    index=1
)
forecast_period = {"7 days": 7, "15 days": 15, "1 month (30 days)": 30}[forecast_choice]

for comm in selected_comms:
    df_c = df_f[df_f["commodity_name"] == comm].copy()
    if df_c.empty:
        continue

    df_prophet = df_c.groupby("arrival_date")["avg_price"].mean().reset_index()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)
    if len(df_prophet) > 60:
        df_prophet = df_prophet.groupby(pd.Grouper(key="ds", freq="W"))["y"].mean().reset_index()
    df_prophet["y"] = df_prophet["y"].rolling(window=3, min_periods=1).mean()

    with st.spinner(f"⏳ Generating hybrid forecast for {comm}..."):
        model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.03
        )
        try:
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=forecast_period, freq="D")
            prophet_df = model.predict(future)
        except Exception as e:
            st.warning(f"Prophet fit failed for {comm}: {e}. Using fallback forecast.")
            prophet_vals = np.repeat(df_prophet["y"].iloc[-1] if len(df_prophet) else 0.0, forecast_period)
            prophet_df = pd.DataFrame({"ds": pd.date_range(df_prophet["ds"].max() + timedelta(days=1), periods=forecast_period), "yhat": prophet_vals})

    prophet_vals = prophet_df["yhat"].iloc[-forecast_period:].values
    try:
        arima_series = df_prophet["y"].dropna()
        if len(arima_series) >= 10:
            arima_model = ARIMA(arima_series, order=(2, 1, 2)).fit()
            arima_forecast = arima_model.forecast(steps=forecast_period)
            arima_vals = np.array(arima_forecast)
        else:
            arima_vals = np.repeat(arima_series.iloc[-1] if len(arima_series) else 0.0, forecast_period)
    except Exception:
        arima_vals = np.repeat(df_prophet["y"].iloc[-1] if len(df_prophet) else 0.0, forecast_period)

    baseline_avg = df_prophet["y"].iloc[-7:].mean() if len(df_prophet) >= 7 else df_prophet["y"].mean()
    baseline_vals = np.repeat(baseline_avg if not pd.isna(baseline_avg) else 0.0, forecast_period)

    hybrid_forecast = (0.5 * prophet_vals + 0.3 * arima_vals + 0.2 * baseline_vals)

    future_dates = pd.date_range(df_prophet["ds"].max() + timedelta(days=1), periods=forecast_period, freq="D")
    forecast_df = pd.DataFrame({"ds": future_dates, "hybrid": hybrid_forecast})

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Historical", line=dict(color="#00b894", width=2)))
    fig_forecast.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["hybrid"], mode="lines+markers", name="Hybrid Forecast", line=dict(color="#0984e3", width=3, dash="dash")))
    fig_forecast.update_layout(
        title=f"📊 {comm} — {forecast_choice} Hybrid Forecast",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(color="#fff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    latest_val = df_prophet["y"].iloc[-1] if len(df_prophet) else np.nan
    next_forecast_val = hybrid_forecast[-1] if len(hybrid_forecast) else np.nan
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{comm} — Latest", format_inr(latest_val))
    with col2:
        st.metric(f"{comm} — Forecast ({forecast_choice})", format_inr(next_forecast_val))

# -----------------------------
# ⚙️ Volatility Chart
# -----------------------------
st.markdown("### ⚙️ Volatility Distribution by Commodity")
vol_data = (
    df_f.groupby("commodity_name")["avg_price"]
    .apply(lambda x: compute_volatility(x))
    .reset_index(name="volatility")
)
fig_vol = px.bar(
    vol_data,
    x="commodity_name",
    y="volatility",
    color="volatility",
    color_continuous_scale="Viridis",
    title="Volatility (%) by Commodity"
)
fig_vol.update_layout(template="plotly_dark", plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#fff"))
st.plotly_chart(fig_vol, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)
st.caption("📊 FY 2025–26 Data | Hybrid forecasting (Prophet + ARIMA + Baseline) |")

