"""
app.py
------
Streamlit dashboard for the Electricity Load Forecasting project.

Run locally:
    streamlit run app.py

Deploy:
    Push this whole folder to GitHub, then deploy on
    https://share.streamlit.io (Streamlit Community Cloud) pointing
    at app.py. See README.md for full steps.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import build_features, FEATURES

st.set_page_config(page_title="Electricity Load Forecaster", layout="wide")

st.title("⚡ Electricity Load Forecasting Dashboard")
st.caption(
    "A machine-learning model (Random Forest) forecasts short-term "
    "electricity demand from historical load, time-of-day/week/season "
    "patterns, and temperature."
)


@st.cache_data
def load_data():
    df = pd.read_csv("load_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["features"]


df_raw = load_data()
model, feature_list = load_model()
df_feat = build_features(df_raw)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Forecast settings")
horizon = st.sidebar.slider("Hours to forecast ahead", min_value=6, max_value=72, value=24, step=6)
temp_offset = st.sidebar.slider(
    "Temperature adjustment (°C)", min_value=-10, max_value=10, value=0,
    help="Shift the assumed future temperature up/down from its seasonal average, to explore what-if scenarios."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About the model**\n\n"
    "- Algorithm: Random Forest Regressor\n"
    "- Features: hour, day of week, month, weekend flag, temperature, "
    "load 1 hour ago, load same hour yesterday"
)

# ---------------- Historical view ----------------
st.subheader("Historical Load")
recent_days = st.slider("Show last N days of history", 3, 60, 14)
recent = df_raw[df_raw["datetime"] >= df_raw["datetime"].max() - pd.Timedelta(days=recent_days)]
st.line_chart(recent.set_index("datetime")[["load_mw"]])

# ---------------- Forecast ----------------
st.subheader(f"Forecast: next {horizon} hours")

last_known = df_feat.iloc[-1:].copy()
history = df_feat.copy()

future_rows = []
current = history.iloc[-1].to_dict()
last_datetime = history["datetime"].iloc[-1]

for step in range(1, horizon + 1):
    next_dt = last_datetime + pd.Timedelta(hours=step)

    row = {
        "datetime": next_dt,
        "hour": next_dt.hour,
        "day_of_week": next_dt.dayofweek,
        "month": next_dt.month,
        "is_weekend": int(next_dt.dayofweek >= 5),
        "temperature_c": history["temperature_c"].iloc[-24:].mean() + temp_offset,
        "load_lag_1": current["load_mw"] if step == 1 else future_rows[-1]["load_mw"],
        "load_lag_24": history["load_mw"].iloc[-(24 - step)] if step <= 24
            else future_rows[-24]["load_mw"],
    }
    X_row = pd.DataFrame([{f: row[f] for f in feature_list}])
    pred = model.predict(X_row)[0]
    row["load_mw"] = pred
    future_rows.append(row)

forecast_df = pd.DataFrame(future_rows)

combined = pd.concat([
    df_raw[["datetime", "load_mw"]].tail(48).assign(type="History"),
    forecast_df[["datetime", "load_mw"]].assign(type="Forecast"),
])

chart_data = combined.pivot(index="datetime", columns="type", values="load_mw")
st.line_chart(chart_data)

col1, col2, col3 = st.columns(3)
col1.metric("Peak forecasted load", f"{forecast_df['load_mw'].max():.2f} MW")
col2.metric("Average forecasted load", f"{forecast_df['load_mw'].mean():.2f} MW")
col3.metric("Lowest forecasted load", f"{forecast_df['load_mw'].min():.2f} MW")

with st.expander("See raw forecast table"):
    st.dataframe(forecast_df[["datetime", "load_mw", "temperature_c"]])

st.markdown("---")
st.caption(
    "Note: this demo uses synthetic data generated to mimic realistic "
    "daily/weekly/seasonal demand patterns. Swap load_data.csv for a "
    "real utility dataset to use this for actual grid planning."
)
