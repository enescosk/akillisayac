"""Streamlit dashboard for smart electricity meter analytics."""
from __future__ import annotations

# from pathlib import Path  # not used
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
import folium

from anomaly import detect_anomalies
from data_utils import CITIES, CSV_PATH, get_consumption_data
from forecast import forecast_consumption

st.set_page_config(page_title="Smart Electricity Analytics", layout="wide")

# -------------------- Helpers -------------------- #

CITY_COORDS: Dict[str, tuple[float, float]] = {
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "Izmir": (38.4237, 27.1428),
    "Bursa": (40.195, 29.06),
    "Adana": (37.0000, 35.3213),
    "Gaziantep": (37.0662, 37.3833),
    "Konya": (37.8746, 32.4932),
    "Antalya": (36.8969, 30.7133),
    "Kayseri": (38.7225, 35.4875),
    "Mersin": (36.8121, 34.6415),
}

# -------------------- Data Loading -------------------- #

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Load consumption data and compute anomalies."""
    df = get_consumption_data()
    df = detect_anomalies(df)
    return df


def filter_by_city_and_dates(df: pd.DataFrame, city: str, start_date, end_date) -> pd.DataFrame:
    mask_city = df["city"] == city
    mask_date = (df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)
    return df.loc[mask_city & mask_date].copy()


# -------------------- Sidebar -------------------- #

df_all = load_data()

st.sidebar.header("Filters")
city_selected: str = st.sidebar.selectbox("Select city", options=CITIES, index=0)

min_date = df_all["datetime"].dt.date.min()
max_date = df_all["datetime"].dt.date.max()

def_date = (min_date, max_date)
date_range = st.sidebar.date_input(
    "Date range", value=def_date, min_value=min_date, max_value=max_date
)

# Streamlit returns a single `datetime.date` if single selection mode is activated. We
# expect a tuple for start & end. Ensure correct unpacking.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range  # type: ignore[misc]

if start_date > end_date:
    st.sidebar.error("Start date cannot be after end date.")

# -------------------- Line Plot with Anomalies -------------------- #

df_city = filter_by_city_and_dates(df_all, city_selected, start_date, end_date)

st.subheader(f"Electricity Consumption - {city_selected}")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_city["datetime"],
        y=df_city["consumption"],
        mode="lines",
        name="Consumption",
        line=dict(color="#1f77b4"),
    )
)

# Add anomalies
anomalies = df_city[df_city["anomaly"]]
fig.add_trace(
    go.Scatter(
        x=anomalies["datetime"],
        y=anomalies["consumption"],
        mode="markers",
        name="Anomalies",
        marker=dict(color="red", size=6, symbol="x"),
    )
)

fig.update_layout(
    height=400,
    xaxis_title="Datetime",
    yaxis_title="Consumption (kWh)",
    legend=dict(orientation="h", y=-0.2, x=0),
)

st.plotly_chart(fig, use_container_width=True)

# -------------------- Forecast -------------------- #

# We don't cache forecast because Prophet models are lightweight for 7 days of hourly data.
def get_forecast(city_df: pd.DataFrame):
    _, fc = forecast_consumption(city_df)
    return fc

st.subheader("72-hour Forecast")
forecast_df = get_forecast(df_city)

forecast_fig = go.Figure()
forecast_fig.add_trace(
    go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        name="Forecast",
        line=dict(color="#2ca02c"),
    )
)

# Shadow actuals for training period
forecast_fig.add_trace(
    go.Scatter(
        x=df_city["datetime"],
        y=df_city["consumption"],
        mode="lines",
        name="Historical",
        line=dict(color="gray", dash="dot"),
    )
)

forecast_fig.update_layout(
    height=400,
    xaxis_title="Datetime",
    yaxis_title="Consumption (kWh)",
    legend=dict(orientation="h", y=-0.2, x=0),
)

st.plotly_chart(forecast_fig, use_container_width=True)

csv_data = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name=f"{city_selected.lower()}_forecast.csv",
    mime="text/csv",
)

# -------------------- Map View -------------------- #

st.subheader("Average Consumption by City (Last 7 Days)")

avg_df = (
    df_all[df_all["datetime"] >= df_all["datetime"].max() - pd.Timedelta(days=7)]
    .groupby("city")["consumption"]
    .mean()
    .reset_index()
)

m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="cartodbpositron")

max_avg = avg_df["consumption"].max()
for _, row in avg_df.iterrows():
    city = row["city"]
    avg_val = row["consumption"]
    lat, lon = CITY_COORDS[city]
    folium.CircleMarker(
        location=(lat, lon),
        radius=5 + 15 * avg_val / max_avg,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6,
        popup=f"{city}: {avg_val:.1f} kWh",
    ).add_to(m)

st_data = st_folium(m, width=700, height=450)

# -------------------- Utility -------------------- #

st.sidebar.markdown("---")
if st.sidebar.button("Regenerate Data"):
    # Delete CSV to force regeneration
    if CSV_PATH.exists():
        CSV_PATH.unlink()
    st.sidebar.success("Data will be regenerated on next load. Please reload the page.") 