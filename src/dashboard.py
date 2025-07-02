from __future__ import annotations

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from anomaly import detect_anomalies
from data_utils import CITIES, load_consumption
from forecast import forecast_city

# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Smart Electricity Analytics", layout="wide")

st.title("⚡ Smart Electricity Meter Analytics Dashboard")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def get_consumption() -> pd.DataFrame:
    """Load or generate consumption dataset (cached for 1 hour)."""
    return load_consumption()


consumption = get_consumption()
ALL_CITIES = [c["name"] for c in CITIES]

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    selected_cities = st.multiselect(
        "Select cities",
        options=ALL_CITIES,
        default=ALL_CITIES[:5],
    )

    min_date = consumption.index.min().date()
    max_date = consumption.index.max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()
    fc_city = st.selectbox(
        "City for forecast",
        options=selected_cities or ALL_CITIES,
        index=0,
    )
    run_forecast = st.button("Run 72-hour Forecast")

# ---------------------------------------------------------------------------
# Data filtering based on selections
# ---------------------------------------------------------------------------

start_date, end_date = (
    date_range if isinstance(date_range, tuple) else (date_range, date_range)
)
mask = (consumption.index.date >= start_date) & (consumption.index.date <= end_date)
filtered = consumption.loc[mask]

if selected_cities:
    filtered = filtered[selected_cities]

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

anomalies = detect_anomalies(filtered)

# ---------------------------------------------------------------------------
# Time-series plot
# ---------------------------------------------------------------------------

st.subheader("Hourly Consumption")

long_df = filtered.reset_index().melt(
    id_vars="index", var_name="City", value_name="Consumption"
)
fig_ts = px.line(long_df, x="index", y="Consumption", color="City")

for city in filtered.columns:
    city_anoms = anomalies[city]
    if city_anoms.any():
        fig_ts.add_scatter(
            x=filtered.index[city_anoms],
            y=filtered[city][city_anoms],
            mode="markers",
            marker=dict(color="red", size=6, symbol="x"),
            name=f"Anomaly – {city}",
            showlegend=False,
        )

st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Folium map visualization
# ---------------------------------------------------------------------------

st.subheader("Average Consumption Map")

avg_vals = filtered.mean()
low_q, high_q = avg_vals.quantile([0.33, 0.66])

m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="cartodbpositron")

for city in CITIES:
    name, lat, lon = city["name"], city["lat"], city["lon"]
    if name not in avg_vals:
        continue
    val = avg_vals[name]
    color = "green" if val <= low_q else "yellow" if val <= high_q else "red"
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"{name}: {val:.1f} kWh",
    ).add_to(m)

st_folium(m, width=750, height=500)

# ---------------------------------------------------------------------------
# Forecast section
# ---------------------------------------------------------------------------

if run_forecast:
    st.subheader(f"72-hour Forecast for {fc_city}")
    forecast_df = forecast_city(consumption, fc_city)

    fig_fc = px.line(
        forecast_df,
        x="ds",
        y="yhat",
        labels={"ds": "Date", "yhat": "Predicted Consumption (kWh)"},
    )
    st.plotly_chart(fig_fc, use_container_width=True)
