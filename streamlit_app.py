import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

from simulate_consumption import generate_consumption, detect_anomalies, CITIES

DATA_FILE = "weekly_consumption.csv"


def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    else:
        df = generate_consumption()
        df.to_csv(DATA_FILE)
    return df


def forecast_city(series, periods=72):
    df = series.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="H")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]


CITY_COORDS = {
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "Izmir": (38.4237, 27.1428),
    "Bursa": (40.1828, 29.0662),
    "Adana": (37.0000, 35.3213),
    "Gaziantep": (37.0662, 37.3833),
    "Konya": (37.8746, 32.4932),
    "Antalya": (36.8969, 30.7133),
    "Kayseri": (38.7225, 35.4875),
    "Mersin": (36.8121, 34.6415),
}


def main():
    st.title("Electricity Consumption Dashboard")

    data = load_data()
    anomalies = detect_anomalies(data)

    cities = st.sidebar.multiselect("Cities", CITIES, default=["Istanbul", "Ankara", "Izmir"])
    start, end = st.sidebar.date_input(
        "Date range",
        [data.index.min().date(), data.index.max().date()],
    )

    mask = (data.index.date >= start) & (data.index.date <= end)
    filtered = data.loc[mask, cities]
    filtered_anoms = anomalies.loc[mask, cities]

    for city in cities:
        st.subheader(city)
        fig = px.line(filtered, y=city, labels={"index": "Date", city: "Consumption"})
        anom_times = filtered.index[filtered_anoms[city]]
        fig.add_scatter(x=anom_times, y=filtered.loc[anom_times, city], mode="markers", marker_symbol="x", marker_color="red", name="Anomaly")
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox(f"Show forecast for {city}", key=city):
            forecast = forecast_city(data[city])
            ffig = px.line(forecast, x="ds", y="yhat", title=f"Forecast for {city}")
            st.plotly_chart(ffig, use_container_width=True)

    if st.checkbox("Show map of average consumption"):
        avg = data.mean()
        map_df = pd.DataFrame({
            "City": CITIES,
            "Avg": [avg[c] for c in CITIES],
            "lat": [CITY_COORDS[c][0] for c in CITIES],
            "lon": [CITY_COORDS[c][1] for c in CITIES],
        })
        fig = px.scatter_geo(map_df, lat="lat", lon="lon", text="City", size="Avg", scope="europe")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
