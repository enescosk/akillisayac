import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
from pathlib import Path
import folium
from streamlit_folium import st_folium

from simulate_consumption import (
    CITIES,
    detect_anomalies,
    generate_consumption,
    load_or_generate,
)

DATA_PATH = Path("consumption.csv")

CITY_COORDS = {
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9334, 32.8597),
    "Izmir": (38.4237, 27.1428),
    "Bursa": (40.1828, 29.0667),
    "Adana": (36.9914, 35.3308),
    "Gaziantep": (37.0662, 37.3833),
    "Konya": (37.8746, 32.4932),
    "Antalya": (36.8969, 30.7133),
    "Kayseri": (38.7312, 35.4787),
    "Mersin": (36.8121, 34.6415),
}


def load_data() -> pd.DataFrame:
    """Load consumption data from CSV or generate if not present."""
    return load_or_generate(DATA_PATH)


def forecast_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Fit Prophet and return forecast dataframe for the given city."""
    model = Prophet()
    city_df = df[[city]].rename(columns={city: "y"})
    city_df["ds"] = city_df.index
    model.fit(city_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=72, freq="H")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def plot_consumption(df: pd.DataFrame, anomalies: pd.DataFrame, city: str) -> None:
    """Plot consumption with anomaly markers."""
    fig = px.line(df.reset_index(), x="index", y=city, labels={"index": "Time"})
    anom_points = df[city][anomalies[city]]
    fig.add_scatter(
        x=anom_points.index,
        y=anom_points,
        mode="markers",
        marker=dict(color="red", size=8),
        name="Anomaly",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_map(df: pd.DataFrame) -> None:
    """Display average consumption on a Folium map."""
    avg = df.mean()
    m = folium.Map(location=[39.0, 35.0], zoom_start=6)
    for city, avg_val in avg.items():
        lat, lon = CITY_COORDS[city]
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + (avg_val / avg.max()) * 10,
            popup=f"{city}: {avg_val:.1f} kWh",
            color="blue",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)
    st_folium(m, width=700, height=500)


def main() -> None:
    """Run the Streamlit forecasting app."""
    st.set_page_config(page_title="Electricity Forecast", layout="wide")
    st.title("Electricity Consumption Dashboard")

    data = load_data()
    anomalies = detect_anomalies(data)

    st.sidebar.header("Options")
    city = st.sidebar.selectbox("City", CITIES)
    date_range = st.sidebar.date_input(
        "Date range",
        [data.index.min().date(), data.index.max().date()],
        min_value=data.index.min().date(),
        max_value=data.index.max().date(),
    )
    show_table = st.sidebar.checkbox("Show raw data")

    start, end = date_range
    df_range = data.loc[str(start) : str(end)]
    anom_range = anomalies.loc[str(start) : str(end)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{city} Consumption")
        plot_consumption(df_range, anom_range, city)
        if show_table:
            st.dataframe(df_range[[city]])

    with col2:
        st.subheader("Average Consumption Map")
        show_map(df_range)

    st.subheader("72h Forecast")
    forecast = forecast_city(df_range, city)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence",
        )
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast as CSV",
        csv,
        file_name=f"{city}_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
