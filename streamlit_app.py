import pandas as pd
import streamlit as st
from prophet import Prophet


def load_data(path):
    """Load consumption data from a CSV file."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def forecast_city(df, city):
    """Fit Prophet and return forecast dataframe for the given city."""
    model = Prophet()
    city_df = df[[city]].rename(columns={city: "y"})
    city_df["ds"] = city_df.index
    model.fit(city_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=24)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def main():
    """Run the Streamlit forecasting app."""
    st.title("Electricity Consumption Forecast")
    data = load_data("consumption.csv")
    city = st.selectbox("City", data.columns)
    forecast = forecast_city(data, city)
    st.line_chart(forecast.set_index("ds")["yhat"])


if __name__ == "__main__":
    main()
