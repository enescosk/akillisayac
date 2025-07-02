"""Forecasting utilities using Prophet."""

from __future__ import annotations

from typing import Literal, Tuple

import pandas as pd
from prophet import Prophet


def _prepare_prophet_frame(consumption: pd.Series) -> pd.DataFrame:
    """Return DataFrame with Prophet-required columns 'ds' and 'y'."""
    return pd.DataFrame({"ds": consumption.index, "y": consumption.values})


def forecast_city(
    consumption: pd.DataFrame | pd.Series,
    city: str,
    periods: int = 72,
    freq: Literal["h", "H"] = "h",
) -> pd.DataFrame:
    """Forecast *periods* hours ahead for *city* using Prophet.

    Parameters
    ----------
    consumption : wide DataFrame or Series
    city : str
        City name present in *consumption*.
    periods : int, default 72
        Number of hours to forecast.
    freq : str, default "h"
        Frequency string passed to Prophet future frame.

    Returns
    -------
    pd.DataFrame
        Prophet forecast DataFrame with columns like ['ds', 'yhat', ...].
    """
    if isinstance(consumption, pd.DataFrame):
        series = consumption[city]
    else:
        series = consumption

    df = _prepare_prophet_frame(series)
    # Prophet requires timezone-naive 'ds'
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None)

    model = Prophet(daily_seasonality=True, weekly_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(
        periods=periods, freq=freq, include_history=True
    )
    forecast = model.predict(future)
    return forecast


def forecast_consumption(
    city_df: pd.DataFrame, *, periods: int = 72
) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a Prophet model to *city_df* and return (model, forecast_df).

    *city_df* must contain 'datetime' and 'consumption' columns for a single city.
    The returned *forecast_df* follows Prophet output schema including 'yhat'.
    """
    prophet_df = city_df.rename(columns={"datetime": "ds", "consumption": "y"})[
        ["ds", "y"]
    ].copy()

    # Prophet prefers timezone-naive timestamps
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_convert(None)

    m = Prophet(
        daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False
    )
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=periods, freq="H", include_history=True)
    forecast = m.predict(future)
    return m, forecast
