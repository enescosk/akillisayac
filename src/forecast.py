"""Forecasting utilities using Prophet."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from prophet import Prophet


def forecast_consumption(
    city_df: pd.DataFrame, *, periods: int = 72
) -> Tuple[Prophet, pd.DataFrame]:
    """Fit a Prophet model to *city_df* and return (model, forecast_df).

    *city_df* must contain 'datetime' and 'consumption' columns for a single city.
    The returned *forecast_df* follows Prophet output schema including 'yhat'.
    """
    prophet_df = (
        city_df.rename(columns={"datetime": "ds", "consumption": "y"})[
            ["ds", "y"]
        ]
        .copy()
    )

    # Prophet prefers timezone-naive timestamps
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_convert(None)

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=periods, freq="H", include_history=True)
    forecast = m.predict(future)
    return m, forecast 