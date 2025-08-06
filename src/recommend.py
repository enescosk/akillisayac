from __future__ import annotations

import pandas as pd
from typing import List


def generate_suggestions(forecast_df: pd.DataFrame) -> List[str]:
    """Generate two actionable suggestions based on 72-hour forecast.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Prophet forecast output containing at least columns ``ds`` (datetime)
        and ``yhat`` (predicted kWh).
    Returns
    -------
    list[str]
        Two suggestion strings (max 2 sentences each).
    """
    # Ensure datetime and prediction columns exist
    if {"ds", "yhat"}.issubset(forecast_df.columns):
        df = forecast_df.copy()
    else:
        raise ValueError("forecast_df must contain 'ds' and 'yhat' columns")

    # Use local (naive) time for hourly grouping
    df["hour"] = df["ds"].dt.hour
    hourly_mean = df.groupby("hour")["yhat"].mean()

    peak_hour = int(hourly_mean.idxmax())
    off_hour = int(hourly_mean.idxmin())

    # Format hour ranges (peak ±2h)
    peak_start = (peak_hour - 1) % 24
    peak_end = (peak_hour + 1) % 24
    off_start = (off_hour - 1) % 24
    off_end = (off_hour + 1) % 24

    suggestion1 = (
        f"Run high-load appliances (laundry, dishwasher) between {off_start:02d}:00–{off_end:02d}:00 "
        "when demand and tariffs are lowest."
    )
    suggestion2 = (
        f"Limit air-conditioning and other heavy use during {peak_start:02d}:00–{peak_end:02d}:00; "
        "pre-cool/heat beforehand to avoid peak-rate costs."
    )

    return [suggestion1, suggestion2] 