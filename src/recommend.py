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

    # Craft suggestions based on when the peak occurs
    if 11 <= peak_hour <= 16:  # Midday solar-rich period
        suggestion1 = (
            f"Shift laundry/dishwasher runs to {off_start:02d}:00–{off_end:02d}:00 night window to benefit from low tariffs."
        )
        suggestion2 = (
            f"Reduce midday AC usage during {peak_start:02d}:00–{peak_end:02d}:00 by pre-cooling your home in the morning." 
        )
    elif 17 <= peak_hour <= 22:  # Evening peak
        suggestion1 = (
            f"Cook dinner with smaller appliances or earlier to avoid the {peak_start:02d}:00–{peak_end:02d}:00 peak window."
        )
        suggestion2 = (
            f"Run high-load devices between {off_start:02d}:00–{off_end:02d}:00 overnight when demand is lowest." 
        )
    elif 6 <= peak_hour <= 10:  # Morning peak
        suggestion1 = (
            f"Prepare hot water (boiler) after {off_start:02d}:00 when rates drop, avoiding the {peak_start:02d}:00–{peak_end:02d}:00 morning spike."
        )
        suggestion2 = (
            f"Delay starting energy-hungry appliances until mid-day off-peak hours around {off_start:02d}:00–{off_end:02d}:00." 
        )
    else:  # Night or flat profile
        suggestion1 = (
            f"Take advantage of consistently low demand by scheduling appliances during {off_start:02d}:00–{off_end:02d}:00 off-peak hours."
        )
        suggestion2 = (
            f"Maintain efficiency by switching off standby electronics; no significant peaks expected in next 72 h." 
        )

    return [suggestion1, suggestion2] 