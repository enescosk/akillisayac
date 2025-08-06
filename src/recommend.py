from __future__ import annotations

import pandas as pd
from typing import List
import random


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

    # Identify usage category by peak hour
    if 11 <= peak_hour <= 16:  # Midday solar-rich period
        category = "midday"
    elif 17 <= peak_hour <= 22:  # Evening peak
        category = "evening"
    elif 6 <= peak_hour <= 10:  # Morning peak
        category = "morning"
    else:  # Night or flat profile
        category = "flat"

    # Choose two templates and format with hours
    templates = _choose_templates(category, 2)
    suggestion1 = templates[0].format(
        peak_start=peak_start,
        peak_end=peak_end,
        off_start=off_start,
        off_end=off_end,
    )
    suggestion2 = templates[1].format(
        peak_start=peak_start,
        peak_end=peak_end,
        off_start=off_start,
        off_end=off_end,
    )

    return [suggestion1, suggestion2] 


def _choose_templates(category: str, n: int) -> list[str]:
    """Return *n* random templates for given category."""

    templates = {
        "midday": [
            "Run laundry and dishwasher after {off_start:02d}:00 to leverage night tariffs and avoid midday AC peak.",
            "Pre-cool your home in the early morning to reduce AC load between {peak_start:02d}:00-{peak_end:02d}:00.",
            "Charge EVs or batteries during {off_start:02d}:00–{off_end:02d}:00 when demand is minimal.",
        ],
        "evening": [
            "Shift cooking to electric pressure cookers before {peak_start:02d}:00 to dodge evening peak rates.",
            "Run water-heaters or ironing overnight ({off_start:02d}:00–{off_end:02d}:00) instead of the {peak_start:02d}:00–{peak_end:02d}:00 window.",
            "Delay EV charging until after {peak_end:02d}:00 to flatten your evening demand curve.",
        ],
        "morning": [
            "Program coffee machines and boilers after {peak_end:02d}:00 to skip the morning spike.",
            "Do vacuuming or other high-draw chores in the off-peak {off_start:02d}:00–{off_end:02d}:00 slot.",
        ],
        "flat": [
            "Maintain savings by clustering appliance use in the {off_start:02d}:00–{off_end:02d}:00 low-demand window.",
            "Turn off standby electronics overnight; no strong peaks expected, so every kWh counts.",
        ],
    }
    pool = templates.get(category, templates["flat"])
    return random.sample(pool, k=min(n, len(pool)))
