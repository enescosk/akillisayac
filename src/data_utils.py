"""Data utilities for smart electricity meter analytics project.

Provides functions to generate synthetic consumption data, load/save the CSV file,
and ensure the data exists.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytz  # TZ dönüşümleri için gerekli

# Constants
CITIES: List[str] = [
    "Istanbul",
    "Ankara",
    "Izmir",
    "Bursa",
    "Adana",
    "Gaziantep",
    "Konya",
    "Antalya",
    "Kayseri",
    "Mersin",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "consumption.csv"


def _simulate_city_series(city: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Simulate hourly consumption for a single city between *start* and *end*.
    The profile follows a daily sinusoidal pattern with added random noise.
    """
    # If datetime is naive, assign Europe/Istanbul timezone
    tz = pytz.timezone("Europe/Istanbul")
    if start.tzinfo is None:
        start = tz.localize(start)
    if end.tzinfo is None:
        end = tz.localize(end)

    rng = pd.date_range(start=start, end=end, freq="H", tz="Europe/Istanbul")

    hours = np.arange(len(rng))

    # Daily pattern: sin wave over 24h (2π per day)
    daily_cycle = np.sin(2 * np.pi * (hours % 24) / 24)

    # City-specific base load and amplitude for variability
    rng_state = np.random.RandomState(abs(hash(city)) % (2**32 - 1))
    base_load = rng_state.uniform(300, 600)  # kWh baseline
    amplitude = rng_state.uniform(150, 300)

    # Random noise (Gaussian)
    noise = np.random.normal(scale=amplitude * 0.2, size=len(rng))

    consumption = base_load + amplitude * (daily_cycle + 1) + noise
    consumption = np.clip(consumption, a_min=0, a_max=None)

    return pd.DataFrame(
        {
            "datetime": rng,
            "city": city,
            "consumption": consumption.astype(float),
        }
    )


def generate_consumption_data(days: int = 7) -> pd.DataFrame:
    """Generate synthetic consumption data for the past *days* (default 7)."""
    end = datetime.now(pytz.timezone("Europe/Istanbul"))
    start = end - timedelta(days=days)

    frames = [
        _simulate_city_series(city, start=start, end=end) for city in CITIES
    ]
    df = pd.concat(frames, ignore_index=True)
    return df


def save_consumption_csv(df: pd.DataFrame, path: Path = CSV_PATH) -> None:
    """Save *df* to *path*, creating parent directories if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    df_to_save["datetime"] = df_to_save["datetime"].dt.tz_convert(None)
    df_to_save.to_csv(path, index=False)


def load_consumption_csv(path: Path = CSV_PATH) -> pd.DataFrame:
    """Load consumption CSV, parsing dates and ensuring correct dtypes."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_localize("Europe/Istanbul")
    return df


def get_consumption_data(ensure: bool = True) -> pd.DataFrame:
    """Load consumption data; generate and save if missing or *ensure* is False."""
    if CSV_PATH.exists():
        try:
            return load_consumption_csv()
        except Exception:
            pass

    if ensure:
        df = generate_consumption_data()
        save_consumption_csv(df)
        return df

    raise FileNotFoundError(f"Consumption CSV not found at {CSV_PATH}")
