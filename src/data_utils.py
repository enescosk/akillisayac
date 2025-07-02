"""Data utilities for smart electricity meter analytics project.

Provides functions to generate synthetic consumption data, load/save the CSV file,
and ensure the data exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import unicodedata

# ---------------------------------------------------------------------------
# City metadata
# ---------------------------------------------------------------------------
# 81 Turkish provinces with approximate latitude and longitude
CITIES: list[dict[str, float | str]] = [
    {"name": "Adana", "lat": 37.0000, "lon": 35.3213},
    {"name": "Adiyaman", "lat": 37.7648, "lon": 38.2769},
    {"name": "Afyonkarahisar", "lat": 38.7638, "lon": 30.5403},
    {"name": "Agri", "lat": 39.7191, "lon": 43.0519},
    {"name": "Amasya", "lat": 40.6499, "lon": 35.8353},
    {"name": "Ankara", "lat": 39.9334, "lon": 32.8597},
    {"name": "Antalya", "lat": 36.8969, "lon": 30.7133},
    {"name": "Artvin", "lat": 41.1828, "lon": 41.8194},
    {"name": "Aydin", "lat": 37.8400, "lon": 27.8447},
    {"name": "Balikesir", "lat": 39.6484, "lon": 27.8826},
    {"name": "Bilecik", "lat": 40.1500, "lon": 29.9833},
    {"name": "Bingol", "lat": 39.0626, "lon": 40.7696},
    {"name": "Bitlis", "lat": 38.3938, "lon": 42.1235},
    {"name": "Bolu", "lat": 40.7395, "lon": 31.6116},
    {"name": "Burdur", "lat": 37.7203, "lon": 30.2908},
    {"name": "Bursa", "lat": 40.1950, "lon": 29.0600},
    {"name": "Canakkale", "lat": 40.1467, "lon": 26.4100},
    {"name": "Cankiri", "lat": 40.6000, "lon": 33.6167},
    {"name": "Corum", "lat": 40.5506, "lon": 34.9556},
    {"name": "Denizli", "lat": 37.7833, "lon": 29.0937},
    {"name": "Diyarbakir", "lat": 37.9144, "lon": 40.2306},
    {"name": "Edirne", "lat": 41.6771, "lon": 26.5553},
    {"name": "Elazig", "lat": 38.6752, "lon": 39.2232},
    {"name": "Erzincan", "lat": 39.7520, "lon": 39.4928},
    {"name": "Erzurum", "lat": 39.9043, "lon": 41.2679},
    {"name": "Eskisehir", "lat": 39.7767, "lon": 30.5206},
    {"name": "Gaziantep", "lat": 37.0662, "lon": 37.3833},
    {"name": "Giresun", "lat": 40.9128, "lon": 38.3895},
    {"name": "Gumushane", "lat": 40.4603, "lon": 39.4814},
    {"name": "Hakkari", "lat": 37.5833, "lon": 43.7333},
    {"name": "Hatay", "lat": 36.2028, "lon": 36.1600},
    {"name": "Isparta", "lat": 37.7648, "lon": 30.5566},
    {"name": "Mersin", "lat": 36.8065, "lon": 34.6400},
    {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784},
    {"name": "Izmir", "lat": 38.4237, "lon": 27.1428},
    {"name": "Kars", "lat": 40.6100, "lon": 43.0975},
    {"name": "Kastamonu", "lat": 41.3887, "lon": 33.7827},
    {"name": "Kayseri", "lat": 38.7225, "lon": 35.4875},
    {"name": "Kirklareli", "lat": 41.7351, "lon": 27.2249},
    {"name": "Kirsehir", "lat": 39.1480, "lon": 34.1685},
    {"name": "Kocaeli", "lat": 40.8533, "lon": 29.8815},
    {"name": "Konya", "lat": 37.8722, "lon": 32.4923},
    {"name": "Kutahya", "lat": 39.4242, "lon": 29.9833},
    {"name": "Malatya", "lat": 38.3552, "lon": 38.3095},
    {"name": "Manisa", "lat": 38.6191, "lon": 27.4289},
    {"name": "Kahramanmaras", "lat": 37.5858, "lon": 36.9371},
    {"name": "Mardin", "lat": 37.3128, "lon": 40.7339},
    {"name": "Mugla", "lat": 37.2153, "lon": 28.3636},
    {"name": "Mus", "lat": 38.9462, "lon": 41.7539},
    {"name": "Nevsehir", "lat": 38.6248, "lon": 34.7179},
    {"name": "Nigde", "lat": 37.9662, "lon": 34.6796},
    {"name": "Ordu", "lat": 40.9862, "lon": 37.8797},
    {"name": "Rize", "lat": 41.0201, "lon": 40.5234},
    {"name": "Sakarya", "lat": 40.7419, "lon": 30.3270},
    {"name": "Samsun", "lat": 41.2928, "lon": 36.3313},
    {"name": "Siirt", "lat": 37.9450, "lon": 41.9403},
    {"name": "Sinop", "lat": 42.0268, "lon": 35.1628},
    {"name": "Sivas", "lat": 39.7477, "lon": 37.0179},
    {"name": "Tekirdag", "lat": 40.9599, "lon": 27.5152},
    {"name": "Tokat", "lat": 40.3141, "lon": 36.5540},
    {"name": "Trabzon", "lat": 41.0030, "lon": 39.7168},
    {"name": "Tunceli", "lat": 39.1081, "lon": 39.5483},
    {"name": "Sanliurfa", "lat": 37.1671, "lon": 38.7955},
    {"name": "Usak", "lat": 38.6823, "lon": 29.4082},
    {"name": "Van", "lat": 38.5012, "lon": 43.3662},
    {"name": "Yozgat", "lat": 39.8209, "lon": 34.8085},
    {"name": "Zonguldak", "lat": 41.4564, "lon": 31.7987},
    {"name": "Aksaray", "lat": 38.3687, "lon": 34.0360},
    {"name": "Bayburt", "lat": 40.2583, "lon": 40.2279},
    {"name": "Karaman", "lat": 37.1811, "lon": 33.2150},
    {"name": "Kirikkale", "lat": 39.8468, "lon": 33.5153},
    {"name": "Batman", "lat": 37.8812, "lon": 41.1351},
    {"name": "Sirnak", "lat": 37.4187, "lon": 42.4918},
    {"name": "Bartin", "lat": 41.6350, "lon": 32.3370},
    {"name": "Ardahan", "lat": 41.1105, "lon": 42.7022},
    {"name": "Igdir", "lat": 39.9237, "lon": 44.0400},
    {"name": "Yalova", "lat": 40.6500, "lon": 29.2667},
    {"name": "Karabuk", "lat": 41.2061, "lon": 32.6204},
    {"name": "Kilis", "lat": 36.7184, "lon": 37.1150},
    {"name": "Osmaniye", "lat": 37.0742, "lon": 36.2475},
    {"name": "Duzce", "lat": 40.8438, "lon": 31.1565},
]

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "consumption.csv"

# Path to yearly totals csv shipped by user
_TOTALS_CSV = Path(__file__).resolve().parent.parent / "city_consumption.csv"


def _get_hours_last_week(now: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Return hourly DateTimeIndex for the last 7 days ending at *now* (rounded to hour)."""
    if now is None:
        now = datetime.now(tz=timezone.utc).astimezone()
    end = pd.Timestamp(now).floor("h")
    periods = 24 * 7
    return pd.date_range(end=end, periods=periods, freq="h")


def _normalize(name: str) -> str:
    """Return lowercase ASCII version of *name* for fuzzy matching."""
    return (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .replace(" ", "")
    )


def _load_city_totals() -> dict[str, float]:
    """Load yearly city totals from CSV → dict(normalized_name -> kWh)."""
    if not _TOTALS_CSV.exists():
        return {}
    df = pd.read_csv(_TOTALS_CSV)
    # Expect columns: city, total_mwh OR Turkish header variations
    # Identify first two columns regardless of names
    col_city, col_value = df.columns[:2]
    totals = {}
    for _, row in df.iterrows():
        name = str(row[col_city]).strip()
        try:
            value_mwh = float(str(row[col_value]).replace(".", ""))
        except ValueError:
            continue
        totals[_normalize(name)] = value_mwh * 1000  # convert to kWh
    return totals


_CITY_TOTALS_KWH: dict[str, float] | None = None


def _get_city_totals() -> dict[str, float]:
    global _CITY_TOTALS_KWH
    if _CITY_TOTALS_KWH is None:
        _CITY_TOTALS_KWH = _load_city_totals()
    return _CITY_TOTALS_KWH


def generate_consumption(hours: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Generate simulated hourly consumption for all 81 cities.

    The profile follows a daily sinusoidal pattern (two harmonics) plus Gaussian noise. Each
    city gets a random offset (heterogeneity).
    """
    if hours is None:
        hours = _get_hours_last_week()

    np.random.seed(42)
    base_pattern = 100 + 20 * np.sin(2 * np.pi * hours.hour / 24)
    base_pattern += 10 * np.sin(4 * np.pi * hours.hour / 24)

    consumption = pd.DataFrame(index=hours)
    totals = _get_city_totals()

    hrs_in_year = 365 * 24
    h_len = len(hours)

    for city in CITIES:
        city_offset = np.random.normal(0, 5)
        noise = np.random.normal(0, 3, size=h_len)
        series = base_pattern + city_offset + noise

        # Scale to yearly totals if available
        norm_name = _normalize(city["name"])
        if norm_name in totals and totals[norm_name] > 0:
            target_total = totals[norm_name] * (h_len / hrs_in_year)
            factor = target_total / series.sum()
            series = series * factor

        consumption[city["name"]] = series

    return consumption


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_consumption(consumption: pd.DataFrame, path: str | Path | None = None) -> None:
    """Save *consumption* to CSV under *path* (default project data folder)."""
    if path is None:
        path = _DATA_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    consumption.to_csv(path)


def load_consumption(path: str | Path | None = None, *, force: bool = False) -> pd.DataFrame:
    """Load consumption from CSV; generate and save if not present or *force*.

    Data will also be regenerated automatically if the city totals CSV is newer
    (i.e., yearly data updated) than the cached consumption file.
    """
    if path is None:
        path = _DATA_PATH
    path = Path(path)

    totals_mtime = _TOTALS_CSV.stat().st_mtime if _TOTALS_CSV.exists() else None

    regenerate = force or (not path.exists())
    if not regenerate and totals_mtime is not None:
        regenerate = totals_mtime > path.stat().st_mtime

    if regenerate:
        data = generate_consumption()
        save_consumption(data, path)
        return data

    return pd.read_csv(path, index_col=0, parse_dates=True)
