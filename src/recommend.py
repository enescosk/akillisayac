from __future__ import annotations

import pandas as pd
from typing import List
import random


def _template_pool() -> dict[str, list[str]]:
    return {
        "midday": [
            "Öğle pikinde ({peak_start:02d}:00–{peak_end:02d}:00) klimayı azaltmak için evi sabahın serin saatlerinde önceden soğutun.",
            "Çamaşır/bulaşık makinelerini gece {off_start:02d}:00–{off_end:02d}:00 tarifesinde çalıştırarak öğle pik tüketiminden kaçının.",
        ],
        "evening": [
            "Akşam pikine ({peak_start:02d}:00–{peak_end:02d}:00) girmeden yemeği erken pişirip yüksek güçlü aletleri gece {off_start:02d}:00 sonrası çalıştırın.",
            "Elektrikli araç şarjını {off_start:02d}:00–{off_end:02d}:00 düşük tarife saatlerine kaydırın; şebeke yükü azalsın fatura düşsün.",
        ],
        "morning": [
            "Sabah pikinde ({peak_start:02d}:00–{peak_end:02d}:00) kettle/termosifon yerine geceden su ısıtın; tüketimi {off_start:02d}:00 sonrası dağıtın.",
            "Yoğun sabah saatleri yerine elektrikli süpürgeyi öğleden sonra kullanarak talep profilinizi düzleştirin.",
        ],
        "flat": [
            "Geceleri {off_start:02d}:00–{off_end:02d}:00 arasında ağır cihaz kullanımını toplamak faturanızı düşürür.",
            "Stand-by cihazları yatmadan önce kapatın; pik dışı saatlerde bile gereksiz tüketimden kaçınırsınız.",
        ],
    }


def generate_suggestions(forecast_df: pd.DataFrame) -> List[str]:
    """Return 2 varied suggestions tailored to forecast peak/off-peak pattern."""
    if not {"ds", "yhat"}.issubset(forecast_df.columns):
        raise ValueError("forecast_df must contain 'ds' and 'yhat' columns")

    df = forecast_df.copy()
    df["hour"] = df["ds"].dt.hour
    hourly_mean = df.groupby("hour")["yhat"].mean()
    peak_hour = int(hourly_mean.idxmax())
    off_hour = int(hourly_mean.idxmin())

    peak_start = (peak_hour - 1) % 24
    peak_end = (peak_hour + 1) % 24
    off_start = (off_hour - 1) % 24
    off_end = (off_hour + 1) % 24

    # category by peak time
    if 11 <= peak_hour <= 16:
        category = "midday"
    elif 17 <= peak_hour <= 22:
        category = "evening"
    elif 6 <= peak_hour <= 10:
        category = "morning"
    else:
        category = "flat"

    templates = _template_pool()[category]
    chosen = random.sample(templates, k=2 if len(templates) >= 2 else len(templates))

    return [t.format(peak_start=peak_start, peak_end=peak_end, off_start=off_start, off_end=off_end) for t in chosen]
