from __future__ import annotations

import pandas as pd
from typing import List
import random


def _template_pool() -> dict[str, list[str]]:
    return {
        "midday": [
            "Öğle pikinde ({peak_start:02d}:00–{peak_end:02d}:00) klimayı azaltmak için evi sabahın serin saatlerinde önceden soğutun.",
            "Çamaşır/bulaşık makinelerini gece {off_start:02d}:00–{off_end:02d}:00 tarifesinde çalıştırarak öğle pik tüketiminden kaçının.",
            "{city}’de güneş paneliniz varsa öğlen fazlasını şebekeye satarak en yüksek geri dönüşü alın.",
            "Öğle pikini dengelemek için ofis cihazlarını zamanlayıcı ile sabah {off_start:02d}:00 sonrası çalıştırın.",
            "Soğutmayı {peak_start:02d}:00’dan önce tamamlayarak öğle tarifesindeki pahalı kWh’lerden kaçının.",
        ],
        "evening": [
            "Akşam pikine ({peak_start:02d}:00–{peak_end:02d}:00) girmeden yemeği erken pişirip yüksek güçlü aletleri gece {off_start:02d}:00 sonrası çalıştırın.",
            "Elektrikli araç şarjını {off_start:02d}:00–{off_end:02d}:00 düşük tarife saatlerine kaydırın; şebeke yükü azalsın fatura düşsün.",
            "Akşam pik oranı yüksek olduğu için {city}’de akıllı prizlerle TV/konsol kullanımını {peak_end:02d}:00 sonrası erteleyin.",
            "Yemek pişirmede Airfryer kullanarak {peak_start:02d}:00–{peak_end:02d}:00 arası fırın yükünü %30 azaltın.",
            "Su ısıtıcısını zamanlayıcıyla gece {off_start:02d}:00’da devreye alarak akşam pikini düşürün.",
        ],
        "morning": [
            "Sabah pikinde ({peak_start:02d}:00–{peak_end:02d}:00) kettle/termosifon yerine geceden su ısıtın; tüketimi {off_start:02d}:00 sonrası dağıtın.",
            "Yoğun sabah saatleri yerine elektrikli süpürgeyi öğleden sonra kullanarak talep profilinizi düzleştirin.",
            "Isıtıcıyı {off_start:02d}:00’dan sonra çalıştırıp sabah piki öncesi evi ısıtın.",
            "Sabah duşu için boyleri gece tarifesinde ısıtıp sabah pikindeki direnci devre dışı bırakın.",
        ],
        "flat": [
            "Geceleri {off_start:02d}:00–{off_end:02d}:00 arasında ağır cihaz kullanımını toplamak faturanızı düşürür.",
            "Stand-by cihazları yatmadan önce kapatın; pik dışı saatlerde bile gereksiz tüketimden kaçınırsınız.",
            "{city}’de tüketim dalgalı değil; tarife tasarrufu için faturanızı tek zamanlı yerine çok zamanlıya geçirmeyi düşünün.",
            "Pik farkı düşük olduğu için tasarrufu cihaz verimliliği ve stand-by azaltımıyla sağlayın.",
        ],
        "rising": [
            "Son 3 günde tüketiminiz % {trend_pct} arttı; enerji yoğun işlemleri erteleyip verimliliği artırarak artışı sınırlayın.",
            "Talep artış trendine karşı tarifeleri gözden geçirip çok zamanlı plana geçmeyi değerlendirin (artış %{trend_pct}).",
        ],
        "high_peak_ratio": [
            "Pik/off-peak oranı %{peak_ratio} olduğundan cihazları gece {off_start:02d}:00–{off_end:02d}:00 arasında çalıştırmak büyük tasarruf sağlar.",
            "Tüketiminiz pik saate göre %{peak_ratio} kat artıyor; akıllı prizlerle otomatik zamanlama yapın.",
        ],
    }


def generate_suggestions(forecast_df: pd.DataFrame, city: str | None = None) -> List[str]:
    """Return 2 varied suggestions tailored to forecast peak/off-peak pattern."""
    if not {"ds", "yhat"}.issubset(forecast_df.columns):
        raise ValueError("forecast_df must contain 'ds' and 'yhat' columns")

    df = forecast_df.copy()
    df["hour"] = df["ds"].dt.hour
    hourly_mean = df.groupby("hour")["yhat"].mean()
    peak_hour = int(hourly_mean.idxmax())
    off_hour = int(hourly_mean.idxmin())

    # Peak ratio (how many times bigger than off peak)
    peak_ratio = hourly_mean.loc[peak_hour] / max(hourly_mean.loc[off_hour], 1e-6)

    # Trend: compare last 24h vs first 24h average
    first_24 = df.iloc[:24]["yhat"].mean()
    last_24 = df.iloc[-24:]["yhat"].mean()
    trend_pct = int((last_24 - first_24) / first_24 * 100)

    peak_start = (peak_hour - 1) % 24
    peak_end = (peak_hour + 1) % 24
    off_start = (off_hour - 1) % 24
    off_end = (off_hour + 1) % 24

    # category by metrics
    if 11 <= peak_hour <= 16:
        category = "midday"
    elif 17 <= peak_hour <= 22:
        category = "evening"
    elif 6 <= peak_hour <= 10:
        category = "morning"
    else:
        category = "flat"

    # Additional category overlays
    extra_keys: list[str] = []
    if peak_ratio > 2:
        extra_keys.append("high_peak_ratio")
    if trend_pct > 5:
        extra_keys.append("rising")

    templates = _template_pool()[category] + sum((_template_pool()[k] for k in extra_keys), [])

    chosen = random.sample(templates, k=2 if len(templates) >= 2 else len(templates))

    return [
        t.format(
            peak_start=peak_start,
            peak_end=peak_end,
            off_start=off_start,
            off_end=off_end,
            trend_pct=trend_pct,
            peak_ratio=int(peak_ratio),
            city=city or "Şehriniz",
        )
        for t in chosen
    ]
