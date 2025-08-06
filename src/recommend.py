from __future__ import annotations

import pandas as pd
from typing import List


def generate_suggestions(forecast_df: pd.DataFrame) -> List[str]:
    """Generate up to 10 actionable suggestions based on 72-hour forecast."""
    # Validate required columns
    if not {"ds", "yhat"}.issubset(forecast_df.columns):
        raise ValueError("forecast_df must contain 'ds' and 'yhat' columns")

    df = forecast_df.copy()
    df["hour"] = df["ds"].dt.hour
    hourly_mean = df.groupby("hour")["yhat"].mean()

    peak_hour = int(hourly_mean.idxmax())
    off_hour = int(hourly_mean.idxmin())

    overall_avg = df["yhat"].mean()
    peak_avg = hourly_mean.loc[peak_hour]
    percent_above = int((peak_avg - overall_avg) / overall_avg * 100)

    peak_start = (peak_hour - 1) % 24
    peak_end = (peak_hour + 1) % 24
    off_start = (off_hour - 1) % 24
    off_end = (off_hour + 1) % 24

    suggestions: List[str] = []

    # 1. Shift heavy loads away from peak
    suggestions.append(
        f"Pik saatler ({peak_start:02d}:00–{peak_end:02d}:00) tüketiminiz günlük ortalamanın %{percent_above} üzerinde; ağır cihaz kullanımını {off_start:02d}:00–{off_end:02d}:00 aralığına kaydırın."
    )

    # 2. Use night tariff
    suggestions.append(
        f"Çamaşır ve bulaşık makinelerini gece tarifesinin başladığı {off_start:02d}:00–{off_end:02d}:00 arasında çalıştırarak faturanızı düşürebilirsiniz."
    )

    # 3. EV charging optimization
    suggestions.append(
        f"Elektrikli araç şarjını pik sonrası {peak_end:02d}:00’dan sonra başlatın; indirimli tarifeden faydalanırsınız ve şebeke yükünü azaltırsınız."
    )

    # 4. Anomaly alert
    suggestions.append(
        "Tüketimde ani artış tespit edilirse anında bildirim alacak şekilde uygulama ayarlarınızı etkinleştirin; böylece arıza veya kaçak hızla fark edilir."
    )

    # 5. HVAC pre-cool / pre-heat
    suggestions.append(
        f"Klimanızı pik öncesi ({peak_start:02d}:00 öncesi) çalıştırıp pike düşük yükle girerek hem konfor hem tasarruf sağlayın."
    )

    # 6. Demand response incentive
    suggestions.append(
        "Yoğun saatlerde talebi %10 azaltmanız durumunda sağlayıcıların sunduğu talep-yanıt indirimlerinden yararlanabilirsiniz."
    )

    # 7. Solar export planning
    suggestions.append(
        f"Güneş paneli üretim fazlanızı {off_start:02d}:00–{off_end:02d}:00 arasında şebekeye satarak en yüksek geri ödemeyi elde edebilirsiniz."
    )

    # 8. Device level audit
    suggestions.append(
        "Haftalık cihaz bazlı rapordan en çok enerji çeken cihazları tespit edip kullanım sürelerini kısaltın."
    )

    # 9. Market hedging
    suggestions.append(
        "72 saatlik tahmini kullanarak gün-öncesi piyasada uygun fiyattan enerji satın almayı planlayın."
    )

    # 10. Preventive maintenance
    suggestions.append(
        "Sayaç ve batarya ömrünü izleyip kritik seviyeye gelmeden bakım planlayarak kesintileri önleyin."
    )

    return suggestions
