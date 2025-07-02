import numpy as np
import pandas as pd

"""Utilities to simulate electricity consumption and detect anomalies."""

from pathlib import Path
import matplotlib.pyplot as plt

CITIES = [
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

HOURS = pd.date_range(start="2024-01-01", periods=24 * 7, freq="h")


def generate_consumption(cities=CITIES, hours=HOURS):
    """Return simulated hourly consumption data for given cities."""
    np.random.seed(42)
    base_pattern = 100 + 20 * np.sin(2 * np.pi * hours.hour / 24)
    base_pattern += 10 * np.sin(4 * np.pi * hours.hour / 24)
    consumption = pd.DataFrame(index=hours)
    for city in cities:
        city_variation = np.random.normal(0, 5)
        noise = np.random.normal(0, 3, size=len(hours))
        consumption[city] = base_pattern + city_variation + noise
    return consumption


def detect_anomalies(consumption, threshold=2):
    """Compute z-scores and return a boolean DataFrame of anomalies."""
    z_scores = (consumption - consumption.mean()) / consumption.std()
    return z_scores.abs() > threshold


def save_consumption(consumption, path="consumption.csv"):
    """Save the generated consumption DataFrame to CSV."""
    consumption.to_csv(path)


def load_or_generate(path="consumption.csv"):
    """Load consumption data from CSV or generate and save if absent."""
    file = Path(path)
    if file.exists():
        return pd.read_csv(file, index_col=0, parse_dates=True)
    data = generate_consumption()
    save_consumption(data, path)
    return data


def main():
    """Generate data, detect anomalies and plot selected cities."""
    consumption = generate_consumption()
    anomalies = detect_anomalies(consumption)

    fig, ax = plt.subplots(figsize=(10, 6))
    for city in ["Istanbul", "Ankara", "Izmir"]:
        ax.plot(consumption.index, consumption[city], label=city)
        ax.plot(
            consumption.index[anomalies[city]],
            consumption[city][anomalies[city]],
            "x",
            markersize=8,
        )

    ax.set_title("Hourly Electricity Consumption")
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    save_consumption(consumption)


if __name__ == "__main__":
    main()
