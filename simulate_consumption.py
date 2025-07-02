import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

CITIES = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Adana",
    "Gaziantep", "Konya", "Antalya", "Kayseri", "Mersin"
]


def generate_consumption(cities=CITIES, start="2024-01-01", periods=24*7, freq="h"):
    hours = pd.date_range(start=start, periods=periods, freq=freq)
    base = 100 + 20 * np.sin(2 * np.pi * hours.hour / 24) + 10 * np.sin(4 * np.pi * hours.hour / 24)
    data = pd.DataFrame(index=hours)
    for city in cities:
        city_variation = np.random.normal(0, 5)
        noise = np.random.normal(0, 3, size=len(hours))
        data[city] = base + city_variation + noise
    return data


def detect_anomalies(df, threshold=2):
    z = (df - df.mean()) / df.std()
    return z.abs() > threshold


def main():
    consumption = generate_consumption()
    anomalies = detect_anomalies(consumption)
    consumption.to_csv("weekly_consumption.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for city in ["Istanbul", "Ankara", "Izmir"]:
        ax.plot(consumption.index, consumption[city], label=city)
        ax.plot(consumption.index[anomalies[city]], consumption[city][anomalies[city]], "x", markersize=8)

    ax.set_title("Hourly Electricity Consumption")
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
