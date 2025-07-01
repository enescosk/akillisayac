import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define cities
cities = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Adana",
    "Gaziantep", "Konya", "Antalya", "Kayseri", "Mersin"
]

hours = pd.date_range(start="2024-01-01", periods=24*7, freq="h")

# Create base daily pattern
base_pattern = 100 + 20 * np.sin(2 * np.pi * (hours.hour) / 24) + 10 * np.sin(4 * np.pi * (hours.hour) / 24)

# Generate consumption data for each city
consumption = pd.DataFrame(index=hours)
for city in cities:
    city_variation = np.random.normal(0, 5)  # Different average level per city
    noise = np.random.normal(0, 3, size=len(hours))
    consumption[city] = base_pattern + city_variation + noise

# Compute z-scores and detect anomalies per city
z_scores = (consumption - consumption.mean()) / consumption.std()
anomalies = (z_scores.abs() > 2)

# Plot Istanbul, Ankara, and Izmir
fig, ax = plt.subplots(figsize=(10, 6))
for city in ["Istanbul", "Ankara", "Izmir"]:
    ax.plot(consumption.index, consumption[city], label=city)
    # Mark anomaly points
    ax.plot(consumption.index[anomalies[city]], consumption[city][anomalies[city]], 'x', markersize=8)

ax.set_title("Hourly Electricity Consumption")
ax.set_xlabel("Time")
ax.set_ylabel("Consumption (kWh)")
ax.legend()
plt.tight_layout()
plt.show()
