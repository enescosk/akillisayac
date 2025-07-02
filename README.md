# Electricity Consumption Simulation

This project simulates hourly electricity consumption for ten major Turkish cities over one week. It provides anomaly detection and a Streamlit dashboard for interactive visualization and forecasting.

## Features

- **Data Simulation**: Generates realistic consumption patterns using sinusoidal daily trends with random noise.
- **Anomaly Detection**: Uses z-scores to flag values with an absolute score greater than 2.
- **CSV Export**: Data is saved to `weekly_consumption.csv`.
- **Streamlit Dashboard**:
  - Line charts per city with anomaly markers.
  - Date range and city selection filters.
  - Optional forecasting of the next three days using Prophet.
  - Map visualization of average consumption per city.

## Usage

1. Install dependencies (pandas, numpy, matplotlib, plotly, prophet, streamlit).
2. Run the simulation script:

```bash
python3 simulate_consumption.py
```

3. Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser, where you can explore consumption data, anomalies, forecasts, and the map view.
