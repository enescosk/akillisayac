# Smart Electricity Meter Analytics

This project simulates hourly electricity consumption for 10 major Turkish cities, detects anomalies, and offers interactive analytics and forecasting via a Streamlit dashboard.

## Features

* **Data Simulation** – Generates realistic hourly consumption data for the last 7 days and stores it in `data/consumption.csv`.
* **Anomaly Detection** – Flags abnormal readings using Z-score (|z| > 2).
* **Forecasting** – 72-hour consumption forecast powered by Prophet.
* **Interactive Dashboard** –
  * City selection & date range filtering
  * Line plot of consumption with anomalies (Plotly)
  * Downloadable forecast CSV
  * Folium map of average city consumption
* **Modular Code** – Components split into reusable modules (`data_utils.py`, `anomaly.py`, `forecast.py`).

## Quick-start

1. **Clone the repository** (skip if you already have the files).
2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   > Prophet relies on a working C/C++ tool-chain. On macOS you may need Xcode command-line tools: `xcode-select --install`.

4. **Launch the dashboard**:

   ```bash
   streamlit run src/dashboard.py
   ```

5. **Regenerate data** (optional) – Use the "Regenerate Data" button in the sidebar to create fresh synthetic data.
   The CSV will be saved to `data/consumption.csv`.

## Project Structure

```text
├── data/                 # Auto-generated CSV lives here
├── src/
│   ├── anomaly.py        # Z-score anomaly detection
│   ├── dashboard.py      # Streamlit app entry-point
│   ├── data_utils.py     # Data simulation, I/O helpers
│   └── forecast.py       # Prophet forecasting utilities
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

Enjoy exploring smart meter analytics! 🎛️