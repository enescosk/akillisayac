"""Anomaly detection using Z-score thresholding."""
from __future__ import annotations

import pandas as pd

Z_THRESHOLD = 2.0


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with added 'zscore' and 'anomaly' columns.

    Z-score is computed within each city separately.
    A point is an anomaly if |z| > Z_THRESHOLD.
    """
    result = df.copy()
    result["zscore"] = result.groupby("city")["consumption"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    result["anomaly"] = result["zscore"].abs() > Z_THRESHOLD
    return result 