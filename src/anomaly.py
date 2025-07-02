"""Anomaly detection using Z-score thresholding."""

from __future__ import annotations

import pandas as pd


def detect_anomalies(consumption: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Return boolean DataFrame indicating where abs(Z-score) > *threshold*.

    Parameters
    ----------
    consumption : pd.DataFrame
        Wide DataFrame with cities as columns and datetime index.
    threshold : float, default 2
        Threshold on absolute Z-score to mark anomaly.

    Returns
    -------
    pd.DataFrame[bool]
        Same shape as *consumption*, True where anomaly.
    """
    z_scores = (consumption - consumption.mean()) / consumption.std()
    return z_scores.abs() > threshold
