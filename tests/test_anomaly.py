import pandas as pd
from src.anomaly import detect_anomalies


def test_detect_anomalies_simple():
    df = pd.DataFrame({'A': [0, 0, 0, 10, 0]})
    result = detect_anomalies(df, threshold=2)
    assert result['A'].sum() == 1
    assert result['A'].iloc[3]
