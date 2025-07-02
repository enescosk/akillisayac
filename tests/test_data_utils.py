import pandas as pd
from src.data_utils import generate_consumption, CITIES, _get_hours_last_week


def test_generate_consumption_shape():
    hours = _get_hours_last_week()
    df = generate_consumption(hours=hours)
    assert df.shape == (len(hours), len(CITIES))
