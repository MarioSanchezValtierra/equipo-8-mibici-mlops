import numpy as np
import pandas as pd
import pytest

from mibici_mlops.config import FEATURE_COLUMNS, WEATHER_COLUMNS
from mibici_mlops.features import add_history_features, split_by_date


def test_lags_use_calendar_days(toy_data):
    result = add_history_features(toy_data, 1, [1, 7], [7])
    row = result.loc[result["Fecha"] == "2025-01-11"].iloc[0]
    assert np.isnan(row["lag_1"])
    assert row["lag_7"] == result.loc[result["Fecha"] == "2025-01-04", "target"].iloc[0]


def test_lag_shorter_than_horizon_fails(toy_data):
    with pytest.raises(ValueError):
        add_history_features(toy_data, 3, [1, 7], [7])


def test_history_does_not_use_future_target(toy_data):
    original = add_history_features(toy_data, 1, [1, 7], [7, 28])
    changed = toy_data.copy()
    changed.loc[changed["Fecha"] >= "2025-01-20", "target"] = 99999
    updated = add_history_features(changed, 1, [1, 7], [7, 28])
    before = original["Fecha"] <= "2025-01-20"
    for column in ["lag_1", "lag_7", "roll_mean_7", "roll_mean_28"]:
        pd.testing.assert_series_equal(
            original.loc[before, column], updated.loc[before, column], check_names=False
        )


def test_duplicate_station_dates_fail(toy_data):
    repeated = pd.concat([toy_data, toy_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        add_history_features(repeated, 1, [1, 7], [7])


def test_split_by_date_keeps_test_separate(toy_data):
    development, test = split_by_date(toy_data, pd.Timestamp("2025-01-01"))
    assert development["Fecha"].max() < pd.Timestamp("2025-01-01")
    assert test["Fecha"].min() >= pd.Timestamp("2025-01-01")
    assert len(development) + len(test) == len(toy_data)


def test_final_features_exclude_observed_weather():
    assert len(FEATURE_COLUMNS) == 18
    assert not set(WEATHER_COLUMNS).intersection(FEATURE_COLUMNS)
    assert "weather_zone" in FEATURE_COLUMNS
