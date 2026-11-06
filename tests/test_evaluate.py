import numpy as np

from mibici_mlops.features import add_history_features
from mibici_mlops.modeling.evaluate import (
    predict_seasonal_naive,
    predict_station_weekday,
    walk_forward_splits,
)


def test_walk_forward_has_no_overlap(toy_data):
    for cutoff, train, validation in walk_forward_splits(toy_data["Fecha"], "2025-01-01", 2, 1):
        assert toy_data.loc[train, "Fecha"].max() < cutoff
        assert toy_data.loc[validation, "Fecha"].min() >= cutoff
        assert not set(train) & set(validation)


def test_seasonal_naive_uses_last_week(toy_data):
    data = add_history_features(toy_data, 1, [7], [7])
    train, validation = data.iloc[:40], data.iloc[40:].copy()
    validation.loc[validation.index[0], "lag_7"] = np.nan
    predictions = predict_seasonal_naive(train, validation)
    assert predictions.iloc[0] == predict_station_weekday(train, validation).iloc[0]
    has_lag = validation["lag_7"].notna()
    assert (predictions[has_lag] == validation.loc[has_lag, "lag_7"]).all()
