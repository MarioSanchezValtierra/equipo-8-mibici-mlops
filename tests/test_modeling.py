import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mibici_mlops.config import (
    FEATURE_COLUMNS, LAGS, POI_COLUMNS, ROLLING_WINDOWS, SEED, STATION_COLUMNS, TARGET_COLUMN,
)
from mibici_mlops.features import add_history_features
from mibici_mlops.modeling.predict import build_next_day_rows
from mibici_mlops.modeling.train import build_model_pipeline


def test_pipeline_preprocessing_and_prediction(toy_data):
    data = toy_data.copy()
    data["month"] = data["Fecha"].dt.month
    data["day_of_month"] = data["Fecha"].dt.day
    data["weather_zone"] = 1
    for column in POI_COLUMNS:
        data[column] = 0
    data = add_history_features(data, 1, LAGS, ROLLING_WINDOWS)
    pipeline = build_model_pipeline(
        RandomForestRegressor(n_estimators=3, max_depth=2, random_state=SEED), FEATURE_COLUMNS
    )
    pipeline.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    predictions = pipeline.predict(data[FEATURE_COLUMNS].iloc[:3])
    assert len(predictions) == 3
    assert np.isfinite(predictions).all()


def test_next_day_rows_use_only_recent_stations():
    dates = pd.date_range("2025-01-01", periods=35)
    data = pd.DataFrame({
        "Origen_Id": [1] * len(dates) + [2],
        "Fecha": list(dates) + [pd.Timestamp("2024-01-01")],
    })
    for column in STATION_COLUMNS:
        data[column] = 0
    result = build_next_day_rows(data, pd.Timestamp("2025-02-05"), active_days=28)
    assert result["Origen_Id"].tolist() == [1]
    assert result["Fecha"].eq(pd.Timestamp("2025-02-05")).all()
