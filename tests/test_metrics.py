"""Pruebas unitarias de las métricas y de la línea base ingenua."""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops.modeling.evaluate import (
    NAIVE_BASELINE, add_skill, predict_seasonal_naive, predict_station_weekday, regression_metrics,
)


def test_metrics_with_known_values():
    metrics = regression_metrics([0, 2, 4], [1, 2, 7])
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(10 / 3))


def test_perfect_prediction_has_zero_error():
    assert regression_metrics([3, 5, 8], [3, 5, 8]) == {"mae": 0.0, "rmse": 0.0}


def test_rmse_is_never_below_mae():
    rng = np.random.default_rng(0)
    metrics = regression_metrics(rng.poisson(10, 200), rng.poisson(10, 200))
    assert metrics["rmse"] >= metrics["mae"]


def test_naive_uses_lag_7_and_falls_back_to_station_weekday():
    train = pd.DataFrame({"Origen_Id": [1, 1, 2], "day_of_week": [0, 0, 0], "target": [10, 20, 40]})
    new = pd.DataFrame({"Origen_Id": [1, 1, 3], "day_of_week": [0, 0, 0], "lag_7": [7.0, None, None]})
    naive = predict_seasonal_naive(train, new)
    assert naive.tolist() == pytest.approx([7.0, 15.0, 70 / 3])  # 70/3: promedio global del lunes


def test_station_weekday_baseline_covers_new_stations():
    train = pd.DataFrame({"Origen_Id": [1, 2], "day_of_week": [3, 3], "target": [10, 30]})
    new = pd.DataFrame({"Origen_Id": [99], "day_of_week": [3]})
    assert predict_station_weekday(train, new).iloc[0] == pytest.approx(20)


def test_skill_is_improvement_over_naive():
    results = pd.DataFrame({"cutoff": ["2025-01", "2025-02"], "mae": [5.0, 9.0]})
    baseline = pd.DataFrame({"cutoff": ["2025-01", "2025-02"],
                             "model": [NAIVE_BASELINE] * 2, "mae": [10.0, 9.0]})
    assert add_skill(results, baseline)["skill_vs_naive"].tolist() == pytest.approx([0.5, 0.0])
