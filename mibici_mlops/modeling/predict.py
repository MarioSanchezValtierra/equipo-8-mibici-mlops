"""Carga del modelo registrado y predicción del día siguiente."""

import mlflow
import mlflow.sklearn
import pandas as pd

from mibici_mlops.config import (
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    HORIZON_DAYS,
    LAGS,
    ROLLING_WINDOWS,
    STATION_COLUMNS,
    TARGET_COLUMN,
)
from mibici_mlops.features import add_calendar_features, add_history_features


def load_registered_model(model_name: str, alias: str):
    return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")


def build_next_day_rows(data: pd.DataFrame, target_date, active_days: int = 28) -> pd.DataFrame:
    # Una fila por estación activa con el calendario de la fecha a predecir.
    recent = data.loc[
        (data["Fecha"] < target_date)
        & (data["Fecha"] >= target_date - pd.Timedelta(days=active_days))
    ]
    if recent.empty:
        raise ValueError("No hay estaciones con historial reciente para predecir.")
    stations = recent.sort_values("Fecha").groupby("Origen_Id")[STATION_COLUMNS].last()
    rows = stations.reset_index()
    rows["Fecha"] = pd.Timestamp(target_date)
    rows["day_of_week"] = rows["Fecha"].dt.day_of_week
    rows["month"] = rows["Fecha"].dt.month
    return rows


def predict_next_day(model, data: pd.DataFrame, target_date) -> pd.DataFrame:
    # Predecir las salidas por estación para target_date con la historia disponible.
    target_date = pd.Timestamp(target_date)
    history = data.loc[data["Fecha"] < target_date, ["Origen_Id", "Fecha", TARGET_COLUMN]]
    new_rows = build_next_day_rows(data.loc[data["Fecha"] < target_date], target_date)
    combined = pd.concat([history, new_rows[["Origen_Id", "Fecha"]]], ignore_index=True)
    with_history = add_history_features(combined, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    features = new_rows.merge(
        with_history.loc[with_history["Fecha"] == target_date, ["Origen_Id"] + HISTORY_COLUMNS],
        on="Origen_Id",
    )
    features = add_calendar_features(features)
    if features["Origen_Id"].duplicated().any():
        raise ValueError("La predicción contiene estaciones duplicadas.")
    features["predicted_trips"] = model.predict(features[FEATURE_COLUMNS])
    return features[["Origen_Id", "Fecha", "predicted_trips"] + HISTORY_COLUMNS]
