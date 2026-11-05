"""Validación walk-forward, métricas, líneas base y evaluación por segmento."""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger("mibici_mlops")


NAIVE_BASELINE = "Ingenuo estacional (lag_7)"


def walk_forward_splits(dates: pd.Series, first_cutoff, n_folds: int, horizon: int):
    # Cortes mensuales: entrenar hasta el corte y validar el mes siguiente.
    splits = []
    for cutoff in pd.date_range(first_cutoff, periods=n_folds, freq="MS"):
        train_end = cutoff - pd.Timedelta(days=horizon - 1)
        validation_end = cutoff + pd.DateOffset(months=1)
        train_positions = np.flatnonzero(dates < train_end)
        validation_positions = np.flatnonzero((dates >= cutoff) & (dates < validation_end))
        if not len(train_positions) or not len(validation_positions):
            raise ValueError(f"Ventana sin datos de entrenamiento o validación: {cutoff.date()}")
        splits.append((cutoff, train_positions, validation_positions))
    return splits


def describe_splits(data: pd.DataFrame, splits) -> pd.DataFrame:
    rows = []
    for cutoff, train_positions, validation_positions in splits:
        train_stations = set(data.loc[train_positions, "Origen_Id"])
        validation_stations = set(data.loc[validation_positions, "Origen_Id"])
        rows.append({
            "cutoff": cutoff.date(),
            "train_end": data.loc[train_positions, "Fecha"].max().date(),
            "train_rows": len(train_positions),
            "validation_rows": len(validation_positions),
            "new_stations": len(validation_stations - train_stations),
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def predict_station_weekday(train_data, data, target_column="target"):
    # Promedio de la estación ese día de la semana; si no hay, promedio global del día.
    means = train_data.groupby(["Origen_Id", "day_of_week"])[target_column].mean()
    keys = pd.MultiIndex.from_frame(data[["Origen_Id", "day_of_week"]])
    predictions = pd.Series(means.reindex(keys).to_numpy(), index=data.index)
    weekday_means = train_data.groupby("day_of_week")[target_column].mean()
    return predictions.fillna(data["day_of_week"].map(weekday_means))


def predict_seasonal_naive(train_data, data, target_column="target"):
    # Mismo día de la semana anterior; si falta, promedio estación-día.
    fallback = predict_station_weekday(train_data, data, target_column)
    return data["lag_7"].fillna(fallback)


def predict_global_weekday(train_data, data, target_column="target"):
    weekday_means = train_data.groupby("day_of_week")[target_column].mean()
    return data["day_of_week"].map(weekday_means)


BASELINES = {
    NAIVE_BASELINE: predict_seasonal_naive,
    "Promedio estación-día": predict_station_weekday,
    "Promedio global por día (versión anterior)": predict_global_weekday,
}


def evaluate_baselines(data, splits, target_column="target") -> pd.DataFrame:
    rows = []
    for cutoff, train_positions, validation_positions in splits:
        train_data = data.loc[train_positions]
        validation_data = data.loc[validation_positions]
        for name, predict in BASELINES.items():
            predictions = predict(train_data, validation_data, target_column)
            rows.append({
                "cutoff": cutoff.date(),
                "model": name,
                **regression_metrics(validation_data[target_column], predictions),
            })
    return pd.DataFrame(rows)


def evaluate_walk_forward(pipeline, data, splits, columns, model_name, target_column="target"):
    # Reentrenar una copia del pipeline en cada ventana y medir el error.
    rows = []
    for cutoff, train_positions, validation_positions in splits:
        train_data = data.loc[train_positions]
        validation_data = data.loc[validation_positions]
        start = time.perf_counter()
        fitted = clone(pipeline).fit(train_data[columns], train_data[target_column])
        fit_seconds = time.perf_counter() - start
        predictions = fitted.predict(validation_data[columns])
        rows.append({
            "cutoff": cutoff.date(),
            "model": model_name,
            **regression_metrics(validation_data[target_column], predictions),
            "fit_seconds": fit_seconds,
            "train_rows": len(train_positions),
            "validation_rows": len(validation_positions),
        })
        logger.info("%s | %s | MAE %.3f", model_name, cutoff.date(), rows[-1]["mae"])
    return pd.DataFrame(rows)


def add_skill(results: pd.DataFrame, baseline_results: pd.DataFrame) -> pd.DataFrame:
    # Mejora sobre el ingenuo estacional en la misma ventana.
    naive_mae = baseline_results.loc[
        baseline_results["model"] == NAIVE_BASELINE
    ].set_index("cutoff")["mae"]
    results = results.copy()
    results["skill_vs_naive"] = 1 - results["mae"] / results["cutoff"].map(naive_mae)
    return results


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("model")
        .agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            skill_vs_naive=("skill_vs_naive", "mean"),
        )
        .sort_values("mae")
    )


def evaluate_by_segment(y_true, model_predictions, naive_predictions, segments) -> pd.DataFrame:
    # Métricas de prueba para todas las estaciones, con historia y nuevas.
    rows = []
    for segment in ["Todas", "Con historia", "Nueva"]:
        mask = np.ones(len(y_true), dtype=bool) if segment == "Todas" else segments == segment
        if not mask.any():
            continue
        model_metrics = regression_metrics(y_true[mask], model_predictions[mask])
        naive_metrics = regression_metrics(y_true[mask], naive_predictions[mask])
        rows.append({
            "stations": segment,
            "rows": int(mask.sum()),
            "model_mae": model_metrics["mae"],
            "model_rmse": model_metrics["rmse"],
            "naive_mae": naive_metrics["mae"],
            "skill_vs_naive": 1 - model_metrics["mae"] / naive_metrics["mae"],
        })
    return pd.DataFrame(rows)
