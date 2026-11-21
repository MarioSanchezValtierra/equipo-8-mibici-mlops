"""Variables del modelo: historia de la demanda, calendario y separación temporal."""

import pandas as pd


def add_history_features(data, horizon, lags, windows, target_column="target"):
    # Rezagos y promedios móviles calculados por fecha, no por fila.
    invalid_lags = [lag for lag in lags if lag < horizon]
    if invalid_lags:
        raise ValueError(
            f"Los rezagos {invalid_lags} usarían información futura con horizonte {horizon}."
        )
    if data.duplicated(["Origen_Id", "Fecha"]).any():
        raise ValueError("Hay fechas duplicadas por estación; no se pueden crear rezagos.")
    if not lags or not windows or any(window <= 0 for window in windows):
        raise ValueError("Los rezagos y ventanas deben ser listas no vacías de enteros positivos.")
    demand_grid = data.pivot(index="Fecha", columns="Origen_Id", values=target_column).asfreq("D")
    available = demand_grid.shift(horizon)

    history = {f"lag_{lag}": demand_grid.shift(lag) for lag in lags}
    for window in windows:
        history[f"roll_mean_{window}"] = available.rolling(window, min_periods=1).mean()
        history[f"active_days_{window}"] = available.rolling(window, min_periods=1).count()

    date_positions = demand_grid.index.get_indexer(data["Fecha"])
    station_positions = demand_grid.columns.get_indexer(data["Origen_Id"])
    result = data.copy()
    for name, values in history.items():
        result[name] = values.to_numpy()[date_positions, station_positions]
    return result


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    # Componentes de la fecha que se conocen antes de predecir.
    result = data.copy()
    result["day_of_month"] = result["Fecha"].dt.day
    result["year"] = result["Fecha"].dt.year
    return result


def build_model_dataset(data, horizon, lags, windows, target_column="target"):
    # Tabla de modelado: orden estable, historia y calendario.
    ordered = data.sort_values(["Origen_Id", "Fecha"], kind="stable").reset_index(drop=True)
    with_history = add_history_features(ordered, horizon, lags, windows, target_column)
    return add_calendar_features(with_history)


def split_by_date(data: pd.DataFrame, test_start: pd.Timestamp):
    # Desarrollo antes de la fecha de corte; prueba desde ella.
    development = data.loc[data["Fecha"] < test_start].reset_index(drop=True)
    test = data.loc[data["Fecha"] >= test_start].reset_index(drop=True)
    return development, test


# Días oficiales de descanso (Ley Federal del Trabajo, art. 74), la lista del notebook 05.
HOLIDAY_DATES = [101, 501, 916, 1225]   # mes * 100 + día
HOLIDAY_MONDAYS = [21, 33, 113]         # mes * 10 + número de lunes: 1.º de febrero, 3.º de marzo y de noviembre


def add_holidays(features: pd.DataFrame) -> pd.DataFrame:
    # Se calcula con month, day_of_month y day_of_week, así la API recibe las mismas 18 variables.
    features = features.copy()
    date_key = features["month"] * 100 + features["day_of_month"]
    monday_key = features["month"] * 10 + (features["day_of_month"] - 1) // 7 + 1
    holiday = date_key.isin(HOLIDAY_DATES) | ((features["day_of_week"] == 0) & monday_key.isin(HOLIDAY_MONDAYS))
    features["is_holiday"] = holiday.astype(int)
    return features
