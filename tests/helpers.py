"""Datos sintéticos y utilidades que comparten las pruebas."""

import numpy as np
import pandas as pd

from mibici_mlops.config import POI_COLUMNS, SEED
from mibici_mlops.modeling.deploy import to_instance


def make_station_days(n_stations: int = 4, start="2025-01-01", end="2025-04-30", seed=SEED):
    """Demanda diaria con patrón semanal y un nivel distinto por estación."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    rows = []
    for station in range(1, n_stations + 1):
        weekly = np.where(dates.day_of_week < 5, 1.0, 0.5)
        rows.append(pd.DataFrame({
            "Origen_Id": station, "Fecha": dates, "target": rng.poisson(10 * station * weekly),
        }))
    data = pd.concat(rows, ignore_index=True)
    data["day_of_week"] = data["Fecha"].dt.day_of_week
    data["month"] = data["Fecha"].dt.month
    data["weather_zone"] = data["Origen_Id"] % 2
    for i, column in enumerate(POI_COLUMNS):
        data[column] = (data["Origen_Id"] * (i + 1)) % 5
    return data


def start_client(model_uri, monkeypatch):
    """Servicio de mibici_mlops/serving/app.py apuntando al modelo indicado."""
    from fastapi.testclient import TestClient

    from mibici_mlops.serving import app as service

    monkeypatch.setattr(service, "MODEL_URI", str(model_uri))
    return TestClient(service.app)
