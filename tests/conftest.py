"""Datos y modelos pequeños compartidos por las pruebas."""

import json

import numpy as np
import pandas as pd
import pytest

from helpers import make_station_days, start_client
from mibici_mlops.config import FEATURE_COLUMNS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN
from mibici_mlops.features import build_model_dataset


@pytest.fixture
def toy_data() -> pd.DataFrame:
    """Una estación, tres meses, con el 10 de enero faltante a propósito."""
    dates = pd.date_range("2024-12-01", "2025-02-28")
    data = pd.DataFrame({"Origen_Id": 1, "Fecha": dates, "target": np.arange(1, len(dates) + 1)})
    data = data.loc[data["Fecha"] != "2025-01-10"].reset_index(drop=True)
    data["day_of_week"] = data["Fecha"].dt.day_of_week
    return data


@pytest.fixture(scope="session")
def model_dataset():
    return build_model_dataset(make_station_days(), 1, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)


@pytest.fixture(scope="session")
def trained_pipeline(model_dataset):
    from mibici_mlops.modeling.train import train_final_model

    return train_final_model("Random Forest", {"n_estimators": 10, "max_depth": 5},
                             model_dataset, FEATURE_COLUMNS, TARGET_COLUMN)


@pytest.fixture(scope="session")
def mlflow_model_dir(trained_pipeline, model_dataset, tmp_path_factory):
    """El modelo pequeño guardado como lo deja export_model.py: carpeta MLflow, metadata.json y ui_days.csv."""
    import mlflow.sklearn

    from mibici_mlops.modeling.export_model import export_ui_days

    root = tmp_path_factory.mktemp("serving")
    mlflow.sklearn.save_model(trained_pipeline, str(root / "model"),
                              serialization_format="cloudpickle",
                              pip_requirements=["scikit-learn"])
    (root / "metadata.json").write_text(
        json.dumps({"model_name": "mibici_salidas_diarias", "model_version": "prueba"}))
    export_ui_days(model_dataset, model_dataset.iloc[0:0], root, start=pd.Timestamp("2025-04-01"))
    return root / "model"


@pytest.fixture
def client(mlflow_model_dir, monkeypatch):
    with start_client(mlflow_model_dir, monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    with start_client(tmp_path / "no_existe", monkeypatch) as test_client:
        yield test_client
