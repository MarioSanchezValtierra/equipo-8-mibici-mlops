"""Integración: del CSV crudo a las métricas, con limpieza, demanda diaria, variables, modelo y predicción."""

import json

from helpers import make_station_days, start_client, to_instance
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.config import (
    FEATURE_COLUMNS, LAGS, POI_COLUMNS, ROLLING_WINDOWS, SEED, SERVING_DIR, TARGET_COLUMN,
)
from mibici_mlops.dataset import build_daily_demand
from mibici_mlops.features import build_model_dataset, split_by_date
from mibici_mlops.ingestion import read_csv_chunks
from mibici_mlops.modeling.evaluate import predict_seasonal_naive, regression_metrics
from mibici_mlops.modeling.train import train_final_model
from mibici_mlops.preprocessing import (
    add_trip_features, clean_strategy_b, count_hourly_demand, normalize_chunk,
)

pytestmark = pytest.mark.integration
TEST_START = pd.Timestamp("2025-04-01")


def write_raw_trips(path, station_days: pd.DataFrame) -> int:
    """Pasar la demanda diaria a viajes individuales con el formato de datos abiertos."""
    rng = np.random.default_rng(SEED)
    rows, trip_id = [], 1
    for record in station_days.itertuples():
        for _ in range(record.target):
            start = record.Fecha + pd.Timedelta(minutes=int(rng.integers(6 * 60, 22 * 60)))
            rows.append({
                "Viaje_Id": trip_id, "Usuario_Id": int(rng.integers(1, 5000)),
                "Genero": rng.choice(["M", "F"]), "Año_de_nacimiento": int(rng.integers(1960, 2005)),
                "Inicio_del_viaje": start.strftime("%Y-%m-%d %H:%M:%S"),
                "Fin_del_viaje": (start + pd.Timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
                "Origen_Id": record.Origen_Id, "Destino_Id": 1 + record.Origen_Id % 4,
            })
            trip_id += 1
    dirty = [
        {**rows[0]},                                                    # Viaje_Id repetido
        {**rows[1], "Viaje_Id": trip_id, "Origen_Id": 407},             # estación excluida
        {**rows[2], "Viaje_Id": trip_id + 1, "Inicio_del_viaje": "no es fecha"},
        {**rows[3], "Viaje_Id": trip_id + 2, "Inicio_del_viaje": "2019-05-01 08:00:00"},
    ]
    pd.DataFrame(rows + dirty).to_csv(path, index=False)
    return len(rows)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    station_days = make_station_days()
    raw_path = tmp_path_factory.mktemp("raw") / "datos_abiertos_2025_01.csv"
    valid_rows = write_raw_trips(raw_path, station_days)

    # 1. Carga y limpieza por bloques, como en el pipeline
    reader, _ = read_csv_chunks(raw_path, chunk_size=5_000)
    hourly_parts, seen_ids, kept = [], set(), 0
    for chunk in reader:
        cleaned, summary, new_ids = clean_strategy_b(normalize_chunk(chunk), seen_ids)
        seen_ids |= new_ids
        kept += summary["kept_rows"]
        hourly_parts.append(count_hourly_demand(add_trip_features(cleaned)))
    hourly = pd.concat(hourly_parts).groupby(level=[0, 1]).sum().reset_index()

    # 2. Demanda diaria y atributos de las estaciones
    daily = build_daily_demand(hourly)
    stations = station_days.drop_duplicates("Origen_Id")[["Origen_Id", "weather_zone"] + POI_COLUMNS]
    integrated = daily.merge(stations, on="Origen_Id", how="left", validate="many_to_one")

    # 3. Variables, separación temporal, entrenamiento y predicción
    dataset = build_model_dataset(integrated, 1, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    development, test = split_by_date(dataset, TEST_START)
    model = train_final_model("Random Forest", {"n_estimators": 30, "max_depth": 6},
                              development, FEATURE_COLUMNS, TARGET_COLUMN)
    return {"station_days": station_days, "valid_rows": valid_rows, "kept": kept, "daily": daily,
            "development": development, "test": test, "model": model,
            "predictions": model.predict(test[FEATURE_COLUMNS])}


def test_cleaning_removes_only_the_dirty_rows(pipeline_run):
    assert pipeline_run["kept"] == pipeline_run["valid_rows"]


def test_daily_demand_matches_the_source(pipeline_run):
    expected = pipeline_run["station_days"].query("target > 0")[["Origen_Id", "Fecha", "target"]]
    merged = expected.merge(pipeline_run["daily"], on=["Origen_Id", "Fecha"], suffixes=("", "_obtenido"))
    assert len(merged) == len(expected)
    assert (merged["target"] == merged["target_obtenido"]).all()


def test_features_have_the_model_columns(pipeline_run):
    test = pipeline_run["test"]
    assert list(test[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert test["lag_7"].notna().all()  # en abril ya hay más de 14 días de historia


def test_split_is_temporal(pipeline_run):
    assert pipeline_run["development"]["Fecha"].max() < TEST_START
    assert pipeline_run["test"]["Fecha"].min() >= TEST_START


def test_model_beats_the_seasonal_naive(pipeline_run):
    test, development = pipeline_run["test"], pipeline_run["development"]
    model_metrics = regression_metrics(test[TARGET_COLUMN], pipeline_run["predictions"])
    naive_metrics = regression_metrics(test[TARGET_COLUMN],
                                       predict_seasonal_naive(development, test, TARGET_COLUMN))
    assert np.isfinite(list(model_metrics.values())).all()
    assert model_metrics["mae"] < naive_metrics["mae"]


def test_api_serves_the_trained_model(pipeline_run, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    mlflow.sklearn.save_model(pipeline_run["model"], str(model_dir),
                              serialization_format="cloudpickle", pip_requirements=["scikit-learn"])
    test = pipeline_run["test"]
    payload = {"instances": [to_instance(r) for r in test.to_dict("records")]}
    with start_client(model_dir, monkeypatch) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    served = np.array([p["predicted_trips"] for p in response.json()["predictions"]])
    assert served == pytest.approx(pipeline_run["predictions"].clip(min=0), abs=1e-4)
    served_mae = regression_metrics(test[TARGET_COLUMN], served)["mae"]
    assert served_mae == pytest.approx(
        regression_metrics(test[TARGET_COLUMN], pipeline_run["predictions"].clip(min=0))["mae"], abs=1e-4)


@pytest.mark.skipif(not (SERVING_DIR / "reference_sample.csv").is_file(),
                    reason="Falta models/serving (se crea con modeling/export_model.py)")
def test_serving_model_through_the_api_matches_reference(monkeypatch):
    meta = json.loads((SERVING_DIR / "metadata.json").read_text())
    reference = pd.read_csv(SERVING_DIR / "reference_sample.csv")
    payload = {"instances": [to_instance(r) for r in reference.to_dict("records")]}
    with start_client(SERVING_DIR / "model", monkeypatch) as client:
        body = client.post("/predict", json=payload).json()
    assert body["model_version"] == meta["model_version"]
    served = np.array([p["predicted_trips"] for p in body["predictions"]])
    assert served == pytest.approx(reference["reference_prediction"].clip(lower=0), abs=1e-4)
