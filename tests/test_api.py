"""Pruebas del servicio: contrato de entrada y salida, y manejo de errores."""

import math

from helpers import to_instance
import pytest

from mibici_mlops.config import FEATURE_COLUMNS
from mibici_mlops.serving.app import EXAMPLE_INSTANCE, StationDayFeatures


def test_schema_has_the_model_features_in_order():
    assert list(StationDayFeatures.model_fields) == FEATURE_COLUMNS


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True,
                    "model_name": "mibici_salidas_diarias", "model_version": "prueba"}


def test_example_instance_gets_a_prediction(client):
    response = client.post("/predict", json={"instances": [EXAMPLE_INSTANCE]})
    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert prediction["Origen_Id"] == EXAMPLE_INSTANCE["Origen_Id"]
    assert prediction["predicted_trips"] >= 0


def test_api_matches_the_pipeline(client, model_dataset, trained_pipeline):
    rows = model_dataset.iloc[-20:]
    response = client.post("/predict", json={"instances": [to_instance(r) for r in rows.to_dict("records")]})
    served = [p["predicted_trips"] for p in response.json()["predictions"]]
    expected = trained_pipeline.predict(rows[FEATURE_COLUMNS]).clip(min=0)
    assert served == pytest.approx(expected, abs=1e-4)


def test_new_station_without_history_is_accepted(client):
    instance = {**EXAMPLE_INSTANCE, "Origen_Id": 9999, "lag_1": None, "lag_7": None, "lag_14": None,
                "roll_mean_7": None, "roll_mean_28": None, "active_days_7": 0, "active_days_28": 0}
    response = client.post("/predict", json={"instances": [instance]})
    assert response.status_code == 200
    assert math.isfinite(response.json()["predictions"][0]["predicted_trips"])


@pytest.mark.parametrize("change", [
    {"day_of_week": 7}, {"month": 13}, {"active_days_7": 8}, {"lag_1": -3},
    {"Origen_Id": "estacion"}, {"variable_extra": 1},
])
def test_invalid_values_return_422(client, change):
    response = client.post("/predict", json={"instances": [{**EXAMPLE_INSTANCE, **change}]})
    assert response.status_code == 422


def test_missing_feature_returns_422(client):
    instance = {k: v for k, v in EXAMPLE_INSTANCE.items() if k != "month"}
    assert client.post("/predict", json={"instances": [instance]}).status_code == 422


def test_empty_request_returns_422(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_without_model_returns_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    response = client_without_model.post("/predict", json={"instances": [EXAMPLE_INSTANCE]})
    assert response.status_code == 503


def test_openapi_documents_predict(client):
    schema = client.get("/openapi.json").json()
    assert "post" in schema["paths"]["/predict"]
    assert "StationDayFeatures" in schema["components"]["schemas"]


def test_station_day_prediction_matches_the_pipeline(client, model_dataset, trained_pipeline):
    row = model_dataset.loc[model_dataset["Fecha"] == "2025-04-15"].iloc[[0]]
    response = client.get("/predict/estacion",
                          params={"origen_id": int(row["Origen_Id"].iloc[0]), "fecha": "2025-04-15"})
    assert response.status_code == 200
    body = response.json()
    expected = max(float(trained_pipeline.predict(row[FEATURE_COLUMNS])[0]), 0)
    assert body["predicted_trips"] == pytest.approx(expected, abs=0.01)
    assert body["actual_trips"] == float(row["target"].iloc[0])
    assert list(body["features"]) == FEATURE_COLUMNS
    assert body["estimated"] is False


def test_station_day_without_history_returns_404(client):
    response = client.get("/predict/estacion", params={"origen_id": 1, "fecha": "2019-01-01"})
    assert response.status_code == 404


def test_future_day_uses_the_last_known_history(client, model_dataset):
    station = model_dataset.loc[model_dataset["Origen_Id"] == 1].set_index("Fecha")["target"]
    response = client.get("/predict/estacion", params={"origen_id": 1, "fecha": "2025-05-14"})
    assert response.status_code == 200
    body = response.json()
    assert body["estimated"] is True and body["history_end"] == "2025-04-30"
    assert body["actual_trips"] is None and body["predicted_trips"] >= 0
    features = body["features"]
    assert (features["day_of_week"], features["month"], features["day_of_month"]) == (2, 5, 14)
    assert features["lag_1"] == station["2025-04-30"]
    assert features["lag_7"] == station["2025-04-24"]
    assert features["roll_mean_7"] == pytest.approx(station["2025-04-24":"2025-04-30"].mean())
    assert features["active_days_28"] == 28


def test_ui_page_lists_stations_and_dates(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert 'option value="1"' in response.text
    assert 'min="2025-04-01"' in response.text and 'value="2025-04-30"' in response.text


def test_ui_without_model_returns_503(client_without_model):
    assert client_without_model.get("/ui").status_code == 503


def test_root_redirects_to_the_page(client):
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui"


def test_root_without_model_redirects_to_docs(client_without_model):
    assert client_without_model.get("/", follow_redirects=False).headers["location"] == "/docs"
