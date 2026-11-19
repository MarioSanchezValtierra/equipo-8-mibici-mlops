"""Reproducibilidad: misma semilla y mismo modelo guardado dan los mismos resultados."""

import json

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.config import FEATURE_COLUMNS, REGISTERED_MODEL_NAME, SERVING_DIR, TARGET_COLUMN
from mibici_mlops.modeling.evaluate import regression_metrics
from mibici_mlops.modeling.train import train_final_model

needs_serving = pytest.mark.skipif(
    not (SERVING_DIR / "reference_sample.csv").is_file(),
    reason="Falta models/serving (se crea con modeling/export_model.py)",
)


def test_same_seed_gives_same_predictions(model_dataset):
    params = {"n_estimators": 10, "max_depth": 5}
    first = train_final_model("Random Forest", params, model_dataset, FEATURE_COLUMNS)
    second = train_final_model("Random Forest", params, model_dataset, FEATURE_COLUMNS)
    np.testing.assert_allclose(first.predict(model_dataset[FEATURE_COLUMNS]),
                               second.predict(model_dataset[FEATURE_COLUMNS]), rtol=0, atol=1e-9)


def test_mlflow_format_keeps_predictions(mlflow_model_dir, trained_pipeline, model_dataset):
    loaded = mlflow.sklearn.load_model(str(mlflow_model_dir))
    np.testing.assert_allclose(loaded.predict(model_dataset[FEATURE_COLUMNS]),
                               trained_pipeline.predict(model_dataset[FEATURE_COLUMNS]),
                               rtol=0, atol=1e-9)


@needs_serving
def test_serving_metadata_describes_the_model():
    meta = json.loads((SERVING_DIR / "metadata.json").read_text())
    assert meta["model_name"] == REGISTERED_MODEL_NAME
    assert meta["registry_uri"] == f"models:/{REGISTERED_MODEL_NAME}/{meta['model_version']}"
    assert meta["features"] == FEATURE_COLUMNS
    assert meta["seed"] == 42


@needs_serving
def test_serving_model_reproduces_reference_sample():
    reference = pd.read_csv(SERVING_DIR / "reference_sample.csv")
    model = mlflow.sklearn.load_model(str(SERVING_DIR / "model"))
    predictions = model.predict(reference[FEATURE_COLUMNS])
    np.testing.assert_allclose(predictions, reference["reference_prediction"], rtol=0, atol=1e-6)
    current = regression_metrics(reference[TARGET_COLUMN], predictions)
    expected = regression_metrics(reference[TARGET_COLUMN], reference["reference_prediction"])
    assert current == pytest.approx(expected, abs=1e-9)
