"""Exportar el champion de MLflow a models/serving para el servicio y la imagen de Docker."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd

from mibici_mlops.config import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    INTEGRATED_PATH,
    LAGS,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    ROLLING_WINDOWS,
    SEED,
    SERVING_DIR,
    TARGET_COLUMN,
    TEST_START,
)
from mibici_mlops import monitoring
from mibici_mlops.features import build_model_dataset, split_by_date

PACKAGES = ["scikit-learn", "pandas", "numpy", "scipy", "joblib", "cloudpickle", "pyarrow", "mlflow"]
REFERENCE_ROWS = 500
UI_DAYS_FILE = "ui_days.csv"
UI_START = pd.Timestamp("2026-09-01")


def environment_versions(packages: list[str] = PACKAGES) -> dict:
    """Versión de Python y de las librerías que afectan al modelo serializado."""
    environment = {"python": platform.python_version()}
    for package in packages:
        try:
            environment[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            environment[package] = None
    return environment


def sha256(path: Path, block: int = 1024 * 1024) -> str:
    """Hash del archivo, para comprobar que los datos son los del entrenamiento."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_serving_model(version, test: pd.DataFrame, integrated_sha: str, environment: dict,
                         serving_dir: Path = SERVING_DIR) -> pd.DataFrame:
    """Copiar una versión del registro sin volver a serializarla, con su metadata y muestra de referencia."""
    uri = f"models:/{REGISTERED_MODEL_NAME}/{version.version}"
    model_dir = serving_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(model_dir)))
    if downloaded.resolve() != model_dir.resolve():
        for item in downloaded.iterdir():
            shutil.move(str(item), model_dir / item.name)
        shutil.rmtree(downloaded)

    metadata_info = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(version.version),
        "model_alias": MODEL_ALIAS,
        "registry_uri": uri,
        "run_id": version.run_id,
        "features": FEATURE_COLUMNS,
        "seed": SEED,
        "integrated_sha256": integrated_sha,
        "environment": environment,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (serving_dir / "metadata.json").write_text(json.dumps(metadata_info, indent=2, ensure_ascii=False))

    # Muestra fija de la prueba 2026 con la predicción del modelo exportado
    served_model = mlflow.sklearn.load_model(str(model_dir))
    sample = test.sample(n=min(REFERENCE_ROWS, len(test)), random_state=SEED).sort_values(["Fecha", "Origen_Id"])
    reference = sample[["Fecha"] + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    reference["reference_prediction"] = served_model.predict(sample[FEATURE_COLUMNS])
    reference.to_csv(serving_dir / "reference_sample.csv", index=False)
    return reference


def export_ui_days(dataset: pd.DataFrame, month: pd.DataFrame, serving_dir: Path = SERVING_DIR,
                   start: pd.Timestamp = UI_START) -> pd.DataFrame:
    """Variables ya calculadas por estación y día para la página /ui, con los viajes reales."""
    columns = ["Fecha"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    days = pd.concat([dataset.loc[dataset["Fecha"] >= start, columns], month[columns]], ignore_index=True)
    days = days.drop_duplicates(["Origen_Id", "Fecha"], keep="last").sort_values(["Fecha", "Origen_Id"])
    days["Fecha"] = days["Fecha"].dt.strftime("%Y-%m-%d")
    serving_dir.mkdir(parents=True, exist_ok=True)
    days.to_csv(serving_dir / UI_DAYS_FILE, index=False)
    return days


def main() -> None:
    """Exportar la versión que tiene el alias champion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    integrated = pd.read_parquet(INTEGRATED_PATH)
    dataset = build_model_dataset(integrated, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    development, test = split_by_date(dataset, TEST_START)
    export_serving_model(version, test, sha256(INTEGRATED_PATH), environment_versions())
    print(f"Exportado models:/{REGISTERED_MODEL_NAME}/{version.version} a {SERVING_DIR / 'model'}")

    # Días de la página /ui: septiembre del integrado y el mes monitoreado (octubre)
    if monitoring.MONITOR_CSV.is_file():
        hourly, _ = monitoring.load_month_hourly(monitoring.MONITOR_CSV, monitoring.MONITOR_START,
                                                 monitoring.MONITOR_END)
        month_daily = monitoring.build_month_daily(hourly, integrated, set(development["Origen_Id"]))
        month = monitoring.build_month_dataset(integrated, month_daily, monitoring.MONITOR_START)
    else:
        month = dataset.iloc[0:0]
    days = export_ui_days(dataset, month)
    print(f"{UI_DAYS_FILE}: {len(days):,} filas, {days['Fecha'].min()} a {days['Fecha'].max()}")


if __name__ == "__main__":
    main()
