"""Etapa de datos de la Fase 2: ejecución de la Fase 1, demanda diaria, caché e integración."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mibici_mlops.config import (
    ALLOW_UNVERIFIED_CACHE,
    CHUNK_SIZE,
    CONTRACT_DTYPES,
    DAILY_PATH,
    DATA_DIR,
    DATA_END,
    DATA_START,
    EXPECTED_MONTHS,
    INTEGRATED_PATH,
    POI_CATEGORIES,
    POI_COLUMNS,
    POI_PATH,
    POI_RADIUS_METERS,
    PROCESSED_DATA_DIR,
    SAVE_TRIPS,
    STATIONS_PATH,
    STATION_ZONES_PATH,
    VALIDATION_REPORTS_DIR,
    WEATHER_COLUMNS,
    WEATHER_GRID_DEGREES,
    WEATHER_PATH,
)
from mibici_mlops.checks import check_integration, check_poi_counts, validate_contract
from mibici_mlops.external import load_or_build_poi_counts, load_or_download_weather
from mibici_mlops.ingestion import discover_monthly_files
from mibici_mlops.pipeline import MiBiciPipeline
from mibici_mlops.tracking import calculate_sha256

logger = logging.getLogger("mibici_mlops")


def find_latest_valid_run(validation_dir: Path, expected_files: int) -> Path | None:
    # Devolver la ejecución completa más reciente de la Fase 1.
    candidates = []
    for hourly_path in validation_dir.rglob("hourly_demand.parquet"):
        manifest_path = hourly_path.parent / "manifest.json"
        if not manifest_path.is_file() or hourly_path.stat().st_size == 0:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("files_processed") == expected_files
            and manifest.get("full_period") is True
            and manifest.get("finished_utc")
        ):
            candidates.append((manifest["finished_utc"], hourly_path.parent))
    if not candidates:
        return None
    return max(candidates)[1]


def check_processed_run(run_dir: Path, expected_months: pd.PeriodIndex) -> pd.Series:
    # Validar que la ejecución sea consistente antes de reutilizarla.
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    kept_rows = manifest["rows"]["kept_rows"]
    hourly = pq.read_table(run_dir / "hourly_demand.parquet", columns=["Cantidad_viajes"])
    hourly_total = int(hourly.column("Cantidad_viajes").to_numpy().sum())

    expected = {f"datos_abiertos_{m.year}_{m.month:02d}.parquet" for m in expected_months}
    trip_files = sorted((run_dir / "trips").glob("*.parquet"))
    found = {path.name for path in trip_files if path.stat().st_size > 0}

    parquet_rows, unreadable = 0, []
    for path in trip_files:
        try:
            parquet_rows += pq.ParquetFile(path).metadata.num_rows
        except (OSError, ValueError):
            unreadable.append(path.name)

    return pd.Series({
        "Demanda horaria = viajes conservados": hourly_total == kept_rows,
        "Cobertura mensual completa": found == expected,
        "Parquet legibles": not unreadable,
        "Filas de Parquet = viajes conservados": parquet_rows == kept_rows,
    })


def cache_manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def build_cache_signature(run_dir: Path) -> dict:
    # Identificar las entradas y reglas que determinan el dataset integrado.
    inputs = {
        "hourly_demand": run_dir / "hourly_demand.parquet",
        "station_catalog": STATIONS_PATH,
        "weather": WEATHER_PATH,
        "station_zones": STATION_ZONES_PATH,
        "poi": POI_PATH,
    }
    return {
        "schema_version": 1,
        "study_start": str(DATA_START.date()),
        "study_end": str(DATA_END.date()),
        "weather_grid_degrees": WEATHER_GRID_DEGREES,
        "poi_radius_meters": POI_RADIUS_METERS,
        "poi_categories": POI_CATEGORIES,
        "contract_dtypes": CONTRACT_DTYPES,
        "sources": {
            name: {"sha256": calculate_sha256(path), "size": path.stat().st_size}
            for name, path in inputs.items()
        },
    }


def cache_is_current(path: Path, signature: dict) -> bool:
    manifest_path = cache_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        return saved.get("signature") == signature and saved.get("sha256") == calculate_sha256(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_cache_manifest(path: Path, signature: dict) -> None:
    manifest = {"signature": signature, "sha256": calculate_sha256(path)}
    cache_manifest_path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_daily_demand(hourly_data: pd.DataFrame) -> pd.DataFrame:
    # Pasar de viajes por estación y hora a viajes por estación y día.
    daily = hourly_data.copy()
    daily["Fecha"] = daily["Hora"].dt.normalize()
    daily = daily.groupby(["Origen_Id", "Fecha"], as_index=False)["Cantidad_viajes"].sum()
    daily = daily.rename(columns={"Cantidad_viajes": "target"})
    daily["day_of_week"] = daily["Fecha"].dt.day_of_week
    daily["month"] = daily["Fecha"].dt.month
    return daily


def load_station_coordinates(stations_path: Path) -> pd.DataFrame:
    # Leer el catálogo y conservar las estaciones con coordenadas.
    stations = pd.read_csv(stations_path)
    stations = stations[["id", "latitude", "longitude"]].rename(columns={"id": "Origen_Id"})
    return stations.dropna(subset=["latitude", "longitude"])


def integrate_daily_sources(daily_data, weather_data, weather_zones, poi_data):
    # Unir clima por zona y fecha, y POI por estación, sin duplicar filas.
    integrated = daily_data.copy()
    integrated["Fecha"] = pd.to_datetime(integrated["Fecha"])
    integrated = integrated.merge(
        weather_zones, on="Origen_Id", how="left", validate="many_to_one"
    )
    integrated = integrated.merge(
        weather_data[["time", "weather_zone"] + WEATHER_COLUMNS],
        left_on=["Fecha", "weather_zone"],
        right_on=["time", "weather_zone"],
        how="left",
        validate="many_to_one",
    )
    integrated = integrated.drop(columns=["time"])
    return integrated.merge(poi_data, on="Origen_Id", how="left", validate="many_to_one")


def prepare_model_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Reutilizar una ejecución válida o procesar los CSV si no existe ninguna.
    run_dir = find_latest_valid_run(VALIDATION_REPORTS_DIR, len(EXPECTED_MONTHS))

    if run_dir is None:
        logger.info("No existe una ejecución válida de la Fase 1. Procesando CSV.")

        raw_dir = DATA_DIR / "raw"
        files = discover_monthly_files(raw_dir)

        if len(files) != len(EXPECTED_MONTHS):
            raise ValueError(
                f"Se esperaban {len(EXPECTED_MONTHS)} CSV y se encontraron {len(files)}."
            )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        report_dir = VALIDATION_REPORTS_DIR / f"mibici_{run_id}"

        pipeline = MiBiciPipeline(CHUNK_SIZE, SAVE_TRIPS)
        pipeline.process_dataset(
            files,
            report_dir,
            True,
            "MiBici optimizado",
        )

        run_dir = report_dir

    run_checks = check_processed_run(run_dir, EXPECTED_MONTHS)
    if not run_checks.all():
        raise ValueError(f"La ejecución {run_dir.name} no pasa las validaciones:\n{run_checks}")
    logger.info("Ejecución de la Fase 1 utilizada: %s", run_dir.name)

    sources_ready = all(path.is_file() for path in (
        WEATHER_PATH, STATION_ZONES_PATH, POI_PATH, STATIONS_PATH,
    ))
    signature = build_cache_signature(run_dir) if sources_ready else None

    if INTEGRATED_PATH.is_file() and not force:
        if signature is not None and cache_is_current(INTEGRATED_PATH, signature):
            integrated = pd.read_parquet(INTEGRATED_PATH)
            report = validate_contract(integrated, CONTRACT_DTYPES, POI_COLUMNS)
            if report["passed"].all():
                logger.info("Dataset integrado reutilizado: %s", INTEGRATED_PATH)
                return integrated, report
            logger.warning("El integrado no pasa las pruebas de calidad; se reconstruye.")
        elif ALLOW_UNVERIFIED_CACHE:
            integrated = pd.read_parquet(INTEGRATED_PATH)
            report = validate_contract(integrated, CONTRACT_DTYPES, POI_COLUMNS)
            if report["passed"].all():
                logger.warning("Reutilizando Parquet sin procedencia verificada: %s", INTEGRATED_PATH)
                return integrated, report
        else:
            logger.info("No hay manifiesto vigente; se reconstruye el integrado una vez.")

    daily = build_daily_demand(pd.read_parquet(run_dir / "hourly_demand.parquet"))
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)

    stations = load_station_coordinates(STATIONS_PATH)
    stations = stations[stations["Origen_Id"].isin(daily["Origen_Id"])]
    weather, weather_zones = load_or_download_weather(stations)
    poi_data = load_or_build_poi_counts(stations)
    check_poi_counts(poi_data, set(daily["Origen_Id"]))

    integrated = integrate_daily_sources(daily, weather, weather_zones, poi_data)
    check_integration(daily, integrated)
    report = validate_contract(integrated, CONTRACT_DTYPES, POI_COLUMNS)

    if not report["passed"].all():
        raise ValueError(f"El dataset integrado no pasa el contrato:\n{report}")

    integrated.to_parquet(INTEGRATED_PATH, index=False)
    save_cache_manifest(INTEGRATED_PATH, build_cache_signature(run_dir))
    logger.info("Dataset integrado guardado: %s", INTEGRATED_PATH)

    return integrated, report