"""Versionado SHA-256 de archivos de datos y resultados, y procedencia del dataset."""

import json
import shutil
from pathlib import Path

import pandas as pd

from mibici_mlops.config import (
    CSV_SNAPSHOT_MAX_BYTES,
    DAILY_PATH,
    DATA_DIR,
    EXPECTED_MONTHS,
    FILE_VERSION_DIR,
    FILE_VERSION_INDEX,
    FILE_VERSION_TABLE,
    INTEGRATED_PATH,
    OSM_PATH,
    POI_PATH,
    PROJECT_ROOT,
    STATIONS_PATH,
    STATION_ZONES_PATH,
    VALIDATION_REPORTS_DIR,
    WEATHER_PATH,
)
from mibici_mlops.dataset import find_latest_valid_run
from mibici_mlops.tracking import calculate_sha256


def version_csv_sources(root: Path, version_dir: Path) -> pd.DataFrame:
    """Inventariar todos los CSV sin alterar los archivos originales."""
    version_dir.mkdir(parents=True, exist_ok=True)
    index_path = version_dir / "csv_versions.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    current = {}
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() == ".csv"):
        if not path.is_file():
            continue
        relative = (
            path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_relative_to(PROJECT_ROOT) else path.relative_to(root).as_posix()
        )
        stat = path.stat()
        old = previous.get(relative, {})
        if old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            sha256 = old["sha256"]
        else:
            sha256 = calculate_sha256(path)
        history = old.get("versions", [])
        if sha256 not in history:
            history.append(sha256)
        current[relative] = {
            "status": "present",
            "sha256": sha256,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "versions": history,
        }
        rows.append({
            "file": relative, "sha256": sha256, "size_bytes": stat.st_size,
            "version": history.index(sha256) + 1, "known_versions": len(history),
        })
        # Los CSV mensuales de viajes (data/raw/<año>/) nunca se copian, sin importar su tamaño.
        if stat.st_size <= CSV_SNAPSHOT_MAX_BYTES and not path.name.startswith("datos_abiertos_"):
            snapshot = version_dir / "snapshots" / path.stem / f"{path.stem}_{sha256[:12]}.csv"
            if not snapshot.is_file():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, snapshot)
    # Conservar el historial de archivos que ya no están presentes.
    for relative, old in previous.items():
        if relative not in current:
            current[relative] = {**old, "status": "missing"}
    index_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    inventory = pd.DataFrame(rows, columns=["file", "sha256", "size_bytes", "version", "known_versions"])
    inventory.to_csv(version_dir / "csv_versions.csv", index=False)
    return inventory


def project_file_record(path: Path, role: str, used: bool = True) -> dict:
    """Describir un archivo sin copiarlo ni modificarlo."""
    path = Path(path)
    relative = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
    record = {"path": relative, "role": role, "used": used, "exists": path.is_file()}
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = calculate_sha256(path)
    return record


def build_data_provenance(inventory: pd.DataFrame) -> dict:
    """Relacionar las entradas conocidas con el integrado y su origen."""
    phase1_run = find_latest_valid_run(VALIDATION_REPORTS_DIR, len(EXPECTED_MONTHS))
    if phase1_run is None:
        raise FileNotFoundError("No se encontró la ejecución completa de la Fase 1.")
    used_sources = [
        (STATIONS_PATH, "catalogo_estaciones"),
        (WEATHER_PATH, "clima_exploratorio_y_zonas"),
        (STATION_ZONES_PATH, "zonas_estaticas"),
        (POI_PATH, "lugares_de_interes"),
        (phase1_run / "hourly_demand.parquet", "demanda_horaria_fase1"),
        (phase1_run / "manifest.json", "manifiesto_fase1"),
        (INTEGRATED_PATH, "dataset_integrado"),
    ]
    if DAILY_PATH.is_file():
        used_sources.append((DAILY_PATH, "demanda_diaria_intermedia"))
    if OSM_PATH.is_file():
        used_sources.append((OSM_PATH, "fuente_osm_para_poi_si_se_reconstruye"))
    files = [project_file_record(path, role, used=not role.startswith("fuente_osm_para_poi")) for path, role in used_sources]
    # Los CSV mensuales fueron procesados en la Fase 1; el manifiesto indica la corrida usada.
    monthly = inventory[inventory["file"].str.contains("datos_abiertos_", regex=False)]
    for row in monthly.itertuples(index=False):
        files.append({
            "path": row.file, "role": "csv_mensual_inventariado_fase1",
            "used": "upstream_no_verificado", "exists": True,
            "sha256": row.sha256, "size_bytes": int(row.size_bytes),
            "version": int(row.version),
        })
    by_path = {row.file: row for row in inventory.itertuples(index=False)}
    for file in files:
        if file["path"] in by_path:
            row = by_path[file["path"]]
            file["version"] = int(row.version)
            file["sha256"] = row.sha256
    return {
        "description": "Entradas conocidas y procedencia del dataset integrado de MiBici",
        "phase1_run": phase1_run.relative_to(PROJECT_ROOT).as_posix(),
        "integrated_sha256": calculate_sha256(INTEGRATED_PATH),
        "inventory_csv_count": len(inventory),
        "notes": [
            "El inventario completo contiene CSV encontrados, no necesariamente utilizados.",
            "Los CSV mensuales son candidatos de origen; la corrida procesada de Fase 1 es la entrada comprobada.",
            "Los archivos Parquet grandes no se duplican en MLflow.",
            "Los CSV externos pueden haberse reutilizado desde cache, sin descargarse en esta corrida.",
        ],
        "files": files,
    }


def version_used_files(sources: list[tuple[Path, str, str]]) -> pd.DataFrame:
    """Versionar archivos usados o generados; conservar el historial de cada ruta."""
    FILE_VERSION_DIR.mkdir(parents=True, exist_ok=True)
    previous = (
        json.loads(FILE_VERSION_INDEX.read_text(encoding="utf-8"))
        if FILE_VERSION_INDEX.is_file() else {}
    )
    records = dict(previous)
    rows = []
    for path, role, stage in sources:
        path = Path(path)
        relative = (
            path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_relative_to(PROJECT_ROOT) else str(path)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {relative}")
        stat = path.stat()
        old = records.get(relative, {})
        # El hash solo se reutiliza si no cambiaron tamaño ni fecha de modificación.
        if old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            sha256 = old["sha256"]
        else:
            sha256 = calculate_sha256(path)
        versions = list(old.get("versions", []))
        if sha256 not in versions:
            versions.append(sha256)
        version = versions.index(sha256) + 1
        records[relative] = {
            "sha256": sha256,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "versions": versions,
            "version": version,
            "role": role,
            "stage": stage,
            "status": "present",
        }
        rows.append({
            "file": relative, "role": role, "stage": stage,
            "sha256": sha256, "version": version, "size_bytes": stat.st_size,
        })
    FILE_VERSION_INDEX.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Conservar también los archivos registrados en llamadas anteriores.
    all_rows = [
        {"file": name, "role": item["role"], "stage": item["stage"],
         "sha256": item["sha256"], "version": item["version"],
         "size_bytes": item["size_bytes"]}
        for name, item in records.items()
        if item.get("status") == "present"
    ]
    table = pd.DataFrame(all_rows, columns=[
        "file", "role", "stage", "sha256", "version", "size_bytes"
    ]).drop_duplicates(subset=["file"])
    table.to_csv(FILE_VERSION_TABLE, index=False)
    return table


def collect_data_files() -> list[tuple[Path, str, str]]:
    """Reunir archivos del pipeline y fuentes de la Fase 1 sin asumir uso no demostrado."""
    files = []
    # Inventario completo de los CSV presentes en data, incluso los nuevos.
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".csv":
            files.append((path, "csv_inventariado", "inventario_datos"))
    # Fuentes e intermedios de la ejecución actual.
    for path, role in [
        (STATIONS_PATH, "catalogo_estaciones"),
        (WEATHER_PATH, "clima_exploratorio"),
        (STATION_ZONES_PATH, "zonas_estaticas"),
        (POI_PATH, "poi"),
        (DAILY_PATH, "demanda_diaria"),
        (INTEGRATED_PATH, "dataset_integrado"),
    ]:
        if path.is_file():
            files.append((path, role, "preparacion_datos"))
    phase1 = find_latest_valid_run(VALIDATION_REPORTS_DIR, len(EXPECTED_MONTHS))
    if phase1 is not None:
        for path in phase1.rglob("*"):
            if path.is_file():
                files.append((path, "salida_fase_1", "fase_1"))
    # El PBF se documenta como fuente disponible; no se declara usado si solo se reutilizó POI.
    if OSM_PATH.is_file():
        files.append((OSM_PATH, "osm_fuente_disponible", "fuente_externa"))
    return files
