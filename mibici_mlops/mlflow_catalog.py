"""Orden de MLflow para todas las fases: descripción y etiquetas de cada experimento, linaje de datos
de las ejecuciones de la Fase 3, marca de ejecuciones vigentes y el resumen de la entrega final."""
import json
from pathlib import Path

import mlflow
import pandas as pd

from mibici_mlops.config import (
    CHALLENGER_EXPERIMENT,
    DOCKERHUB_REPO,
    MLFLOW_ARTIFACT_DIR,
    MODEL_ALIAS,
    PHASE2_MODEL_VERSION,
    RELEASES,
    CSV_VERSION_DIR,
    FILE_VERSION_INDEX,
    FILE_VERSION_TABLE,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELING_EXPERIMENT,
    MONITORING_EXPERIMENT,
    PROJECT_ROOT,
    PROVENANCE_DIR,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
)

SUMMARY_EXPERIMENT = "MiBici_Entrega_Final"
SUMMARY_RUN = "Entrega final | Random Forest | resumen de las tres fases"
TRACE_TAGS = ["git_commit", "git_branch", "git_dirty", "implementation_commit", "python_version",
              "sklearn_version", "data_sha256", "data_manifest", "dataset_role", "dataset_path",
              "dataset_sha256"]

# Descripción y etiquetas de cada experimento. Solo se agregan; las ejecuciones no se tocan.
EXPERIMENTS = {
    MLFLOW_EXPERIMENT: (
        "Fase 1. Carga, limpieza e integración de los viajes de MiBici con inventario de archivos, "
        "versiones y procedencia de los datos.",
        {"phase": "fase_1", "content": "datos", "status": "vigente"},
    ),
    MODELING_EXPERIMENT: (
        "Fase 2. Líneas base, validación walk-forward, búsqueda de hiperparámetros y modelo final "
        "registrado como mibici_salidas_diarias v7.",
        {"phase": "fase_2", "content": "modelado", "status": "vigente"},
    ),
    "MiBici_Monitoreo_Drift": (
        "Fase 3, desarrollo en el notebook 07. Corridas de prueba del monitoreo de octubre. "
        f"El resultado final está en {MONITORING_EXPERIMENT}.",
        {"phase": "fase_3", "content": "monitoreo", "status": "desarrollo"},
    ),
    "MiBici_Champion_Challenger": (
        "Fase 3, desarrollo en el notebook 07. Entrenamiento de los challengers y registro de las "
        f"versiones nuevas. La comparación final está en {CHALLENGER_EXPERIMENT}.",
        {"phase": "fase_3", "content": "champion_challenger", "status": "desarrollo"},
    ),
    MONITORING_EXPERIMENT: (
        "Fase 3. Monitoreo de octubre de 2026 con la v7: data drift por variable, tres escenarios "
        "(rezagos faltantes, medias desplazadas y cambio estacional), alertas A1 a A5 y acción "
        "propuesta. Cada escenario es una ejecución hija.",
        {"phase": "fase_3", "content": "monitoreo", "status": "vigente"},
    ),
    CHALLENGER_EXPERIMENT: (
        "Fase 3. Champion contra challenger (selección en septiembre, prueba en octubre) y la "
        "documentación del champion que sirve la API y la imagen de Docker.",
        {"phase": "fase_3", "content": "champion_challenger", "status": "vigente"},
    ),
    SUMMARY_EXPERIMENT: (
        "Entrega final. Una ejecución que junta las tres fases: métricas, gráficas, tablas, decisión del "
        "champion, reproducibilidad, servicio, Docker y pruebas, con el enlace a cada ejecución de origen.",
        {"phase": "fases_1_2_3", "content": "resumen", "status": "vigente"},
    ),
}

REGISTERED_MODEL_DESCRIPTION = (
    "Salidas diarias por estación de MiBici, un día adelante. La v7 es el modelo final de la Fase 2. "
    "El alias champion marca la versión que sirve la API."
)


def versions_by_experiment(client: mlflow.MlflowClient) -> dict:
    """Versiones del Model Registry agrupadas por el experimento de la ejecución que las creó."""
    found = {}
    for version in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        try:
            experiment_id = client.get_run(version.run_id).info.experiment_id
        except Exception:
            continue
        found.setdefault(experiment_id, []).append(int(version.version))
    return {key: sorted(value) for key, value in found.items()}


def document_experiments(tracking_uri: str = MLFLOW_TRACKING_URI) -> pd.DataFrame:
    """Pone descripción y etiquetas a los experimentos que existan y a las versiones del registro."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    versions = versions_by_experiment(client)
    rows = []
    for name, (description, tags) in EXPERIMENTS.items():
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            continue
        registered = versions.get(experiment.experiment_id, [])
        if registered:
            description += " Versiones del registro creadas aquí: " + ", ".join(f"v{v}" for v in registered) + "."
        client.set_experiment_tag(experiment.experiment_id, "mlflow.note.content", description)
        for key, value in tags.items():
            client.set_experiment_tag(experiment.experiment_id, key, value)
        for version in registered:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "phase", tags["phase"])
            client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "source_experiment", name)
        rows.append({"experimento": name, **tags, "versiones": registered,
                     "ejecuciones": len(client.search_runs([experiment.experiment_id], max_results=5000))})

    model = client.get_registered_model(REGISTERED_MODEL_NAME)
    if not model.description:
        client.update_registered_model(REGISTERED_MODEL_NAME, REGISTERED_MODEL_DESCRIPTION)
    return pd.DataFrame(rows)


def mark_previous_runs(experiment_name: str, run_name: str) -> int:
    """Marca como anteriores las ejecuciones que ya existían con el mismo nombre, con sus hijas.

    No borra nada: solo cambia la etiqueta `run_status` a "anterior". La ejecución nueva se registra
    con "vigente", así el filtro `tags.run_status = "vigente"` deja una sola por tipo.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return 0
    escaped = run_name.replace("'", "\\'")
    previous = client.search_runs([experiment.experiment_id], f"attributes.run_name = '{escaped}'")
    for run in previous:
        children = client.search_runs([experiment.experiment_id],
                                      f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        for item in [run, *children]:
            client.set_tag(item.info.run_id, "run_status", "anterior")
    return len(previous)


def log_data_lineage(project_root: Path, data_path: Path, data_sha256: str, role: str,
                     datasets: dict) -> None:
    """Mismo linaje que las ejecuciones de la Fase 2: etiquetas del dataset, inventario, versiones,
    procedencia y los conjuntos usados en la pestaña Datasets.

    `datasets` va de contexto a (nombre, DataFrame, fuente), por ejemplo
    {"monitoring": ("mibici_monitoreo_2026_10", month, csv_path)}.
    """
    project_root = Path(project_root)
    mlflow.set_tags({
        "dataset_role": role,
        "dataset_path": data_path.relative_to(project_root).as_posix(),
        "dataset_sha256": data_sha256,
    })
    files = {
        "data/inventory": [CSV_VERSION_DIR / "csv_versions.csv", CSV_VERSION_DIR / "csv_versions.json"],
        "data/versions": [FILE_VERSION_TABLE, FILE_VERSION_INDEX],
        "data/lineage": [PROVENANCE_DIR / "data_provenance.json"],
    }
    phase1_run = None
    for folder, paths in files.items():
        for path in paths:
            local = project_root / path.relative_to(PROJECT_ROOT)
            if local.is_file():
                mlflow.log_artifact(str(local), folder)
                if local.name == "data_provenance.json":
                    phase1_run = json.loads(local.read_text(encoding="utf-8")).get("phase1_run")
    mlflow.log_dict({
        "integrated_path": data_path.relative_to(project_root).as_posix(),
        "integrated_sha256": data_sha256,
        "phase1_run": phase1_run,
        "datasets": {context: {"name": name, "rows": len(frame), "source": str(source)}
                     for context, (name, frame, source) in datasets.items()},
    }, "data/lineage/dataset_reference.json")
    for context, (name, frame, source) in datasets.items():
        dataset = mlflow.data.from_pandas(
            frame.reset_index(drop=True), source=str(source), name=name,
            targets=TARGET_COLUMN if TARGET_COLUMN in frame.columns else None,
        )
        mlflow.log_input(dataset, context=context)


def log_artifact_catalog(files: dict, project_root: Path) -> list:
    """Sube cada archivo a su carpeta y deja `artifact_catalog.json` con el mismo formato de la Fase 2."""
    from mibici_mlops.tracking import calculate_sha256

    project_root = Path(project_root)
    catalog = []
    for folder, paths in files.items():
        for path in paths:
            path = Path(path)
            if not path.is_file():
                continue
            mlflow.log_artifact(str(path), folder)
            try:
                source = path.resolve().relative_to(project_root.resolve()).as_posix()
            except ValueError:
                source = str(path)
            catalog.append({"file": path.name, "source_path": source, "artifact_path": f"{folder}/{path.name}",
                            "size_bytes": path.stat().st_size, "sha256": calculate_sha256(path)})
    mlflow.log_dict({"files": catalog}, "artifact_catalog.json")
    return catalog


def source_tags(module_path: str) -> dict:
    """La columna Source muestra el archivo que generó la ejecución, como main.py en la Fase 2."""
    return {"mlflow.source.name": module_path, "mlflow.source.type": "LOCAL", "run_status": "vigente"}


def trace_tags(run_id: str) -> dict:
    """Etiquetas de trazabilidad (commit, entorno y SHA-256 de los datos) de una ejecución, para repetirlas."""
    tags = mlflow.MlflowClient().get_run(run_id).data.tags
    return {key: tags[key] for key in TRACE_TAGS if key in tags}


# Las corridas hechas desde notebooks (Fase 1 y notebook 07) no guardaron
# git_commit ni data_sha256 con ese nombre; se buscan nombres alternos y si
# no hay ninguno se escribe "no registrado" en lugar de dejar la celda vacía.
DATA_HASH_TAGS = ["data_sha256", "dataset_sha256", "data_hash", "integrated_sha256"]
COMMIT_TAGS = ["git_commit", "implementation_commit", "mlflow.source.git.commit"]


def first_tag(run, keys: list) -> str:
    for key in keys:
        value = run.data.tags.get(key, "")
        if value:
            return value
    return ""


def short_value(value: str, size: int) -> str:
    return f"`{value[:size]}`" if value else "no registrado"


def current_run(client, experiment_name: str, run_name: str | None = None, stage: str | None = None):
    """La ejecución vigente más reciente de un experimento, por nombre o por la etiqueta stage."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    filters = []
    if run_name:
        filters.append(f"attributes.run_name = '{run_name}'")
    if stage:
        filters.append(f"tags.stage = '{stage}'")
    runs = client.search_runs([experiment.experiment_id], " and ".join(filters),
                              order_by=["attributes.start_time DESC"], max_results=20)
    current = [run for run in runs if run.data.tags.get("run_status", "vigente") == "vigente"]
    return (current or runs or [None])[0]


def run_link(run) -> str:
    return f"#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"


def copy_run_artifacts(client, run, destination: str, workdir: Path, max_bytes: int = 50_000_000) -> list:
    """Copia todos los artefactos de una ejecución a una carpeta del resumen. Devuelve los omitidos por tamaño."""
    skipped = []
    pending = [""]
    while pending:
        folder = pending.pop()
        for item in client.list_artifacts(run.info.run_id, folder or None):
            if item.is_dir:
                pending.append(item.path)
            elif item.file_size and item.file_size > max_bytes:
                skipped.append(item.path)
            else:
                local = Path(client.download_artifacts(run.info.run_id, item.path, str(workdir)))
                parent = Path(item.path).parent.as_posix()
                mlflow.log_artifact(str(local), destination if parent == "." else f"{destination}/{parent}")
    return skipped


def log_final_summary(project_root: Path = PROJECT_ROOT, verification: pd.DataFrame | None = None,
                      extra_files: dict | None = None) -> str:
    """Ejecución resumen de la entrega final en su propio experimento.

    Junta las métricas, parámetros y artefactos de las ejecuciones de origen: datos (Fase 1), modelo
    final v7 (Fase 2), monitoreo con sus escenarios, champion contra challenger y el champion.
    `verification` es la tabla de revisiones del notebook 08 y `extra_files` va de carpeta a archivos.
    """
    import os
    import tempfile

    from mibici_mlops import tracking

    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"  # sin una barra por archivo copiado
    from mibici_mlops.config import INTEGRATED_PATH

    project_root = Path(project_root)
    tracking.setup_mlflow(MLFLOW_TRACKING_URI, SUMMARY_EXPERIMENT, project_root / MLFLOW_ARTIFACT_DIR.name)
    mlflow.set_experiment(SUMMARY_EXPERIMENT)
    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    phase2 = client.get_model_version(REGISTERED_MODEL_NAME, str(PHASE2_MODEL_VERSION))
    sources = {
        "fase_1_datos": current_run(client, MLFLOW_EXPERIMENT),
        "fase_2_modelo_final": client.get_run(phase2.run_id),
        "fase_3_champion_entrenamiento": client.get_run(champion.run_id),
        "fase_3_monitoreo": current_run(client, MONITORING_EXPERIMENT, stage="monitoring"),
        "fase_3_champion_challenger": current_run(client, CHALLENGER_EXPERIMENT, stage="champion_challenger"),
        "fase_3_champion": current_run(client, CHALLENGER_EXPERIMENT, stage="champion"),
    }
    sources = {key: run for key, run in sources.items() if run is not None}
    scenarios = []
    if "fase_3_monitoreo" in sources:
        parent = sources["fase_3_monitoreo"]
        scenarios = client.search_runs([parent.info.experiment_id],
                                       f"tags.mlflow.parentRunId = '{parent.info.run_id}'")

    data_path = project_root / INTEGRATED_PATH.relative_to(PROJECT_ROOT)
    data_sha256 = tracking.calculate_sha256(data_path)
    mark_previous_runs(SUMMARY_EXPERIMENT, SUMMARY_RUN)
    with mlflow.start_run(run_name=SUMMARY_RUN, tags=source_tags("mibici_mlops/mlflow_catalog.py")) as run:
        tracking.log_run_context(project_root, data_path, data_sha256)
        log_data_lineage(project_root, data_path, data_sha256, "summary", {})
        # Los mismos conjuntos de datos de las ejecuciones de origen, para verlos en la columna Dataset
        seen, inputs = set(), []
        for source in sources.values():
            for item in (source.inputs.dataset_inputs if source.inputs else []):
                key = (item.dataset.name, item.dataset.digest)
                if key not in seen:
                    seen.add(key)
                    inputs.append(item)
        if inputs:
            client.log_inputs(run.info.run_id, datasets=inputs)
        mlflow.set_tags({
            "stage": "final_summary", "phase": "fases_1_2_3", "model_role": "champion",
            "model_version": champion.version, "phase2_model_version": phase2.version,
            "registered_model_name": REGISTERED_MODEL_NAME, "docker_image": DOCKERHUB_REPO,
            "docker_tag_champion": RELEASES.get(str(champion.version), "sin publicar"),
            **{f"run_{key}": source.info.run_id for key, source in sources.items()},
        })
        decision = sources.get("fase_3_champion_challenger")
        decision_text = decision.data.tags.get("decision", "") if decision else ""
        if decision_text:
            mlflow.set_tag("decision", decision_text)
        mlflow.set_tag("mlflow.note.content", (
            f"Resumen de la entrega final. Champion: {REGISTERED_MODEL_NAME} v{champion.version}; modelo de la "
            f"Fase 2: v{phase2.version}. Decisión en octubre de 2026: {decision_text or 'sin registrar'}. "
            "Cada carpeta de artefactos copia una ejecución de origen; README.md tiene los enlaces."
        ))
        mlflow.log_params({
            "champion_version": champion.version, "phase2_version": phase2.version,
            "docker_repository": DOCKERHUB_REPO,
            **{f"docker_tag_v{version}": tag for version, tag in RELEASES.items()},
            "api_endpoints": "POST /predict, GET /predict/estacion, GET /ui, GET /health, /docs, /redoc",
        })

        rows = []
        for key, source in sources.items():
            values = {f"{key}.{name}": value for name, value in source.data.metrics.items()}
            mlflow.log_metrics({name.replace(".", "__"): float(v) for name, v in values.items()
                                if pd.notna(v) and abs(v) != float("inf")})
            rows += [{"ejecución": key, "métrica": name, "valor": value}
                     for name, value in source.data.metrics.items()]
        for child in scenarios:
            name = child.data.tags.get("scenario", child.info.run_name)
            rows += [{"ejecución": f"escenario {name}", "métrica": metric, "valor": value}
                     for metric, value in child.data.metrics.items()]
        metrics_table = pd.DataFrame(rows)
        mlflow.log_table(metrics_table, "resumen/metricas_por_ejecucion.json")

        runs_table = pd.DataFrame([{
            "ejecución": key, "nombre": source.info.run_name,
            "experimento": client.get_experiment(source.info.experiment_id).name,
            "run_id": source.info.run_id, "enlace": run_link(source),
            "data_sha256": first_tag(source, DATA_HASH_TAGS),
            "git_commit": first_tag(source, COMMIT_TAGS),
        } for key, source in sources.items()] + [{
            "ejecución": "escenario", "nombre": child.info.run_name,
            "experimento": MONITORING_EXPERIMENT, "run_id": child.info.run_id, "enlace": run_link(child),
            "data_sha256": first_tag(child, DATA_HASH_TAGS), "git_commit": first_tag(child, COMMIT_TAGS),
        } for child in scenarios])
        mlflow.log_table(runs_table, "resumen/ejecuciones.json")

        alerts = pd.DataFrame([{
            "escenario": child.data.tags.get("scenario", child.info.run_name),
            "tipo": child.data.tags.get("scenario_type", ""), "acción": child.data.tags.get("action", ""),
            **{key: value for key, value in child.data.tags.items() if key.startswith("alert_")},
        } for child in scenarios])
        if not alerts.empty:
            mlflow.log_table(alerts, "resumen/alertas_por_escenario.json")
        if verification is not None:
            mlflow.log_table(verification, "resumen/verificacion_notebook_08.json")

        skipped = {}
        with tempfile.TemporaryDirectory() as workdir:
            for key, source in sources.items():
                skipped[key] = copy_run_artifacts(client, source, key, Path(workdir))
        log_artifact_catalog(extra_files or {}, project_root)

        lines = [
            "# Entrega final MiBici", "",
            f"Champion: `models:/{REGISTERED_MODEL_NAME}/{champion.version}` (alias `{MODEL_ALIAS}`). "
            f"Modelo de la Fase 2: v{phase2.version}. Decisión: {decision_text or 'sin registrar'}.",
            f"Imagen: `{DOCKERHUB_REPO}`, etiquetas " + ", ".join(f"v{v}: {t}" for v, t in RELEASES.items()) + ".", "",
            "## Ejecuciones de origen", "",
            "| Parte | Ejecución | Experimento | Enlace | SHA-256 de datos | Commit |", "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [f"| {r.ejecución} | {r.nombre.replace('|', '/')} | {r.experimento} | [abrir]({r.enlace}) | {short_value(r.data_sha256, 12)} | {short_value(r.git_commit, 10)} |"
                  for r in runs_table.itertuples()]
        lines += ["", "## Carpetas", "",
                  "Cada carpeta tiene una copia completa de los artefactos de su ejecución de origen, con su misma estructura.", ""]
        lines += [f"- `{key}/`: {source.info.run_name}" for key, source in sources.items()]
        lines += ["- `resumen/`: métricas de todas las ejecuciones, lista de ejecuciones con enlaces, alertas por escenario y revisiones del notebook 08.",
                  "", "Las métricas llevan el prefijo de su ejecución, por ejemplo `fase_3_champion__oct_mae`."]
        if any(skipped.values()):
            lines += ["", "Archivos no copiados por tamaño (siguen en su ejecución de origen):"]
            lines += [f"- {key}: {path}" for key, paths in skipped.items() for path in paths]
        mlflow.log_text("\n".join(lines), "README.md")
    return run.info.run_id


def main() -> None:
    """Resumen de la entrega final y descripción de todos los experimentos, desde la terminal."""
    reports = PROJECT_ROOT / "reports"
    extra = {
        "verificacion/pruebas": sorted((reports / "tests").glob("*.txt")),
        "verificacion/reproducibilidad": sorted(p for p in (reports / "reproducibility").glob("*") if p.is_file()),
    }
    print("Resumen:", log_final_summary(extra_files=extra))
    print(document_experiments().to_string(index=False))


if __name__ == "__main__":
    main()
