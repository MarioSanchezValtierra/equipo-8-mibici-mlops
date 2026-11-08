"""Registro en MLflow: ejecución de la Fase 1, experimentos de modelado y Model Registry."""

import hashlib
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from mibici_mlops.config import (
    CSV_VERSION_DIR,
    EXCLUDED_STATIONS,
    EXCLUSION_REASONS,
    FILE_VERSION_INDEX,
    FILE_VERSION_TABLE,
    FIRST_CUTOFF,
    HORIZON_DAYS,
    LAGS,
    MAIN_FEATURE_SET,
    MLFLOW_EXPERIMENT,
    MODEL_ALIAS,
    MODEL_FIGURES_DIR,
    N_FOLDS,
    POI_COLUMNS,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
    ROLLING_WINDOWS,
    SEED,
    STUDY_END,
    STUDY_START,
    TARGET_COLUMN,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
)
from mibici_mlops.plots import save_figure


MLFLOW_ARTIFACT_GUIDE = """# Guía de artefactos de MiBici

- `data/inventory`: listado de CSV encontrados y sus versiones SHA-256.
- `data/inventory/model_reports`: versiones de los CSV de resultados.
- `data/lineage`: procedencia y referencias de datasets.
- `data/versions`: SHA-256 y versiones de CSV, Parquet, PBF y demás archivos de datos.
- `data/outputs`: hashes y versiones de los archivos de resultados de esta ejecución.
- `data/quality`: comprobaciones de calidad del dataset.
- `eda/fase_1`: gráficas exploratorias recuperadas del análisis inicial (demanda, estaciones y fuentes externas).
- `eda/fase_2`: gráficas complementarias del notebook de modelado (demanda, calidad temporal, estaciones y POI).
- Esta separación indica la etapa del análisis; no implica que sean copias exactas de los PNG originales de Fase 1.
- `modeling/validation`: resultados y gráficas de validación temporal.
- `modeling/comparison`: comparación de modelos y líneas base.
- `modeling/test`: resultados de la prueba independiente de 2026.
- `model`: pipeline serializado y firma de entrada de MLflow.
- `environment`: entorno, versiones de librerías y commit.
- `performance`: tiempos por etapa (`execution_times.csv`), gráfica comparativa (`execution_times.png`), resumen y versiones SHA-256.
- `documentation`: configuración y parámetros de cada ejecución y del modelo final.

Los CSV originales y los Parquet grandes no se copian a MLflow. El manifiesto registra
su ruta y hash cuando están disponibles. Los CSV del inventario no son necesariamente
todos los que se utilizaron para entrenar.

Las gráficas PNG se consultan en Artifacts. Los valores de MAE y RMSE por mes
se registran también como series de métricas para consultarlos en Model metrics.
Las gráficas de demanda por día, mes, evolución mensual y cobertura anual se
registran también como series en Model metrics. Sus pasos se explican en
`eda/metric_series_guide.json`. Histogramas, POI y relaciones con clima se
conservan como PNG en Artifacts, porque no tienen una escala temporal común.
System metrics corresponde al consumo de recursos, si está habilitado.
"""


def collect_environment(project_root: Path) -> dict:
    """Describir el entorno real sin modificar los datos ni el pipeline."""
    versions = {}
    for package in ('mlflow', 'pandas', 'numpy', 'pyarrow', 'matplotlib', 'IPython'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    def git_value(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ['git', *arguments], cwd=project_root,
                capture_output=True, text=True, check=True,
            ).stdout.strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git_value('rev-parse', 'HEAD')
    branch = git_value('branch', '--show-current')
    status = git_value('status', '--porcelain', '--untracked-files=normal')
    # None significa que Git no estuvo disponible; False significa árbol limpio.
    dirty = bool(status) if commit is not None else None
    return {
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'packages': versions,
        'git_commit': commit,
        'git_branch': branch or 'detached_or_unavailable',
        'git_dirty': dirty,
    }


def log_environment(project_root: Path) -> None:
    """Guardar versiones y commit en el run de MLflow como tags y JSON."""
    environment = collect_environment(project_root)
    mlflow.set_tag('python_version', environment['python_version'])
    mlflow.set_tag(
        'implementation_commit',
        environment['git_commit'] or 'unavailable',
    )
    mlflow.set_tag('git_commit', environment['git_commit'] or 'unavailable')
    mlflow.set_tag('git_branch', environment['git_branch'])
    mlflow.set_tag('git_dirty', str(environment['git_dirty']).lower())
    mlflow.log_dict(environment, 'environment/runtime.json')


def calculate_sha256(file_path: Path) -> str:
    """Calcular el SHA-256 sin cargar el archivo completo en memoria."""
    sha256 = hashlib.sha256()

    with file_path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            sha256.update(block)

    return sha256.hexdigest()


def log_datasets(csv_files: list[Path], splits: pd.DataFrame) -> None:
    """Registrar en MLflow los CSV originales y los conjuntos procesados."""
    if not csv_files:
        raise ValueError('No se encontraron archivos CSV originales.')

    source_records = []

    for file_path in sorted(csv_files):
        source_records.append({
            'file_name': file_path.name,
            'size_bytes': file_path.stat().st_size,
            'sha256': calculate_sha256(file_path),
        })

    source_manifest = pd.DataFrame(source_records)

    if source_manifest['file_name'].duplicated().any():
        raise ValueError('Existen nombres de archivos CSV duplicados.')

    mlflow.log_table(
        source_manifest,
        'datasets/source_files_manifest.json',
    )

    source_dataset = mlflow.data.from_pandas(
        source_manifest,
        name='mibici_source_files',
    )
    mlflow.log_input(source_dataset, context='source')

    contexts = {
        'train': 'training',
        'validation': 'validation',
        'test': 'testing',
    }

    for name, context in contexts.items():
        dataset_info = pd.DataFrame([{
            'split': name,
            'sha256': str(splits.loc[name, 'SHA256']),
            'rows': int(splits.loc[name, 'Filas']),
            'trips': int(splits.loc[name, 'Viajes']),
        }])

        dataset = mlflow.data.from_pandas(
            dataset_info,
            name=f'mibici_{name}_metadata',
        )

        mlflow.log_input(dataset, context=context)
        mlflow.log_table(
            dataset_info,
            f'datasets/{name}_metadata.json',
        )


def log_eda_to_mlflow(tables: dict, monthly_trips: pd.Series) -> None:
    """Registrar indicadores, series y tablas existentes del EDA de train."""
    stats = tables['Estadísticas de demanda']['Cantidad_viajes']

    mlflow.log_metrics({
        'eda_train_mean_demand': float(stats['mean']),
        'eda_train_median_demand': float(stats['50%']),
        'eda_train_std_demand': float(stats['std']),
        'eda_train_p95_demand': float(stats['95%']),
        'eda_train_max_demand': float(stats['max']),
        'eda_train_station_count': len(tables['Demanda por estación']),
    })

    hourly = tables['Demanda por hora'].sort_values('Hora_del_dia')

    for _, row in hourly.iterrows():
        mlflow.log_metric(
            'eda_train_demand_by_hour',
            float(row['Promedio_viajes']),
            step=int(row['Hora_del_dia']),
        )

    weekdays = tables['Demanda por día'].sort_values('Dia_semana')

    for _, row in weekdays.iterrows():
        mlflow.log_metric(
            'eda_train_demand_by_weekday',
            float(row['Promedio_viajes']),
            step=int(row['Dia_semana']),
        )

    months = monthly_trips.sort_index()

    for step, value in enumerate(months):
        mlflow.log_metric(
            'eda_train_monthly_trips',
            float(value),
            step=step,
        )

    mlflow.log_table(
        hourly,
        artifact_file='eda_data/demand_by_hour.json',
    )
    mlflow.log_table(
        weekdays,
        artifact_file='eda_data/demand_by_weekday.json',
    )
    mlflow.log_table(
        tables['Demanda por estación'].head(10),
        artifact_file='eda_data/top_10_stations.json',
    )

    month_table = pd.DataFrame({
        'Mes': months.index.astype(str),
        'Cantidad_viajes': months.to_numpy(),
    })

    mlflow.log_table(
        month_table,
        artifact_file='eda_data/monthly_trips.json',
    )


def log_to_mlflow(
    results,
    splits: pd.DataFrame,
    project_root: Path,
    run_label: str,
    chunk_size: int,
    run_all_files: bool,
    tables: dict,
    monthly_trips: pd.Series,
    csv_files: list[Path],
) -> str:
    """Registrar parámetros, métricas y artefactos de la ejecución."""

    tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    project_run_id = os.environ.get('MLFLOW_RUN_ID')

    if project_run_id:
        # MLflow Projects ya creó una ejecución.
        # Recuperamos su experimento en lugar de seleccionar otro.
        client = mlflow.tracking.MlflowClient()
        project_run = client.get_run(project_run_id)
        experiment_id = project_run.info.experiment_id
    else:
        # Ejecución directa desde Python.
        # Seleccionamos el experimento configurado para MiBici.
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                MLFLOW_EXPERIMENT,
                artifact_location=(project_root / 'mlruns').as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id

    report_dir = results['report_dir']
    totals = results['totals']

    run_name = f'{run_label} | {report_dir.name}'

    if project_run_id:
        run_context = mlflow.start_run(
            run_id=project_run_id,
            run_name=run_name,
        )
    else:
        run_context = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        )

    with run_context as run:
        mlflow.set_tag("mlflow.note.content", "Fase 1: preparación y trazabilidad de datos MiBici. Ver artefactos y manifiestos de esta ejecución.")
        log_environment(project_root)
        log_datasets(csv_files, splits)

        mlflow.log_params({
            'run_label': run_label,
            'chunk_size': chunk_size,
            'run_all_files': run_all_files,
            'csv_files': len(results['file_summary']),
            'study_start': STUDY_START.date(),
            'train_end': TRAIN_END.date(),
            'validation_end': VALIDATION_END.date(),
            'study_end': STUDY_END.date(),
            'excluded_stations': sorted(EXCLUDED_STATIONS),
            **{
                f'{name}_sha256': splits.loc[name, 'SHA256']
                for name in splits.index
            },
        })

        excluded = {
            'excluded_' + re.sub(r'\W+', '_', reason).lower(): totals[reason]
            for reason in EXCLUSION_REASONS
        }

        mlflow.log_metrics({
            **{
                key: totals[key]
                for key in ('original_rows', 'kept_rows', 'excluded_rows')
            },
            **excluded,
            **{
                f'{name}_rows': splits.loc[name, 'Filas']
                for name in splits.index
            },
            **{
                f'{name}_trips': splits.loc[name, 'Viajes']
                for name in splits.index
            },
        })

        for file_name in (
            'manifest.json',
            'file_summary.csv',
            'timing_summary.csv',
        ):
            mlflow.log_artifact(
                report_dir / file_name,
                'reportes',
            )

        mlflow.log_artifact(
            report_dir / 'splits' / 'split_summary.csv',
            'splits',
        )

        mlflow.log_artifacts(
            report_dir / 'eda_tables',
            'eda_tables',
        )

        mlflow.log_artifacts(
            report_dir / 'eda_figures',
            'eda_figures',
        )

        log_eda_to_mlflow(tables, monthly_trips)

    print('Ejecución de MLflow:', run.info.run_id)
    return run.info.run_id


def setup_mlflow(tracking_uri: str, experiment_name: str, artifact_dir: Path) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
    return experiment.experiment_id


def log_run_context(project_root: Path, data_path: Path, data_sha256: str) -> None:
    # Información compartida por todas las ejecuciones.
    environment = collect_environment(project_root)
    for package in ("scikit-learn", "scipy", "joblib"):
        environment["packages"][package] = metadata.version(package)
    mlflow.set_tags({
        "implementation_commit": environment["git_commit"] or "unavailable",
        "git_commit": environment["git_commit"] or "unavailable",
        "git_branch": environment["git_branch"],
        "git_dirty": str(environment["git_dirty"]).lower(),
        "python_version": environment["python_version"],
        "sklearn_version": environment["packages"]["scikit-learn"],
        "data_sha256": data_sha256,
        "data_manifest": "data/lineage/data_provenance.json",
    })
    mlflow.log_dict(environment, "environment/runtime.json")
    # Configuración reproducible y documentación de parámetros comunes.
    run_parameters = {
        "data_file": data_path.name,
        "horizon_days": HORIZON_DAYS,
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "first_cutoff": FIRST_CUTOFF.date(),
        "test_start": TEST_START.date(),
        "n_folds": N_FOLDS,
        "seed": SEED,
    }
    mlflow.log_dict(
        {
            "parameters": {name: str(value) for name, value in run_parameters.items()},
            "data_sha256": data_sha256,
            "data_path": str(data_path),
            "environment": environment,
        },
        "documentation/run_configuration.json",
    )
    mlflow.log_params({
        "data_file": data_path.name,
        "horizon_days": HORIZON_DAYS,
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "first_cutoff": FIRST_CUTOFF.date(),
        "test_start": TEST_START.date(),
        "n_folds": N_FOLDS,
        "seed": SEED,
    })


def log_data_documentation(context: dict) -> None:
    """Guardar el inventario y la procedencia en la ejecución padre.

    `context` lo arma `workflow.build_run_context`: rutas, SHA-256, procedencia,
    conteos de inventario y los datasets de MLflow de desarrollo y prueba.
    """
    mlflow.log_artifact(str(CSV_VERSION_DIR / "csv_versions.csv"), "data/inventory")
    mlflow.log_artifact(str(CSV_VERSION_DIR / "csv_versions.json"), "data/inventory")
    mlflow.log_artifact(str(context["provenance_path"]), "data/lineage")
    mlflow.log_artifact(str(FILE_VERSION_TABLE), "data/versions")
    mlflow.log_artifact(str(FILE_VERSION_INDEX), "data/versions")
    mlflow.log_dict({
        "integrated_path": str(context["data_path"].relative_to(PROJECT_ROOT)),
        "integrated_sha256": context["data_sha256"],
        "phase1_run": context["phase1_run"],
        "csv_inventory_count": context["csv_inventory_count"],
        "file_inventory_count": context["file_inventory_count"],
        "file_versions_manifest": "data/versions/file_versions.csv",
    }, "data/lineage/dataset_reference.json")


def log_dataset_reference(context: dict, role: str = "validation") -> None:
    """Vincular cada corrida con el dataset integrado, sin copiar sus filas."""
    mlflow.set_tags({
        "dataset_role": role,
        "dataset_path": str(context["data_path"].relative_to(PROJECT_ROOT)),
        "dataset_sha256": context["data_sha256"],
    })
    log_data_documentation(context)
    # Mismo dataset de desarrollo para los modelos y las líneas base.
    mlflow.log_input(
        context["development_input"],
        context="training" if role == "training_and_testing" else "validation",
    )
    if role == "training_and_testing":
        mlflow.log_input(context["test_input"], context="testing")


def log_report_artifacts(files: list[Path]) -> list[dict]:
    """Separar los archivos por contenido, no por la carpeta donde se generaron."""
    eda_demand = {
        "demand_distribution.png", "monthly_demand.png", "monthly_demand_trend.png",
        "weekday_demand.png", "day_of_month_demand.png", "holiday_demand.png",
        "missing_periods.png", "shared_missing_dates.png", "demand_correlation.png",
    }
    eda_stations = {
        "station_ranking.png", "annual_station_coverage.png",
        "station_attribute_consistency.png",
    }
    eda_external = {
        "weather_distribution.png", "weather_vs_demand.png", "station_poi.png",
        "top_station_demand_vs_poi.png", "bottom_station_demand_vs_poi.png",
    }
    modeling_validation = {"mae_by_fold.png", "final_walk_forward.csv"}
    modeling_comparison = {"model_comparison.csv", "mlflow_model_comparison.png"}
    modeling_test = {"test_results_2026.csv", "test_predictions_2026.png"}
    quality = {"quality_report.csv"}
    inventory = {"csv_versions.csv", "csv_versions.json"}
    recorded = set()
    catalog = []
    for path in files:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Artefacto no encontrado: {path}")
        if path.name in inventory:
            folder = "data/inventory/model_reports" if "model_reports" in path.parts else "data/inventory"
        elif path.name in quality:
            folder = "data/quality"
        elif path.name in eda_demand:
            folder = "eda/fase_1/demanda"
        elif path.name in eda_stations:
            folder = "eda/fase_1/estaciones"
        elif path.name in eda_external:
            folder = "eda/fase_1/fuentes_externas"
        elif path.name in modeling_validation:
            folder = "modeling/validation"
        elif path.name in modeling_comparison:
            folder = "modeling/comparison"
        elif path.name in modeling_test:
            folder = "modeling/test"
        else:
            folder = "reports/other"
        # Los análisis complementarios del notebook 05 van en Fase 2.
        phase2_eda = {
            "monthly_demand_trend.png": "demanda",
            "day_of_month_demand.png": "demanda",
            "holiday_demand.png": "demanda",
            "missing_periods.png": "calidad_temporal",
            "shared_missing_dates.png": "calidad_temporal",
            "annual_station_coverage.png": "estaciones",
            "station_attribute_consistency.png": "estaciones",
            "top_station_demand_vs_poi.png": "fuentes_externas",
            "bottom_station_demand_vs_poi.png": "fuentes_externas",
        }
        if path.name in phase2_eda:
            folder = f"eda/fase_2/{phase2_eda[path.name]}"
        # Evitar duplicados en una misma carpeta, sin descartar ningún archivo.
        key = (folder, path.name)
        if key not in recorded:
            mlflow.log_artifact(str(path), folder)
            recorded.add(key)
            catalog.append({
                "file": path.name,
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "artifact_path": f"{folder}/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            })
    return catalog


def log_walk_forward(results, run_name, context, params=None, columns=None, tags=None) -> str:
    # Corrida principal con el resumen y una corrida hija por ventana.
    with mlflow.start_run(run_name=run_name) as run:
        log_run_context(context["project_root"], context["data_path"], context["data_sha256"])
        mlflow.set_tag("mlflow.note.content", "Validación temporal walk-forward. Cada ejecución hija corresponde a una ventana; los resultados se encuentran en modeling/validation.")
        mlflow.set_tags({"stage": "walk_forward", **(tags or {})})
        log_dataset_reference(context, role="walk_forward")
        if params:
            mlflow.log_params({f"model_{k}": v for k, v in params.items()})
        if columns:
            mlflow.log_param("n_features", len(columns))
            mlflow.log_dict({"features": list(columns)}, "features.json")
        summary = {
            "mae_mean": results["mae"].mean(),
            "mae_std": results["mae"].std(),
            "rmse_mean": results["rmse"].mean(),
            "skill_vs_naive_mean": results["skill_vs_naive"].mean(),
        }
        if "fit_seconds" in results:
            summary["fit_seconds_total"] = results["fit_seconds"].sum()
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if pd.notna(v) and np.isfinite(v)})
        for step, row in enumerate(results.itertuples(index=False)):
            mlflow.log_metric("fold_mae", row.mae, step=step)
            mlflow.log_metric("fold_rmse", row.rmse, step=step)
            with mlflow.start_run(run_name=f"{run_name} | {row.cutoff}", nested=True):
                mlflow.log_param("cutoff", row.cutoff)
                mlflow.set_tag("dataset_sha256", context["data_sha256"])
                fold_metrics = {"mae": row.mae, "rmse": row.rmse, "skill_vs_naive": row.skill_vs_naive}
                for name in ("fit_seconds", "train_rows", "validation_rows"):
                    if hasattr(row, name):
                        fold_metrics[name] = getattr(row, name)
                mlflow.log_metrics({k: float(v) for k, v in fold_metrics.items() if pd.notna(v) and np.isfinite(v)})
        mlflow.log_table(results.astype({"cutoff": str}), "modeling/validation/walk_forward_results.json")
        # Una gráfica por modelo, guardada en la corrida padre y sin ventanas.
        if not results.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(results["cutoff"].astype(str), results["mae"], marker="o", label="MAE")
            ax.plot(results["cutoff"].astype(str), results["rmse"], marker="s", label="RMSE")
            ax.set(title="Error por ventana temporal", xlabel="Inicio de validación", ylabel="Viajes")
            ax.tick_params(axis="x", rotation=60)
            ax.legend()
            fig.tight_layout()
            chart_path = MODEL_FIGURES_DIR / f"mlflow_{run.info.run_id}_fold_errors.png"
            save_figure(fig, chart_path)
            mlflow.log_artifact(str(chart_path), artifact_path="modeling/validation")
        return run.info.run_id


def log_grid_search(search, model_name: str, context: dict) -> str:
    with mlflow.start_run(run_name=f"Búsqueda | {model_name}") as run:
        log_run_context(context["project_root"], context["data_path"], context["data_sha256"])
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("mlflow.note.content", "Búsqueda de hiperparámetros. Cada ejecución hija registra una configuración y sus métricas de validación.")
        log_dataset_reference(context, role="tuning")
        results = pd.DataFrame(search.cv_results_)
        for i, row in results.iterrows():
            with mlflow.start_run(run_name=f"{model_name} | combinación {i}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
                mlflow.log_metrics({
                    "mae": -row["mean_test_mae"],
                    "mae_std": row["std_test_mae"],
                    "rmse": -row["mean_test_rmse"],
                    "fit_seconds": row["mean_fit_time"],
                })
        return run.info.run_id


def log_eda_metric_series(data: pd.DataFrame, poi_columns: list[str]) -> None:
    """Guardar indicadores de EDA que sí tienen sentido como métricas numéricas."""
    guide = {}

    def log_series(name: str, values: pd.Series, meaning: str, unit: str) -> None:
        # MLflow usa pasos numéricos; las etiquetas originales quedan en el catálogo.
        series = pd.Series(values)
        for step, value in enumerate(series):
            if pd.notna(value) and np.isfinite(value):
                mlflow.log_metric(name, float(value), step=step)
        guide[name] = {
            "description": meaning,
            "unit": unit,
            "source": "Dataset integrado utilizado en esta ejecución",
            "interpretation": "Cada punto corresponde a la etiqueta de la misma posición; no es un instante de entrenamiento.",
            "labels": [str(label) for label in series.index],
            "step": "Posición de la etiqueta en labels, empezando en 0",
        }

    def log_value(name: str, value: float, meaning: str, unit: str) -> None:
        if pd.notna(value) and np.isfinite(value):
            mlflow.log_metric(name, float(value))
            guide[name] = {"description": meaning, "unit": unit, "source": "Dataset integrado utilizado en esta ejecución", "step": "Métrica escalar: no tiene serie temporal"}

    # Definiciones explícitas: evitar descripciones como "Resumen: p90_trips".
    summary_definitions = {
        "records": ("Registros estación-día disponibles en el dataset integrado", "registros"),
        "stations": ("Estaciones de origen distintas en el dataset integrado", "estaciones"),
        "dates": ("Fechas distintas con registros en el dataset integrado", "días"),
        "total_trips": ("Suma de viajes de todos los registros estación-día", "viajes"),
        "mean_trips": ("Promedio de viajes por registro estación-día", "viajes/estación-día"),
        "median_trips": ("Mediana de viajes por registro estación-día", "viajes/estación-día"),
        "std_trips": ("Desviación estándar muestral de viajes por estación-día", "viajes/estación-día"),
        "min_trips": ("Menor número de viajes registrado en una estación-día", "viajes"),
        "max_trips": ("Mayor número de viajes registrado en una estación-día", "viajes"),
        "p10_trips": ("Percentil 10 de viajes por estación-día", "viajes/estación-día"),
        "p90_trips": ("Percentil 90 de viajes por estación-día", "viajes/estación-día"),
        "p95_trips": ("Percentil 95 de viajes por estación-día", "viajes/estación-día"),
        "zero_trip_rows": ("Registros estación-día con cero viajes", "registros"),
        "zero_trip_fraction": ("Proporción de registros estación-día con cero viajes", "proporción 0–1"),
    }
    target = data["target"]
    summary_values = {
        "records": len(data),
        "stations": data["Origen_Id"].nunique(),
        "dates": data["Fecha"].nunique(),
        "total_trips": target.sum(),
        "mean_trips": target.mean(),
        "median_trips": target.median(),
        "std_trips": target.std(),
        "min_trips": target.min(),
        "max_trips": target.max(),
        "p10_trips": target.quantile(0.10),
        "p90_trips": target.quantile(0.90),
        "p95_trips": target.quantile(0.95),
        "zero_trip_rows": target.eq(0).sum(),
        "zero_trip_fraction": target.eq(0).mean(),
    }
    for name, value in summary_values.items():
        description, unit = summary_definitions[name]
        log_value(f"eda_fase1_{name}", value, description, unit)

    log_series(
        "eda_fase1_weekday_mean_trips",
        data.groupby("day_of_week")["target"].mean().reindex(range(7)),
        "Promedio de viajes por día de la semana; 0=lunes", "viajes/estación-día",
    )
    log_series(
        "eda_fase1_month_mean_trips",
        data.groupby("month")["target"].mean().reindex(range(1, 13)),
        "Promedio de viajes por mes del año", "viajes/estación-día",
    )
    station_mean = data.groupby("Origen_Id")["target"].mean()
    for name, value in {
        "min": station_mean.min(),
        "p10": station_mean.quantile(0.10),
        "median": station_mean.median(),
        "p90": station_mean.quantile(0.90),
        "max": station_mean.max(),
    }.items():
        log_value(f"eda_fase1_station_mean_{name}", value,
                  f"Estadístico {name} de los promedios diarios de viajes calculados por estación", "viajes/estación-día")

    # Fase 2: tendencia y cobertura; una observación por mes o por año.
    monthly_group = data.groupby(data["Fecha"].dt.to_period("M"))
    monthly = monthly_group["target"].agg(["mean", "sum", "size"])
    monthly["stations"] = monthly_group["Origen_Id"].nunique()
    for column, meaning, unit in (
        ("mean", "Demanda media por estación-día observado", "viajes/estación-día"),
        ("sum", "Total mensual de viajes", "viajes"),
        ("size", "Registros estación-día observados", "registros"),
        ("stations", "Estaciones con registros en el mes", "estaciones"),
    ):
        log_series(f"eda_fase2_monthly_{column}", monthly[column], meaning, unit)

    annual_group = data.groupby(data["Fecha"].dt.year)
    annual = annual_group["target"].agg(["sum", "mean", "size"])
    annual["stations"] = annual_group["Origen_Id"].nunique()
    for column, unit in (("sum", "viajes"), ("mean", "viajes/estación-día"),
                         ("size", "registros"), ("stations", "estaciones")):
        log_series(f"eda_fase2_annual_{column}", annual[column],
                   f"Estadístico anual {column} de demanda o cobertura de estaciones", unit)

    log_series(
        "eda_fase2_day_of_month_mean_trips",
        data.groupby(data["Fecha"].dt.day)["target"].mean().reindex(range(1, 32)),
        "Demanda media por día del mes", "viajes/estación-día",
    )

    # Cobertura: faltantes dentro de la vida observada de cada estación.
    spans = data.groupby("Origen_Id")["Fecha"].agg(["min", "max"])
    all_dates = pd.date_range(data["Fecha"].min(), data["Fecha"].max(), freq="D")
    starts = spans["min"].value_counts().reindex(all_dates, fill_value=0).cumsum()
    ends = spans["max"].value_counts().reindex(all_dates, fill_value=0).cumsum().shift(1, fill_value=0)
    expected = starts.sub(ends)
    observed = data.groupby("Fecha")["Origen_Id"].nunique().reindex(all_dates, fill_value=0)
    missing = expected.sub(observed).clip(lower=0)
    monthly_missing = missing.groupby(missing.index.to_period("M")).sum()
    log_series("eda_fase2_monthly_missing_station_days", monthly_missing,
               "Suma mensual de estación-días sin registro dentro de su periodo observado",
               "estación-días")
    log_value("eda_fase2_missing_station_days_total", missing.sum(),
              "Total de estación-días sin registro", "estación-días")
    log_value("eda_fase2_missing_station_days_fraction", missing.sum() / expected.sum()
              if expected.sum() else 0,
              "Fracción de estación-días sin registro", "proporción")

    previous = data[["Origen_Id", "Fecha"]].sort_values(
        ["Origen_Id", "Fecha"]
    ).groupby("Origen_Id")["Fecha"].shift()
    gap_days = (data["Fecha"] - previous).dt.days.sub(1)
    # El desplazamiento anterior conserva los índices originales.
    gap_days = gap_days.loc[gap_days > 0]
    log_value("eda_fase2_missing_periods_count", len(gap_days),
              "Cantidad de intervalos sin registros", "intervalos")
    log_value("eda_fase2_missing_periods_max_days", gap_days.max() if len(gap_days) else 0,
              "Duración máxima de un intervalo sin registros", "días")

    # Calidad: únicamente indicadores calculables con el dataset integrado.
    log_value("quality_duplicate_station_dates",
              data.duplicated(["Origen_Id", "Fecha"]).sum(),
              "Filas con estación y fecha repetidas", "filas")
    log_value("quality_target_nulls", target.isna().sum(),
              "Valores nulos en la demanda", "filas")
    log_value("quality_negative_trips", target.lt(0).sum(),
              "Filas con demanda negativa", "filas")

    # POI: resumen de disponibilidad y consistencia, no barras por categoría.
    present_poi = [column for column in poi_columns if column in data.columns]
    if present_poi:
        station_poi = data.drop_duplicates("Origen_Id").set_index("Origen_Id")[present_poi]
        log_value("eda_fase2_poi_columns", len(present_poi),
                  "Número de columnas de POI", "columnas")
        log_value("eda_fase2_stations_with_any_poi",
                  station_poi.fillna(0).gt(0).any(axis=1).sum(),
                  "Estaciones con al menos un POI registrado", "estaciones")
        changes = data.groupby("Origen_Id")[present_poi].nunique(dropna=False).gt(1)
        log_value("quality_stations_with_changing_poi", changes.any(axis=1).sum(),
                  "Estaciones cuyos atributos POI cambian en el tiempo", "estaciones")
    if "weather_zone" in data.columns:
        log_value("eda_fase2_weather_zones", data["weather_zone"].nunique(),
                  "Zonas meteorológicas presentes", "zonas")

    # Temperatura y lluvia son exploratorias; nunca se agregan a las 18 variables.
    for column, unit in (("temperature_2m_mean", "°C"),
                         ("precipitation_sum", "mm")):
        if column in data.columns:
            values = data[column]
            log_value(f"eda_fase1_{column}_missing_fraction", values.isna().mean(),
                      f"Fracción de datos faltantes de {column}", "proporción")
            log_value(f"eda_fase1_{column}_mean", values.mean(),
                      f"Media exploratoria de {column}", unit)

    # Las etiquetas y las unidades permiten interpretar cada serie en MLflow.
    mlflow.log_dict(guide, "eda/metric_series_guide.json")
    # Guía legible sin abrir JSON; MLflow no admite descripciones nativas por métrica.
    lines = [
        "# Diccionario de métricas exploratorias",
        "",
        "Todas estas métricas se calculan sobre el dataset integrado de esta ejecución.",
        "Una fila representa una estación en una fecha; `target` es su número de viajes.",
        "Fase 1 y Fase 2 indican el origen del análisis, no datasets independientes.",
        "Las proporciones se expresan entre 0 y 1; multiplícalas por 100 para obtener %.",
        "Las series utilizan `step` como índice; consulta las etiquetas para interpretar fechas y categorías.",
        "Las métricas describen los datos disponibles, no necesariamente la población total de viajes.",
        "Las visualizaciones originales permanecen como PNG en Artifacts.",
        "",
        "| Métrica | Qué mide | Unidad |",
        "|---|---|---|",
    ]
    for name, info in sorted(guide.items()):
        lines.append(
            f"| `{name}` | {info['description']} | {info['unit']} |"
        )
    lines.extend([
        "",
        "## Series y etiquetas",
        "En `metric_series_guide.json`, `labels` asigna a cada step su mes, año o día.",
        "No compares valores con unidades distintas en el mismo eje.",
    ])
    mlflow.log_text("\n".join(lines), "documentation/EDA_METRICS_README.md")


def log_final_model(pipeline, model_name, params, validation, test, example, context, artifacts):
    # Corrida del modelo final, con el modelo registrado en el Model Registry.
    with mlflow.start_run(run_name=f"Modelo final | {model_name}") as run:
        log_run_context(context["project_root"], context["data_path"], context["data_sha256"])
        mlflow.set_tags({"stage": "final_model", "model_family": model_name})
        mlflow.set_tag("mlflow.note.content", (
            f"Modelo final {model_name} para pronosticar salidas diarias por estación de MiBici. "
            "Validación temporal en 2025, prueba independiente en 2026, "
            "comparación con líneas base y registro en Model Registry. "
            "Los parámetros, métricas, versiones de datos y tiempos están en este run."
        ))
        log_dataset_reference(context, role="training_and_testing")
        mlflow.log_text(MLFLOW_ARTIFACT_GUIDE, "README.md")
        mlflow.log_params({f"model_{k}": v for k, v in params.items()})
        mlflow.log_param("feature_set", MAIN_FEATURE_SET)
        mlflow.log_param("n_features", len(example.columns))
        mlflow.log_param("registered_model_name", REGISTERED_MODEL_NAME)
        mlflow.log_param("registered_model_alias", MODEL_ALIAS)
        mlflow.log_dict(
            {
                "model_family": model_name,
                "model_parameters": {key: str(value) for key, value in params.items()},
                "feature_set": MAIN_FEATURE_SET,
                "features": list(example.columns),
                "target": TARGET_COLUMN,
                "forecast_horizon_days": HORIZON_DAYS,
                "validation_folds": N_FOLDS,
                "test_start": str(TEST_START.date()),
                "random_seed": SEED,
            },
            "documentation/model_configuration.json",
        )
        mlflow.log_dict({"features": list(example.columns)}, "features.json")
        # Los datasets ya se vincularon mediante log_dataset_reference.
        final_metrics = {
            "validation_mae_mean": validation["mae"].mean(),
            "validation_mae_std": validation["mae"].std(),
            "validation_rmse_mean": validation["rmse"].mean(),
            "validation_skill_vs_naive": validation["skill_vs_naive"].mean(),
            "final_fit_seconds": float(artifacts.get("fit_seconds", 0.0)),
        }
        mlflow.log_metrics({k: float(v) for k, v in final_metrics.items() if pd.notna(v) and np.isfinite(v)})
        for row in test.itertuples(index=False):
            prefix = {"Todas": "test", "Con historia": "test_known", "Nueva": "test_new"}[row.stations]
            segment_metrics = {
                f"{prefix}_mae": row.model_mae,
                f"{prefix}_rmse": row.model_rmse,
                f"{prefix}_naive_mae": row.naive_mae,
                f"{prefix}_skill_vs_naive": row.skill_vs_naive,
                f"{prefix}_rows": row.rows,
            }
            mlflow.log_metrics({k: float(v) for k, v in segment_metrics.items() if pd.notna(v) and np.isfinite(v)})
        # Estas series se pueden graficar directamente en Model metrics.
        for step, row in enumerate(validation.itertuples(index=False)):
            mlflow.log_metric("validation_fold_mae", float(row.mae), step=step)
            mlflow.log_metric("validation_fold_rmse", float(row.rmse), step=step)
        log_eda_metric_series(artifacts["eda_data"], POI_COLUMNS)
        artifact_catalog = log_report_artifacts(artifacts["files"])
        mlflow.log_dict({"files": artifact_catalog}, "artifact_catalog.json")
        mlflow.log_artifact(str(FILE_VERSION_TABLE), "data/versions")
        mlflow.log_artifact(str(FILE_VERSION_INDEX), "data/versions")
        if "output_version_table" in artifacts:
            mlflow.log_artifact(str(artifacts["output_version_table"]), "data/outputs")
            mlflow.log_artifact(str(artifacts["output_version_index"]), "data/outputs")
        signature = infer_signature(example, pipeline.predict(example))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        return run.info.run_id, model_info


def promote_model(model_name: str, version: str, alias: str, description: str, tags: dict) -> None:
    client = mlflow.MlflowClient()
    client.update_model_version(model_name, version, description=description)
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, str(value))
    client.set_registered_model_alias(model_name, alias, version)
