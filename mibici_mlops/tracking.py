"""Registro de la ejecución y del EDA en MLflow."""

import hashlib
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import mlflow
import pandas as pd

from mibici_mlops.config import (
    EXCLUDED_STATIONS,
    EXCLUSION_REASONS,
    MLFLOW_EXPERIMENT,
    STUDY_END,
    STUDY_START,
    TRAIN_END,
    VALIDATION_END,
)


def collect_environment(project_root: Path) -> dict:
    """Describir el entorno real sin modificar los datos ni el pipeline."""
    versions = {}
    for package in ('mlflow', 'pandas', 'numpy', 'pyarrow', 'matplotlib', 'IPython'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    return {
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'packages': versions,
        'git_commit': commit,
    }


def log_environment(project_root: Path) -> None:
    """Guardar versiones y commit en el run de MLflow como tags y JSON."""
    environment = collect_environment(project_root)
    mlflow.set_tag('python_version', environment['python_version'])
    mlflow.set_tag(
        'implementation_commit',
        environment['git_commit'] or 'unavailable',
    )
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