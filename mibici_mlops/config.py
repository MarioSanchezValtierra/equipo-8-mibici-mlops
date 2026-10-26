"""Rutas, periodo de estudio, cortes de la separación, reglas de limpieza y esquemas."""

from pathlib import Path

import pandas as pd
import pyarrow as pa


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'
MODELS_DIR = PROJECT_ROOT / 'models'


def locate_project_root(start: Path) -> Path:
    """Buscar la raíz del repositorio sin importar módulos propios."""
    for directory in (start, *start.parents):
        if (directory / 'pyproject.toml').is_file() and (directory / 'data' / 'raw').is_dir():
            return directory
    raise FileNotFoundError(
        'No se encontró pyproject.toml y data/raw en los directorios superiores.'
    )


# Periodo de estudio
STUDY_START = pd.Timestamp('2020-01-01')
STUDY_END = pd.Timestamp('2026-10-01')

# Cortes de la separación temporal; cada fecha es el inicio del conjunto siguiente
TRAIN_END = pd.Timestamp('2025-01-01')
VALIDATION_END = pd.Timestamp('2026-01-01')

SPLIT_PERIODS = {
    'train': (STUDY_START, TRAIN_END),
    'validation': (TRAIN_END, VALIDATION_END),
    'test': (VALIDATION_END, STUDY_END),
}

# Configuración del procesamiento y MLflow
MLFLOW_EXPERIMENT = 'MiBici'
EXPECTED_MONTHS = pd.period_range('2020-01', '2026-09', freq='M')
CHUNK_SIZE = 50_000
RUN_ALL_FILES = True  # True: 81 CSV; False: prueba con dos meses
SAVE_TRIPS = True      # Escritura Parquet por bloques

# Reglas de limpieza
EXCLUDED_STATIONS = frozenset({407, 124, 405})
VALID_GENDERS = frozenset({'M', 'F'})
MIN_AGE, MAX_AGE = 10, 100

# Formatos de fecha
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
ALTERNATIVE_DATETIME_FORMAT = '%d/%m/%Y %H:%M'

# Nombres y columnas del dataset
COLUMN_MAPPING = {'A}äe_nacimiento': 'Año_de_nacimiento'}

REQUIRED_COLUMNS = (
    'Viaje_Id',
    'Usuario_Id',
    'Genero',
    'Año_de_nacimiento',
    'Inicio_del_viaje',
    'Fin_del_viaje',
    'Origen_Id',
    'Destino_Id',
)

INTEGER_COLUMNS = (
    'Viaje_Id',
    'Usuario_Id',
    'Origen_Id',
    'Destino_Id',
)

EXCLUSION_REASONS = (
    'Viaje_Id inválido',
    'Viaje_Id duplicado',
    'Inicio inválido',
    'Fuera del periodo',
    'Origen inválido',
    'Estación excluida',
)

# Esquema de los viajes procesados
TRIP_PARQUET_SCHEMA = pa.schema([
    pa.field('Viaje_Id', pa.int64()),
    pa.field('Usuario_Id', pa.int64()),
    pa.field('Genero', pa.string()),
    pa.field('Año_de_nacimiento', pa.int64()),
    pa.field('Inicio_del_viaje', pa.timestamp('ns')),
    pa.field('Fin_del_viaje', pa.timestamp('ns')),
    pa.field('Origen_Id', pa.int64()),
    pa.field('Destino_Id', pa.int64()),
    pa.field('duracion_minutos', pa.float64()),
    pa.field('categoria_duracion', pa.string()),
    pa.field('tipo_recorrido', pa.string()),
])

# Esquema de la demanda horaria
COMPLETE_SCHEMA = pa.schema([
    pa.field('Origen_Id', pa.int32()),
    pa.field('Hora', pa.timestamp('ns')),
    pa.field('Cantidad_viajes', pa.int32()),
])