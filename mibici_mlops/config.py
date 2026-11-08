"""Rutas, periodo de estudio, cortes de la separación, reglas de limpieza y esquemas."""

from pathlib import Path

import pandas as pd
import pyarrow as pa


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

# Configuración de Fase 2, extraída del notebook 06 (versión 3).
# Fase 2: rutas
VALIDATION_REPORTS_DIR = PROJECT_ROOT / "reports" / "validation"
MODEL_REPORTS_DIR = PROJECT_ROOT / "reports" / "modeling"
MODEL_FIGURES_DIR = FIGURES_DIR / "modeling"
STATIONS_PATH = DATA_DIR / "raw" / "Manual_datos" / "nomenclatura_2026_09.csv"
WEATHER_DIR = DATA_DIR / "raw" / "weather"
WEATHER_PATH = WEATHER_DIR / "weather_daily_zones_2020_2026.csv"
STATION_ZONES_PATH = WEATHER_DIR / "mibici_station_weather_zones.csv"
POI_DIR = DATA_DIR / "raw" / "pois"
OSM_PATH = POI_DIR / "mexico-latest.osm.pbf"
POI_PATH = POI_DIR / "mibici_station_pois.csv"
DAILY_PATH = PROCESSED_DATA_DIR / "mibici_daily.parquet"
INTEGRATED_PATH = PROCESSED_DATA_DIR / "mibici_integrated.parquet"
FINAL_MODEL_PATH = MODELS_DIR / "mibici_final_model.joblib"

# Fase 2: fuentes externas
DATA_START = STUDY_START
DATA_END = STUDY_END - pd.Timedelta(days=1)  # STUDY_END es exclusivo: último día 2026-09-30
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLUMNS = ["temperature_2m_mean", "precipitation_sum"]
WEATHER_TIMEZONE = "America/Mexico_City"
WEATHER_GRID_DEGREES = 0.05
OSM_URL = "https://download.geofabrik.de/north-america/mexico-latest.osm.pbf"
POI_RADIUS_METERS = 500
POI_BBOX_MARGIN = 0.01
METRIC_CRS = "EPSG:32613"  # UTM 13N, distancias en metros
POI_CATEGORIES = {
    "education": {"amenity": ["school", "university", "college"]},
    "transport": {"railway": ["station", "tram_stop"], "public_transport": ["station"]},
    "commerce": {"shop": ["supermarket", "mall"], "amenity": ["marketplace"]},
    "work": {"office": None},
    "health": {"amenity": ["hospital", "clinic"]},
    "recreation": {"leisure": ["park", "sports_centre", "pitch"]},
}
POI_COLUMNS = [f"{category}_count" for category in POI_CATEGORIES]

# Fase 2: contrato del dataset integrado
CONTRACT_DTYPES = {
    "Origen_Id": "int64",
    "Fecha": "datetime64[us]",
    "target": "int64",
    "day_of_week": "int32",
    "month": "int32",
    "weather_zone": "int64",
    "temperature_2m_mean": "float64",
    "precipitation_sum": "float64",
    **{column: "int64" for column in POI_COLUMNS},
}

# Fase 2: problema y validación temporal
TARGET_COLUMN = "target"
HORIZON_DAYS = 1  # predecimos el día siguiente
LAGS = [1, 7, 14]  # deben ser >= HORIZON_DAYS
ROLLING_WINDOWS = [7, 28]
FIRST_CUTOFF = TRAIN_END  # primer mes evaluado: enero de 2025
TEST_START = VALIDATION_END  # 2026 se reserva para la prueba final
N_FOLDS = 12
TUNING_FOLD_STEP = 2  # la búsqueda usa una de cada dos ventanas
SEED = 42

# Fase 2: variables
STATION_ID_COLUMNS = ["Origen_Id"]
ONEHOT_COLUMNS = ["day_of_week", "month", "weather_zone"]
CALENDAR_COLUMNS = ["day_of_week", "month", "day_of_month"]
STATION_COLUMNS = ["weather_zone"] + POI_COLUMNS
HISTORY_COLUMNS = [f"lag_{lag}" for lag in LAGS] + [
    f"{prefix}_{window}" for window in ROLLING_WINDOWS for prefix in ("roll_mean", "active_days")
]
FEATURE_SETS = {
    "original_11_variables": ["day_of_week", "month"] + STATION_COLUMNS + WEATHER_COLUMNS,
    "calendario_estacion": STATION_ID_COLUMNS + CALENDAR_COLUMNS + STATION_COLUMNS,
    "calendario_estacion_historia": (
        STATION_ID_COLUMNS + CALENDAR_COLUMNS + STATION_COLUMNS + HISTORY_COLUMNS
    ),
    "historia_y_clima_observado": (
        STATION_ID_COLUMNS + CALENDAR_COLUMNS + STATION_COLUMNS + HISTORY_COLUMNS + WEATHER_COLUMNS
    ),
}
MAIN_FEATURE_SET = "calendario_estacion_historia"
FEATURE_COLUMNS = FEATURE_SETS[MAIN_FEATURE_SET]

# Fase 2: modelos
MODEL_NAMES = ["Random Forest", "HistGradientBoosting"]
RF_INITIAL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_samples": 0.5,  # cada árbol usa 50 % de las filas
    "random_state": SEED,
    "n_jobs": -1,
}
HGB_INITIAL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "learning_rate": 0.1,
    "random_state": SEED,
}
# Se usan cuando RUN_GRID_SEARCH = False. Random Forest: mejor configuración de la búsqueda
# de la versión anterior. HistGradientBoosting: punto de partida, todavía sin búsqueda propia.
FINAL_MODEL = "Random Forest"
BEST_PARAMS = {
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    "HistGradientBoosting": {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "loss": "poisson",
    },
}
PARAMETER_GRIDS = {
    "Random Forest": [
        {
            "model__n_estimators": [150],
            "model__max_depth": [12, 15, 18],
            "model__min_samples_leaf": [5],
            "model__max_features": [0.7, 1.0],
        },
        {
            "model__n_estimators": [150],
            "model__max_depth": [15],
            "model__min_samples_leaf": [3, 8],
            "model__max_features": [0.7],
        },
    ],
    "HistGradientBoosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_iter": [200, 400],
        "model__max_leaf_nodes": [31, 63],
        "model__loss": ["squared_error", "poisson"],
    },
}

# Fase 2: interruptores de las partes que tardan
FORCE_REBUILD_DATA = False  # True: reconstruir el Parquet integrado aunque sea válido
ALLOW_UNVERIFIED_CACHE = False  # False: no reutilizar caché sin manifiesto verificable
COMPARE_FEATURE_SETS = False  # comparar los cuatro conjuntos de variables (~15 min)
RUN_WEATHER_REFERENCE = False  # modelo con clima observado (~4 min)
RUN_GRID_SEARCH = False  # búsqueda de hiperparámetros (~1 h)

# Fase 2: versionado y procedencia de los datos
CSV_VERSION_DIR = PROJECT_ROOT / "reports" / "data_versioning"
CSV_SNAPSHOT_MAX_BYTES = 10 * 1024 * 1024  # los CSV pequeños se copian por versión
FILE_VERSION_DIR = CSV_VERSION_DIR / "all_files"
FILE_VERSION_INDEX = FILE_VERSION_DIR / "file_versions.json"
FILE_VERSION_TABLE = FILE_VERSION_DIR / "file_versions.csv"
OUTPUT_VERSION_DIR = CSV_VERSION_DIR / "model_outputs"
PROVENANCE_DIR = PROJECT_ROOT / "reports" / "mlflow_provenance"

# Fase 2: MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
MODELING_EXPERIMENT = "MiBici_Fase2_Modelado"
REGISTERED_MODEL_NAME = "mibici_salidas_diarias"
MODEL_ALIAS = "champion"

# Fase 2: resultados del notebook 05 (MAE) para comprobar la equivalencia
REFERENCE_RESULTS = {
    "Ingenuo estacional (lag_7)": 7.224,
    "Promedio estación-día": 7.703,
    "Promedio global por día (versión anterior)": 22.986,
    "HistGradientBoosting | calendario_estacion_historia": 5.440,
    "Random Forest | calendario_estacion_historia": 5.456,
    "Modelo final, 12 ventanas de 2025": 5.438,
    "Modelo final, prueba 2026 (todas)": 5.896,
    "Ingenuo estacional, prueba 2026 (todas)": 7.957,
    "Modelo final, prueba 2026 (nuevas)": 3.511,
}
