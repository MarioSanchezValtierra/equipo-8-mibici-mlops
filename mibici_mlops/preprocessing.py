"""Normalización, limpieza B, variables del viaje y demanda por estación y hora."""

import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa

from mibici_mlops.config import (
    ALTERNATIVE_DATETIME_FORMAT, COLUMN_MAPPING, DATETIME_FORMAT, EXCLUDED_STATIONS,
    EXCLUSION_REASONS, INTEGER_COLUMNS, MAX_AGE, MIN_AGE, REQUIRED_COLUMNS, STUDY_END,
    STUDY_START, TRIP_PARQUET_SCHEMA, VALID_GENDERS,
)


INTEGER_PATTERN = re.compile(r'[+-]?\d+(?:\.0+)?\Z')
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def parse_integer(value):
    """Parse exact integer text without rounding large identifiers."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not INTEGER_PATTERN.fullmatch(text):
        return pd.NA
    try:
        number = int(text.split('.')[0])
    except ValueError:
        return pd.NA
    return number if INT64_MIN <= number <= INT64_MAX else pd.NA


def normalize_integer_column(values: pd.Series) -> pd.Series:
    """Return a nullable Int64 series preserving the original index."""
    return pd.Series(pd.array([parse_integer(v) for v in values], dtype='Int64'), index=values.index)


def normalize_datetime_column(values: pd.Series) -> pd.Series:
    """Accept the two date formats identified in the original notebook."""
    primary = pd.to_datetime(values, format=DATETIME_FORMAT, errors='coerce')
    missing = primary.isna() & values.notna()
    if missing.any():
        primary.loc[missing] = pd.to_datetime(
            values.loc[missing], format=ALTERNATIVE_DATETIME_FORMAT, errors='coerce'
        )
    return primary


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize the eight trip columns without changing the input."""
    if not isinstance(chunk, pd.DataFrame):
        raise TypeError('chunk debe ser un DataFrame')
    result = chunk.rename(columns=COLUMN_MAPPING).copy()
    if not result.columns.is_unique:
        raise ValueError('Hay columnas repetidas después de normalizar encabezados')
    missing = set(REQUIRED_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    for column in INTEGER_COLUMNS:
        result[column] = normalize_integer_column(result[column])
    result['Año_de_nacimiento'] = pd.to_numeric(
        result['Año_de_nacimiento'], errors='coerce'
    ).astype('Float64')
    result['Genero'] = result['Genero'].astype('string')
    for column in ('Inicio_del_viaje', 'Fin_del_viaje'):
        result[column] = normalize_datetime_column(result[column])
    return result


def classify_duration(duration_minutes: pd.Series) -> pd.Categorical:
    """Classify trip duration using the original seven categories."""
    categories = [
        'Muy corto', 'Corto', 'Habitual', 'Largo',
        'Muy largo', 'Prolongado', 'Extremo',
    ]
    values = pd.to_numeric(duration_minutes, errors='coerce').astype('float64').copy()
    values = values.mask(values < 0)
    values = values.mask(values == 180, np.nextafter(180.0, -np.inf))
    values = values.mask(values == 1440, np.nextafter(1440.0, -np.inf))
    return pd.cut(
        values, bins=[-np.inf, 1, 5, 15, 30, 180, 1440, np.inf],
        labels=categories, right=False,
    )


def add_trip_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add trip duration, duration class and station relation."""
    result = data.copy()
    start = pd.to_datetime(result['Inicio_del_viaje'], errors='coerce')
    end = pd.to_datetime(result['Fin_del_viaje'], errors='coerce')
    duration = (end - start).dt.total_seconds().div(60).mask(lambda x: x < 0)
    result['duracion_minutos'] = duration
    result['categoria_duracion'] = classify_duration(duration)
    origin = result['Origen_Id'].astype('Int64')
    destination = result['Destino_Id'].astype('Int64')
    valid = origin.notna() & destination.notna()
    same = origin.eq(destination).fillna(False) & valid
    different = origin.ne(destination).fillna(False) & valid
    result['tipo_recorrido'] = pd.Series(pd.NA, index=result.index, dtype='string')
    result.loc[same, 'tipo_recorrido'] = 'Misma estación'
    result.loc[different, 'tipo_recorrido'] = 'Estaciones diferentes'
    return result


def clean_strategy_b(data: pd.DataFrame, seen_ids: set[int] | None = None):
    """Apply strategy B; return cleaned rows, counters and new unique IDs."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data debe ser un DataFrame')
    if seen_ids is not None and not isinstance(seen_ids, set):
        raise TypeError('seen_ids debe ser set o None')
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    previous = set() if seen_ids is None else seen_ids
    result = data.copy()
    trip_id = result['Viaje_Id'].astype('Int64')
    origin = result['Origen_Id'].astype('Int64')
    destination = result['Destino_Id'].astype('Int64')
    start = pd.to_datetime(result['Inicio_del_viaje'], errors='coerce')
    reason = pd.Series(pd.NA, index=result.index, dtype='string')
    invalid_id = (trip_id.isna() | trip_id.le(0)).fillna(True)
    reason.loc[invalid_id] = EXCLUSION_REASONS[0]
    # Buscar solo los IDs del bloque en el conjunto, sin convertir el conjunto completo
    ids = trip_id.to_numpy(dtype='int64', na_value=-1).tolist()
    in_previous = pd.Series([value in previous for value in ids], index=trip_id.index)
    duplicated = (trip_id.duplicated(keep='first') | in_previous).fillna(False)
    reason.loc[reason.isna() & duplicated] = EXCLUSION_REASONS[1]
    reason.loc[reason.isna() & start.isna()] = EXCLUSION_REASONS[2]
    outside = (start.lt(STUDY_START) | start.ge(STUDY_END)).fillna(False)
    reason.loc[reason.isna() & outside] = EXCLUSION_REASONS[3]
    invalid_origin = (origin.isna() | origin.le(0)).fillna(True)
    reason.loc[reason.isna() & invalid_origin] = EXCLUSION_REASONS[4]
    excluded = (origin.isin(EXCLUDED_STATIONS) | destination.isin(EXCLUDED_STATIONS)).fillna(False)
    reason.loc[reason.isna() & excluded] = EXCLUSION_REASONS[5]
    new_ids = set(trip_id.loc[~invalid_id & ~duplicated].dropna().astype('int64').tolist())
    cleaned = result.loc[reason.isna()].copy()
    gender = cleaned['Genero'].astype('string').str.strip().str.upper()
    valid_gender = gender.isin(VALID_GENDERS).fillna(False)
    cleaned['Genero'] = gender.where(valid_gender, 'No especificado')
    birth_year = pd.to_numeric(cleaned['Año_de_nacimiento'], errors='coerce')
    age = pd.to_datetime(cleaned['Inicio_del_viaje']).dt.year - birth_year
    invalid_birth = (birth_year.notna() & ((birth_year % 1 != 0) | age.lt(MIN_AGE) | age.gt(MAX_AGE))).fillna(False)
    cleaned['Año_de_nacimiento'] = birth_year.mask(invalid_birth).astype('Int64')
    summary = {
        'original_rows': len(result),
        'kept_rows': len(cleaned),
        'excluded_rows': int(reason.notna().sum()),
        **{key: int(reason.eq(key).sum()) for key in EXCLUSION_REASONS},
        'corrected_genders': int((~valid_gender).sum()),
        'corrected_birth_years': int(invalid_birth.sum()),
    }
    return cleaned, summary, new_ids


def count_hourly_demand(trips: pd.DataFrame) -> pd.Series:
    """Count trips by origin station and hour of departure."""
    if trips.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=['Origen_Id', 'Hora'])
        return pd.Series([], index=index, dtype='int64', name='Cantidad_viajes')
    valid = (
        trips['Viaje_Id'].notna() & trips['Viaje_Id'].gt(0)
        & trips['Origen_Id'].notna() & trips['Origen_Id'].gt(0)
        & trips['Inicio_del_viaje'].notna()
        & trips['Inicio_del_viaje'].ge(STUDY_START)
        & trips['Inicio_del_viaje'].lt(STUDY_END)
    ).fillna(False)
    selected = trips.loc[valid, ['Origen_Id', 'Inicio_del_viaje']].copy()
    selected['Hora'] = selected['Inicio_del_viaje'].dt.floor('h')
    return selected.groupby(['Origen_Id', 'Hora']).size().rename('Cantidad_viajes')


def as_arrow_table(trips: pd.DataFrame) -> pa.Table:
    """Convertir los viajes transformados al esquema Parquet."""
    selected = trips.loc[:, TRIP_PARQUET_SCHEMA.names].copy()
    for column in ('Genero', 'categoria_duracion', 'tipo_recorrido'):
        selected[column] = selected[column].astype('string')
    return pa.Table.from_pandas(
        selected, schema=TRIP_PARQUET_SCHEMA, preserve_index=False, safe=True
    )


def merge_hourly(target: Counter, hourly: pd.Series) -> None:
    """Acumular conteos por estación y hora sin conservar los viajes."""
    target.update({(int(station), hour.isoformat()): int(value)
                   for (station, hour), value in hourly.items()})
