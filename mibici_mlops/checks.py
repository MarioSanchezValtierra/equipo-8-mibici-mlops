"""Validaciones de la Fase 1 (conteos) y de la Fase 2 (contrato T1-T9, POI e integración)."""

import pandas as pd

from mibici_mlops.config import (
    EXCLUSION_REASONS,
)


def check_counts(summary, demand_total: int, label: str) -> None:
    """Comprobar que registros, motivos de exclusión y demanda concilien."""
    if summary['original_rows'] != summary['kept_rows'] + summary['excluded_rows']:
        raise AssertionError(f'{label}: los registros no concilian')
    if sum(summary[key] for key in EXCLUSION_REASONS) != summary['excluded_rows']:
        raise AssertionError(f'{label}: los motivos de exclusión no concilian')
    if demand_total != summary['kept_rows']:
        raise AssertionError(f'{label}: la demanda no coincide con los viajes conservados')


def validate_contract(data: pd.DataFrame, dtypes: dict, poi_columns: list[str]) -> pd.DataFrame:
    # Pruebas T1-T9 del dataset integrado.
    has_schema = set(data.columns) == set(dtypes)
    if not has_schema:
        missing = sorted(set(dtypes) - set(data.columns))
        extra = sorted(set(data.columns) - set(dtypes))
        raise ValueError(f"Esquema inesperado. Faltan: {missing}; sobran: {extra}")
    text_columns = data.select_dtypes(include="object").columns
    empty_texts = sum(data[c].astype(str).str.strip().eq("").sum() for c in text_columns)
    checks = {
        "T1 - Esquema": has_schema,
        "T2 - Tipos": has_schema and all(
            pd.api.types.is_datetime64_any_dtype(data[c]) if t.startswith("datetime64")
            else str(data[c].dtype) == t
            for c, t in dtypes.items()
        ),
        "T3 - Textos vacíos": empty_texts == 0,
        "T4 - Valores inválidos": has_schema and (
            (data["Origen_Id"] > 0).all()
            and (data["weather_zone"] >= 0).all()
            and (data[poi_columns] >= 0).all().all()
        ),
        "T5 - Nulos": not data.isna().any().any(),
        "T6 - Rangos": has_schema and (
            data["day_of_week"].between(0, 6).all()
            and data["month"].between(1, 12).all()
            and (data["precipitation_sum"] >= 0).all()
        ),
        "T7 - Fechas consistentes": has_schema and (
            data["Fecha"].dt.day_of_week.eq(data["day_of_week"]).all()
            and data["Fecha"].dt.month.eq(data["month"]).all()
        ),
        "T8 - Duplicados": not data.duplicated(["Origen_Id", "Fecha"]).any(),
        "T9 - Objetivo": (data["target"] >= 0).all()
        and pd.api.types.is_integer_dtype(data["target"]),
    }
    return pd.DataFrame({"test": list(checks), "passed": [bool(v) for v in checks.values()]})


def check_poi_counts(poi_data: pd.DataFrame, station_ids: set) -> None:
    # Validar los conteos de POI antes de integrarlos.
    count_columns = poi_data.columns.drop("Origen_Id")
    if set(poi_data["Origen_Id"]) != set(station_ids):
        raise ValueError("Los identificadores de los POI no coinciden con las estaciones.")
    if poi_data["Origen_Id"].duplicated().any():
        raise ValueError("Hay estaciones duplicadas en los POI.")
    if poi_data.isna().any().any():
        raise ValueError("Hay conteos de POI faltantes.")
    if (poi_data[count_columns] < 0).any().any():
        raise ValueError("Hay conteos de POI negativos.")
    if not all(pd.api.types.is_integer_dtype(poi_data[c]) for c in count_columns):
        raise ValueError("Hay conteos de POI que no son enteros.")


def check_integration(daily_data: pd.DataFrame, integrated: pd.DataFrame) -> None:
    # La integración no debe cambiar filas, claves ni viajes.
    keys = ["Origen_Id", "Fecha", "target"]
    if len(integrated) != len(daily_data):
        raise ValueError("La integración cambió la cantidad de registros.")
    if integrated["target"].sum() != daily_data["target"].sum():
        raise ValueError("La integración cambió el total de viajes.")
    if not integrated[keys].equals(daily_data[keys]):
        raise ValueError("La integración cambió estaciones, fechas o viajes.")
