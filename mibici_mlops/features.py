"""Transformaciones reutilizables para enriquecer los viajes de MiBici.

Las funciones retoman transformaciones del notebook
``01_MiBici_EDA_Functional.ipynb``. Trabajan sobre una copia de los datos para
conservar intacto el DataFrame de entrada.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ALTERNATIVE_DATETIME_FORMAT = "%d/%m/%Y %H:%M"
REQUIRED_COLUMNS = {
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
}

app = typer.Typer(help="Construcción de variables para los datos de viajes de MiBici.")


def _parse_datetime(values: pd.Series) -> pd.Series:
    """Interpreta los dos formatos de fecha identificados en el notebook EDA."""
    parsed = pd.to_datetime(values, format=DATETIME_FORMAT, errors="coerce")
    fallback_mask = parsed.isna() & values.notna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            values.loc[fallback_mask],
            format=ALTERNATIVE_DATETIME_FORMAT,
            errors="coerce",
        )
    return parsed


def classify_duration(duration_minutes: pd.Series) -> pd.Series:
    """Clasifica la duración del viaje con los límites del notebook funcional.

    Las duraciones negativas o faltantes se conservan como valores nulos. Los
    valores exactos de 180 y 1,440 minutos siguen las categorías definidas en
    el notebook para mantener consistencia con el análisis original.
    """
    if not isinstance(duration_minutes, pd.Series):
        duration_minutes = pd.Series(duration_minutes)

    categories = [
        "Muy corto",
        "Corto",
        "Habitual",
        "Largo",
        "Muy largo",
        "Prolongado",
        "Extremo",
    ]
    boundaries = [-np.inf, 1, 5, 15, 30, 180, 1440, np.inf]
    adjusted = pd.to_numeric(duration_minutes, errors="coerce").astype("float64").copy()
    # Conservamos la clasificación de los límites exactos tal como se definió en el notebook.
    adjusted = adjusted.mask(adjusted == 180, np.nextafter(180.0, -np.inf))
    adjusted = adjusted.mask(adjusted == 1440, np.nextafter(1440.0, -np.inf))
    # Una duración negativa indica que las fechas requieren revisión; no la clasificamos.
    adjusted = adjusted.mask(adjusted < 0)
    return pd.cut(adjusted, bins=boundaries, labels=categories, right=False)


def add_trip_features(data: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de duración y tipo de recorrido sin reemplazar columnas originales.

    Añade ``duracion_minutos``, ``categoria_duracion`` y ``tipo_recorrido``.
    Antes de llamar la función, los nombres de las columnas deben estar
    normalizados como se usan en el análisis exploratorio.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias para crear variables: {missing}")

    result = data.copy()
    start_time = _parse_datetime(result["Inicio_del_viaje"])
    end_time = _parse_datetime(result["Fin_del_viaje"])
    duration = (end_time - start_time).dt.total_seconds() / 60
    duration = duration.mask(duration < 0)

    result["duracion_minutos"] = duration
    result["categoria_duracion"] = classify_duration(duration)

    origin_id = pd.to_numeric(result["Origen_Id"], errors="coerce")
    destination_id = pd.to_numeric(result["Destino_Id"], errors="coerce")
    valid_stations = origin_id.notna() & destination_id.notna()
    same_station = origin_id.eq(destination_id).fillna(False) & valid_stations
    different_stations = origin_id.ne(destination_id).fillna(False) & valid_stations

    # Si falta algún identificador de estación, dejamos el tipo de recorrido sin clasificar.
    route_type = pd.Series(pd.NA, index=result.index, dtype="string")
    route_type.loc[same_station] = "Misma estación"
    route_type.loc[different_stations] = "Estaciones diferentes"
    result["tipo_recorrido"] = route_type
    return result


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="CSV de viajes con encabezados normalizados."),
    output_path: Path = typer.Argument(..., help="Ruta del CSV enriquecido de salida."),
    encoding: str = typer.Option("utf-8", help="Codificación del CSV de entrada, por ejemplo utf-8 o cp1252."),
    chunk_size: int = typer.Option(200_000, min=1, help="Filas por bloque para limitar el uso de memoria."),
) -> None:
    """Aplica las transformaciones a un CSV, leyendo los registros por bloques."""
    if not input_path.is_file():
        raise typer.BadParameter(f"No se encontró el archivo: {input_path}", param_hint="input_path")
    if input_path.resolve() == output_path.resolve():
        raise typer.BadParameter("La salida debe ser diferente del archivo original.", param_hint="output_path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    rows_written = 0
    try:
        for chunk in pd.read_csv(input_path, encoding=encoding, chunksize=chunk_size):
            enriched = add_trip_features(chunk)
            enriched.to_csv(output_path, mode="a", index=False, header=rows_written == 0)
            rows_written += len(enriched)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    logger.success("Se generaron variables para {} registros: {}", f"{rows_written:,}", output_path)


if __name__ == "__main__":
    app()
