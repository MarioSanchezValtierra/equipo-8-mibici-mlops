"""Resúmenes exploratorios de la demanda horaria de MiBici.

El módulo trabaja con el Parquet horario que utiliza el notebook EDA y requiere
las columnas ``Origen_Id``, ``Hora`` y ``Cantidad_viajes``. Lee la información
por lotes para evitar cargar toda la tabla en memoria. El análisis sigue siendo
horario; la demanda diaria por estación queda pendiente para una siguiente etapa.
"""
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mibici_mlops.config import PROCESSED_DATA_DIR

REQUIRED_COLUMNS = {"Origen_Id", "Hora", "Cantidad_viajes"}
WEEKDAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MONTH_NAMES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
DEFAULT_INPUT = PROCESSED_DATA_DIR / "mibici_hourly_complete.parquet"
app = typer.Typer(help="Análisis exploratorio por lotes para la demanda horaria de MiBici.")


def _merge_histogram(previous: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Agrega conteos enteros no negativos a un histograma de frecuencias."""
    if values.size == 0:
        return previous
    if np.any(values < 0):
        raise ValueError("Cantidad_viajes no puede contener valores negativos.")
    current = np.bincount(values.astype(np.int64, copy=False))
    if len(previous) < len(current):
        previous = np.pad(previous, (0, len(current) - len(previous)))
    previous[:len(current)] += current
    return previous


def histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    """Calcula un percentil exacto a partir de un histograma de conteos enteros."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile debe estar entre 0 y 100.")
    cumulative = np.cumsum(histogram)
    if cumulative.size == 0 or cumulative[-1] == 0:
        return float("nan")

    position = (int(cumulative[-1]) - 1) * percentile / 100

    def value_at(index: int) -> int:
        return int(np.searchsorted(cumulative, index + 1, side="left"))

    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    return value_at(lower) + (value_at(upper) - value_at(lower)) * (position - lower)


def summarize_hourly_batches(batches: Iterable[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Calcula tablas del EDA a partir de lotes, sin retener todos los registros.

    Parámetros
    ----------
    batches:
        Secuencia de DataFrames con las tres columnas del Parquet horario
        generado durante la preparación de los datos.

    Devuelve
    -------
    dict[str, pandas.DataFrame]
        Tablas de demanda por hora, día de la semana, estación y mes; comparación
        entre días laborables y fin de semana; horas pico y conteos atípicos.
    """
    hour_totals = np.zeros(24, dtype=np.int64)
    hour_counts = np.zeros(24, dtype=np.int64)
    weekday_totals = np.zeros(7, dtype=np.int64)
    weekday_counts = np.zeros(7, dtype=np.int64)
    weekday_histograms = [np.zeros(1, dtype=np.int64) for _ in range(7)]
    day_hour_totals = np.zeros((7, 24), dtype=np.int64)
    day_hour_counts = np.zeros((7, 24), dtype=np.int64)

    station_totals: dict[int, int] = defaultdict(int)
    station_counts: dict[int, int] = defaultdict(int)
    station_maxima: dict[int, int] = defaultdict(int)
    station_histograms: dict[int, np.ndarray] = {}
    station_hour_totals: dict[int, np.ndarray] = {}
    station_hour_counts: dict[int, np.ndarray] = {}
    month_totals: dict[pd.Period, int] = defaultdict(int)
    demand_histogram = np.zeros(1, dtype=np.int64)
    rows_seen = 0

    # Acumulamos los resultados por lote para no cargar todos los viajes en memoria.
    for batch_number, batch in enumerate(batches, start=1):
        missing = sorted(REQUIRED_COLUMNS.difference(batch.columns))
        if missing:
            raise ValueError(f"El lote {batch_number} no contiene columnas requeridas: {missing}")
        chunk = batch[list(REQUIRED_COLUMNS)].copy()
        chunk["Hora"] = pd.to_datetime(chunk["Hora"], errors="coerce")
        chunk["Origen_Id"] = pd.to_numeric(chunk["Origen_Id"], errors="coerce")
        chunk["Cantidad_viajes"] = pd.to_numeric(chunk["Cantidad_viajes"], errors="coerce")
        if chunk[["Hora", "Origen_Id", "Cantidad_viajes"]].isna().any().any():
            raise ValueError(f"El lote {batch_number} contiene Hora, Origen_Id o Cantidad_viajes nulos/no convertibles.")
        if (chunk["Origen_Id"] <= 0).any():
            raise ValueError(f"El lote {batch_number} contiene identificadores de estación no positivos.")
        values_float = chunk["Cantidad_viajes"].to_numpy(dtype="float64")
        if np.any(values_float < 0) or np.any(values_float % 1 != 0):
            raise ValueError(f"El lote {batch_number} tiene Cantidad_viajes negativo o no entero.")
        values = values_float.astype(np.int64)
        hours = chunk["Hora"].dt.hour.to_numpy(dtype=np.int64)
        weekdays = chunk["Hora"].dt.dayofweek.to_numpy(dtype=np.int64)

        np.add.at(hour_totals, hours, values)
        np.add.at(hour_counts, hours, 1)
        np.add.at(weekday_totals, weekdays, values)
        np.add.at(weekday_counts, weekdays, 1)
        np.add.at(day_hour_totals, (weekdays, hours), values)
        np.add.at(day_hour_counts, (weekdays, hours), 1)
        demand_histogram = _merge_histogram(demand_histogram, values)
        rows_seen += len(chunk)

        for day in range(7):
            weekday_histograms[day] = _merge_histogram(
                weekday_histograms[day], values[weekdays == day]
            )

        periods = chunk["Hora"].dt.to_period("M")
        month_chunk = pd.Series(values, index=periods).groupby(level=0).sum()
        for period, total in month_chunk.items():
            month_totals[period] += int(total)

        chunk["_hour_of_day"] = hours
        for station_id, station_rows in chunk.groupby("Origen_Id", sort=False):
            sid = int(station_id)
            station_values = station_rows["Cantidad_viajes"].to_numpy(dtype=np.int64)
            station_totals[sid] += int(station_values.sum())
            station_counts[sid] += int(len(station_values))
            station_maxima[sid] = max(station_maxima[sid], int(station_values.max()))
            station_histograms[sid] = _merge_histogram(
                station_histograms.get(sid, np.zeros(1, dtype=np.int64)), station_values
            )
            if sid not in station_hour_totals:
                station_hour_totals[sid] = np.zeros(24, dtype=np.int64)
                station_hour_counts[sid] = np.zeros(24, dtype=np.int64)
            by_hour = station_rows.groupby("_hour_of_day")["Cantidad_viajes"].agg(["sum", "count"])
            for hour, row in by_hour.iterrows():
                station_hour_totals[sid][int(hour)] += int(row["sum"])
                station_hour_counts[sid][int(hour)] += int(row["count"])

    if rows_seen == 0:
        raise ValueError("No se recibieron registros para analizar.")

    hourly_pattern = pd.DataFrame({
        "Hora_del_dia": np.arange(24),
        "Promedio_viajes": np.divide(hour_totals, hour_counts, out=np.full(24, np.nan), where=hour_counts > 0),
        "Registros": hour_counts,
    })
    weekday_pattern = pd.DataFrame({
        "Dia_semana": np.arange(7),
        "Promedio_viajes": np.divide(weekday_totals, weekday_counts, out=np.full(7, np.nan), where=weekday_counts > 0),
        "Mediana_viajes": [histogram_percentile(hist, 50) for hist in weekday_histograms],
        "Registros": weekday_counts,
        "Nombre_dia": WEEKDAY_NAMES,
    })

    station_rows_out = []
    station_outliers_out = []
    peak_rows_out = []
    for sid in sorted(station_totals):
        hist = station_histograms[sid]
        q1 = histogram_percentile(hist, 25)
        median = histogram_percentile(hist, 50)
        q3 = histogram_percentile(hist, 75)
        upper_limit = q3 + 1.5 * (q3 - q1)
        first_outlier = max(0, int(np.floor(upper_limit)) + 1)
        outlier_count = int(hist[first_outlier:].sum()) if first_outlier < len(hist) else 0
        station_rows_out.append({
            "Origen_Id": sid,
            "Promedio_viajes": station_totals[sid] / station_counts[sid],
            "Mediana_viajes": median,
            "Total_viajes": station_totals[sid],
            "Horas_observadas": station_counts[sid],
            "Maximo_viajes": station_maxima[sid],
        })
        station_outliers_out.append({
            "Origen_Id": sid,
            "Q1": q1,
            "Mediana": median,
            "Q3": q3,
            "Limite_IQR": upper_limit,
            "Horas_atipicas": outlier_count,
        })
        counts = station_hour_counts[sid]
        sums = station_hour_totals[sid]
        averages = np.divide(sums, counts, out=np.full(24, np.nan), where=counts > 0)
        if np.isfinite(averages).any():
            peak_hour = int(np.nanargmax(averages))
            peak_rows_out.append({
                "Origen_Id": sid,
                "Hora_pico": peak_hour,
                "Promedio_pico": float(averages[peak_hour]),
                "Promedio_global": station_totals[sid] / station_counts[sid],
                "Total_viajes": station_totals[sid],
                "Horas_observadas": station_counts[sid],
            })

    # Se preparan tablas independientes para comparar demanda por estación, meses y horarios.
    station_demand = pd.DataFrame(station_rows_out).sort_values(
        "Promedio_viajes", ascending=False, ignore_index=True
    )
    station_outliers = pd.DataFrame(station_outliers_out).sort_values(
        "Horas_atipicas", ascending=False, ignore_index=True
    )
    station_peak_hours = pd.DataFrame(peak_rows_out).sort_values(
        "Promedio_pico", ascending=False, ignore_index=True
    )

    monthly_demand = pd.DataFrame([
        {"Periodo": str(period), "Cantidad_viajes": total}
        for period, total in sorted(month_totals.items())
    ])
    # Esta variación describe cambios entre meses; no debe interpretarse como predicción.
    monthly_demand["Variacion_pct"] = monthly_demand["Cantidad_viajes"].pct_change() * 100
    monthly_periods = pd.PeriodIndex(monthly_demand["Periodo"], freq="M")
    monthly_demand["Mes_numero"] = monthly_periods.month
    monthly_demand["Dias_mes"] = monthly_periods.days_in_month
    monthly_demand["Promedio_diario"] = monthly_demand["Cantidad_viajes"] / monthly_demand["Dias_mes"]
    seasonality = (
        monthly_demand.groupby("Mes_numero", as_index=False)["Promedio_diario"].mean()
        .rename(columns={"Promedio_diario": "Promedio_diario_medio"})
    )
    seasonality["Nombre_mes"] = seasonality["Mes_numero"].map(lambda n: MONTH_NAMES[n - 1])

    workday_sum = day_hour_totals[:5].sum(axis=0)
    workday_count = day_hour_counts[:5].sum(axis=0)
    weekend_sum = day_hour_totals[5:].sum(axis=0)
    weekend_count = day_hour_counts[5:].sum(axis=0)
    day_type_hour = pd.DataFrame({
        "Hora_del_dia": np.arange(24),
        "Lunes_a_viernes": np.divide(workday_sum, workday_count, out=np.full(24, np.nan), where=workday_count > 0),
        "Fin_de_semana": np.divide(weekend_sum, weekend_count, out=np.full(24, np.nan), where=weekend_count > 0),
        "Registros_laborables": workday_count,
        "Registros_fin_semana": weekend_count,
    })

    q1 = histogram_percentile(demand_histogram, 25)
    median = histogram_percentile(demand_histogram, 50)
    q3 = histogram_percentile(demand_histogram, 75)
    upper_limit = q3 + 1.5 * (q3 - q1)
    first_outlier = max(0, int(np.floor(upper_limit)) + 1)
    outlier_count = int(demand_histogram[first_outlier:].sum()) if first_outlier < len(demand_histogram) else 0
    distribution = pd.DataFrame([{
        "Registros_estacion_hora": rows_seen,
        "Viajes_contabilizados": int(sum(station_totals.values())),
        "Horas_sin_viajes": int(demand_histogram[0]) if len(demand_histogram) else 0,
        "Promedio_viajes": float(sum(station_totals.values()) / rows_seen),
        "Q1": q1,
        "Mediana": median,
        "Q3": q3,
        "Limite_IQR": upper_limit,
        "Registros_atipicos": outlier_count,
    }])

    return {
        "demanda_por_hora": hourly_pattern,
        "demanda_por_dia_semana": weekday_pattern,
        "demanda_por_estacion": station_demand,
        "demanda_mensual": monthly_demand,
        "estacionalidad_por_mes": seasonality,
        "laborables_vs_fin_semana": day_type_hour,
        "hora_pico_por_estacion": station_peak_hours,
        "distribucion_demanda": distribution,
        "valores_atipicos_por_estacion": station_outliers,
    }


def iter_parquet_batches(input_path: Path, batch_size: int = 200_000):
    """Lee el Parquet horario por lotes con PyArrow y entrega cada lote como DataFrame."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Para leer Parquet instala las dependencias del proyecto, incluido pyarrow.") from exc

    parquet_file = pq.ParquetFile(input_path)
    missing = sorted(REQUIRED_COLUMNS.difference(parquet_file.schema.names))
    if missing:
        raise ValueError(f"El Parquet no contiene columnas requeridas: {missing}")
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=sorted(REQUIRED_COLUMNS)):
        yield batch.to_pandas()


def analyze_hourly_parquet(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = PROCESSED_DATA_DIR,
    batch_size: int = 200_000,
) -> dict[str, pd.DataFrame]:
    """Analiza el Parquet horario y guarda las tablas resumen en archivos CSV."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el Parquet de demanda horaria: {input_path}")
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_hourly_batches(iter_parquet_batches(input_path, batch_size=batch_size))
    for table_name, table in summaries.items():
        table.to_csv(output_dir / f"eda_{table_name}.csv", index=False)
    logger.success("EDA completado: {} tablas guardadas en {}", len(summaries), output_dir)
    return summaries


@app.command()
def main(
    input_path: Path = typer.Option(DEFAULT_INPUT, help="Parquet horario completo generado en el procesamiento."),
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Carpeta para guardar las tablas CSV del EDA."),
    batch_size: int = typer.Option(200_000, min=1, help="Filas por lote de lectura Parquet."),
) -> None:
    """Genera resúmenes del Parquet horario sin cargar todo el archivo en memoria."""
    analyze_hourly_parquet(input_path=input_path, output_dir=output_dir, batch_size=batch_size)


if __name__ == "__main__":
    app()
