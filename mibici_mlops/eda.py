"""Resúmenes del EDA de train, comunes a las dos versiones."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mibici_mlops.config import TRAIN_END
from mibici_mlops.ingestion import file_month


WEEKDAYS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
MONTHS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']


def demand_summary(complete_path: Path) -> dict:
    """Collect EDA statistics in one streaming pass over complete station-hours."""
    totals = {name: Counter() for name in ('hour', 'weekday', 'station', 'weekday_hour', 'station_hour')}
    counts = {name: Counter() for name in totals}
    hist = Counter()
    weekday_hist = [Counter() for _ in range(7)]
    station_hist = {}
    maximum = Counter()
    # Guardar únicamente los 20 registros de mayor demanda
    top = pd.DataFrame({
        'Origen_Id': pd.Series(dtype='int32'),
        'Hora': pd.Series(dtype='datetime64[ns]'),
        'Cantidad_viajes': pd.Series(dtype='int32'),
    })
    keys = {
        'hour': ['hour'], 'weekday': ['weekday'], 'station': ['Origen_Id'],
        'weekday_hour': ['weekday', 'hour'], 'station_hour': ['Origen_Id', 'hour'],
    }
    for batch in pq.ParquetFile(complete_path).iter_batches(batch_size=100_000):
        data = batch.to_pandas()
        data['hour'] = data['Hora'].dt.hour
        data['weekday'] = data['Hora'].dt.dayofweek
        for name, columns in keys.items():
            grouped = data.groupby(columns)['Cantidad_viajes'].agg(['sum', 'count'])
            totals[name].update(grouped['sum'].to_dict())
            counts[name].update(grouped['count'].to_dict())
        hist.update(data['Cantidad_viajes'].value_counts().to_dict())
        for day, group in data.groupby('weekday'):
            weekday_hist[day].update(group['Cantidad_viajes'].value_counts().to_dict())
        for station_id, group in data.groupby('Origen_Id'):
            maximum[int(station_id)] = max(maximum[int(station_id)], int(group['Cantidad_viajes'].max()))
            station_hist.setdefault(int(station_id), Counter()).update(
                group['Cantidad_viajes'].value_counts().to_dict()
            )
        top = pd.concat([top, data.nlargest(20, 'Cantidad_viajes')[
            ['Origen_Id', 'Hora', 'Cantidad_viajes']
        ]], ignore_index=True).nlargest(20, 'Cantidad_viajes')
    result = {}
    for name in keys:
        frame = pd.DataFrame({
            'total': pd.Series(totals[name]), 'count': pd.Series(counts[name])
        })
        frame['mean'] = frame['total'] / frame['count']
        result[name] = frame.sort_index()
    result.update(hist=hist, weekday_hist=weekday_hist,
                  maximum=pd.Series(maximum), station_hist=station_hist,
                  top=top.reset_index(drop=True))
    return result


def hist_percentile(histogram: Counter, percentile: float) -> float:
    """Calculate a percentile with linear interpolation, without expanding records."""
    values = np.array(sorted(histogram))
    cumulative = np.cumsum([histogram[value] for value in values])
    position = (cumulative[-1] - 1) * percentile / 100
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    low_value = values[np.searchsorted(cumulative, lower + 1)]
    high_value = values[np.searchsorted(cumulative, upper + 1)]
    return float(low_value + (high_value - low_value) * (position - lower))


def box_stats(histogram: Counter, label: str) -> dict:
    """Prepare an IQR boxplot from frequencies."""
    q1, median, q3 = [hist_percentile(histogram, p) for p in (25, 50, 75)]
    limit = q3 + 1.5 * (q3 - q1)
    return dict(label=label, q1=q1, med=median, q3=q3,
                whislo=min(histogram), whishi=max(v for v in histogram if v <= limit),
                fliers=[])


def prepare_eda(results, splits: pd.DataFrame):
    """Calcular los resúmenes del EDA con el conjunto train."""
    train_path = Path(splits.loc['train', 'Archivo'])
    summary = demand_summary(train_path)
    monthly_trips = pd.Series({
        file_month(Path(file_name)): kept_rows
        for file_name, kept_rows in zip(results['file_summary']['archivo'],
                                        results['file_summary']['kept_rows'])
    })
    # El análisis mensual se limita a los meses de train
    monthly_trips = monthly_trips[monthly_trips.index < TRAIN_END.to_period('M')]
    assert sum(summary['hist'].values()) == splits.loc['train', 'Filas']
    assert sum(value * count for value, count in summary['hist'].items()) == splits.loc['train', 'Viajes']
    return summary, monthly_trips
