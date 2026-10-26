"""Dataset de modelado: horas completas por estación y separación train/validation/test."""

import hashlib
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici_mlops.config import COMPLETE_SCHEMA, SPLIT_PERIODS


def fill_station_hours(station_id: int, station: pd.DataFrame) -> pa.Table:
    """Completar con ceros las horas de una estación entre su primer y último viaje."""
    hours = pd.date_range(station['Hora'].min(), station['Hora'].max(), freq='h')
    values = station.set_index('Hora')['Cantidad_viajes'].reindex(hours, fill_value=0)
    return pa.Table.from_pydict({
        'Origen_Id': np.full(len(hours), station_id, dtype=np.int32),
        'Hora': hours.to_numpy(),
        'Cantidad_viajes': values.to_numpy(dtype=np.int32),
    }, schema=COMPLETE_SCHEMA)


def complete_station_hours(observed_path: Path, complete_path: Path) -> None:
    """Escribir la tabla completa procesando una estación a la vez."""
    pending = pd.DataFrame()
    with pq.ParquetWriter(complete_path, COMPLETE_SCHEMA, compression='zstd') as writer:
        for batch in pq.ParquetFile(observed_path).iter_batches(batch_size=100_000):
            block = pd.concat([pending, batch.to_pandas()], ignore_index=True)
            # La última estación del lote puede continuar en el siguiente
            if block.empty:
                continue
            last_station = block['Origen_Id'].iloc[-1]
            ready = block.loc[block['Origen_Id'] != last_station]
            pending = block.loc[block['Origen_Id'] == last_station]
            for station_id, station in ready.groupby('Origen_Id', sort=True):
                writer.write_table(fill_station_hours(station_id, station))
        if not pending.empty:
            writer.write_table(fill_station_hours(int(pending['Origen_Id'].iloc[0]), pending))


def check_sorted_keys(data: pd.DataFrame, previous: tuple | None) -> None:
    """Comprobar que las claves (Origen_Id, Hora) estén ordenadas y no se repitan."""
    stations = data['Origen_Id'].to_numpy(dtype=np.int64)
    hours = data['Hora'].to_numpy()
    if previous is not None:
        # Incluir la última clave del bloque anterior para revisar la unión entre bloques
        stations = np.r_[previous[0], stations]
        hours = np.r_[np.datetime64(previous[1], 'ns'), hours]
    station_step = np.diff(stations)
    hour_step = np.diff(hours)
    ordered = (station_step > 0) | ((station_step == 0) & (hour_step > np.timedelta64(0, 'ns')))
    if not ordered.all():
        raise ValueError('Hay claves (Origen_Id, Hora) repetidas o fuera de orden')


def file_sha256(path: Path) -> str:
    """Calcular el SHA-256 de un archivo para identificar la versión del dataset."""
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def split_dataset(complete_path: Path, output_dir: Path, total_trips: int,
                  run_all_files: bool) -> pd.DataFrame:
    """Separar la demanda estación-hora por fecha y escribir un Parquet por conjunto."""
    output_dir.mkdir()
    rows, trips = Counter(), Counter()
    first_hour, last_hour = {}, {}
    previous = None
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(pq.ParquetWriter(
                output_dir / f'{name}.parquet', COMPLETE_SCHEMA, compression='zstd'))
            for name in SPLIT_PERIODS
        }
        for batch in pq.ParquetFile(complete_path).iter_batches(batch_size=100_000):
            data = batch.to_pandas()
            check_sorted_keys(data, previous)
            previous = (data['Origen_Id'].iloc[-1], data['Hora'].iloc[-1])
            for name, (start, end) in SPLIT_PERIODS.items():
                selected = data.loc[data['Hora'].ge(start) & data['Hora'].lt(end)]
                if selected.empty:
                    continue
                writers[name].write_table(
                    pa.Table.from_pandas(selected, schema=COMPLETE_SCHEMA, preserve_index=False))
                rows[name] += len(selected)
                trips[name] += int(selected['Cantidad_viajes'].sum())
                first_hour[name] = min(first_hour.get(name, selected['Hora'].min()), selected['Hora'].min())
                last_hour[name] = max(last_hour.get(name, selected['Hora'].max()), selected['Hora'].max())

    # Validar la separación antes de usar los conjuntos
    if sum(rows.values()) != pq.ParquetFile(complete_path).metadata.num_rows:
        raise ValueError('Hay filas fuera de los periodos de train, validation y test')
    if sum(trips.values()) != total_trips:
        raise ValueError('Los viajes de los conjuntos no coinciden con los viajes conservados')
    empty = [name for name in SPLIT_PERIODS if rows[name] == 0]
    if run_all_files and empty:
        raise ValueError(f'Conjuntos sin registros: {empty}')

    splits = pd.DataFrame([{
        'Conjunto': name, 'Inicio': start.date(), 'Fin_exclusivo': end.date(),
        'Filas': rows[name], 'Viajes': trips[name],
        'Primera_hora': first_hour.get(name), 'Ultima_hora': last_hour.get(name),
        'Archivo': str(output_dir / f'{name}.parquet'),
        'SHA256': file_sha256(output_dir / f'{name}.parquet'),
    } for name, (start, end) in SPLIT_PERIODS.items()]).set_index('Conjunto')
    splits.to_csv(output_dir / 'split_summary.csv')
    print(splits[['Inicio', 'Fin_exclusivo', 'Filas', 'Viajes', 'Primera_hora', 'Ultima_hora']].to_string())
    return splits
