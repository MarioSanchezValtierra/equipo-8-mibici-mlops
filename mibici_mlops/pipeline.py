"""Procesamiento de los CSV con funciones."""

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici_mlops.checks import check_counts
from mibici_mlops.config import EXPECTED_MONTHS, TRIP_PARQUET_SCHEMA
from mibici_mlops.ingestion import read_csv_chunks
from mibici_mlops.preprocessing import (
    add_trip_features, as_arrow_table, clean_strategy_b, count_hourly_demand, merge_hourly,
    normalize_chunk,
)
from mibici_mlops.reporting import timed

try:
    import resource  # Solo existe en Linux y WSL
except ImportError:
    resource = None


def process_file(file_path: Path, chunk_size: int, seen_ids: set[int],
                 output_dir: Path, save_trips: bool = True):
    """Procesar un CSV por bloques, medir cada etapa y escribir un Parquet."""
    times = Counter()
    totals = Counter()
    hourly_totals = Counter()
    output_path = output_dir / f'{file_path.stem}.parquet'
    writer = None
    chunks = 0
    file_start = time.perf_counter()
    with timed(times, 'Apertura CSV'):
        reader, encoding = read_csv_chunks(file_path, chunk_size)
    chunk_iterator = iter(reader)
    try:
        while True:
            with timed(times, 'Lectura CSV'):
                raw = next(chunk_iterator, None)
            if raw is None:
                break
            chunks += 1
            with timed(times, 'Normalización'):
                normalized = normalize_chunk(raw)
            with timed(times, 'Limpieza B'):
                cleaned, summary, new_ids = clean_strategy_b(normalized, seen_ids)
            with timed(times, 'Variables'):
                trips = add_trip_features(cleaned)
            with timed(times, 'Demanda horaria'):
                hourly = count_hourly_demand(trips)
            check_counts(summary, int(hourly.sum()), f'{file_path.name}, bloque {chunks}')
            with timed(times, 'Escritura Parquet'):
                if save_trips:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path, TRIP_PARQUET_SCHEMA, compression='snappy'
                        )
                    writer.write_table(as_arrow_table(trips))
            with timed(times, 'Actualización IDs'):
                seen_ids.update(new_ids)
            with timed(times, 'Acumulación horaria'):
                merge_hourly(hourly_totals, hourly)
            totals.update(summary)
            del raw, normalized, cleaned, trips, hourly
    finally:
        if writer is not None:
            writer.close()
    times['Tiempo total'] = time.perf_counter() - file_start
    if writer is not None and pq.ParquetFile(output_path).metadata.num_rows != totals['kept_rows']:
        raise AssertionError('El Parquet no coincide con los viajes conservados')
    return {
        'file': file_path.name, 'encoding': encoding, 'chunks': chunks,
        'summary': dict(totals), 'timings': dict(times),
        'hourly': hourly_totals, 'parquet': str(output_path) if writer is not None else None,
    }


def process_dataset(selected_files, report_dir, chunk_size, save_trips, run_all_files, run_label):
    """Procesar los CSV mensuales, validar los conteos y guardar los reportes."""
    if not selected_files:
        raise ValueError('No hay archivos seleccionados')
    report_dir.mkdir(parents=True, exist_ok=False)
    trips_dir = report_dir / 'trips'
    trips_dir.mkdir()
    seen_ids = set()
    all_hourly = Counter()
    all_summaries = Counter()
    file_rows = []
    timing_rows = []
    run_started = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()

    for file_number, file_path in enumerate(selected_files, start=1):
        file_start = time.perf_counter()
        file_result = process_file(file_path, chunk_size, seen_ids, trips_dir, save_trips)
        all_summaries.update(file_result['summary'])
        all_hourly.update(file_result['hourly'])
        file_rows.append({'archivo': file_result['file'], 'bloques': file_result['chunks'],
                          'codificacion': file_result['encoding'], **file_result['summary']})
        timing_rows.append({'archivo': file_result['file'], **file_result['timings']})
        print(f'{file_number:02d}/{len(selected_files):02d} {file_path.name}: '
              f"{file_result['summary']['original_rows']:,} leídos, "
              f"{file_result['summary']['kept_rows']:,} conservados, "
              f'{time.perf_counter() - file_start:.1f} s')
    elapsed = time.perf_counter() - run_start
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None

    file_summary = pd.DataFrame(file_rows)
    timing_summary = pd.DataFrame(timing_rows)
    hourly_output = pd.DataFrame(
        [(station, hour, count) for (station, hour), count in all_hourly.items()],
        columns=['Origen_Id', 'Hora', 'Cantidad_viajes'],
    ).sort_values(['Origen_Id', 'Hora']).reset_index(drop=True)
    if not hourly_output.empty:
        hourly_output['Hora'] = pd.to_datetime(hourly_output['Hora'])

    # Validar las tablas que se van a guardar antes de escribir los reportes
    check_counts(file_summary.sum(numeric_only=True),
                 int(hourly_output['Cantidad_viajes'].sum()), 'Ejecución completa')

    file_summary.to_csv(report_dir / 'file_summary.csv', index=False)
    timing_summary.to_csv(report_dir / 'timing_summary.csv', index=False)
    hourly_output.to_parquet(report_dir / 'hourly_demand.parquet', index=False)
    manifest = {
        'version': run_label, 'started_utc': run_started,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'chunk_size': chunk_size, 'files_expected': len(EXPECTED_MONTHS),
        'files_processed': len(selected_files), 'full_period': run_all_files,
        'rows': dict(all_summaries), 'elapsed_seconds': elapsed,
        'process_peak_rss_kib': peak_rss_kib,
        'unique_trip_ids_seen': len(seen_ids),
        'python': sys.version.split()[0], 'pandas': pd.__version__, 'pyarrow': pa.__version__,
        'note': 'ru_maxrss mide el pico RSS del proceso en Linux; incluye otras operaciones del kernel.',
    }
    (report_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('Archivos procesados:', len(selected_files))
    print('Filas leídas:', f"{all_summaries['original_rows']:,}")
    print('Viajes conservados:', f"{all_summaries['kept_rows']:,}")
    print('Demanda horaria:', f'{sum(all_hourly.values()):,}')
    print('Tiempo total:', f'{elapsed:.2f} s')
    if peak_rss_kib is not None and sys.platform.startswith('linux'):
        print('Memoria pico del proceso (RSS):', f'{peak_rss_kib / 1024:.1f} MiB')
    print('Reportes:', report_dir)
    print(file_summary.to_string(index=False))
    return {
        'file_summary': file_summary, 'timing_summary': timing_summary,
        'totals': all_summaries, 'report_dir': report_dir,
    }
