"""Prueba de integración del pipeline funcional."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from mibici_mlops.pipeline import process_file


def test_integration(test_rows):
    """Procesar dos archivos pequeños y comprobar los IDs vistos entre archivos."""
    with TemporaryDirectory(prefix='mibici_v1_test_') as directory:
        root = Path(directory)
        input_dir = root / 'input'
        output_dir = root / 'output'
        input_dir.mkdir()
        output_dir.mkdir()
        first_path = input_dir / 'datos_abiertos_2026_08.csv'
        second_path = input_dir / 'datos_abiertos_2026_09.csv'
        test_rows.iloc[:2].to_csv(first_path, index=False)
        test_rows.iloc[2:].to_csv(second_path, index=False)
        seen = set()
        a = process_file(first_path, 2, seen, output_dir)
        b = process_file(second_path, 2, seen, output_dir)
        assert a['summary']['kept_rows'] + b['summary']['kept_rows'] == 5
        assert b['summary']['Viaje_Id duplicado'] == 1
        assert pq.ParquetFile(a['parquet']).metadata.num_rows == 2
        assert pq.ParquetFile(b['parquet']).metadata.num_rows == 3
        assert sum(a['hourly'].values()) + sum(b['hourly'].values()) == 5
        assert len(seen) == 7
