"""Prueba de los cortes de la separación temporal."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici_mlops.config import COMPLETE_SCHEMA, VALIDATION_END
from mibici_mlops.dataset import split_dataset


def test_split():
    """Probar los cortes de la separación con horas alrededor de cada fecha límite."""
    hours = pd.to_datetime(['2024-12-31 23:00', '2025-01-01 00:00',
                            '2025-12-31 23:00', '2026-01-01 00:00'])
    table = pd.DataFrame({
        'Origen_Id': np.full(4, 10, dtype=np.int32), 'Hora': hours,
        'Cantidad_viajes': np.array([1, 2, 3, 4], dtype=np.int32),
    })
    with TemporaryDirectory(prefix='mibici_split_test_') as directory:
        root = Path(directory)
        pq.write_table(pa.Table.from_pandas(table, schema=COMPLETE_SCHEMA, preserve_index=False),
                       root / 'complete.parquet')
        splits = split_dataset(root / 'complete.parquet', root / 'splits', 10, True)
        assert splits['Filas'].tolist() == [1, 2, 1]
        assert splits['Viajes'].tolist() == [1, 5, 4]
        assert pd.read_parquet(root / 'splits' / 'test.parquet')['Hora'].min() == VALIDATION_END
        # Una clave repetida debe detener la separación
        repeated = pd.concat([table, table.tail(1)], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(repeated, schema=COMPLETE_SCHEMA, preserve_index=False),
                       root / 'repeated.parquet')
        try:
            split_dataset(root / 'repeated.parquet', root / 'splits_repeated', 14, True)
        except ValueError:
            pass
        else:
            raise AssertionError('La clave repetida no se detectó')
