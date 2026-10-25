"""Inventario de los CSV mensuales y lectura por bloques."""

import codecs
import re
from pathlib import Path

import pandas as pd

from mibici_mlops.config import EXPECTED_MONTHS


MONTHLY_NAME = re.compile(r'^datos_abiertos_(\d{4})_(\d{2})\.csv$', re.IGNORECASE)


def file_month(file_path: Path) -> pd.Period | None:
    """Obtener el mes indicado en el nombre de un archivo mensual."""
    match = MONTHLY_NAME.fullmatch(file_path.name)
    if match is None:
        return None
    return pd.Period(f'{match.group(1)}-{match.group(2)}', freq='M')


def discover_monthly_files(raw_dir: Path) -> list[Path]:
    """Encontrar los CSV mensuales y rechazar meses repetidos."""
    monthly = {}
    for path in raw_dir.rglob('*.csv'):
        month = file_month(path)
        if month is None:
            continue
        if month in monthly:
            raise ValueError(f'Mes duplicado: {month}: {monthly[month]} y {path}')
        monthly[month] = path
    expected = set(EXPECTED_MONTHS)
    missing = sorted(expected - set(monthly))
    extra = sorted(set(monthly) - expected)
    if missing or extra:
        raise ValueError(f'Inventario incorrecto. Faltantes: {missing}; adicionales: {extra}')
    return [monthly[month] for month in EXPECTED_MONTHS]


def read_csv_chunks(file_path: Path, chunk_size: int):
    """Select a verified encoding and read CSV records in bounded chunks."""
    if chunk_size < 1:
        raise ValueError('chunk_size debe ser positivo')
    encoding = 'utf-8'
    decoder = codecs.getincrementaldecoder(encoding)('strict')
    try:
        with file_path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                decoder.decode(block)
            decoder.decode(b'', final=True)
    except UnicodeDecodeError:
        encoding = 'cp1252'
    reader = pd.read_csv(
        file_path, encoding=encoding, dtype='string', chunksize=chunk_size
    )
    return reader, encoding
