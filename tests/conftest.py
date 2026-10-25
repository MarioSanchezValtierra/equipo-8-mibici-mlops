"""Datos de prueba compartidos por las pruebas de comportamiento y de integración."""

import pandas as pd
import pytest


@pytest.fixture
def test_rows() -> pd.DataFrame:
    """Ocho viajes con un duplicado, una estación excluida y un origen vacío."""
    return pd.DataFrame({
        'Viaje_Id': ['1', '2', '2', '3', '4', '5', '6', '7'],
        'Usuario_Id': ['101'] * 8,
        'Genero': ['M', None, None, 'F', 'M', 'F', 'M', 'F'],
        'Año_de_nacimiento': ['1995', None, None, '1990', '2000', '1985', '1900', '2000'],
        'Inicio_del_viaje': [f'2026-09-01 {hour:02d}:00:00' for hour in range(8, 16)],
        'Fin_del_viaje': [f'2026-09-01 {hour:02d}:20:00' for hour in range(8, 16)],
        'Origen_Id': ['10', '10', '10', '407', '20', None, '30', '40'],
        'Destino_Id': ['10', '15', '15', '25', '30', '40', '35', '45'],
    })
