"""Datos pequeños compartidos por las pruebas."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_data() -> pd.DataFrame:
    """Una estación, tres meses, con el 10 de enero faltante a propósito."""
    dates = pd.date_range("2024-12-01", "2025-02-28")
    data = pd.DataFrame({"Origen_Id": 1, "Fecha": dates, "target": np.arange(1, len(dates) + 1)})
    data = data.loc[data["Fecha"] != "2025-01-10"].reset_index(drop=True)
    data["day_of_week"] = data["Fecha"].dt.day_of_week
    return data
