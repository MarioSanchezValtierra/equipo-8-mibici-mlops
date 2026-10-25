import pandas as pd
import pytest

from mibici_mlops.eda import summarize_hourly_batches


def test_summarize_hourly_batches_builds_expected_tables():
    sample = pd.DataFrame({
        "Origen_Id": [10, 10, 20, 20],
        "Hora": pd.to_datetime([
            "2024-01-01 08:00:00",  # Lunes
            "2024-01-01 18:00:00",  # Lunes
            "2024-01-06 12:00:00",  # Sábado
            "2024-01-07 12:00:00",  # Domingo
        ]),
        "Cantidad_viajes": [2, 4, 1, 3],
    })
    results = summarize_hourly_batches([sample.iloc[:2], sample.iloc[2:]])

    assert results["demanda_por_hora"].loc[8, "Promedio_viajes"] == 2
    assert results["demanda_por_hora"].loc[18, "Promedio_viajes"] == 4
    assert results["demanda_por_dia_semana"].loc[0, "Promedio_viajes"] == 3
    assert results["demanda_por_dia_semana"].loc[5, "Promedio_viajes"] == 1
    assert results["demanda_por_estacion"].loc[0, "Origen_Id"] == 10
    assert results["demanda_por_estacion"].loc[0, "Total_viajes"] == 6
    assert results["demanda_mensual"].loc[0, "Cantidad_viajes"] == 10
    assert results["laborables_vs_fin_semana"].loc[12, "Fin_de_semana"] == 2
    assert results["distribucion_demanda"].loc[0, "Viajes_contabilizados"] == 10


def test_summarize_hourly_batches_rejects_missing_columns():
    with pytest.raises(ValueError, match="columnas requeridas"):
        summarize_hourly_batches([pd.DataFrame({"Origen_Id": [1]})])


def test_summarize_hourly_batches_rejects_negative_demand():
    sample = pd.DataFrame({
        "Origen_Id": [1],
        "Hora": pd.to_datetime(["2024-01-01 08:00:00"]),
        "Cantidad_viajes": [-1],
    })
    with pytest.raises(ValueError, match="Cantidad_viajes negativo o no entero"):
        summarize_hourly_batches([sample])
