import pandas as pd

from mibici_mlops.plots import plot_demand_by_hour


def test_plot_demand_by_hour_saves_png(tmp_path):
    table = pd.DataFrame({
        "Hora_del_dia": [8, 18],
        "Promedio_viajes": [1.2, 2.5],
        "Registros": [100, 120],
    })
    output = plot_demand_by_hour(table, tmp_path / "hourly.png")
    assert output.is_file()
    assert output.stat().st_size > 0
