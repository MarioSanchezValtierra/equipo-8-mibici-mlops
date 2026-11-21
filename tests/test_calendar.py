"""Pruebas de la variable de días festivos (add_holidays en mibici_mlops/features.py)."""

import pandas as pd

from mibici_mlops.features import add_holidays

HOLIDAYS_2025 = ["2025-01-01", "2025-02-03", "2025-03-17", "2025-05-01",
                 "2025-09-16", "2025-11-17", "2025-12-25"]


def calendar(dates) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(dates))
    return pd.DataFrame({"Fecha": dates, "month": dates.dt.month,
                         "day_of_month": dates.dt.day, "day_of_week": dates.dt.day_of_week})


def test_marks_the_phase2_holidays_of_2025():
    year = calendar(pd.date_range("2025-01-01", "2025-12-31"))
    marked = year.loc[add_holidays(year)["is_holiday"] == 1, "Fecha"].dt.strftime("%Y-%m-%d")
    assert marked.tolist() == HOLIDAYS_2025


def test_monday_holidays_depend_on_the_weekday():
    # 10 de febrero de 2025 es lunes, pero es el segundo lunes: no es festivo
    assert add_holidays(calendar(["2025-02-10"]))["is_holiday"].iloc[0] == 0


def test_does_not_change_the_input():
    data = calendar(["2025-09-16"])
    add_holidays(data)
    assert "is_holiday" not in data
