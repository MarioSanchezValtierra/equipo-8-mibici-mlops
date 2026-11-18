"""Pruebas unitarias del monitoreo de drift (mibici_mlops/monitoring.py)."""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops import monitoring

BASE = {"baseline_mae": 6.0, "baseline_missing": 0.10, "reference_demand": 30.0}


def period(**changes) -> pd.Series:
    """Un periodo sin problemas; cada prueba cambia solo lo que quiere revisar."""
    row = {"mae": 6.0, "mejora_vs_ingenuo": 0.25, "sesgo": 0.0, "lag_1_vacío": 0.10,
           "proporción_drift": 0.0, "psi_máximo": 0.02,
           "cambio_estaciones_activas": 0.0, "cambio_viajes_por_día": 0.0}
    return pd.Series({**row, **changes})


@pytest.fixture
def samples():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"estable": rng.normal(10, 2, 5_000), "desplazada": rng.normal(10, 2, 5_000)})
    current = pd.DataFrame({"estable": rng.normal(10, 2, 5_000), "desplazada": rng.normal(13, 2, 5_000)})
    return reference, current


def test_psi_is_near_zero_for_the_same_distribution(samples):
    reference, current = samples
    assert monitoring.psi(reference["estable"], current["estable"]) < 0.02


def test_psi_detects_a_shift(samples):
    reference, current = samples
    assert monitoring.psi(reference["desplazada"], current["desplazada"]) > monitoring.PSI_LEVELS[1]


def test_psi_ignores_missing_values():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0] * 50)
    current = pd.concat([reference, pd.Series([np.nan] * 100)])
    assert monitoring.psi(reference, current) == pytest.approx(0.0, abs=1e-9)


def test_drift_table_flags_only_the_shifted_variable(samples):
    reference, current = samples
    table = monitoring.drift_table(reference, current, ["estable", "desplazada"]).set_index("variable")
    assert not table.loc["estable", "drift"]
    assert table.loc["desplazada", "drift"]


def test_constant_variable_does_not_break_the_table():
    reference = pd.DataFrame({"zona": [3] * 100})
    assert not monitoring.drift_table(reference, reference.copy(), ["zona"]).loc[0, "drift"]


def test_level_uses_warning_and_critical_limits():
    assert [monitoring.level(v, (0.1, 0.2)) for v in (0.05, 0.15, 0.25)] == [0, 1, 2]


def test_daily_operation_counts_stations_and_trips():
    data = pd.DataFrame({"Fecha": ["d1", "d1", "d2"], "Origen_Id": [1, 2, 1], "target": [5, 7, 4]})
    assert monitoring.daily_operation(data) == (1.5, 8.0)


def test_normal_period_has_no_alerts():
    levels = monitoring.alert_levels(period(), **BASE)
    assert set(levels.values()) == {0}
    assert monitoring.decide(levels) == "Sin acción"


@pytest.mark.parametrize("changes, alert, expected", [
    ({"lag_1_vacío": 0.20}, "A1_datos_faltantes", 1),
    ({"lag_1_vacío": 0.30}, "A1_datos_faltantes", 2),
    ({"sesgo": -4.0}, "A2_sesgo", 1),
    ({"sesgo": 7.0}, "A2_sesgo", 2),
    ({"mae": 7.5}, "A3_desempeño", 1),
    ({"mae": 7.5, "mejora_vs_ingenuo": 0.10}, "A3_desempeño", 2),
    ({"psi_máximo": 0.15}, "A4_drift", 1),
    ({"proporción_drift": 0.5}, "A4_drift", 1),
    ({"psi_máximo": 0.40}, "A4_drift", 2),
    ({"cambio_estaciones_activas": 0.20}, "A5_operación", 1),
    ({"cambio_viajes_por_día": -0.35}, "A5_operación", 2),
])
def test_each_alert_and_level(changes, alert, expected):
    levels = monitoring.alert_levels(period(**changes), **BASE)
    assert levels[alert] == expected
    assert all(value == 0 for name, value in levels.items() if name != alert)


def test_data_problems_come_before_retraining():
    levels = monitoring.alert_levels(period(lag_1_vacío=0.30, mae=8.0, mejora_vs_ingenuo=0.0), **BASE)
    assert monitoring.decide(levels, previous_a3=2) == "Revisar el pipeline de datos (no reentrenar)"


def test_retrain_only_after_two_critical_months():
    levels = monitoring.alert_levels(period(mae=8.0, mejora_vs_ingenuo=0.10), **BASE)
    assert monitoring.decide(levels, previous_a3=1) == "Vigilar y confirmar con el siguiente mes"
    assert monitoring.decide(levels, previous_a3=2) == "Reentrenar: A3 crítica dos meses seguidos"


def test_operation_change_without_performance_loss():
    levels = monitoring.alert_levels(period(cambio_viajes_por_día=0.60), **BASE)
    assert monitoring.decide(levels) == "Revisar con operación el cambio de la red"


def test_warnings_are_only_watched():
    levels = monitoring.alert_levels(period(psi_máximo=0.15), **BASE)
    assert monitoring.decide(levels) == "Vigilar"
