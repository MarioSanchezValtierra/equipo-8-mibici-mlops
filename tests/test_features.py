import pandas as pd
import pytest

from mibici_mlops.features import add_trip_features, classify_duration


def test_classify_duration_preserves_notebook_boundaries():
    values = pd.Series([0.5, 1, 5, 15, 30, 180, 180.01, 1440, 1440.01, -1, None])
    result = classify_duration(values).astype("string").tolist()
    assert result[:9] == [
        "Muy corto", "Corto", "Habitual", "Largo", "Muy largo",
        "Muy largo", "Prolongado", "Prolongado", "Extremo",
    ]
    assert pd.isna(result[9])
    assert pd.isna(result[10])


def test_add_trip_features_adds_columns_without_changing_input():
    original = pd.DataFrame({
        "Inicio_del_viaje": ["2024-01-01 08:00:00", "2024-01-01 10:00:00"],
        "Fin_del_viaje": ["2024-01-01 08:02:00", "2024-01-01 10:05:00"],
        "Origen_Id": [10, 20],
        "Destino_Id": [10, 21],
        "Viaje_Id": [100, 101],
    })
    enriched = add_trip_features(original)
    assert "duracion_minutos" not in original.columns
    assert enriched["duracion_minutos"].tolist() == [2.0, 5.0]
    assert enriched["tipo_recorrido"].astype("string").tolist() == [
        "Misma estación", "Estaciones diferentes"
    ]
    assert enriched["categoria_duracion"].astype("string").tolist() == [
        "Corto", "Habitual"
    ]


def test_add_trip_features_requires_notebook_columns():
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        add_trip_features(pd.DataFrame({"Viaje_Id": [1]}))
