"""Pruebas de normalización, limpieza B y variables del viaje."""

import pandas as pd
from pandas.testing import assert_frame_equal

from mibici_mlops.preprocessing import (
    add_trip_features, classify_duration, clean_strategy_b, count_hourly_demand, normalize_chunk,
    normalize_integer_column,
)


def test_behavior(test_rows):
    """Comprobar la normalización, la limpieza B entre bloques y las categorías de duración."""
    source = test_rows
    untouched = source.copy(deep=True)
    first = normalize_chunk(source.iloc[:2])
    second = normalize_chunk(source.iloc[2:])
    seen = set()
    cleaned_1, summary_1, new_1 = clean_strategy_b(first, seen)
    seen.update(new_1)
    cleaned_2, summary_2, new_2 = clean_strategy_b(second, seen)
    seen.update(new_2)
    assert summary_1['kept_rows'] + summary_2['kept_rows'] == 5
    assert summary_2['Viaje_Id duplicado'] == 1
    assert summary_2['Estación excluida'] == 1
    assert summary_2['Origen inválido'] == 1
    assert int(count_hourly_demand(add_trip_features(cleaned_1)).sum()) == len(cleaned_1)
    assert_frame_equal(source, untouched)
    assert seen == {1, 2, 3, 4, 5, 6, 7}
    durations = pd.Series([0, 1, 5, 15, 30, 180, 1440, 1441, -1])
    labels = classify_duration(durations).astype('string').tolist()
    assert labels[:8] == [
        'Muy corto', 'Corto', 'Habitual', 'Largo', 'Muy largo',
        'Muy largo', 'Prolongado', 'Extremo',
    ] and pd.isna(labels[-1])
    assert normalize_integer_column(pd.Series(['9223372036854775807', '9223372036854775808'])).tolist()[0] == 9223372036854775807
    assert pd.isna(normalize_integer_column(pd.Series(['9223372036854775808'])).iloc[0])
