"""Pruebas de la clase MiBiciPipeline."""

import pytest

from mibici_mlops.pipeline import MiBiciPipeline


def test_pipeline_initialization():
    """Comprobar la configuración y el estado inicial."""
    pipeline = MiBiciPipeline(chunk_size=2, save_trips=True)

    assert pipeline.chunk_size == 2
    assert pipeline.save_trips is True
    assert pipeline.seen_ids == set()


def test_pipeline_independent_instances():
    """Comprobar que las instancias no comparten identificadores."""
    first = MiBiciPipeline(chunk_size=2, save_trips=True)
    second = MiBiciPipeline(chunk_size=2, save_trips=False)

    first.seen_ids.add(123)

    assert first.seen_ids == {123}
    assert second.seen_ids == set()


def test_pipeline_invalid_chunk_size():
    """Rechazar tamaños de bloque inválidos."""
    with pytest.raises(ValueError):
        MiBiciPipeline(chunk_size=0, save_trips=True)