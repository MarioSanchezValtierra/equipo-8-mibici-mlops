"""Tiempos por etapa y reportes de rendimiento y cobertura."""

import time
from collections import Counter
from contextlib import contextmanager

import matplotlib.pyplot as plt

from mibici_mlops.config import EXPECTED_MONTHS


@contextmanager
def timed(times: Counter, stage: str):
    """Sumar al contador el tiempo que tarda una etapa."""
    start = time.perf_counter()
    try:
        yield
    finally:
        times[stage] += time.perf_counter() - start


def show_performance(results, run_label):
    """Summarize elapsed time by pipeline stage."""
    stage_columns = [column for column in results["timing_summary"].columns
                     if column not in ('archivo', 'Tiempo total')]
    stage_totals = results["timing_summary"][stage_columns].sum().sort_values(ascending=False)
    performance_table = stage_totals.rename_axis('Etapa').reset_index(name='Segundos')
    performance_table['Porcentaje etapas'] = (
        performance_table['Segundos'] / performance_table['Segundos'].sum() * 100
        if performance_table['Segundos'].sum() else 0.0
    )
    print(performance_table.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(performance_table['Etapa'], performance_table['Segundos'])
    ax.invert_yaxis()
    ax.set_xlabel('Tiempo acumulado (s)')
    ax.set_title(f'{run_label} | Tiempo por etapa')
    plt.tight_layout()
    plt.show()


def show_coverage(selected_files, run_all_files, run_label):
    """Report whether the full study period was processed."""
    full_run = run_all_files and len(selected_files) == len(EXPECTED_MONTHS)
    print('Cobertura completa:', 'SÍ' if full_run else 'NO, prueba con dos meses')
    print(f'Cobertura de {run_label}:', '81 CSV seleccionados' if full_run else 'Prueba parcial')
