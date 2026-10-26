"""Validaciones de integridad de los datos procesados."""

from mibici_mlops.config import EXCLUSION_REASONS


def check_counts(summary, demand_total: int, label: str) -> None:
    """Comprobar que registros, motivos de exclusión y demanda concilien."""
    if summary['original_rows'] != summary['kept_rows'] + summary['excluded_rows']:
        raise AssertionError(f'{label}: los registros no concilian')
    if sum(summary[key] for key in EXCLUSION_REASONS) != summary['excluded_rows']:
        raise AssertionError(f'{label}: los motivos de exclusión no concilian')
    if demand_total != summary['kept_rows']:
        raise AssertionError(f'{label}: la demanda no coincide con los viajes conservados')
