"""Punto de entrada del pipeline MiBici con OOP y funciones."""

from datetime import datetime, timezone
from pathlib import Path

from mibici_mlops.config import CHUNK_SIZE, RUN_ALL_FILES, SAVE_TRIPS, locate_project_root
from mibici_mlops.dataset import complete_station_hours, split_dataset
from mibici_mlops.eda import prepare_eda
from mibici_mlops.ingestion import discover_monthly_files
from mibici_mlops.pipeline import MiBiciPipeline
from mibici_mlops.plots import (
    build_eda_tables,
    plot_demand_distribution,
    plot_demand_extremes,
    plot_demand_temporal,
    save_eda_tables,
    show_eda_tables,
)
from mibici_mlops.reporting import show_coverage, show_performance
from mibici_mlops.tracking import log_to_mlflow


RUN_LABEL = "MiBici optimizado"


def main(
    run_all_files: bool = RUN_ALL_FILES,
    chunk_size: int = CHUNK_SIZE,
    raw_dir: Path | None = None,
) -> dict:
    """Ejecutar el pipeline MiBici, generar reportes y registrar MLflow."""
    # 1. Rutas del proyecto
    project_root = locate_project_root(Path.cwd().resolve())
    data_dir = raw_dir if raw_dir is not None else project_root / "data" / "raw"
    report_base = project_root / "reports" / "validation"

    # 2. Inventario: los 81 CSV mensuales en orden cronológico
    files = discover_monthly_files(data_dir)
    selected_files = files if run_all_files else files[:2]

    # 3. Procesamiento por bloques mediante MiBiciPipeline
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    pipeline = MiBiciPipeline(chunk_size, SAVE_TRIPS)

    results = pipeline.process_dataset(
        selected_files,
        report_base / f"mibici_{run_id}",
        run_all_files,
        RUN_LABEL,
    )

    # 4. Dataset de modelado: horas sin viajes de cada estación en cero
    report_dir = results["report_dir"]

    complete_station_hours(
        report_dir / "hourly_demand.parquet",
        report_dir / "hourly_complete.parquet",
    )

    # 5. Separación temporal: train, validation y test
    splits = split_dataset(
        report_dir / "hourly_complete.parquet",
        report_dir / "splits",
        results["totals"]["kept_rows"],
        run_all_files,
    )

    results["splits"] = splits

    # 6. EDA con train: resúmenes, tablas y gráficas
    summary, monthly_trips = prepare_eda(results, splits)

    tables = build_eda_tables(summary, monthly_trips)

    show_eda_tables(tables)
    save_eda_tables(tables, report_dir)

    figure_dir = report_dir / "eda_figures"

    plot_demand_distribution(summary, output_dir=figure_dir)
    plot_demand_temporal(summary, monthly_trips, output_dir=figure_dir)
    plot_demand_extremes(summary, output_dir=figure_dir)

    # 7. Rendimiento y cobertura de la ejecución
    show_performance(results, RUN_LABEL)
    show_coverage(selected_files, run_all_files, RUN_LABEL)

    # 8. Registro de la ejecución en MLflow
    log_to_mlflow(
        results,
        splits,
        project_root,
        RUN_LABEL,
        chunk_size,
        run_all_files,
        tables,
        monthly_trips,
        selected_files,
    )

    return results


if __name__ == "__main__":
    results = main()