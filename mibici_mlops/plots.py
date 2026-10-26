"""Gráficas reutilizables para el análisis exploratorio de la demanda de MiBici."""
from pathlib import Path

import matplotlib
# Permite guardar las figuras desde la terminal sin abrir una ventana gráfica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer
from loguru import logger

from mibici_mlops.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Generación de figuras del análisis exploratorio de MiBici.")


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_demand_by_hour(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "demanda_por_hora.png") -> Path:
    """Grafica el promedio de viajes por estación-hora para cada hora del día."""
    _require_columns(summary, {"Hora_del_dia", "Promedio_viajes"}, "demanda por hora")
    data = summary.sort_values("Hora_del_dia")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["Hora_del_dia"], data["Promedio_viajes"], marker="o", linewidth=1.8)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Promedio de viajes por estación-hora")
    ax.set_title("Demanda promedio de MiBici por hora del día")
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, output_path)


def plot_demand_by_weekday(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "demanda_por_dia_semana.png") -> Path:
    """Grafica el promedio de viajes por día de la semana."""
    _require_columns(summary, {"Nombre_dia", "Promedio_viajes"}, "demanda por día de semana")
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(data["Nombre_dia"], data["Promedio_viajes"])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Día de la semana")
    ax.set_ylabel("Promedio de viajes por estación-hora")
    ax.set_title("Demanda promedio de MiBici por día de la semana")
    ax.tick_params(axis="x", rotation=20)
    ymax = data["Promedio_viajes"].max()
    if pd.notna(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.18)
    return _save_figure(fig, output_path)


def plot_monthly_demand(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "demanda_mensual.png") -> Path:
    """Grafica el total de viajes contabilizados por mes."""
    _require_columns(summary, {"Periodo", "Cantidad_viajes"}, "demanda mensual")
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["Periodo"].astype(str), data["Cantidad_viajes"], linewidth=1.6)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Viajes contabilizados")
    ax.set_title("Evolución mensual de la demanda de MiBici")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(True, alpha=0.3)
    return _save_figure(fig, output_path)


def plot_workday_vs_weekend(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "laborables_vs_fin_semana.png") -> Path:
    """Compara los promedios por hora entre días laborables y fin de semana."""
    _require_columns(summary, {"Hora_del_dia", "Lunes_a_viernes", "Fin_de_semana"}, "comparación laboral vs. fin de semana")
    data = summary.sort_values("Hora_del_dia")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(data["Hora_del_dia"], data["Lunes_a_viernes"], marker="o", label="Lunes a viernes")
    ax.plot(data["Hora_del_dia"], data["Fin_de_semana"], marker="o", label="Fin de semana")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Promedio de viajes por estación-hora")
    ax.set_title("Demanda horaria: días laborales y fin de semana")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_figure(fig, output_path)


def plot_top_stations(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "estaciones_mayor_demanda.png", top_n: int = 10) -> Path:
    """Grafica las estaciones con mayor demanda promedio por hora observada."""
    _require_columns(summary, {"Origen_Id", "Promedio_viajes"}, "demanda por estación")
    if top_n < 1:
        raise ValueError("top_n debe ser mayor que cero.")
    data = summary.nlargest(top_n, "Promedio_viajes").sort_values("Promedio_viajes")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(data["Origen_Id"].astype(str), data["Promedio_viajes"])
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Promedio de viajes por hora observada")
    ax.set_ylabel("Estación de origen")
    ax.set_title(f"{min(top_n, len(data))} estaciones con mayor demanda promedio")
    return _save_figure(fig, output_path)


def plot_monthly_seasonality(summary: pd.DataFrame, output_path: Path = FIGURES_DIR / "estacionalidad_mensual.png") -> Path:
    """Grafica el promedio diario por mes del calendario, comparando distintos años."""
    _require_columns(summary, {"Nombre_mes", "Promedio_diario_medio"}, "estacionalidad por mes")
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(data["Nombre_mes"], data["Promedio_diario_medio"])
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.set_xlabel("Mes del año")
    ax.set_ylabel("Promedio diario de viajes")
    ax.set_title("Estacionalidad de la demanda por mes del año")
    ax.tick_params(axis="x", rotation=35)
    return _save_figure(fig, output_path)


def _require_columns(data: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"La tabla de {label} no contiene columnas requeridas: {missing}")


def generate_plots(summary_dir: Path = PROCESSED_DATA_DIR, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Lee los CSV de resumen generados por ``mibici_mlops.eda`` y guarda las gráficas."""
    summary_dir = Path(summary_dir)
    figures_dir = Path(figures_dir)
    # Cada tabla de resumen se conecta con una función de gráfica y un nombre de salida.
    tasks = [
        ("eda_demanda_por_hora.csv", plot_demand_by_hour),
        ("eda_demanda_por_dia_semana.csv", plot_demand_by_weekday),
        ("eda_demanda_mensual.csv", plot_monthly_demand),
        ("eda_laborables_vs_fin_semana.csv", plot_workday_vs_weekend),
        ("eda_demanda_por_estacion.csv", plot_top_stations),
        ("eda_estacionalidad_por_mes.csv", plot_monthly_seasonality),
    ]
    generated = []
    for filename, plotter in tasks:
        csv_path = summary_dir / filename
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"No se encontró {csv_path}. Primero ejecuta el módulo mibici_mlops.eda."
            )
        table = pd.read_csv(csv_path)
        if filename == "eda_estacionalidad_por_mes.csv":
            table = table.sort_values("Mes_numero")
        generated.append(plotter(table, output_path=figures_dir / filename.replace("eda_", "").replace(".csv", ".png")))
    logger.success("Se generaron {} gráficas en {}", len(generated), figures_dir)
    return generated


@app.command()
def main(
    summary_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Carpeta con las tablas CSV producidas por eda.py."),
    figures_dir: Path = typer.Option(FIGURES_DIR, help="Carpeta de salida de las gráficas."),
) -> None:
    """Genera las gráficas a partir de las tablas CSV que produce el módulo EDA."""
    generate_plots(summary_dir=summary_dir, figures_dir=figures_dir)


if __name__ == "__main__":
    app()
