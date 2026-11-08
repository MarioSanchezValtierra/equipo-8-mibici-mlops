"""Gráficas del EDA (Fase 1) y del modelado (Fase 2). Las de la Fase 2 se guardan y se cierran."""

import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from mibici_mlops.eda import MONTHS, WEEKDAYS, box_stats, hist_percentile


def build_eda_tables(summary: dict, monthly_trips: pd.Series) -> dict:
    """Build EDA tables from existing summaries without rereading Parquet."""
    histogram = summary['hist']
    total = sum(histogram.values())
    mean = sum(value * count for value, count in histogram.items()) / total
    variance = sum(count * (value - mean) ** 2 for value, count in histogram.items())
    variance /= max(total - 1, 1)
    stats = pd.DataFrame({'Cantidad_viajes': {
        'count': total, 'mean': mean, 'std': np.sqrt(variance),
        **{f'{p}%': hist_percentile(histogram, p) for p in (0, 25, 50, 75, 90, 95, 99, 100)}
    }}).rename(index={'0%': 'min', '100%': 'max'})

    hourly = summary['hour'].reindex(range(24))
    hourly_table = pd.DataFrame({
        'Hora_del_dia': range(24), 'Promedio_viajes': hourly['mean'].to_numpy(),
        'Registros': hourly['count'].to_numpy()
    })
    weekday = summary['weekday'].reindex(range(7))
    weekday_table = pd.DataFrame({
        'Dia_semana': range(7), 'Nombre_dia': WEEKDAYS,
        'Promedio_viajes': weekday['mean'].to_numpy(),
        'Mediana_viajes': [hist_percentile(hist, 50) for hist in summary['weekday_hist']],
        'Registros': weekday['count'].to_numpy()
    })
    station = summary['station'].copy()
    station_table = pd.DataFrame({
        'Origen_Id': station.index,
        'Promedio_viajes': station['mean'].to_numpy(),
        'Mediana_viajes': [hist_percentile(summary['station_hist'][int(i)], 50) for i in station.index],
        'Total_viajes': station['total'].to_numpy(),
        'Horas_observadas': station['count'].to_numpy(),
        'Maximo_viajes': summary['maximum'].reindex(station.index).to_numpy()
    }).sort_values('Promedio_viajes', ascending=False).reset_index(drop=True)

    monthly = monthly_trips.sort_index().to_frame('Cantidad_viajes')
    monthly['Variacion_pct'] = monthly['Cantidad_viajes'].pct_change() * 100
    season = pd.DataFrame({
        'Mes_numero': monthly.index.month,
        'Año': monthly.index.year,
        'Promedio_diario': monthly['Cantidad_viajes'].to_numpy() / monthly.index.days_in_month
    })
    seasonal = season.groupby('Mes_numero')['Promedio_diario'].agg(
        Promedio_diario='mean', Mediana_diaria='median', Años_observados='count'
    ).reindex(range(1, 13))
    seasonal.index = MONTHS
    seasonal.index.name = 'Mes'

    day_hour = summary['weekday_hour']
    profiles = {}
    for label, days in [('Lunes a viernes', range(5)), ('Fin de semana', range(5, 7))]:
        group = day_hour.loc[list(days)].groupby(level=1)[['total', 'count']].sum()
        profiles[label] = (group['total'] / group['count']).reindex(range(24))
    day_type = pd.DataFrame(profiles)
    day_type.index.name = 'Hora_del_dia'

    q1, q3 = hist_percentile(histogram, 25), hist_percentile(histogram, 75)
    limit = q3 + 1.5 * (q3 - q1)
    outliers = sum(count for value, count in histogram.items() if value > limit)
    outlier_stats = pd.DataFrame({'Valor': {
        'Q1': q1, 'Q3': q3, 'IQR': q3 - q1, 'Limite_superior': limit,
        'Registros_analizados': total, 'Valores_atipicos': outliers,
        'Porcentaje_atipicos': outliers / total * 100
    }})

    station_outliers = []
    for station_id, station_hist in summary['station_hist'].items():
        a, b = hist_percentile(station_hist, 25), hist_percentile(station_hist, 75)
        station_limit = b + 1.5 * (b - a)
        count = sum(station_hist.values())
        extreme = sum(n for value, n in station_hist.items() if value > station_limit)
        station_outliers.append({
            'Origen_Id': station_id, 'Q1': a, 'Q3': b,
            'Limite_superior': station_limit, 'Valores_atipicos': extreme,
            'Porcentaje_atipicos': extreme / count * 100
        })
    station_extremes = station_table.set_index('Origen_Id').join(
        pd.DataFrame(station_outliers).set_index('Origen_Id')
    )
    station_extremes['Relacion_maximo_promedio'] = (
        station_extremes['Maximo_viajes'] /
        station_extremes['Promedio_viajes'].replace(0, np.nan)
    )
    station_extremes = station_extremes.sort_values('Maximo_viajes', ascending=False)

    station_hour = summary['station_hour']['mean'].unstack().reindex(columns=range(24))
    peak_hour = station_hour.idxmax(axis=1)
    peak_value = station_hour.max(axis=1)
    general_mean = summary['station']['mean'].reindex(station_hour.index)
    peaks = pd.DataFrame({
        'Origen_Id': station_hour.index, 'Hora_pico': peak_hour.to_numpy(),
        'Promedio_hora_pico': peak_value.to_numpy(),
        'Promedio_general': general_mean.to_numpy(),
        'Relacion_pico_promedio': (peak_value / general_mean.replace(0, np.nan)).to_numpy()
    }).sort_values('Origen_Id').reset_index(drop=True)

    return {
        'Estadísticas de demanda': stats,
        'Demanda por hora': hourly_table,
        'Demanda por día': weekday_table,
        'Demanda por estación': station_table,
        'Estadísticas mensuales': monthly.describe().round(2),
        'Meses con más viajes': monthly.nlargest(5, 'Cantidad_viajes'),
        'Meses con menos viajes': monthly.nsmallest(5, 'Cantidad_viajes'),
        'Días laborales y fines de semana': day_type,
        'Estacionalidad': seasonal,
        'Estadísticas de valores atípicos': outlier_stats,
        'Registros extremos': summary['top'],
        'Valores extremos por estación': station_extremes.head(20),
        'Horas pico por estación': peaks
    }


def show_eda_tables(tables: dict) -> None:
    """Display EDA tables in the same order as the original analysis."""
    for title, table in tables.items():
        print(f'\n{title}')
        display(table.round(2))


def save_eda_tables(tables: dict, output_dir: Path) -> None:
    """Guardar cada tabla del EDA en un CSV con el nombre de la tabla."""
    output_dir = output_dir / 'eda_tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    for title, table in tables.items():
        file_name = re.sub(r'\W+', '_', title).strip('_').lower()
        table.to_csv(output_dir / f'{file_name}.csv', index=True)


def finish_plot(ax, title, xlabel, ylabel, output_dir=None):
    """Apply shared labels and show a figure."""
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_number = len(list(output_dir.glob("figure_*.png"))) + 1
        ax.figure.savefig(output_dir / f"figure_{figure_number:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
    else:
        plt.show()


def plot_series(values, title, xlabel, ylabel, kind='bar', horizontal=False,
                percentage=False, output_dir=None):
    """Plot a bar or line chart with optional percentage labels."""
    fig, ax = plt.subplots(figsize=(11, 5))
    if kind == 'line':
        ax.plot(values.index, values.values, marker='o', markersize=3)
    elif horizontal:
        bars = ax.barh(values.index.astype(str), values.values)
    else:
        bars = ax.bar(values.index.astype(str), values.values)
    if percentage and kind == 'bar':
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
        ax.set_ylim(0, max(values.max() * 1.15, 1))
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
    if not horizontal and kind == 'bar' and len(values) > 12:
        ax.tick_params(axis='x', rotation=45)
    finish_plot(ax, title, xlabel, ylabel, output_dir=output_dir)


def plot_heatmap(values, title, xlabel, ylabel, decimals=1, output_dir=None):
    """Plot a heatmap with numeric labels and missing-value handling."""
    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(values.columns)), values.columns)
    ax.set_yticks(range(len(values.index)), values.index)
    if len(values.columns) == 12:
        ax.tick_params(axis='x', rotation=45)
    numeric = values.to_numpy(dtype=float)
    low, high = image.get_clim()
    midpoint = (low + high) / 2
    for row in range(numeric.shape[0]):
        for col in range(numeric.shape[1]):
            value = numeric[row, col]
            if np.isfinite(value):
                color = 'white' if value > midpoint else 'black'
                ax.text(col, row, f'{value:,.{decimals}f}',
                        ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(image, ax=ax, label='Demanda promedio')
    finish_plot(ax, title, xlabel, ylabel, output_dir=output_dir)


def plot_demand_distribution(summary: dict, output_dir=None) -> None:
    """Show demand distribution by hour, weekday and station."""
    histogram = summary['hist']
    total = sum(histogram.values())
    ranges = [histogram[0], sum(histogram[i] for i in (1, 2)),
              sum(histogram[i] for i in (3, 4, 5)),
              sum(histogram[i] for i in range(6, 11)),
              sum(n for value, n in histogram.items() if value > 10)]
    distribution = pd.Series(np.array(ranges) / total * 100,
                             index=['0', '1–2', '3–5', '6–10', '>10'])
    plot_series(distribution, 'Distribución de demanda por estación-hora',
                'Viajes por estación-hora', 'Registros (%)', percentage=True, output_dir=output_dir)

    hour = summary['hour']['mean'].reindex(range(24))
    plot_series(hour, 'Demanda promedio por hora', 'Hora', 'Viajes por estación-hora', 'line', output_dir=output_dir)
    weekday = summary['weekday']['mean'].reindex(range(7))
    weekday.index = WEEKDAYS
    plot_series(weekday, 'Demanda promedio por día', 'Día', 'Viajes por estación-hora', output_dir=output_dir)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bxp([box_stats(hist, WEEKDAYS[i]) for i, hist in enumerate(summary['weekday_hist'])],
           showfliers=False)
    finish_plot(ax, 'Distribución de demanda por día', 'Día', 'Viajes por estación-hora', output_dir=output_dir)

    stations = summary['station']['mean'].sort_values()
    for title, values in [('Mayor', stations.tail(10)), ('Menor', stations.head(10))]:
        plot_series(values, f'{title} demanda promedio por estación',
                    'Viajes por estación-hora', 'Estación', horizontal=True, output_dir=output_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(stations.to_numpy(), tick_labels=['Estaciones'], showfliers=False)
    finish_plot(ax, 'Distribución entre estaciones', '', 'Viajes por estación-hora', output_dir=output_dir)


def plot_demand_temporal(summary: dict, monthly_trips: pd.Series, output_dir=None) -> None:
    """Show monthly evolution and weekday-hour patterns."""
    monthly = monthly_trips.sort_index()
    dates = monthly.index.to_timestamp()
    plot_series(pd.Series(monthly.values, index=dates), 'Evolución mensual',
                'Mes', 'Cantidad de viajes', 'line', output_dir=output_dir)
    plot_series(pd.Series(monthly.pct_change().values * 100, index=dates),
                'Variación mensual', 'Mes', 'Variación (%)', 'line', output_dir=output_dir)

    day_hour = summary['weekday_hour']
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, days in [('Lunes a viernes', range(5)), ('Fin de semana', range(5, 7))]:
        selected = day_hour.loc[list(days)].groupby(level=1)[['total', 'count']].sum()
        profile = selected['total'] / selected['count']
        ax.plot(profile.index, profile.values, marker='o', label=label)
    ax.legend()
    ax.set_xticks(range(24))
    finish_plot(ax, 'Días laborales y fines de semana', 'Hora', 'Viajes por estación-hora', output_dir=output_dir)
    heatmap = day_hour['mean'].unstack().reindex(index=range(7), columns=range(24))
    heatmap.index = WEEKDAYS
    plot_heatmap(heatmap, 'Demanda por día y hora', 'Hora', 'Día', output_dir=output_dir)

    season = pd.DataFrame({'year': monthly.index.year, 'month': monthly.index.month,
                           'daily': monthly.values / monthly.index.days_in_month})
    average = season.groupby('month')['daily'].mean().reindex(range(1, 13))
    average.index = MONTHS
    plot_series(average, 'Estacionalidad mensual', 'Mes', 'Viajes diarios', 'line', output_dir=output_dir)
    year_month = season.pivot(index='year', columns='month', values='daily').reindex(columns=range(1, 13))
    year_month.columns = MONTHS
    plot_heatmap(year_month, 'Demanda diaria por mes y año', 'Mes', 'Año', decimals=0, output_dir=output_dir)


def plot_demand_extremes(summary: dict, output_dir=None) -> None:
    """Show extreme station-hour demand and station profiles."""
    histogram = summary['hist']
    total = sum(histogram.values())
    q1, q3 = hist_percentile(histogram, 25), hist_percentile(histogram, 75)
    threshold = q3 + 1.5 * (q3 - q1)
    outliers = sum(n for value, n in histogram.items() if value > threshold)
    print(f'Atípicos: {outliers:,} de {total:,} ({outliers / total:.2%}); límite IQR: {threshold:.2f}')
    extreme = summary['top']['Cantidad_viajes'].iloc[::-1]
    extreme.index = extreme.index.astype(str)
    plot_series(extreme, '20 registros de mayor demanda', 'Viajes', 'Registro', horizontal=True, output_dir=output_dir)

    top_stations = summary['maximum'].nlargest(20).index
    comparison = pd.DataFrame({'Promedio': summary['station'].loc[top_stations, 'mean'],
                               'Máximo': summary['maximum'].loc[top_stations]})
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.plot.bar(ax=ax)
    ax.tick_params(axis='x', rotation=45)
    finish_plot(ax, 'Demanda promedio y máxima por estación', 'Estación', 'Viajes por estación-hora', output_dir=output_dir)

    station_hour = summary['station_hour']['mean'].unstack().reindex(columns=range(24))
    peaks = station_hour.idxmax(axis=1).value_counts().reindex(range(24), fill_value=0)
    plot_series(peaks, 'Hora pico por estación', 'Hora', 'Estaciones', output_dir=output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    for station_id in summary['station']['total'].nlargest(10).index:
        ax.plot(station_hour.columns, station_hour.loc[station_id], label=str(station_id))
    ax.set_xticks(range(24))
    ax.legend(title='Estación', bbox_to_anchor=(1.02, 1), loc='upper left')
    finish_plot(ax, 'Patrones horarios de las 10 estaciones con más viajes',
                'Hora', 'Viajes por estación-hora', output_dir=output_dir)


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mae_by_fold(results: pd.DataFrame, models: list[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    for model_name in models:
        model_rows = results.loc[results["model"] == model_name]
        ax.plot(model_rows["cutoff"].astype(str), model_rows["mae"], marker="o", label=model_name)
    ax.set_title("MAE por ventana de validación (mes evaluado)")
    ax.set_xlabel("Inicio del mes evaluado")
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    return save_figure(fig, path)


def plot_test_predictions(review: pd.DataFrame, target_column: str, path: Path) -> Path:
    columns = [target_column, "predicted_trips", "naive_prediction"]
    daily_totals = review.groupby("Fecha")[columns].sum()
    station = (
        review.loc[review["segment"] == "Con historia"]
        .groupby("Origen_Id")[target_column].mean().idxmax()
    )
    station_series = review.loc[review["Origen_Id"] == station].set_index("Fecha")[columns]
    styles = [
        (target_column, "Viajes reales", "black", 1.6),
        ("predicted_trips", "Predicción del modelo", "tab:blue", 1.2),
        ("naive_prediction", "Ingenuo estacional (lag_7)", "tab:orange", 1.0),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for column, label, color, width in styles:
        axes[0].plot(daily_totals.index, daily_totals[column], label=label, color=color, linewidth=width)
        axes[1].plot(station_series.index, station_series[column], label=label, color=color, linewidth=width)
    axes[0].set_title("Viajes diarios totales en 2026 (suma de todas las estaciones)")
    axes[1].set_title(f"Estación {station} (mayor demanda media en 2026)")
    for ax in axes:
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Viajes por día")
        ax.grid(alpha=0.3)
        ax.legend()
    return save_figure(fig, path)


def plot_exploratory_figures(data: pd.DataFrame, poi_columns: list[str], output_dir: Path) -> dict:
    """Guardar las gráficas exploratorias del notebook 05 sin mostrar ventanas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["target"], bins=60, edgecolor="black")
    ax.set(title="Distribución de viajes diarios por estación", xlabel="Viajes diarios", ylabel="Frecuencia")
    saved["demand_distribution"] = save_figure(fig, output_dir / "demand_distribution.png")

    weekday = data.groupby("day_of_week")["target"].mean().reindex(range(7))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], weekday)
    ax.set(title="Promedio de viajes por día de la semana", ylabel="Viajes por estación y día")
    saved["weekday_demand"] = save_figure(fig, output_dir / "weekday_demand.png")

    monthly = data.groupby("month")["target"].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, 13), monthly)
    ax.set(title="Promedio de viajes por mes", xlabel="Mes", ylabel="Viajes por estación y día")
    saved["monthly_demand"] = save_figure(fig, output_dir / "monthly_demand.png")

    station_mean = data.groupby("Origen_Id")["target"].mean().sort_values()
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    station_mean.tail(10).plot.bar(ax=axes[0], title="10 estaciones con mayor demanda media")
    station_mean.head(10).plot.bar(ax=axes[1], title="10 estaciones con menor demanda media")
    for ax in axes:
        ax.set(xlabel="Estación", ylabel="Viajes por día")
    fig.tight_layout()
    saved["station_ranking"] = save_figure(fig, output_dir / "station_ranking.png")

    if set(poi_columns).issubset(data.columns):
        station_poi = data.drop_duplicates("Origen_Id").set_index("Origen_Id")[poi_columns]
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        station_poi.reindex(station_mean.tail(10).index).plot.bar(stacked=True, ax=axes[0], title="POI en estaciones de mayor demanda")
        station_poi.reindex(station_mean.head(10).index).plot.bar(stacked=True, ax=axes[1], title="POI en estaciones de menor demanda")
        fig.tight_layout()
        saved["station_poi"] = save_figure(fig, output_dir / "station_poi.png")

    if {"temperature_2m_mean", "precipitation_sum"}.issubset(data.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(data["temperature_2m_mean"].dropna(), bins=35)
        axes[0].set(title="Distribución de temperatura (EDA)", xlabel="Temperatura media (°C)")
        axes[1].hist(data["precipitation_sum"].dropna(), bins=35)
        axes[1].set(title="Distribución de precipitación (EDA)", xlabel="Precipitación (mm)")
        saved["weather_distribution"] = save_figure(fig, output_dir / "weather_distribution.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        temp_bins = pd.cut(data["temperature_2m_mean"], bins=10)
        data.groupby(temp_bins, observed=True)["target"].mean().plot.bar(ax=axes[0])
        axes[0].set(title="Demanda media por temperatura (EDA)", xlabel="Rango de temperatura")
        rain_bins = [-0.001, 0, 1, 5, 20, float("inf")]
        rain_labels = ["Sin lluvia", "0-1", "1-5", "5-20", ">20"]
        rain = pd.cut(data["precipitation_sum"], bins=rain_bins, labels=rain_labels, include_lowest=True)
        data.groupby(rain, observed=True)["target"].mean().plot.bar(ax=axes[1])
        axes[1].set(title="Demanda media por lluvia (EDA)", xlabel="Precipitación (mm)")
        fig.tight_layout()
        saved["weather_vs_demand"] = save_figure(fig, output_dir / "weather_vs_demand.png")
    return saved


def plot_additional_eda(data: pd.DataFrame, poi_columns: list[str], output_dir: Path) -> dict:
    """Recuperar las visualizaciones de calidad y EDA del notebook 05."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    annual = data.assign(year=data["Fecha"].dt.year).groupby("year").agg(
        records=("target", "size"),
        stations=("Origen_Id", "nunique"),
        trips=("target", "sum"),
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(annual.index.astype(str), annual["stations"])
    ax.set(title="Estaciones con registros por año", xlabel="Año", ylabel="Número de estaciones")
    saved["annual_station_coverage"] = save_figure(fig, output_dir / "annual_station_coverage.png")

    # Las fechas son únicas por estación y día según el contrato de datos.
    station_dates = data[["Origen_Id", "Fecha"]].sort_values(["Origen_Id", "Fecha"])
    previous = station_dates.groupby("Origen_Id")["Fecha"].shift()
    gap_days = (station_dates["Fecha"] - previous).dt.days.sub(1)
    gap_days = gap_days.loc[gap_days > 0]
    gap_groups = pd.cut(
        gap_days,
        bins=[0, 1, 7, 30, float("inf")],
        labels=["1 día", "2 a 7 días", "8 a 30 días", "Más de 30 días"],
    )
    gap_counts = gap_groups.value_counts(sort=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(gap_counts.index.astype(str), gap_counts.values)
    ax.set(title="Periodos consecutivos sin registros", xlabel="Duración", ylabel="Cantidad de periodos")
    saved["missing_periods"] = save_figure(fig, output_dir / "missing_periods.png")

    # Diferencia de cobertura por fecha sin expandir todos los intervalos faltantes.
    # Para cada estación se cuentan los días desde su primera hasta su última fecha.
    station_span = data.groupby("Origen_Id")["Fecha"].agg(["min", "max"])
    dates = pd.date_range(data["Fecha"].min(), data["Fecha"].max(), freq="D")
    starts = station_span["min"].value_counts().reindex(dates, fill_value=0).cumsum()
    ends = station_span["max"].value_counts().reindex(dates, fill_value=0).cumsum().shift(1, fill_value=0)
    expected = starts.sub(ends)
    observed = data.groupby("Fecha")["Origen_Id"].nunique().reindex(dates, fill_value=0)
    missing = expected.sub(observed).clip(lower=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(missing.index, missing.values)
    ax.set(title="Estaciones sin registros por fecha (dentro de su periodo observado)",
           xlabel="Fecha", ylabel="Estaciones sin registros")
    ax.grid(alpha=0.3)
    saved["shared_missing_dates"] = save_figure(fig, output_dir / "shared_missing_dates.png")

    day_of_month = data.groupby(data["Fecha"].dt.day)["target"].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(day_of_month.index, day_of_month.values, marker="o")
    ax.set(title="Demanda promedio según el día del mes", xlabel="Día del mes",
           ylabel="Viajes promedio por estación-día observado")
    ax.set_xticks(range(1, 32, 2))
    ax.grid(alpha=0.2)
    saved["day_of_month_demand"] = save_figure(fig, output_dir / "day_of_month_demand.png")

    # Evolución de la demanda por mes calendario (distinta del promedio por mes del año).
    monthly = data.groupby(data["Fecha"].dt.to_period("M"))["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(monthly.index.to_timestamp(), monthly.values)
    ax.set(title="Evolución mensual de la demanda media por estación", xlabel="Mes",
           ylabel="Viajes promedio por estación-día observado")
    ax.grid(alpha=0.3)
    saved["monthly_demand_trend"] = save_figure(fig, output_dir / "monthly_demand_trend.png")

    if set(poi_columns).issubset(data.columns):
        station = data.groupby("Origen_Id")["target"].mean().rename("mean_target").to_frame()
        station_poi = data.drop_duplicates("Origen_Id").set_index("Origen_Id")[poi_columns]
        station = station.join(station_poi, how="inner")
        correlations = station[poi_columns].corrwith(station["mean_target"]).dropna().sort_values()
        if not correlations.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(correlations.index, correlations.values)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title="Correlación descriptiva entre POI y demanda media por estación",
                   xlabel="Categoría de POI", ylabel="Correlación de Pearson")
            ax.tick_params(axis="x", rotation=45)
            saved["poi_demand_correlation"] = save_figure(fig, output_dir / "poi_demand_correlation.png")

        for label, selected in (("top", station.nlargest(10, "mean_target")),
                                ("bottom", station.nsmallest(10, "mean_target"))):
            if selected.empty:
                continue
            totals = selected[poi_columns].sum(axis=1)
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].bar(selected.index.astype(str), selected["mean_target"])
            axes[0].set(title=f"Demanda media diaria: estaciones {label}", ylabel="Viajes diarios")
            axes[1].bar(selected.index.astype(str), totals)
            axes[1].set(title="Cantidad total de POI", xlabel="ID de estación", ylabel="POI en 500 m")
            saved[f"{label}_station_demand_vs_poi"] = save_figure(
                fig, output_dir / f"{label}_station_demand_vs_poi.png"
            )

        attribute_changes = data.groupby("Origen_Id")[["weather_zone"] + poi_columns].nunique()
        inconsistent = attribute_changes.gt(1).sum()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(inconsistent.index, inconsistent.values)
        ax.set(title="Estaciones con atributos de zona o POI variables en el tiempo",
               xlabel="Atributo", ylabel="Estaciones con cambios")
        ax.tick_params(axis="x", rotation=45)
        saved["station_attribute_consistency"] = save_figure(
            fig, output_dir / "station_attribute_consistency.png"
        )

    return saved


def plot_holiday_demand(data: pd.DataFrame, output_dir: Path) -> dict:
    """Reproducir el análisis descriptivo de días oficiales de descanso del notebook 05."""
    from datetime import timedelta

    def nth_monday(year: int, month: int, occurrence: int) -> date:
        first = date(year, month, 1)
        return first + timedelta(days=(7 - first.weekday()) % 7 + 7 * (occurrence - 1))

    records = []
    for year in range(int(data["Fecha"].dt.year.min()), int(data["Fecha"].dt.year.max()) + 1):
        holidays = [
            (date(year, 1, 1), "Año Nuevo"),
            (nth_monday(year, 2, 1), "Constitución"),
            (nth_monday(year, 3, 3), "Natalicio de Benito Juárez"),
            (date(year, 5, 1), "Día del Trabajo"),
            (date(year, 9, 16), "Independencia"),
            (nth_monday(year, 11, 3), "Revolución Mexicana"),
            (date(year, 12, 25), "Navidad"),
        ]
        if year == 2024:
            holidays.append((date(year, 10, 1), "Cambio de gobierno federal"))
        records.extend((pd.Timestamp(day), name) for day, name in holidays)

    holiday_dates = pd.DataFrame(records, columns=["Fecha", "holiday_name"])
    daily = data.groupby("Fecha")["target"].mean().rename("mean_demand").reset_index()
    holidays = daily.merge(holiday_dates, on="Fecha", how="inner", validate="one_to_one")
    summary = holidays.groupby("holiday_name")["mean_demand"].mean().sort_values()
    if summary.empty:
        return {}
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(summary.index, summary.values)
    ax.set(title="Demanda promedio de MiBici por festividad (2020–2026)",
           xlabel="Viajes diarios promedio por estación", ylabel="Festividad")
    return {"holiday_demand": save_figure(fig, output_dir / "holiday_demand.png")}
