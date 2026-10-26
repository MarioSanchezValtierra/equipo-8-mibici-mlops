"""Tablas y gráficas del EDA de la versión funcional."""

import re
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
