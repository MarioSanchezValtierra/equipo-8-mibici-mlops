"""Monitoreo mensual del modelo: desempeño, data drift, escenarios sintéticos, alertas y acción."""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from mibici_mlops import mlflow_catalog, preprocessing, tracking
from mibici_mlops.config import (
    CHUNK_SIZE,
    DRIFT_REPORTS_DIR,
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    HORIZON_DAYS,
    INTEGRATED_PATH,
    LAGS,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    MONITORING_DATA_DIR,
    MONITORING_EXPERIMENT,
    PHASE2_MODEL_VERSION,
    POI_COLUMNS,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
    ROLLING_WINDOWS,
    SEED,
    TARGET_COLUMN,
    TEST_START,
)
from mibici_mlops.dataset import build_daily_demand
from mibici_mlops.features import build_model_dataset, split_by_date
from mibici_mlops.ingestion import read_csv_chunks
from mibici_mlops.modeling.evaluate import predict_seasonal_naive, regression_metrics

# Mes monitoreado. El escenario 3 baja la demanda desde DROP_START.
MONITOR_START = pd.Timestamp("2026-10-01")
MONITOR_END = pd.Timestamp("2026-11-01")
DROP_START = pd.Timestamp("2026-10-16")
MONITOR_CSV = MONITORING_DATA_DIR / "2026" / "datos_abiertos_2026_10.csv"

# Pruebas estadísticas por variable
KS_THRESHOLD = 0.1
WASSERSTEIN_THRESHOLD = 0.1
DATASET_DRIFT_SHARE = 0.3

# Niveles de alerta: (advertencia, crítica)
MISSING_LEVELS = (0.05, 0.15)     # puntos de lag_1 vacío por arriba de la línea base
BIAS_LEVELS = (0.10, 0.20)        # sesgo como fracción de la demanda media de referencia
PSI_LEVELS = (0.10, 0.25)
OPERATION_LEVELS = (0.15, 0.30)   # cambio relativo de estaciones activas o viajes por día
MAE_ALERT_INCREASE = 0.20
SKILL_ALERT_MIN = 0.20
LEVEL_NAMES = {0: "-", 1: "advertencia", 2: "crítica"}

HISTORY_INPUT_COLUMNS = ["Origen_Id", "Fecha", TARGET_COLUMN, "day_of_week", "month",
                         "weather_zone"] + POI_COLUMNS
MONITORED = ["weather_zone"] + POI_COLUMNS + HISTORY_COLUMNS + [TARGET_COLUMN, "prediction"]
BASELINE_NAME = "Línea base (prueba ene-sep 2026)"
SCENARIO_TYPES = {
    BASELINE_NAME: "referencia",
    "Octubre real": "datos reales",
    "1. Rezagos faltantes": "entrada alterada",
    "2. Medias desplazadas": "entrada alterada",
    "3. Cambio estacional": "cambio real en la demanda",
}


# Datos del mes


def load_month_hourly(csv_path: Path, start: pd.Timestamp, end: pd.Timestamp):
    """Leer y limpiar un CSV mensual con la limpieza de la fase 1; devuelve viajes por hora."""
    original_period = (preprocessing.STUDY_START, preprocessing.STUDY_END)
    preprocessing.STUDY_START, preprocessing.STUDY_END = start, end
    try:
        reader, _ = read_csv_chunks(csv_path, chunk_size=CHUNK_SIZE)
        hourly_parts, summaries, seen_ids = [], [], set()
        for chunk in reader:
            cleaned, summary, new_ids = preprocessing.clean_strategy_b(
                preprocessing.normalize_chunk(chunk), seen_ids)
            seen_ids |= new_ids
            summaries.append(summary)
            hourly_parts.append(preprocessing.count_hourly_demand(
                preprocessing.add_trip_features(cleaned)))
    finally:
        preprocessing.STUDY_START, preprocessing.STUDY_END = original_period
    hourly = pd.concat(hourly_parts).groupby(level=[0, 1]).sum().reset_index()
    return hourly, pd.DataFrame(summaries).sum()


def build_month_daily(hourly: pd.DataFrame, integrated: pd.DataFrame,
                      training_stations: set) -> pd.DataFrame:
    """Demanda diaria del mes con los atributos más recientes de cada estación."""
    daily = build_daily_demand(hourly)
    attributes = (integrated.sort_values("Fecha").groupby("Origen_Id")[["weather_zone"] + POI_COLUMNS]
                  .last().reset_index())
    daily = daily.merge(attributes, on="Origen_Id", how="left", validate="many_to_one")
    daily["new_station"] = ~daily["Origen_Id"].isin(training_stations)
    return daily


def build_month_dataset(integrated: pd.DataFrame, month_daily: pd.DataFrame,
                        start: pd.Timestamp) -> pd.DataFrame:
    """Variables del mes usando la historia del integrado para los rezagos de los primeros días."""
    combined = pd.concat([integrated[HISTORY_INPUT_COLUMNS], month_daily[HISTORY_INPUT_COLUMNS]],
                         ignore_index=True)
    dataset = build_model_dataset(combined, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    month = dataset.loc[dataset["Fecha"] >= start].reset_index(drop=True)
    month["new_station"] = month["Origen_Id"].isin(
        month_daily.loc[month_daily["new_station"], "Origen_Id"])
    return month


def prepare_month(model, integrated: pd.DataFrame, development: pd.DataFrame, csv_path: Path,
                  start: pd.Timestamp, end: pd.Timestamp):
    """Tabla del mes con la predicción del modelo en uso y la del ingenuo estacional."""
    hourly, cleaning = load_month_hourly(csv_path, start, end)
    month_daily = build_month_daily(hourly, integrated, set(development["Origen_Id"]))
    month = build_month_dataset(integrated, month_daily, start)
    month["predicted"] = np.clip(model.predict(month[FEATURE_COLUMNS]), 0, None)
    month["naive"] = predict_seasonal_naive(development, month, TARGET_COLUMN).to_numpy()
    return month, month_daily, cleaning


def monthly_performance(data: pd.DataFrame, predictions, naive) -> pd.DataFrame:
    """MAE, RMSE y mejora frente al ingenuo estacional para cada mes."""
    table = data[["Fecha", TARGET_COLUMN]].assign(pred=predictions, naive=naive)
    rows = []
    for month, part in table.groupby(table["Fecha"].dt.to_period("M")):
        model = regression_metrics(part[TARGET_COLUMN], part["pred"])
        baseline = regression_metrics(part[TARGET_COLUMN], part["naive"])
        rows.append({"mes": str(month), "filas": len(part), "mae": model["mae"],
                     "rmse": model["rmse"], "mae_ingenuo": baseline["mae"],
                     "mejora_vs_ingenuo": 1 - model["mae"] / baseline["mae"]})
    return pd.DataFrame(rows)


# Drift


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Índice de estabilidad con cortes por cuantiles de la referencia."""
    reference, current = reference.dropna(), current.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [reference.min() - 0.5, reference.max() + 0.5]]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """KS, Wasserstein normalizado y PSI por variable; drift si KS o Wasserstein pasan 0.1."""
    rows = []
    for column in columns:
        ref, cur = reference[column].dropna(), current[column].dropna()
        ks = ks_2samp(ref, cur)
        scale = ref.std() if ref.std() > 0 else 1.0
        rows.append({
            "variable": column,
            "media_ref": ref.mean(), "media_actual": cur.mean(),
            "ks": ks.statistic, "ks_pvalue": ks.pvalue,
            "wasserstein_norm": wasserstein_distance(ref, cur) / scale,
            "psi": psi(ref, cur),
        })
    table = pd.DataFrame(rows)
    table["drift"] = (table["ks"] > KS_THRESHOLD) | (table["wasserstein_norm"] > WASSERSTEIN_THRESHOLD)
    return table


def seasonal_diagnosis(drift_vs_test: pd.DataFrame, drift_vs_season: pd.DataFrame) -> pd.DataFrame:
    """Separar cambios de temporada de cambios que también aparecen contra el mismo mes del año pasado."""
    comparison = drift_vs_test[["variable", "media_ref", "media_actual", "ks", "wasserstein_norm",
                                "psi", "drift"]].merge(
        drift_vs_season[["variable", "media_ref", "ks", "wasserstein_norm", "drift"]].rename(columns={
            "media_ref": "media_oct_2025", "ks": "ks_vs_2025",
            "wasserstein_norm": "wasserstein_vs_2025", "drift": "drift_vs_2025"}),
        on="variable",
    )
    comparison["diagnóstico"] = np.select(
        [comparison["drift"] & comparison["drift_vs_2025"],
         comparison["drift"] & ~comparison["drift_vs_2025"],
         ~comparison["drift"] & comparison["drift_vs_2025"]],
        ["revisar", "estacional", "cambio frente a 2025"],
        default="estable",
    )
    return comparison


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> dict:
    """Reporte HTML de Evidently con el mismo umbral de proporción de variables."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset(columns=MONITORED, drift_share=DATASET_DRIFT_SHARE)])
    snapshot = report.run(reference_data=reference[MONITORED], current_data=current[MONITORED])
    snapshot.save_html(str(path))
    summary = snapshot.dict()["metrics"][0]["value"]
    return {"variables_con_drift": int(summary["count"]), "proporción": summary["share"],
            "drift_del_conjunto": summary["share"] > DATASET_DRIFT_SHARE, "reporte": path.name}


# Escenarios sintéticos


def make_scenarios(current: pd.DataFrame, month_daily: pd.DataFrame, integrated: pd.DataFrame,
                   start: pd.Timestamp, drop_start: pd.Timestamp, seed: int = SEED) -> dict:
    """Escenarios 1 y 2 alteran solo la entrada; el 3 cambia la demanda y recalcula los rezagos."""
    rng = np.random.default_rng(seed)
    lag_columns = [f"lag_{lag}" for lag in LAGS]
    shifted_columns = lag_columns + [c for c in HISTORY_COLUMNS if c.startswith("roll_mean")]

    missing_lags = current.copy()
    lost = rng.random(len(missing_lags)) < 0.30
    missing_lags.loc[lost, lag_columns] = np.nan

    shifted_means = current.copy()
    shifted_means[shifted_columns] = shifted_means[shifted_columns] * 1.5

    seasonal_daily = month_daily.copy()
    dropped = seasonal_daily["Fecha"] >= drop_start
    seasonal_daily.loc[dropped, TARGET_COLUMN] = np.round(
        seasonal_daily.loc[dropped, TARGET_COLUMN] * 0.6).astype(int)
    combined = pd.concat([integrated[HISTORY_INPUT_COLUMNS], seasonal_daily[HISTORY_INPUT_COLUMNS]],
                         ignore_index=True)
    seasonal = build_model_dataset(combined, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    seasonal = seasonal.loc[seasonal["Fecha"] >= start].reset_index(drop=True)
    return {"Octubre real": current, "1. Rezagos faltantes": missing_lags,
            "2. Medias desplazadas": shifted_means, "3. Cambio estacional": seasonal}


def evaluate_scenario(model, development: pd.DataFrame, reference: pd.DataFrame,
                      name: str, data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Error, sesgo, rezagos vacíos y proporción de variables con drift de un escenario."""
    data = data.copy()
    data["prediction"] = np.clip(model.predict(data[FEATURE_COLUMNS]), 0, None)
    naive = predict_seasonal_naive(development, data, TARGET_COLUMN)
    metrics = regression_metrics(data[TARGET_COLUMN], data["prediction"])
    drift = drift_table(reference, data, MONITORED)
    row = {
        "escenario": name, "mae": metrics["mae"], "rmse": metrics["rmse"],
        "mejora_vs_ingenuo": 1 - metrics["mae"] / regression_metrics(data[TARGET_COLUMN], naive)["mae"],
        "sesgo": float((data["prediction"] - data[TARGET_COLUMN]).mean()),
        "lag_1_vacío": float(data["lag_1"].isna().mean()),
        "variables_con_drift": int(drift["drift"].sum()),
        "proporción_drift": float(drift["drift"].mean()),
    }
    return row, data


# Alertas y acción


def level(value: float, limits: tuple[float, float]) -> int:
    """0 sin alerta, 1 advertencia, 2 crítica."""
    warning, critical = limits
    return 2 if value > critical else 1 if value > warning else 0


def daily_operation(data: pd.DataFrame, target: str = TARGET_COLUMN) -> tuple[float, float]:
    """Estaciones activas por día y viajes totales por día."""
    by_day = data.groupby("Fecha")
    return by_day["Origen_Id"].nunique().mean(), by_day[target].sum().mean()


def performance_level(mae: float, skill: float, baseline_mae: float) -> int:
    """A3: advertencia con una de las dos condiciones, crítica con las dos."""
    return int(mae > (1 + MAE_ALERT_INCREASE) * baseline_mae) + int(skill < SKILL_ALERT_MIN)


def alert_levels(row: pd.Series, baseline_mae: float, baseline_missing: float,
                 reference_demand: float) -> dict:
    """Nivel de las cinco alertas para un periodo o escenario (una fila de la tabla de monitoreo)."""
    drift = level(row["psi_máximo"], PSI_LEVELS)
    if row["proporción_drift"] > DATASET_DRIFT_SHARE:
        drift = max(drift, 1)
    operation = max(abs(row["cambio_estaciones_activas"]), abs(row["cambio_viajes_por_día"]))
    return {
        "A1_datos_faltantes": level(row["lag_1_vacío"] - baseline_missing, MISSING_LEVELS),
        "A2_sesgo": level(abs(row["sesgo"]) / reference_demand, BIAS_LEVELS),
        "A3_desempeño": performance_level(row["mae"], row["mejora_vs_ingenuo"], baseline_mae),
        "A4_drift": drift,
        "A5_operación": level(operation, OPERATION_LEVELS),
    }


def decide(levels, previous_a3: int = 0) -> str:
    """Acción propuesta. Los problemas de datos van primero porque reentrenar no los arregla."""
    data_problem = max(levels["A1_datos_faltantes"], levels["A2_sesgo"])
    if data_problem == 2:
        return "Revisar el pipeline de datos (no reentrenar)"
    if data_problem == 1:
        return "Revisar los datos de entrada"
    if levels["A3_desempeño"] == 2:
        if previous_a3 == 2:
            return "Reentrenar: A3 crítica dos meses seguidos"
        return "Vigilar y confirmar con el siguiente mes"
    if levels["A5_operación"] == 2:
        return "Revisar con operación el cambio de la red"
    if max(levels.values() if isinstance(levels, dict) else levels) >= 1:
        return "Vigilar"
    return "Sin acción"


# Flujo completo


def run_monitoring(model, integrated: pd.DataFrame, dataset: pd.DataFrame,
                   development: pd.DataFrame, test: pd.DataFrame, csv_path: Path,
                   start: pd.Timestamp, end: pd.Timestamp, drop_start: pd.Timestamp,
                   output_dir: Path = DRIFT_REPORTS_DIR) -> dict:
    """Monitoreo de un mes con el modelo que estuvo en uso; guarda tablas y reportes en output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    month, month_daily, cleaning = prepare_month(model, integrated, development, csv_path, start, end)

    # Desempeño mensual contra el ingenuo estacional
    pred_test = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
    test_naive = predict_seasonal_naive(development, test, TARGET_COLUMN).to_numpy()
    performance = pd.concat([monthly_performance(test, pred_test, test_naive),
                             monthly_performance(month, month["predicted"], month["naive"])],
                            ignore_index=True)
    baseline_metrics = regression_metrics(test[TARGET_COLUMN], pred_test)
    baseline_skill = 1 - baseline_metrics["mae"] / regression_metrics(test[TARGET_COLUMN], test_naive)["mae"]

    # Drift contra la prueba y contra el mismo mes de 2025
    reference_test = test.assign(prediction=pred_test)
    season_start, season_end = start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    same_month_last_year = dataset.loc[(dataset["Fecha"] >= season_start)
                                       & (dataset["Fecha"] < season_end)].copy()
    same_month_last_year["prediction"] = np.clip(
        model.predict(same_month_last_year[FEATURE_COLUMNS]), 0, None)
    current = month.assign(prediction=month["predicted"])
    drift_vs_test = drift_table(reference_test, current, MONITORED)
    drift_vs_season = drift_table(same_month_last_year, current, MONITORED)
    comparison = seasonal_diagnosis(drift_vs_test, drift_vs_season)
    comparison.to_csv(output_dir / "drift_october_2026.csv", index=False)
    evidently = pd.DataFrame([
        {"referencia": "vs_prueba_2026",
         **evidently_report(reference_test, current, output_dir / "evidently_vs_prueba_2026.html")},
        {"referencia": "vs_octubre_2025",
         **evidently_report(same_month_last_year, current, output_dir / "evidently_vs_octubre_2025.html")},
    ])

    # Escenarios
    scenarios = make_scenarios(current, month_daily, integrated, start, drop_start)
    rows, scenario_data = [], {}
    for name, data in scenarios.items():
        row, scenario_data[name] = evaluate_scenario(model, development, reference_test, name, data)
        rows.append(row)
    baseline_row = {
        "escenario": BASELINE_NAME, "mae": baseline_metrics["mae"], "rmse": baseline_metrics["rmse"],
        "mejora_vs_ingenuo": baseline_skill,
        "sesgo": float((reference_test["prediction"] - reference_test[TARGET_COLUMN]).mean()),
        "lag_1_vacío": float(test["lag_1"].isna().mean()), "variables_con_drift": 0,
        "proporción_drift": 0.0,
    }
    scenario_results = pd.DataFrame([baseline_row] + rows)
    scenario_results["mae_vs_base"] = scenario_results["mae"] / baseline_metrics["mae"] - 1
    scenario_results.insert(1, "tipo", scenario_results["escenario"].map(SCENARIO_TYPES))
    scenario_results.to_csv(output_dir / "scenario_results.csv", index=False)

    # Alertas
    reference_demand = reference_test[TARGET_COLUMN].mean()
    baseline_missing = float(test["lag_1"].isna().mean())
    reference_active, reference_trips = daily_operation(test)
    extra = []
    for name, data in {BASELINE_NAME: reference_test, **scenario_data}.items():
        active, trips = daily_operation(data)
        extra.append({
            "escenario": name,
            "psi_máximo": 0.0 if name == BASELINE_NAME else drift_table(reference_test, data, MONITORED)["psi"].max(),
            "estaciones_activas_día": active, "viajes_por_día": trips,
            "cambio_estaciones_activas": active / reference_active - 1,
            "cambio_viajes_por_día": trips / reference_trips - 1,
        })
    monitor_table = scenario_results.merge(pd.DataFrame(extra), on="escenario").set_index("escenario")
    alerts = pd.DataFrame({
        name: alert_levels(row, baseline_metrics["mae"], baseline_missing, reference_demand)
        for name, row in monitor_table.iterrows()
    }).T

    # Acción, con la regla de los dos meses para A3
    previous_month = (start - pd.DateOffset(months=1)).strftime("%Y-%m")
    previous = performance.loc[performance["mes"] == previous_month].iloc[0]
    previous_a3_real = performance_level(previous["mae"], previous["mejora_vs_ingenuo"],
                                         baseline_metrics["mae"])
    previous_a3 = pd.Series(0, index=alerts.index)
    previous_a3["Octubre real"] = previous_a3_real
    decisions = alerts.replace(LEVEL_NAMES)
    decisions["A3_mes_anterior"] = previous_a3.map(LEVEL_NAMES)
    decisions["acción"] = [decide(alerts.loc[name], previous_a3[name]) for name in alerts.index]
    decisions.to_csv(output_dir / "alerts_and_actions.csv")

    results = {
        "cleaning": cleaning, "month_daily": month_daily, "month": month,
        "performance": performance, "baseline_metrics": baseline_metrics,
        "baseline_skill": baseline_skill, "reference_test": reference_test,
        "same_month_last_year": same_month_last_year, "current": current,
        "comparison": comparison, "evidently": evidently, "scenario_results": scenario_results,
        "scenario_data": scenario_data, "monitor_table": monitor_table, "alerts": alerts,
        "previous_a3": previous_a3, "decisions": decisions,
    }
    plot_monitoring(results, output_dir, drop_start)
    return results


def plot_monitoring(results: dict, output_dir: Path, drop_start: pd.Timestamp) -> None:
    """Las tres gráficas del monitoreo: desempeño por mes, error diario por escenario y demanda."""
    performance, baseline_mae = results["performance"], results["baseline_metrics"]["mae"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(performance["mes"], performance["mae"], marker="o", label="MAE del modelo")
    ax1.plot(performance["mes"], performance["mae_ingenuo"], marker="o", linestyle="--", label="MAE del ingenuo")
    ax1.axhline((1 + MAE_ALERT_INCREASE) * baseline_mae, color="red", linestyle=":", label="Umbral A3")
    ax1.set_title("MAE por mes")
    ax1.set_ylabel("viajes por estación-día")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()
    ax2.bar(performance["mes"], performance["mejora_vs_ingenuo"] * 100,
            color=["tab:red" if v < SKILL_ALERT_MIN else "tab:blue" for v in performance["mejora_vs_ingenuo"]])
    ax2.axhline(SKILL_ALERT_MIN * 100, color="red", linestyle=":", label="Umbral A3")
    ax2.set_title("Mejora frente al ingenuo estacional")
    ax2.set_ylabel("%")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_performance.png", dpi=150)
    plt.close(fig)

    daily_mae = pd.DataFrame({
        name: data.assign(error=(data["prediction"] - data[TARGET_COLUMN]).abs()).groupby("Fecha")["error"].mean()
        for name, data in results["scenario_data"].items()
    })
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in daily_mae.columns:
        ax.plot(daily_mae.index, daily_mae[name], label=name)
    ax.axhline(baseline_mae, color="gray", linestyle="--", label="MAE de la línea base")
    ax.axvline(drop_start, color="black", linestyle=":", label="Inicio del cambio estacional")
    ax.set_title("Error absoluto medio por día, octubre de 2026")
    ax.set_ylabel("viajes por estación-día")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "daily_error_scenarios.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 201, 5)
    for name, data in [("Prueba ene-sep 2026", results["reference_test"]),
                       ("Octubre 2025", results["same_month_last_year"]), ("Octubre 2026", results["current"])]:
        ax.hist(data[TARGET_COLUMN].clip(upper=200), bins=bins, density=True, histtype="step",
                linewidth=1.5, label=f"{name} (media {data[TARGET_COLUMN].mean():.1f})")
    ax.set_title("Distribución de la demanda diaria por estación")
    ax.set_xlabel("viajes por estación-día (recortado en 200)")
    ax.set_ylabel("densidad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "demand_distribution.png", dpi=150)
    plt.close(fig)


MONITORING_ARTIFACT_GUIDE = """# Guía de artefactos del monitoreo

- `monitoring/performance`: MAE y mejora frente al ingenuo por mes, de enero a octubre de 2026.
- `monitoring/drift`: drift por variable con su diagnóstico, distribución de la demanda y reportes de Evidently contra la prueba 2026 y contra octubre de 2025.
- `monitoring/scenarios`: resultados de los escenarios sintéticos y su error diario.
- `monitoring/alerts`: nivel de las alertas A1 a A5 por escenario y la acción propuesta.
- `data`: inventario, versiones y procedencia de los datos, igual que en la Fase 2.
- `environment` y `documentation`: entorno, commit y configuración, igual que en la Fase 2.
- `features.json`: variables del modelo.
- `artifact_catalog.json`: lista de archivos con su origen y SHA-256.

En Model metrics, `monthly_mae` y `monthly_skill_vs_naive` son series por mes (paso 0 = enero de 2026).
Cada escenario tiene una ejecución hija con sus métricas y el nivel de cada alerta en las etiquetas.
"""
MONITORING_ARTIFACTS = {
    "monitoring/performance": ["monthly_performance.png"],
    "monitoring/drift": ["drift_october_2026.csv", "demand_distribution.png",
                         "evidently_vs_prueba_2026.html", "evidently_vs_octubre_2025.html"],
    "monitoring/scenarios": ["scenario_results.csv", "daily_error_scenarios.png"],
    "monitoring/alerts": ["alerts_and_actions.csv"],
}
MONITORING_RUN = "Monitoreo | Random Forest v{version} | octubre 2026"


def log_monitoring(results: dict, model_uri: str, csv_name: str, output_dir: Path = DRIFT_REPORTS_DIR,
                   project_root: Path = PROJECT_ROOT) -> str:
    """Registrar el monitoreo en MLflow con la misma estructura de la Fase 2."""
    thresholds = {
        "missing_warning": MISSING_LEVELS[0], "missing_critical": MISSING_LEVELS[1],
        "bias_warning": BIAS_LEVELS[0], "bias_critical": BIAS_LEVELS[1],
        "mae_alert_increase": MAE_ALERT_INCREASE, "skill_alert_min": SKILL_ALERT_MIN,
        "psi_warning": PSI_LEVELS[0], "psi_critical": PSI_LEVELS[1],
        "operation_warning": OPERATION_LEVELS[0], "operation_critical": OPERATION_LEVELS[1],
        "dataset_drift_share": DATASET_DRIFT_SHARE, "ks_threshold": KS_THRESHOLD,
        "wasserstein_threshold": WASSERSTEIN_THRESHOLD,
    }
    metrics = results["baseline_metrics"]
    data_path = project_root / INTEGRATED_PATH.relative_to(PROJECT_ROOT)
    data_sha256 = tracking.calculate_sha256(data_path)
    model_version = model_uri.rsplit("/", 1)[-1]
    run_name = MONITORING_RUN.format(version=model_version)
    model_columns = [TARGET_COLUMN] + FEATURE_COLUMNS
    tracking.setup_mlflow(MLFLOW_TRACKING_URI, MONITORING_EXPERIMENT, project_root / MLFLOW_ARTIFACT_DIR.name)
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    mlflow_catalog.mark_previous_runs(MONITORING_EXPERIMENT, run_name)
    with mlflow.start_run(run_name=run_name, tags=mlflow_catalog.source_tags("mibici_mlops/monitoring.py")) as parent:
        tracking.log_run_context(project_root, data_path, data_sha256)
        mlflow.set_tags({"stage": "monitoring", "phase": "fase_3", "model_uri": model_uri,
                         "model_version": model_version, "monitor_period": "2026-10",
                         "reference_period": "2026-01 a 2026-09", "seasonal_reference": "2025-10"})
        mlflow.set_tag("mlflow.note.content", (
            f"Monitoreo de octubre de 2026 con {model_uri}, el modelo en uso ese mes. Data drift contra la "
            "prueba 2026 y contra octubre de 2025, tres escenarios sintéticos, alertas con dos niveles y "
            "acción propuesta. Cada escenario es una ejecución hija."
        ))
        mlflow_catalog.log_data_lineage(project_root, data_path, data_sha256, "monitoring", {
            "reference": ("mibici_test_2026", results["reference_test"][model_columns], data_path),
            "monitoring": ("mibici_monitoreo_2026_10", results["month"][model_columns], csv_name),
        })
        mlflow.log_text(MONITORING_ARTIFACT_GUIDE, "README.md")
        mlflow.log_dict({"features": FEATURE_COLUMNS}, "features.json")
        mlflow.log_params({**thresholds, "monitor_csv": csv_name, "monitor_rows": len(results["current"]),
                           "new_stations": int(results["month_daily"].loc[
                               results["month_daily"]["new_station"], "Origen_Id"].nunique())})
        mlflow.log_metrics({"baseline_mae": metrics["mae"], "baseline_rmse": metrics["rmse"],
                            "baseline_skill_vs_naive": results["baseline_skill"]})
        for step, row in enumerate(results["performance"].itertuples(index=False)):
            mlflow.log_metric("monthly_mae", float(row.mae), step=step)
            mlflow.log_metric("monthly_skill_vs_naive", float(row.mejora_vs_ingenuo), step=step)
        mlflow_catalog.log_artifact_catalog(
            {folder: [output_dir / name for name in names] for folder, names in MONITORING_ARTIFACTS.items()},
            project_root)
        for name, row in results["monitor_table"].iloc[1:].iterrows():
            with mlflow.start_run(run_name=f"Escenario | Random Forest v{model_version} | {name}", nested=True,
                                  tags=mlflow_catalog.source_tags("mibici_mlops/monitoring.py")):
                mlflow.set_tags({**mlflow_catalog.trace_tags(parent.info.run_id),
                                 "stage": "monitoring_scenario", "phase": "fase_3", "model_uri": model_uri,
                                 "model_version": model_version, "scenario": name,
                                 "scenario_type": SCENARIO_TYPES[name],
                                 "action": results["decisions"].loc[name, "acción"],
                                 **{f"alert_{flag.split('_')[0]}": LEVEL_NAMES[int(level)].replace("-", "sin alerta")
                                    for flag, level in results["alerts"].loc[name].items()}})
                mlflow.log_param("scenario", name)
                mlflow.log_metrics({
                    "mae": row["mae"], "rmse": row["rmse"], "skill_vs_naive": row["mejora_vs_ingenuo"],
                    "bias": row["sesgo"], "lag_1_missing": row["lag_1_vacío"],
                    "drift_share": row["proporción_drift"], "mae_vs_baseline": row["mae_vs_base"],
                    "psi_max": row["psi_máximo"], "active_stations_change": row["cambio_estaciones_activas"],
                    "daily_trips_change": row["cambio_viajes_por_día"],
                    **{flag: float(value) for flag, value in results["alerts"].loc[name].items()},
                })
    return parent.info.run_id


def load_context(model_version: int = PHASE2_MODEL_VERSION) -> dict:
    """Datos integrados, separación temporal y el modelo que estuvo en uso durante el mes."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    integrated = pd.read_parquet(INTEGRATED_PATH)
    dataset = build_model_dataset(integrated, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN)
    development, test = split_by_date(dataset, TEST_START)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    return {"integrated": integrated, "dataset": dataset, "development": development, "test": test,
            "model": mlflow.sklearn.load_model(model_uri), "model_uri": model_uri,
            "model_version": model_version}


def main() -> None:
    """Monitorear octubre de 2026 con la v7 y registrar el resultado en MLflow."""
    context = load_context()
    results = run_monitoring(context["model"], context["integrated"], context["dataset"],
                             context["development"], context["test"], MONITOR_CSV,
                             MONITOR_START, MONITOR_END, DROP_START)
    run_id = log_monitoring(results, context["model_uri"], MONITOR_CSV.name)
    print(results["decisions"][["A3_mes_anterior", "acción"]].to_string())
    print("Ejecución de monitoreo:", run_id)


if __name__ == "__main__":
    main()
