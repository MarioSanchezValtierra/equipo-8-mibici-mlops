"""Evidencia del champion en MLflow: métricas, gráficas y configuración, con la estructura de la Fase 2."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mibici_mlops import mlflow_catalog, tracking
from mibici_mlops.config import (
    CHALLENGER_EXPERIMENT,
    CHAMPION_REPORTS_DIR,
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    INTEGRATED_PATH,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
    SEED,
    TARGET_COLUMN,
)
from mibici_mlops.features import HOLIDAY_DATES, HOLIDAY_MONDAYS
from mibici_mlops.modeling.challenger import evaluate_month
from mibici_mlops.modeling.evaluate import regression_metrics

CHAMPION_ARTIFACT_GUIDE = """# Guía de artefactos del champion

- `modeling/comparison`: selección en septiembre de la v7 y los challengers, con su gráfica junto a la prueba de octubre.
- `modeling/test`: champion contra la v7 en octubre de 2026, error diario y predicción contra viajes reales.
- `modeling/interpretation`: importancia de variables del champion.
- `data`: inventario, versiones y procedencia de los datos, igual que en la Fase 2.
- `artifact_catalog.json`: lista de archivos con su origen y SHA-256.
- `features.json` y `documentation/model_configuration.json`: variables, hiperparámetros, regla de días festivos y periodo de entrenamiento.
- `environment` y `documentation/run_configuration.json`: entorno, commit y datos, igual que en la Fase 2.

El modelo está en el Model Registry (`mibici_salidas_diarias`, alias `champion`). Esta ejecución documenta
esa versión; no registra una versión nueva. En Model metrics, `october_daily_mae` es una serie por día
(paso 0 = 1 de octubre de 2026) para el champion y `october_daily_mae_v7` para la v7.
"""


def feature_importance(model) -> pd.DataFrame:
    """Importancia de cada variable del Random Forest después del preprocesamiento."""
    preprocessing, forest = model.named_steps["preprocessing"], model.named_steps["model"]
    try:
        names = [name.split("__", 1)[-1] for name in preprocessing.get_feature_names_out()]
    except (AttributeError, ValueError):
        names = [f"variable_{i}" for i in range(len(forest.feature_importances_))]
    table = pd.DataFrame({"variable": names, "importancia": forest.feature_importances_})
    return table.sort_values("importancia", ascending=False, ignore_index=True)


def build_champion_report(champion_model, champion_version, reference_version, month: pd.DataFrame,
                          selection_table: pd.DataFrame | None = None,
                          output_dir: Path = CHAMPION_REPORTS_DIR) -> dict:
    """Comparar el champion con la v7 en el mes monitoreado y guardar tablas y gráficas.

    month es la tabla del mes con las columnas predicted (v7), naive y new_station.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    champion_pred = np.clip(champion_model.predict(month[FEATURE_COLUMNS]), 0, None)
    reference_label = f"v{reference_version} (en uso en octubre)"
    champion_label = f"v{champion_version} (champion)"
    comparison = pd.DataFrame([
        evaluate_month(reference_label, month, month["predicted"].to_numpy(), "2026-10-12"),
        evaluate_month(champion_label, month, champion_pred, "2026-10-12"),
    ]).set_index("modelo")
    comparison.to_csv(output_dir / "october_comparison.csv")

    errors = month[["Fecha", TARGET_COLUMN]].assign(
        v7=(month["predicted"] - month[TARGET_COLUMN]).abs(), champion=np.abs(champion_pred - month[TARGET_COLUMN]))
    daily = errors.groupby("Fecha")[["v7", "champion"]].mean()
    daily.to_csv(output_dir / "october_daily_mae.csv")
    importance = feature_importance(champion_model)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    if selection_table is not None:
        selection_table.to_csv(output_dir / "selection_september.csv")

    # Selección y prueba
    fig, axes = plt.subplots(1, 2 if selection_table is not None else 1, figsize=(12, 4), squeeze=False)
    if selection_table is not None:
        axes[0, 0].bar(selection_table.index, selection_table["mae"], color=["gray", "tab:blue", "tab:green"])
        axes[0, 0].set_title("Selección: MAE en septiembre de 2026")
        axes[0, 0].tick_params(axis="x", rotation=15)
    ax = axes[0, -1]
    ax.bar(comparison.index, comparison["mae"], color=["gray", "tab:green"])
    ax.set_title("Prueba: MAE en octubre de 2026")
    for ax in axes[0]:
        ax.set_ylabel("viajes por estación-día")
    fig.tight_layout()
    fig.savefig(output_dir / "selection_and_test.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["v7"], label=reference_label, color="gray")
    ax.plot(daily.index, daily["champion"], label=champion_label, color="tab:green")
    ax.set_title("Error absoluto medio por día, octubre de 2026")
    ax.set_ylabel("viajes por estación-día")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "october_daily_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    limit = float(np.quantile(month[TARGET_COLUMN], 0.995))
    ax.scatter(month[TARGET_COLUMN], champion_pred, s=4, alpha=0.3)
    ax.plot([0, limit], [0, limit], color="red", linestyle=":")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_title("Champion: predicción contra viajes reales, octubre de 2026")
    ax.set_xlabel("viajes reales")
    ax.set_ylabel("viajes predichos")
    fig.tight_layout()
    fig.savefig(output_dir / "october_pred_vs_real.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15).iloc[::-1]
    ax.barh(top["variable"], top["importancia"])
    ax.set_title("Importancia de variables del champion")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    return {"comparison": comparison, "daily": daily, "importance": importance,
            "selection_table": selection_table, "output_dir": output_dir, "month": month}


def log_champion(report: dict, champion_version, champion_model, output_dir: Path = CHAMPION_REPORTS_DIR,
                 project_root: Path = PROJECT_ROOT) -> str:
    """Ejecución que documenta al champion y enlace desde su versión en el Model Registry."""
    version = str(champion_version.version)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
    reference_row, champion_row = report["comparison"].iloc[0], report["comparison"].iloc[1]
    improvement = 1 - champion_row["mae"] / reference_row["mae"]
    forest = champion_model.named_steps["model"]
    candidate = champion_version.tags.get("candidate") or type(forest).__name__
    train_end = champion_version.tags.get("train_end", "")
    summary = f"{candidate}, entrenado hasta {train_end}" if train_end else candidate

    data_path = project_root / INTEGRATED_PATH.relative_to(PROJECT_ROOT)
    data_sha256 = tracking.calculate_sha256(data_path)
    run_name = f"Modelo champion | Random Forest | v{version}"
    tracking.setup_mlflow(MLFLOW_TRACKING_URI, CHALLENGER_EXPERIMENT, project_root / MLFLOW_ARTIFACT_DIR.name)
    mlflow.set_experiment(CHALLENGER_EXPERIMENT)
    mlflow_catalog.mark_previous_runs(CHALLENGER_EXPERIMENT, run_name)
    with mlflow.start_run(run_name=run_name,
                          tags=mlflow_catalog.source_tags("mibici_mlops/modeling/champion_report.py")) as run:
        tracking.log_run_context(project_root, data_path, data_sha256)
        mlflow.set_tags({"stage": "champion", "phase": "fase_3", "model_role": "champion",
                         "model_version": version, "model_family": type(forest).__name__,
                         "registered_model_name": REGISTERED_MODEL_NAME,
                         "registered_model_version": version, "registered_model_alias": MODEL_ALIAS,
                         "model_uri": model_uri, "model_run_id": champion_version.run_id,
                         "candidate": candidate, "train_end": train_end})
        mlflow_catalog.log_data_lineage(project_root, data_path, data_sha256, "champion_test", {
            "testing": ("mibici_monitoreo_2026_10", report["month"][[TARGET_COLUMN] + FEATURE_COLUMNS],
                        "data/monitoring/2026/datos_abiertos_2026_10.csv"),
        })
        mlflow.set_tag("mlflow.note.content", (
            f"Champion vigente: {model_uri} ({summary}). En octubre de 2026 "
            f"baja el MAE {improvement:.1%} frente a la v7 ({reference_row['mae']:.3f} a {champion_row['mae']:.3f}). "
            "Es el modelo que sirve la API y la imagen de Docker."
        ))
        mlflow.log_text(CHAMPION_ARTIFACT_GUIDE, "README.md")
        mlflow.log_params({f"model_{k}": v for k, v in forest.get_params().items()
                           if k in ("n_estimators", "max_depth", "max_features", "max_samples", "min_samples_leaf")})
        mlflow.log_params({"registered_model_version": version, "n_features": len(FEATURE_COLUMNS),
                           "holiday_feature": champion_model.steps[0][0] == "holidays"})
        mlflow.log_dict({
            "model_family": type(forest).__name__,
            "model_parameters": {k: str(v) for k, v in forest.get_params().items()},
            "pipeline_steps": [name for name, _ in champion_model.steps],
            "features": FEATURE_COLUMNS,
            "derived_features": ["is_holiday"] if champion_model.steps[0][0] == "holidays" else [],
            "holiday_rule": {"dates_month_day": HOLIDAY_DATES, "mondays_month_n": HOLIDAY_MONDAYS},
            "target": TARGET_COLUMN,
            "forecast_horizon_days": HORIZON_DAYS,
            "train_end": train_end,
            "selection_period": "2026-09",
            "test_period": "2026-10",
            "random_seed": SEED,
        }, "documentation/model_configuration.json")
        mlflow.log_dict({"features": FEATURE_COLUMNS}, "features.json")

        metrics = {f"oct_{k}": float(v) for k, v in champion_row.items()}
        metrics.update({f"oct_v7_{k}": float(v) for k, v in reference_row.items()})
        metrics["oct_mae_improvement_vs_v7"] = float(improvement)
        if report["selection_table"] is not None:
            for name, row in report["selection_table"].iterrows():
                prefix = "v7" if name.startswith("v") else ("challenger_b" if "B" in name else "challenger_a")
                metrics.update({f"sep_{prefix}_{k}": float(v) for k, v in row.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        for step, row in enumerate(report["daily"].itertuples(index=False)):
            mlflow.log_metric("october_daily_mae", float(row.champion), step=step)
            mlflow.log_metric("october_daily_mae_v7", float(row.v7), step=step)
        mlflow_catalog.log_artifact_catalog({
            "modeling/comparison": [output_dir / "selection_september.csv", output_dir / "selection_and_test.png"],
            "modeling/test": [output_dir / name for name in (
                "october_comparison.csv", "october_daily_mae.csv", "october_daily_error.png",
                "october_pred_vs_real.png")],
            "modeling/interpretation": [output_dir / "feature_importance.csv", output_dir / "feature_importance.png"],
        }, project_root)

    registry = mlflow.MlflowClient()
    registry.update_model_version(REGISTERED_MODEL_NAME, version, description=(
        f"Champion: {summary}. Octubre 2026: MAE {champion_row['mae']:.3f} "
        f"contra {reference_row['mae']:.3f} de la v7. Evidencia en la ejecución {run.info.run_id} "
        f"del experimento {CHALLENGER_EXPERIMENT}."))
    registry.set_model_version_tag(REGISTERED_MODEL_NAME, version, "documentation_run", run.info.run_id)
    return run.info.run_id


def main() -> None:
    """Documentar en MLflow la versión que tiene el alias champion."""
    import mlflow.sklearn

    from mibici_mlops import monitoring
    from mibici_mlops.config import DRIFT_REPORTS_DIR

    context = monitoring.load_context()
    month, _, _ = monitoring.prepare_month(context["model"], context["integrated"], context["development"],
                                           monitoring.MONITOR_CSV, monitoring.MONITOR_START,
                                           monitoring.MONITOR_END)
    version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    champion_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version.version}")
    selection_path = DRIFT_REPORTS_DIR / "challenger_selection_september.csv"
    selection = pd.read_csv(selection_path, index_col=0) if selection_path.is_file() else None
    report = build_champion_report(champion_model, version.version, context["model_version"], month, selection)
    run_id = log_champion(report, version, champion_model)
    print(report["comparison"].round(3).to_string())
    print(f"Champion v{version.version} documentado en la ejecución {run_id}")


if __name__ == "__main__":
    main()
