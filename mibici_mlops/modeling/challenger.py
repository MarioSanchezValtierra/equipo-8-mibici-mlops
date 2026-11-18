"""Champion contra challenger: selección en un mes, prueba en el siguiente y regla de promoción."""

from pathlib import Path
import time

import cloudpickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mibici_mlops import features as features_module
from mibici_mlops import mlflow_catalog, monitoring, tracking
from mibici_mlops.config import (
    AUTO_PROMOTE,
    CHALLENGER_EXPERIMENT,
    DRIFT_REPORTS_DIR,
    FEATURE_COLUMNS,
    INTEGRATED_PATH,
    MIN_IMPROVEMENT,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    PROJECT_ROOT,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
)
from mibici_mlops.features import add_holidays
from mibici_mlops.modeling.evaluate import predict_seasonal_naive, regression_metrics
from mibici_mlops.modeling.train import build_model_pipeline

SELECTION_START = pd.Timestamp("2026-09-01")
CANDIDATES = {"Challenger A (reentrenado)": False, "Challenger B (+ festivos)": True}
SOURCE_TAG = "challenger_octubre_2026"
MONTH_NAMES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio", 7: "julio",
               8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"}
RF_LOGGED_PARAMS = ("n_estimators", "max_depth", "max_features", "max_samples", "min_samples_leaf")


def make_challenger(base_model, with_holidays: bool) -> Pipeline:
    """Pipeline con los hiperparámetros del modelo de referencia; B calcula is_holiday dentro."""
    if not with_holidays:
        return build_model_pipeline(clone(base_model), FEATURE_COLUMNS)
    base = build_model_pipeline(clone(base_model), FEATURE_COLUMNS + ["is_holiday"])
    return Pipeline([("holidays", FunctionTransformer(add_holidays)), *base.steps])


def fit_challenger(name: str, data: pd.DataFrame, base_model) -> Pipeline:
    start = time.perf_counter()
    model = make_challenger(base_model, CANDIDATES[name]).fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    print(f"{name}: {len(data):,} filas hasta {data['Fecha'].max().date()} | {time.perf_counter() - start:.0f} s")
    return model


def month_metrics(name: str, data: pd.DataFrame, predictions, naive) -> dict:
    """MAE, RMSE, mejora frente al ingenuo y sesgo de un modelo en el mes de selección."""
    predictions = np.clip(predictions, 0, None)
    model = regression_metrics(data[TARGET_COLUMN], predictions)
    return {"modelo": name, "mae": model["mae"], "rmse": model["rmse"],
            "mejora_vs_ingenuo": 1 - model["mae"] / regression_metrics(data[TARGET_COLUMN], naive)["mae"],
            "sesgo": float((predictions - data[TARGET_COLUMN]).mean())}


def evaluate_month(name: str, month: pd.DataFrame, predictions, special_day: str) -> dict:
    """Métricas del mes de prueba, por tipo de estación y en un día especial de referencia."""
    predictions = np.clip(predictions, 0, None)
    y = month[TARGET_COLUMN]
    known, new = ~month["new_station"], month["new_station"]
    special = month["Fecha"] == special_day
    model = regression_metrics(y, predictions)
    return {
        "modelo": name, "mae": model["mae"], "rmse": model["rmse"],
        "mejora_vs_ingenuo": 1 - model["mae"] / regression_metrics(y, month["naive"])["mae"],
        "mae_con_historia": regression_metrics(y[known], predictions[known])["mae"],
        "mae_nuevas": regression_metrics(y[new], predictions[new])["mae"],
        "mae_12_oct": regression_metrics(y[special], predictions[special])["mae"],
        "sesgo": float((predictions - y).mean()),
    }


def run_challenger(reference_model, reference_version, dataset: pd.DataFrame, month: pd.DataFrame,
                   selection_start: pd.Timestamp, train_end: pd.Timestamp,
                   output_dir: Path = DRIFT_REPORTS_DIR, special_day: str = "2026-10-12") -> dict:
    """Elegir entre A y B en el mes de selección y probar al elegido contra la referencia en month."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = reference_model.named_steps["model"]

    # 1. Selección: A y B entrenados hasta el día anterior a selection_start
    train_selection = dataset.loc[dataset["Fecha"] < selection_start]
    selection_month = dataset.loc[(dataset["Fecha"] >= selection_start) & (dataset["Fecha"] < train_end)]
    selection_models = {name: fit_challenger(name, train_selection, base_model) for name in CANDIDATES}
    selection_naive = predict_seasonal_naive(train_selection, selection_month, TARGET_COLUMN)
    selection_table = pd.DataFrame(
        [month_metrics(f"v{reference_version} (referencia)", selection_month,
                       reference_model.predict(selection_month[FEATURE_COLUMNS]), selection_naive)]
        + [month_metrics(name, selection_month, model.predict(selection_month[FEATURE_COLUMNS]), selection_naive)
           for name, model in selection_models.items()]
    ).set_index("modelo")
    best_name = selection_table.loc[list(CANDIDATES), "mae"].idxmin()
    selection_table.to_csv(output_dir / "challenger_selection_september.csv")

    # 2. Prueba: el elegido se reentrena hasta train_end y se compara con la referencia en month
    train_recent = dataset.loc[dataset["Fecha"] < train_end]
    final_model = fit_challenger(best_name, train_recent, base_model)
    month_name = MONTH_NAMES[month["Fecha"].min().month]
    train_month = MONTH_NAMES[train_recent["Fecha"].max().month]
    comparison = pd.DataFrame([
        evaluate_month(f"v{reference_version} (en uso en {month_name})", month,
                       month["predicted"].to_numpy(), special_day),
        evaluate_month(f"{best_name}, entrenado hasta {train_month}", month,
                       final_model.predict(month[FEATURE_COLUMNS]), special_day),
    ]).set_index("modelo")
    comparison.to_csv(output_dir / "challenger_test_october.csv")

    reference_row, challenger_row = comparison.iloc[0], comparison.iloc[1]
    improvement = 1 - challenger_row["mae"] / reference_row["mae"]
    return {
        "selection_table": selection_table, "best_name": best_name, "comparison": comparison,
        "final_model": final_model, "base_model": base_model, "train_rows": len(train_recent),
        "train_end": str(train_recent["Fecha"].max().date()), "improvement": improvement,
        "wins": bool(improvement >= MIN_IMPROVEMENT and challenger_row["rmse"] <= reference_row["rmse"]),
    }


def promotion_decision(results: dict, alias_version, reference_version) -> tuple[str, str]:
    """Aplicar la regla y revisar si el champion actual ya salió de esta comparación."""
    reference_row, challenger_row = results["comparison"].iloc[0], results["comparison"].iloc[1]
    best_name, improvement, wins = results["best_name"], results["improvement"], results["wins"]
    from_this_comparison = alias_version.tags.get("source") == SOURCE_TAG
    same_candidate = alias_version.tags.get("candidate") == best_name

    if wins and from_this_comparison and same_candidate:
        return "ya promovido", f"el champion v{alias_version.version} es {best_name} entrenado hasta septiembre"
    if wins and from_this_comparison:
        return "revisar con el equipo", (f"gana {best_name}, pero el champion v{alias_version.version} es "
                                         f"{alias_version.tags.get('candidate')}")
    if wins:
        return "promover", (f"MAE {improvement:.1%} menor que v{reference_version} y RMSE "
                            f"{reference_row['rmse']:.2f} a {challenger_row['rmse']:.2f}")
    if from_this_comparison:
        return "revisar con el equipo", (f"el challenger ya no gana a la v{reference_version}, pero el "
                                         f"champion es v{alias_version.version}")
    return "mantener champion", (f"mejora de MAE {improvement:.1%} (mínimo {MIN_IMPROVEMENT:.0%}) o RMSE peor "
                                 f"({reference_row['rmse']:.2f} a {challenger_row['rmse']:.2f})")


CHALLENGER_ARTIFACT_GUIDE = """# Guía de artefactos de champion contra challenger

- `modeling/comparison`: MAE, RMSE y mejora frente al ingenuo de la v7 y los dos challengers en septiembre.
- `modeling/test`: el mejor challenger contra la v7 en octubre de 2026.
- `data`: inventario, versiones y procedencia de los datos, igual que en la Fase 2.
- `environment` y `documentation`: entorno, commit y configuración, igual que en la Fase 2.
- `artifact_catalog.json`: lista de archivos con su origen y SHA-256.

La decisión, el candidato y el motivo están en las etiquetas `decision`, `candidate` y `reason`.
"""
CHALLENGER_RUN = "Champion contra challenger | Random Forest | octubre 2026"


def log_challenger(results: dict, decision: str, reason: str, alias_version, reference_uri: str,
                   input_example: pd.DataFrame, auto_promote: bool = AUTO_PROMOTE,
                   output_dir: Path = DRIFT_REPORTS_DIR, project_root: Path = PROJECT_ROOT,
                   month: pd.DataFrame | None = None) -> dict:
    """Registrar la comparación en MLflow y, si se promueve, la versión nueva con el alias champion."""
    promote_now = decision == "promover" and auto_promote
    best_name = results["best_name"]
    data_path = project_root / INTEGRATED_PATH.relative_to(PROJECT_ROOT)
    data_sha256 = tracking.calculate_sha256(data_path)
    tracking.setup_mlflow(MLFLOW_TRACKING_URI, CHALLENGER_EXPERIMENT, project_root / MLFLOW_ARTIFACT_DIR.name)
    mlflow.set_experiment(CHALLENGER_EXPERIMENT)
    mlflow_catalog.mark_previous_runs(CHALLENGER_EXPERIMENT, CHALLENGER_RUN)
    with mlflow.start_run(run_name=CHALLENGER_RUN,
                          tags=mlflow_catalog.source_tags("mibici_mlops/modeling/challenger.py")) as run:
        tracking.log_run_context(project_root, data_path, data_sha256)
        mlflow.set_tag("mlflow.note.content", (
            f"Comparación del champion contra dos challengers. Selección en septiembre, prueba en octubre contra "
            f"{reference_uri}. Decisión: {decision} ({reason})."
        ))
        mlflow.set_tags({"stage": "champion_challenger", "phase": "fase_3", "model_role": "challenger",
                         "model_version": str(alias_version.version), "source": SOURCE_TAG, "reference_uri": reference_uri, "deployed_champion": alias_version.version,
                         "candidate": best_name, "decision": decision, "reason": reason,
                         "auto_promote": str(auto_promote), "selection_period": "2026-09",
                         "test_period": "2026-10", "train_end": results["train_end"]})
        mlflow.log_params({"min_improvement": MIN_IMPROVEMENT, "train_rows": results["train_rows"],
                           "holidays": CANDIDATES[best_name],
                           **{f"rf_{k}": v for k, v in results["base_model"].get_params().items()
                              if k in RF_LOGGED_PARAMS}})
        for name, row in results["selection_table"].iterrows():
            prefix = "v7" if name.startswith("v") else ("challenger_b" if "B" in name else "challenger_a")
            mlflow.log_metrics({f"sep_{prefix}_{k}": float(v) for k, v in row.items()})
        for prefix, (_, row) in zip(["v7", "challenger"], results["comparison"].iterrows()):
            mlflow.log_metrics({f"oct_{prefix}_{k}": float(v) for k, v in row.items()})
        mlflow.log_metrics({"mae_improvement": results["improvement"],
                            "challenger_wins": float(results["wins"])})
        datasets = {}
        if month is not None:
            datasets["testing"] = ("mibici_monitoreo_2026_10", month[[TARGET_COLUMN] + FEATURE_COLUMNS],
                                   monitoring.MONITOR_CSV.name)
        mlflow_catalog.log_data_lineage(project_root, data_path, data_sha256, "challenger_test", datasets)
        mlflow.log_text(CHALLENGER_ARTIFACT_GUIDE, "README.md")
        mlflow.log_dict({"features": FEATURE_COLUMNS}, "features.json")
        mlflow_catalog.log_artifact_catalog({
            "modeling/comparison": [output_dir / "challenger_selection_september.csv"],
            "modeling/test": [output_dir / "challenger_test_october.csv"],
        }, project_root)
        model_info = None
        if promote_now:
            # add_holidays va dentro del modelo porque la imagen no tiene mibici_mlops
            cloudpickle.register_pickle_by_value(features_module)
            try:
                model_info = mlflow.sklearn.log_model(
                    results["final_model"], name="model", registered_model_name=REGISTERED_MODEL_NAME,
                    serialization_format="cloudpickle", input_example=input_example,
                    signature=infer_signature(input_example, results["final_model"].predict(input_example)),
                )
            finally:
                cloudpickle.unregister_pickle_by_value(features_module)

    new_version = None
    if model_info is not None:
        new_version = model_info.registered_model_version
        version_tags = {"source": SOURCE_TAG, "decision": decision, "reason": reason,
                        "replaces_version": alias_version.version, "candidate": best_name,
                        "train_end": results["train_end"]}
        mlflow.MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", new_version)
        tracking.promote_model(REGISTERED_MODEL_NAME, new_version, MODEL_ALIAS,
                               f"{best_name} entrenado hasta {results['train_end']}. {reason}.", version_tags)
    return {"run_id": run.info.run_id, "new_version": new_version, "promoted": new_version is not None}


def main(auto_promote: bool = AUTO_PROMOTE) -> None:
    """Comparar challengers contra la v7 y promover solo si gana y auto_promote es True."""
    context = monitoring.load_context()
    month, _, _ = monitoring.prepare_month(context["model"], context["integrated"], context["development"],
                                           monitoring.MONITOR_CSV, monitoring.MONITOR_START,
                                           monitoring.MONITOR_END)
    results = run_challenger(context["model"], context["model_version"], context["dataset"], month,
                             SELECTION_START, monitoring.MONITOR_START)
    alias_version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    decision, reason = promotion_decision(results, alias_version, context["model_version"])
    logged = log_challenger(results, decision, reason, alias_version, context["model_uri"],
                            month[FEATURE_COLUMNS].head(5), auto_promote, month=month)
    print(results["comparison"].round(3).to_string())
    print(f"Decisión: {decision} | {reason}")
    if logged["promoted"]:
        print(f"Alias {MODEL_ALIAS}: versión {alias_version.version} cambia a {logged['new_version']}")


if __name__ == "__main__":
    main()
