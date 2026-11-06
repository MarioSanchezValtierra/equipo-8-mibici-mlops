"""Orquestación de la Fase 2: llama a los módulos en el mismo orden que el notebook 06.

Este archivo no define lógica de datos ni de modelos: solo conecta las funciones de
`dataset`, `versioning`, `features`, `modeling`, `plots` y `tracking`, mide el tiempo
de cada etapa y registra todo en MLflow.
"""

from collections import Counter
import json
import logging
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")  # sin ventanas: las gráficas solo se guardan como PNG
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mibici_mlops.config import (  # noqa: E402
    BEST_PARAMS,
    COMPARE_FEATURE_SETS,
    CSV_VERSION_DIR,
    DATA_DIR,
    DATA_END,
    DATA_START,
    FEATURE_COLUMNS,
    FEATURE_SETS,
    FILE_VERSION_INDEX,
    FINAL_MODEL_PATH,
    FIRST_CUTOFF,
    FORCE_REBUILD_DATA,
    HISTORY_COLUMNS,
    HORIZON_DAYS,
    INTEGRATED_PATH,
    LAGS,
    MAIN_FEATURE_SET,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_FIGURES_DIR,
    MODEL_NAMES,
    MODEL_REPORTS_DIR,
    MODELING_EXPERIMENT,
    N_FOLDS,
    OUTPUT_VERSION_DIR,
    PARAMETER_GRIDS,
    POI_COLUMNS,
    PROJECT_ROOT,
    PROVENANCE_DIR,
    REFERENCE_RESULTS,
    REGISTERED_MODEL_NAME,
    ROLLING_WINDOWS,
    RUN_GRID_SEARCH,
    RUN_WEATHER_REFERENCE,
    TARGET_COLUMN,
    TEST_START,
    TUNING_FOLD_STEP,
    WEATHER_COLUMNS,
)
from mibici_mlops.dataset import prepare_model_data  # noqa: E402
from mibici_mlops.features import build_model_dataset, split_by_date  # noqa: E402
from mibici_mlops.modeling.evaluate import (  # noqa: E402
    BASELINES,
    NAIVE_BASELINE,
    add_skill,
    describe_splits,
    evaluate_baselines,
    evaluate_by_segment,
    evaluate_walk_forward,
    predict_seasonal_naive,
    summarize_results,
    walk_forward_splits,
)
from mibici_mlops.modeling.predict import load_registered_model, predict_next_day  # noqa: E402
from mibici_mlops.modeling.train import (  # noqa: E402
    best_configuration,
    build_model_pipeline,
    make_model,
    run_grid_search,
    save_model,
    train_final_model,
    tuning_table,
)
from mibici_mlops.plots import (  # noqa: E402
    plot_additional_eda,
    plot_exploratory_figures,
    plot_holiday_demand,
    plot_mae_by_fold,
    plot_test_predictions,
    save_figure,
)
from mibici_mlops.reporting import timed  # noqa: E402
from mibici_mlops.tracking import (  # noqa: E402
    calculate_sha256,
    log_final_model,
    log_grid_search,
    log_walk_forward,
    promote_model,
    setup_mlflow,
)
from mibici_mlops.versioning import (  # noqa: E402
    build_data_provenance,
    collect_data_files,
    version_csv_sources,
    version_used_files,
)

logger = logging.getLogger("mibici_mlops")


def show(title: str, table: pd.DataFrame) -> None:
    # Las tablas se imprimen completas en la terminal, sin depender de IPython.
    print(f"\n{title}\n{table.to_string()}")


def prepare_data(stage_times: Counter) -> dict:
    # Etapa de datos, inventario SHA-256 de los CSV y procedencia del dataset.
    with timed(stage_times, "01_prepare_data"):
        integrated, quality_report = prepare_model_data(force=FORCE_REBUILD_DATA)
    show("Pruebas de calidad T1-T9", quality_report)
    print(f"Dimensiones: {integrated.shape} | Estaciones: {integrated['Origen_Id'].nunique()}")

    with timed(stage_times, "02_csv_source_versioning"):
        csv_inventory = version_csv_sources(DATA_DIR, CSV_VERSION_DIR)
    file_inventory = version_used_files(collect_data_files())
    provenance = build_data_provenance(csv_inventory)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = PROVENANCE_DIR / "data_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CSV inventariados: {len(csv_inventory)} | "
          f"Archivos con SHA-256: {file_inventory['file'].nunique()}")
    return {
        "integrated": integrated,
        "quality_report": quality_report,
        "csv_inventory": csv_inventory,
        "file_inventory": file_inventory,
        "provenance": provenance,
        "provenance_path": provenance_path,
    }


def build_features(integrated: pd.DataFrame, stage_times: Counter) -> dict:
    with timed(stage_times, "02_feature_engineering_and_split"):
        model_dataset = build_model_dataset(
            integrated, HORIZON_DAYS, LAGS, ROLLING_WINDOWS, TARGET_COLUMN
        )
        development, test = split_by_date(model_dataset, TEST_START)
    print("Proporción de valores faltantes en la historia:")
    print(model_dataset[HISTORY_COLUMNS].isna().mean().round(3).to_string())
    for name, part in [("Desarrollo", development), ("Prueba", test)]:
        print(f"{name}: {len(part):,} registros, {part['Fecha'].min().date()} a "
              f"{part['Fecha'].max().date()}, {part['Origen_Id'].nunique()} estaciones")
    return {"model_dataset": model_dataset, "development": development, "test": test}


def run_validation(development: pd.DataFrame, stage_times: Counter) -> dict:
    # Ventanas, líneas base, modelos iniciales, clima opcional y búsqueda opcional.
    with timed(stage_times, "03_validation_splits"):
        splits = walk_forward_splits(development["Fecha"], FIRST_CUTOFF, N_FOLDS, HORIZON_DAYS)
        cv_splits = [(train, validation) for _, train, validation in splits]
    show("Ventanas de validación", describe_splits(development, splits))

    with timed(stage_times, "04_baselines"):
        baseline_results = evaluate_baselines(development, splits, TARGET_COLUMN)
        baseline_results = add_skill(baseline_results, baseline_results)
    show("Líneas base", summarize_results(baseline_results).round(3))

    feature_sets = (
        ["original_11_variables", "calendario_estacion", MAIN_FEATURE_SET]
        if COMPARE_FEATURE_SETS else [MAIN_FEATURE_SET]
    )
    model_runs = {}
    for feature_set in feature_sets:
        columns = FEATURE_SETS[feature_set]
        for model_name in MODEL_NAMES:
            label = f"{model_name} | {feature_set}"
            pipeline = build_model_pipeline(make_model(model_name), columns)
            with timed(stage_times, f"06_cv_{model_name}_{feature_set}"):
                results = evaluate_walk_forward(
                    pipeline, development, splits, columns, label, TARGET_COLUMN
                )
            model_runs[label] = {
                "model_name": model_name,
                "feature_set": feature_set,
                "params": pipeline.named_steps["model"].get_params(),
                "results": add_skill(results, baseline_results),
            }

    model_results = pd.concat([run["results"] for run in model_runs.values()], ignore_index=True)
    all_results = pd.concat([baseline_results, model_results], ignore_index=True)
    model_comparison = summarize_results(all_results)
    show("Comparación de modelos", model_comparison.round(3))

    with timed(stage_times, "07_weather_reference"):
        main_models = model_comparison.loc[
            model_comparison.index.str.endswith(f"| {MAIN_FEATURE_SET}")
        ]
        best_initial_model = main_models.index[0].split(" | ")[0]
        if RUN_WEATHER_REFERENCE:
            weather_set = "historia_y_clima_observado"
            columns = FEATURE_SETS[weather_set]
            label = f"{best_initial_model} | {weather_set}"
            pipeline = build_model_pipeline(make_model(best_initial_model), columns)
            model_runs[label] = {
                "model_name": best_initial_model,
                "feature_set": weather_set,
                "params": pipeline.named_steps["model"].get_params(),
                "results": add_skill(
                    evaluate_walk_forward(
                        pipeline, development, splits, columns, label, TARGET_COLUMN
                    ),
                    baseline_results,
                ),
            }

    grid_searches, tuning_results = {}, None
    with timed(stage_times, "08_hyperparameter_search"):
        if RUN_GRID_SEARCH:
            tuning_splits = cv_splits[::TUNING_FOLD_STEP]
            for model_name, grid in PARAMETER_GRIDS.items():
                grid_searches[model_name] = run_grid_search(
                    model_name, grid, development, FEATURE_COLUMNS, tuning_splits, TARGET_COLUMN
                )
            tuning_results = tuning_table(grid_searches)
            show("Mejores combinaciones", tuning_results.head(10))
        else:
            logger.info("Búsqueda omitida; se usan BEST_PARAMS: %s", BEST_PARAMS)
    final_model_name, final_params = best_configuration(tuning_results)

    with timed(stage_times, "09_final_walk_forward"):
        final_label = f"{final_model_name} | final"
        final_walk_forward = add_skill(
            evaluate_walk_forward(
                build_model_pipeline(make_model(final_model_name, final_params), FEATURE_COLUMNS),
                development, splits, FEATURE_COLUMNS, final_label, TARGET_COLUMN,
            ),
            baseline_results,
        )
    print(f"Modelo final: {final_model_name} {final_params} | "
          f"MAE 12 ventanas: {final_walk_forward['mae'].mean():.3f}")
    return {
        "baseline_results": baseline_results,
        "model_runs": model_runs,
        "all_results": all_results,
        "model_comparison": model_comparison,
        "grid_searches": grid_searches,
        "final_model_name": final_model_name,
        "final_params": final_params,
        "final_label": final_label,
        "final_walk_forward": final_walk_forward,
    }


def train_and_test(development, test, final_model_name, final_params, stage_times) -> dict:
    # Entrenamiento con 2020-2025 y evaluación única en 2026.
    with timed(stage_times, "10_final_fit"):
        final_pipeline = train_final_model(
            final_model_name, final_params, development, FEATURE_COLUMNS, TARGET_COLUMN
        )
    save_model(final_pipeline, FINAL_MODEL_PATH)
    with timed(stage_times, "11_test_prediction"):
        final_predictions = final_pipeline.predict(test[FEATURE_COLUMNS])
    naive_predictions = predict_seasonal_naive(development, test, TARGET_COLUMN).to_numpy()
    segments = np.where(
        test["Origen_Id"].isin(set(development["Origen_Id"])), "Con historia", "Nueva"
    )
    test_results = evaluate_by_segment(
        test[TARGET_COLUMN].to_numpy(), final_predictions, naive_predictions, segments
    )
    show("Prueba 2026", test_results.round(3))

    review = test[["Fecha", "Origen_Id", TARGET_COLUMN]].copy()
    review["predicted_trips"] = final_predictions
    review["naive_prediction"] = naive_predictions
    review["absolute_error"] = (review[TARGET_COLUMN] - review["predicted_trips"]).abs()
    review["segment"] = segments
    return {
        "final_pipeline": final_pipeline,
        "final_predictions": final_predictions,
        "test_results": test_results,
        "prediction_review": review,
    }


def make_figures(integrated, validation, prediction_review, stage_times) -> dict:
    with timed(stage_times, "12_figures"):
        figures = {
            "mae_by_fold": plot_mae_by_fold(
                pd.concat([validation["all_results"], validation["final_walk_forward"]],
                          ignore_index=True),
                [NAIVE_BASELINE, "Promedio estación-día"]
                + [f"{name} | {MAIN_FEATURE_SET}" for name in MODEL_NAMES]
                + [validation["final_label"]],
                MODEL_FIGURES_DIR / "mae_by_fold.png",
            ),
            "test_predictions": plot_test_predictions(
                prediction_review, TARGET_COLUMN, MODEL_FIGURES_DIR / "test_predictions_2026.png"
            ),
        }
        exploration_dir = MODEL_FIGURES_DIR / "exploration"
        figures.update(plot_exploratory_figures(integrated, POI_COLUMNS, exploration_dir))
        figures.update(plot_additional_eda(integrated, POI_COLUMNS, exploration_dir))
        figures.update(plot_holiday_demand(integrated, exploration_dir))
    logger.info("Gráficas guardadas: %s", len(figures))
    return figures


def build_run_context(data: dict, features: dict) -> dict:
    # Todo lo que las funciones de tracking necesitan saber de esta ejecución.
    model_columns = [TARGET_COLUMN] + FEATURE_COLUMNS
    return {
        "project_root": PROJECT_ROOT,
        "data_path": INTEGRATED_PATH,
        "data_sha256": calculate_sha256(INTEGRATED_PATH),
        "provenance_path": data["provenance_path"],
        "phase1_run": data["provenance"]["phase1_run"],
        "csv_inventory_count": len(data["csv_inventory"]),
        "file_inventory_count": int(data["file_inventory"]["file"].nunique()),
        "development_input": mlflow.data.from_pandas(
            features["development"][model_columns], source=str(INTEGRATED_PATH),
            name="mibici_development_2020_2025", targets=TARGET_COLUMN,
        ),
        "test_input": mlflow.data.from_pandas(
            features["test"][model_columns], source=str(INTEGRATED_PATH),
            name="mibici_test_2026", targets=TARGET_COLUMN,
        ),
    }


def log_experiments(data, features, validation, final, figures, stage_times) -> dict:
    # Corridas de validación, modelo final, Model Registry y alias champion.
    with timed(stage_times, "13_mlflow_validation_runs"):
        experiment_id = setup_mlflow(MLFLOW_TRACKING_URI, MODELING_EXPERIMENT, MLFLOW_ARTIFACT_DIR)
        mlflow.set_experiment(experiment_id=experiment_id)
        context = build_run_context(data, features)
        baseline_results = validation["baseline_results"]
        for name in BASELINES:
            log_walk_forward(
                baseline_results.loc[baseline_results["model"] == name],
                f"Línea base | {name}", context, tags={"method": "baseline"},
            )
        for label, run in validation["model_runs"].items():
            log_walk_forward(
                run["results"], label, context, params=run["params"],
                columns=FEATURE_SETS[run["feature_set"]],
                tags={"method": "model", "feature_set": run["feature_set"], "params": "initial"},
            )
        for model_name, search in validation["grid_searches"].items():
            log_grid_search(search, model_name, context)
        log_walk_forward(
            validation["final_walk_forward"], f"Validación | {validation['final_label']}",
            context, params=validation["final_params"], columns=FEATURE_COLUMNS,
            tags={"method": "model", "feature_set": MAIN_FEATURE_SET, "params": "final"},
        )

    comparison = validation["model_comparison"].reset_index()
    fig, ax = plt.subplots(figsize=(11, max(4, len(comparison) * 0.35)))
    ax.barh(comparison.iloc[:, 0].astype(str), comparison["mae"])
    ax.invert_yaxis()
    ax.set(xlabel="MAE promedio (viajes)", title="Comparación de modelos y líneas base")
    fig.tight_layout()
    comparison_path = save_figure(fig, MODEL_FIGURES_DIR / "mlflow_model_comparison.png")

    with timed(stage_times, "14_mlflow_registry"):
        MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_files = []
        for name, table in {
            "model_comparison.csv": validation["model_comparison"].reset_index(),
            "final_walk_forward.csv": validation["final_walk_forward"],
            "test_results_2026.csv": final["test_results"],
            "quality_report.csv": data["quality_report"],
        }.items():
            path = MODEL_REPORTS_DIR / name
            table.to_csv(path, index=False)
            report_files.append(path)
        with timed(stage_times, "14_csv_report_versioning"):
            version_csv_sources(MODEL_REPORTS_DIR, CSV_VERSION_DIR / "model_reports")
        report_files += [
            CSV_VERSION_DIR / "model_reports" / "csv_versions.json",
            CSV_VERSION_DIR / "model_reports" / "csv_versions.csv",
        ]
        output_files = report_files + list(figures.values()) + [comparison_path]
        output_inventory = version_used_files(
            [(path, "resultado_generado", "modelado") for path in output_files]
        )
        OUTPUT_VERSION_DIR.mkdir(parents=True, exist_ok=True)
        output_table = OUTPUT_VERSION_DIR / "file_versions.csv"
        output_index = OUTPUT_VERSION_DIR / "file_versions.json"
        output_inventory.loc[output_inventory["stage"] == "modelado"].to_csv(
            output_table, index=False
        )
        shutil.copy2(FILE_VERSION_INDEX, output_index)

        development = features["development"]
        final_run_id, model_info = log_final_model(
            final["final_pipeline"], validation["final_model_name"], validation["final_params"],
            validation["final_walk_forward"], final["test_results"],
            example=development[FEATURE_COLUMNS].head(5),
            context=context,
            artifacts={
                "files": output_files,
                "fit_seconds": stage_times["10_final_fit"],
                "eda_data": data["integrated"],
                "output_version_table": output_table,
                "output_version_index": output_index,
            },
        )
        model_version = str(model_info.registered_model_version)
        test_all = final["test_results"].set_index("stations").loc["Todas"]
        promote_model(
            REGISTERED_MODEL_NAME, model_version, MODEL_ALIAS,
            description=(
                f"{validation['final_model_name']} para predecir las salidas diarias por "
                f"estación con horizonte de {HORIZON_DAYS} día. Variables: {MAIN_FEATURE_SET} "
                f"({len(FEATURE_COLUMNS)}), sin clima. Entrenado con "
                f"{development['Fecha'].min().date()} a {development['Fecha'].max().date()}."
            ),
            tags={
                "validation_mae": round(validation["final_walk_forward"]["mae"].mean(), 3),
                "test_mae": round(test_all["model_mae"], 3),
                "test_naive_mae": round(test_all["naive_mae"], 3),
                "feature_set": MAIN_FEATURE_SET,
                "data_sha256": context["data_sha256"],
                "data_manifest": "data/lineage/data_provenance.json",
                "source_run_id": final_run_id,
            },
        )
    print(f"Modelo registrado: {REGISTERED_MODEL_NAME}, versión {model_version}, "
          f"alias '{MODEL_ALIAS}' (corrida {final_run_id})")
    return {"final_run_id": final_run_id, "model_version": model_version}


def check_registry(integrated, features, final, stage_times) -> pd.DataFrame:
    # El modelo del Registry debe dar las mismas predicciones que el de memoria.
    test = features["test"]
    with timed(stage_times, "15_registry_prediction"):
        champion = load_registered_model(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        registry_predictions = champion.predict(test[FEATURE_COLUMNS])
        next_date = features["model_dataset"]["Fecha"].max() + pd.Timedelta(days=1)
        next_day = predict_next_day(champion, integrated, next_date)
    if not np.allclose(registry_predictions, final["final_predictions"]):
        raise AssertionError("El modelo del Registry no reproduce las predicciones de 2026.")
    print(f"Predicción para {next_date.date()}: {len(next_day)} estaciones, "
          f"{next_day['predicted_trips'].sum():,.0f} salidas")

    with timed(stage_times, "16_prediction_consistency"):
        last_date = test["Fecha"].max()
        check_day = predict_next_day(champion, integrated, last_date)
        review = final["prediction_review"]
        reference = review.loc[review["Fecha"] == last_date, ["Origen_Id", "predicted_trips"]]
        comparison = reference.merge(
            check_day[["Origen_Id", "predicted_trips"]], on="Origen_Id",
            suffixes=("_test", "_predict"),
        )
    if set(reference["Origen_Id"]) - set(check_day["Origen_Id"]):
        raise AssertionError("Faltan estaciones de evaluación en predict_next_day.")
    if not np.allclose(comparison["predicted_trips_test"], comparison["predicted_trips_predict"]):
        raise AssertionError("predict_next_day no reproduce las predicciones de la prueba.")
    logger.info("Registry y predict_next_day consistentes (%s estaciones)", len(comparison))
    return next_day


def save_timings(stage_times: Counter, final_run_id: str) -> pd.DataFrame:
    # Tabla, métricas y gráfica de tiempos en la corrida del modelo final.
    timing = (
        pd.DataFrame([{"stage": k, "seconds": v} for k, v in stage_times.items()])
        .sort_values("seconds", ascending=False)
        .reset_index(drop=True)
    )
    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timing_path = MODEL_REPORTS_DIR / "execution_times.csv"
    timing.to_csv(timing_path, index=False)
    with mlflow.start_run(run_id=final_run_id):
        mlflow.log_artifact(str(timing_path), artifact_path="performance")
        timing_version = version_used_files([(timing_path, "tiempos_ejecucion", "rendimiento")])
        mlflow.log_dict(timing_version.to_dict(orient="records"), "performance/file_version.json")
        for name, seconds in stage_times.items():
            mlflow.log_metric(f"time_{name}_seconds", float(seconds))
        mlflow.log_metric("time_total_recorded_seconds", float(sum(stage_times.values())))
        mlflow.log_metric("time_stages_count", len(stage_times))
        mlflow.log_dict(
            {
                "unit": "seconds",
                "description": "Tiempo medido por etapa; no incluye trabajo anterior al inicio de los cronómetros.",
                "total_recorded_seconds": float(sum(stage_times.values())),
                "stages": [{"name": k, "seconds": float(v)} for k, v in stage_times.items()],
            },
            "performance/execution_summary.json",
        )
        mlflow.log_text(
            "# Tiempos de ejecución\n\n"
            "Cada `time_<etapa>_seconds` indica segundos medidos para esa etapa.\n"
            "`time_total_recorded_seconds` suma únicamente las etapas instrumentadas.\n"
            "`time_stages_count` es la cantidad de etapas con medición.\n"
            "Consulta `execution_times.csv` para los nombres completos y "
            "`execution_times.png` para compararlos visualmente.\n",
            "performance/README.md",
        )
        fig, ax = plt.subplots(figsize=(11, max(5, len(timing) * 0.32)))
        ax.barh(timing["stage"], timing["seconds"])
        ax.set_xlabel("Tiempo de ejecución (segundos)")
        ax.set_title("Tiempo por etapa del pipeline")
        ax.invert_yaxis()
        fig.tight_layout()
        chart_path = save_figure(fig, MODEL_REPORTS_DIR / "execution_times.png")
        mlflow.log_artifact(str(chart_path), artifact_path="performance")
        chart_version = version_used_files([(chart_path, "grafica_tiempos", "rendimiento")])
        mlflow.log_dict(chart_version.to_dict(orient="records"), "performance/chart_version.json")
    show("Tiempos por etapa (s)", timing.round(2))
    return timing


def compare_with_notebook_05(validation: dict, final: dict) -> pd.DataFrame:
    # Mismas cifras del notebook 05 si no cambió nada del modelo.
    test_by_segment = final["test_results"].set_index("stations")
    obtained = {
        **validation["model_comparison"]["mae"].to_dict(),
        "Modelo final, 12 ventanas de 2025": validation["final_walk_forward"]["mae"].mean(),
        "Modelo final, prueba 2026 (todas)": test_by_segment.loc["Todas", "model_mae"],
        "Ingenuo estacional, prueba 2026 (todas)": test_by_segment.loc["Todas", "naive_mae"],
        "Modelo final, prueba 2026 (nuevas)": (
            test_by_segment.loc["Nueva", "model_mae"]
            if "Nueva" in test_by_segment.index else np.nan
        ),
    }
    table = pd.DataFrame({
        "notebook_05": pd.Series(REFERENCE_RESULTS),
        "workflow": pd.Series(obtained).reindex(list(REFERENCE_RESULTS)),
    })
    table["equivalent"] = (table["workflow"] - table["notebook_05"]).abs() < 0.005
    show("Equivalencia con el notebook 05", table.round(3))
    return table


def run_pipeline() -> dict:
    """Ejecutar la Fase 2 completa: datos, validación, modelo final, MLflow y Registry."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Avisos de versiones de sklearn/MLflow que no cambian resultados; se ocultan para leer la salida.
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    print(f"Raíz del proyecto: {PROJECT_ROOT}")
    print(f"Periodo: {DATA_START.date()} a {DATA_END.date()} | Primer corte: {FIRST_CUTOFF.date()}"
          f" | Prueba desde: {TEST_START.date()}")
    print(f"Variables: {MAIN_FEATURE_SET} ({len(FEATURE_COLUMNS)}), "
          f"clima en el modelo: {any(c in FEATURE_COLUMNS for c in WEATHER_COLUMNS)}")

    stage_times = Counter()
    data = prepare_data(stage_times)
    features = build_features(data["integrated"], stage_times)
    validation = run_validation(features["development"], stage_times)
    final = train_and_test(
        features["development"], features["test"],
        validation["final_model_name"], validation["final_params"], stage_times,
    )
    figures = make_figures(data["integrated"], validation, final["prediction_review"], stage_times)
    registry = log_experiments(data, features, validation, final, figures, stage_times)
    next_day = check_registry(data["integrated"], features, final, stage_times)
    timing = save_timings(stage_times, registry["final_run_id"])
    equivalence = compare_with_notebook_05(validation, final)
    return {
        **registry,
        "test_results": final["test_results"],
        "next_day": next_day,
        "timing": timing,
        "equivalence": equivalence,
    }


if __name__ == "__main__":
    run_pipeline()
