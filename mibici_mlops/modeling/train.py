"""Pipeline de Scikit-Learn, modelos, búsqueda de hiperparámetros y modelo final."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from mibici_mlops.config import (
    BEST_PARAMS,
    FINAL_MODEL,
    HGB_INITIAL_PARAMS,
    HISTORY_COLUMNS,
    ONEHOT_COLUMNS,
    RF_INITIAL_PARAMS,
    SEED,
    STATION_ID_COLUMNS,
)


def build_model_pipeline(
    model,
    columns,
    station_id_columns=STATION_ID_COLUMNS,
    onehot_columns=ONEHOT_COLUMNS,
    history_columns=HISTORY_COLUMNS,
    seed=SEED,
) -> Pipeline:
    # Preprocesamiento según la tabla de tratamiento de variables, más el modelo.
    station = [c for c in columns if c in station_id_columns]
    categorical = [c for c in columns if c in onehot_columns]
    history = [c for c in columns if c in history_columns]
    numeric = [c for c in columns if c not in station + categorical + history]

    transformers = []
    if station:
        transformers.append(
            ("station", TargetEncoder(target_type="continuous", random_state=seed), station)
        )
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    if history:
        transformers.append((
            "history",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            history,
        ))
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def make_model(model_name: str, params: dict | None = None):
    if model_name == "Random Forest":
        return RandomForestRegressor(**{**RF_INITIAL_PARAMS, **(params or {})})
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**{**HGB_INITIAL_PARAMS, **(params or {})})
    raise ValueError(f"Modelo desconocido: {model_name}")


def run_grid_search(model_name, grid, data, columns, splits, target_column="target"):
    # Búsqueda de hiperparámetros con las ventanas temporales; 2026 no participa.
    search = GridSearchCV(
        estimator=build_model_pipeline(make_model(model_name), columns),
        param_grid=grid,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        refit=False,
        cv=splits,
        n_jobs=1,
        verbose=1,
    )
    return search.fit(data[columns], data[target_column])


def tuning_table(searches: dict) -> pd.DataFrame:
    tables = []
    for model_name, search in searches.items():
        results = pd.DataFrame(search.cv_results_)
        results["model"] = model_name
        results["mae"] = -results["mean_test_mae"]
        results["mae_std"] = results["std_test_mae"]
        results["rmse"] = -results["mean_test_rmse"]
        tables.append(results[["model", "params", "mae", "mae_std", "rmse"]])
    return pd.concat(tables, ignore_index=True).sort_values("mae", ignore_index=True)


def best_configuration(tuning_results: pd.DataFrame | None) -> tuple[str, dict]:
    # Mejor combinación de la búsqueda o, sin búsqueda, la guardada en config.
    if tuning_results is None or tuning_results.empty:
        return FINAL_MODEL, BEST_PARAMS[FINAL_MODEL]
    best = tuning_results.iloc[0]
    params = {key.replace("model__", ""): value for key, value in best["params"].items()}
    return best["model"], params


def train_final_model(model_name, params, data, columns, target_column="target") -> Pipeline:
    pipeline = build_model_pipeline(make_model(model_name, params), columns)
    return pipeline.fit(data[columns], data[target_column])


def save_model(pipeline: Pipeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path
