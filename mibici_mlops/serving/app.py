"""API de predicción de salidas diarias por estación de MiBici (Equipo 8)."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "1.1.0"
MAX_INSTANCES = 5_000
MODEL_URI = os.getenv("MODEL_URI", "/app/models/serving/model")
UI_DAYS_FILE = "ui_days.csv"

EXAMPLE_INSTANCE = {
    "Origen_Id": 35, "day_of_week": 2, "month": 10, "day_of_month": 15, "weather_zone": 3,
    "education_count": 4, "transport_count": 1, "commerce_count": 2, "work_count": 6,
    "health_count": 0, "recreation_count": 3,
    "lag_1": 41, "lag_7": 38, "lag_14": 36, "roll_mean_7": 39.4, "active_days_7": 7,
    "roll_mean_28": 37.9, "active_days_28": 28,
}


class StationDayFeatures(BaseModel):
    """Las 18 variables del modelo para una estación y un día."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_INSTANCE})

    Origen_Id: int = Field(..., gt=0, description="Id de la estación de origen")
    day_of_week: int = Field(..., ge=0, le=6, description="0 = lunes, 6 = domingo")
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    weather_zone: int = Field(..., ge=0, description="Zona meteorológica de la estación")
    education_count: int = Field(..., ge=0, description="POI de educación a 500 m")
    transport_count: int = Field(..., ge=0, description="POI de transporte a 500 m")
    commerce_count: int = Field(..., ge=0, description="POI de comercio a 500 m")
    work_count: int = Field(..., ge=0, description="POI de oficinas a 500 m")
    health_count: int = Field(..., ge=0, description="POI de salud a 500 m")
    recreation_count: int = Field(..., ge=0, description="POI de recreación a 500 m")
    lag_1: float | None = Field(None, ge=0, description="Salidas del día anterior")
    lag_7: float | None = Field(None, ge=0, description="Salidas de hace 7 días")
    lag_14: float | None = Field(None, ge=0, description="Salidas de hace 14 días")
    roll_mean_7: float | None = Field(None, ge=0, description="Promedio de los últimos 7 días")
    active_days_7: int = Field(..., ge=0, le=7, description="Días con registro en los últimos 7")
    roll_mean_28: float | None = Field(None, ge=0, description="Promedio de los últimos 28 días")
    active_days_28: int = Field(..., ge=0, le=28, description="Días con registro en los últimos 28")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid",
                              json_schema_extra={"example": {"instances": [EXAMPLE_INSTANCE]}})
    instances: list[StationDayFeatures] = Field(..., min_length=1, max_length=MAX_INSTANCES)


class Prediction(BaseModel):
    Origen_Id: int
    predicted_trips: float = Field(..., description="Viajes que inician en la estación ese día")


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    model_uri: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class StationDayPrediction(BaseModel):
    Origen_Id: int
    Fecha: str
    predicted_trips: float = Field(..., description="Viajes que predice el modelo")
    actual_trips: float | None = Field(None, description="Viajes que hubo en realidad, si ya se conocen")
    model_version: str
    estimated: bool = Field(..., description="True si el día no está en la tabla y se usó la última historia conocida")
    history_end: str = Field(..., description="Último día con viajes reales que se usó para la historia")
    features: dict = Field(..., description="Las 18 variables con las que se predijo")


class LoadedModel:
    def __init__(self, uri: str):
        self.uri = uri
        self.model = mlflow.sklearn.load_model(uri)
        meta_path = Path(uri).parent / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.name = meta.get("model_name", "mibici_salidas_diarias")
        self.version = str(meta.get("model_version", "desconocida"))
        self.features = meta.get("features") or list(self.model.feature_names_in_)
        days_path = Path(uri).parent / UI_DAYS_FILE
        self.days = pd.read_csv(days_path, dtype={"Fecha": str}) if days_path.is_file() else None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(features[self.features]), 0, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = LoadedModel(MODEL_URI)
    except Exception as error:
        app.state.model, app.state.model_error = None, str(error)
    yield


app = FastAPI(
    title="MiBici · Predicción de salidas diarias por estación",
    description="Equipo 8, MLOps. Viajes que iniciarán en cada estación el día siguiente.",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoreo"])
def health(request: Request) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    return HealthResponse(status="ok" if model else "degradado", model_loaded=model is not None,
                          model_name=getattr(model, "name", None),
                          model_version=getattr(model, "version", None))


@app.post("/predict", response_model=PredictResponse, tags=["predicción"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503, f"Modelo no disponible: {request.app.state.model_error}")
    features = pd.DataFrame([row.model_dump() for row in payload.instances])
    history = [c for c in features if c.startswith(("lag_", "roll_"))]
    features[history] = features[history].astype("float64")
    try:
        predictions = model.predict(features)
    except Exception as error:
        raise HTTPException(500, "Error interno al predecir") from error
    return PredictResponse(
        model_name=model.name, model_version=model.version, model_uri=model.uri,
        predictions=[Prediction(Origen_Id=int(s), predicted_trips=round(float(p), 4))
                     for s, p in zip(features["Origen_Id"], predictions)],
    )


def features_from_history(station: pd.DataFrame, date: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    """Variables de un día que no está en la tabla, con la última historia conocida de la estación."""
    # Se calcula como si el día pedido fuera el siguiente al último con datos; el calendario es el del día pedido
    last = station.iloc[-1]
    last_date = pd.Timestamp(last["Fecha"])
    demand = pd.Series(station["target"].to_numpy(), index=pd.to_datetime(station["Fecha"]))
    row = {column: last[column] for column in columns}
    row.update({"day_of_week": date.day_of_week, "month": date.month, "day_of_month": date.day})
    for column in columns:
        if column.startswith("lag_"):
            row[column] = demand.get(last_date - pd.Timedelta(days=int(column[4:]) - 1), np.nan)
        elif column.startswith(("roll_mean_", "active_days_")):
            window = int(column.rsplit("_", 1)[1])
            recent = demand.loc[last_date - pd.Timedelta(days=window - 1):last_date]
            row[column] = recent.mean() if column.startswith("roll_mean_") else len(recent)
    return pd.DataFrame([row])[columns].astype("float64")


@app.get("/predict/estacion", response_model=StationDayPrediction, tags=["interfaz"])
def predict_station_day(request: Request,
                        origen_id: int = Query(..., gt=0, description="Id de la estación"),
                        fecha: str = Query(..., description="Día a predecir, AAAA-MM-DD")) -> StationDayPrediction:
    """Predicción para una estación y un día."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503, f"Modelo no disponible: {request.app.state.model_error}")
    if model.days is None:
        raise HTTPException(503, f"No se encontró {UI_DAYS_FILE} junto al modelo")
    try:
        date = pd.Timestamp(fecha)
    except ValueError as error:
        raise HTTPException(422, f"Fecha inválida: {fecha}. Usa el formato AAAA-MM-DD") from error
    fecha = date.strftime("%Y-%m-%d")

    station = model.days.loc[model.days["Origen_Id"] == origen_id]
    row = station.loc[station["Fecha"] == fecha]
    if not row.empty:
        features, actual, history_end, estimated = row[model.features], row["target"].iloc[0], None, False
    else:
        history = station.loc[station["Fecha"] < fecha]
        if history.empty:
            raise HTTPException(404, f"La estación {origen_id} no tiene historia antes del {fecha}")
        features = features_from_history(history, date, model.features)
        actual, history_end, estimated = None, history["Fecha"].iloc[-1], True
    if history_end is None:
        previous = station.loc[station["Fecha"] < fecha, "Fecha"]
        history_end = previous.iloc[-1] if not previous.empty else fecha

    values = features.iloc[0]
    return StationDayPrediction(
        Origen_Id=origen_id, Fecha=fecha,
        predicted_trips=round(float(model.predict(features)[0]), 2),
        actual_trips=None if actual is None or pd.isna(actual) else float(actual),
        model_version=model.version, estimated=estimated, history_end=history_end,
        features={k: (None if pd.isna(v) else float(v)) for k, v in values.items()},
    )


UI_PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiBici: predicción de salidas</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #1f2933; }
  h1 { font-size: 1.5rem; margin-bottom: .25rem; }
  p.nota { color: #52606d; margin-top: 0; }
  form { display: flex; gap: .75rem; flex-wrap: wrap; align-items: end; margin: 1.5rem 0; }
  label { display: flex; flex-direction: column; font-size: .9rem; gap: .25rem; }
  select, input, button { font-size: 1rem; padding: .45rem .6rem; }
  button { background: #c8102e; color: white; border: 0; border-radius: 4px; cursor: pointer; }
  .resultado { border: 1px solid #d9e2ec; border-radius: 6px; padding: 1rem 1.25rem; }
  .cifra { font-size: 2.25rem; font-weight: 600; }
  table { border-collapse: collapse; margin-top: 1rem; font-size: .9rem; }
  td { padding: .2rem .8rem .2rem 0; }
  .error { color: #c8102e; }
  .aviso { color: #8a5a00; background: #fff7e0; padding: .4rem .6rem; border-radius: 4px; }
</style>
</head>
<body>
<h1>MiBici: viajes que saldrán de una estación</h1>
<p class="nota">Modelo __MODEL__. Hay viajes reales del __START__ al __END__. Para esos días la predicción usa
la demanda real de la estación hasta el día anterior, igual que el pipeline. Para un día posterior se usa
la última historia conocida de la estación y el calendario del día elegido, y se marca como estimación.</p>
<form id="forma">
  <label>Estación <select id="estacion">__OPTIONS__</select></label>
  <label>Día <input type="date" id="fecha" min="__START__" value="__END__"></label>
  <button type="submit">Predecir</button>
</form>
<div id="salida"></div>
<script>
document.getElementById("forma").addEventListener("submit", async (event) => {
  event.preventDefault();
  const estacion = document.getElementById("estacion").value;
  const fecha = document.getElementById("fecha").value;
  const salida = document.getElementById("salida");
  const respuesta = await fetch(`predict/estacion?origen_id=${estacion}&fecha=${fecha}`);
  const datos = await respuesta.json();
  if (!respuesta.ok) { salida.innerHTML = `<p class="error">${datos.detail}</p>`; return; }
  const real = datos.actual_trips === null ? "todavía no se conocen" : datos.actual_trips;
  const aviso = datos.estimated
    ? `<p class="aviso">Estimación: no hay datos de ese día, se usó la historia real hasta el ${datos.history_end}.</p>` : "";
  const v = datos.features;
  salida.innerHTML = `<div class="resultado">
    <div>Estación ${datos.Origen_Id}, ${datos.Fecha}</div>
    <div class="cifra">${datos.predicted_trips} viajes</div>
    ${aviso}
    <div>Viajes reales ese día: ${real}</div>
    <table>
      <tr><td>Viajes el último día con datos (lag_1)</td><td>${v.lag_1 ?? "sin dato"}</td></tr>
      <tr><td>Mismo día de la semana anterior (lag_7)</td><td>${v.lag_7 ?? "sin dato"}</td></tr>
      <tr><td>Promedio de los últimos 7 días</td><td>${v.roll_mean_7 === null ? "sin dato" : v.roll_mean_7.toFixed(1)}</td></tr>
      <tr><td>Promedio de los últimos 28 días</td><td>${v.roll_mean_28 === null ? "sin dato" : v.roll_mean_28.toFixed(1)}</td></tr>
    </table></div>`;
});
</script>
</body>
</html>"""


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui(request: Request) -> HTMLResponse:
    """Página para elegir estación y día y ver la predicción."""
    model = getattr(request.app.state, "model", None)
    if model is None or model.days is None:
        return HTMLResponse("<p>El modelo o el archivo de días no están disponibles. Revisa /health.</p>",
                            status_code=503)
    stations = sorted(model.days["Origen_Id"].unique())
    options = "".join(f'<option value="{s}">{s}</option>' for s in stations)
    page = (UI_PAGE.replace("__OPTIONS__", options)
            .replace("__START__", model.days["Fecha"].min())
            .replace("__END__", model.days["Fecha"].max())
            .replace("__MODEL__", f"{model.name} v{model.version}"))
    return HTMLResponse(page)


@app.get("/", include_in_schema=False)
def home(request: Request) -> RedirectResponse:
    """La raíz lleva a la página /ui; si no hay archivo de días, a la documentación /docs."""
    model = getattr(request.app.state, "model", None)
    return RedirectResponse("/ui" if model is not None and model.days is not None else "/docs")
