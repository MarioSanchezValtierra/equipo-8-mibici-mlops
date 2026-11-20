"""Construir la imagen del servicio, comprobarla con la muestra de referencia y publicarla en DockerHub."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import requests

from mibici_mlops.config import (
    AUTO_DEPLOY,
    DOCKERHUB_REPO,
    FEATURE_COLUMNS,
    IMAGE_NAME,
    PROJECT_ROOT,
    RELEASES,
    REPRODUCIBILITY_DIR,
    SERVING_DIR,
    TARGET_COLUMN,
)
from mibici_mlops.modeling.evaluate import regression_metrics

CONTAINER = "mibici-api"
PORT = 8000
TOLERANCE = 1e-4


def run(command: list[str], check: bool = True) -> str:
    """Ejecutar un comando y devolver su salida."""
    print("$", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 and check:
        print(output[-3000:])
        raise RuntimeError(f"Falló: {' '.join(command)}")
    return output


def docker_available() -> bool:
    """Docker Desktop (o el servicio de Docker en WSL) tiene que estar abierto."""
    try:
        output = run(["docker", "version", "--format", "{{.Server.Version}}"], check=False)
    except FileNotFoundError:
        return False
    return not any(text in output for text in ("failed to connect", "Cannot connect", "error during connect"))


def image_tag(model_version: str, releases: dict = RELEASES) -> str:
    """Etiqueta fija de cada champion; uno nuevo recibe la versión menor siguiente a la más alta."""
    if str(model_version) in releases:
        return releases[str(model_version)]
    return f"1.{max(int(tag.split('.')[1]) for tag in releases.values()) + 1}.0"


def to_instance(row: dict) -> dict:
    """Una fila del dataset en el formato JSON que recibe /predict."""
    instance = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            instance[column] = None
        elif column.startswith(("lag_", "roll_")):
            instance[column] = float(value)
        else:
            instance[column] = int(value)
    return instance


def build_image(image: str = IMAGE_NAME, context: Path = PROJECT_ROOT) -> None:
    """Equivale a docker build -t ml-service:latest . desde la raíz del repositorio."""
    start = time.perf_counter()
    run(["docker", "build", "-t", image, str(context)])
    print(f"Imagen {image} construida en {time.perf_counter() - start:.0f} s")


def start_container(image: str = IMAGE_NAME, container: str = CONTAINER, port: int = PORT,
                    timeout: int = 60) -> dict:
    """Levantar el contenedor y esperar a que /health diga que el modelo está cargado."""
    run(["docker", "rm", "-f", container], check=False)
    run(["docker", "run", "-d", "--name", container, "-p", f"{port}:8000", image])
    for _ in range(timeout):
        try:
            health = requests.get(f"http://localhost:{port}/health", timeout=2).json()
            if health.get("model_loaded"):
                return health
        except requests.RequestException:
            pass
        time.sleep(1)
    print(run(["docker", "logs", container], check=False)[-2000:])
    raise RuntimeError("El contenedor no respondió")


def check_container(reference: pd.DataFrame, port: int = PORT) -> tuple[dict, pd.DataFrame]:
    """Predecir la muestra de referencia dentro del contenedor y compararla con la del modelo."""
    payload = {"instances": [to_instance(r) for r in reference.to_dict(orient="records")]}
    start = time.perf_counter()
    response = requests.post(f"http://localhost:{port}/predict", json=payload, timeout=60)
    latency = time.perf_counter() - start
    body = response.json()
    served = np.array([p["predicted_trips"] for p in body["predictions"]])
    expected = reference["reference_prediction"].clip(lower=0).to_numpy()
    difference = np.abs(served - expected)
    local_metrics = regression_metrics(reference[TARGET_COLUMN], expected)
    docker_metrics = regression_metrics(reference[TARGET_COLUMN], served)
    summary = {
        "model_version": body["model_version"], "model_uri": body["model_uri"],
        "rows": len(reference), "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()), "tolerance": TOLERANCE,
        "reproducible": bool(difference.max() < TOLERANCE),
        "local_mae": local_metrics["mae"], "docker_mae": docker_metrics["mae"],
        "local_rmse": local_metrics["rmse"], "docker_rmse": docker_metrics["rmse"],
        "latency_seconds_500_rows": round(latency, 3),
    }
    comparison = reference[["Fecha", "Origen_Id", TARGET_COLUMN]].assign(
        local_prediction=expected, docker_prediction=served, abs_difference=difference)
    return summary, comparison


def is_published(tag: str, repo: str = DOCKERHUB_REPO) -> bool:
    result = subprocess.run(["docker", "manifest", "inspect", f"{repo}:{tag}"], capture_output=True)
    return result.returncode == 0


def publish(tag: str, image: str = IMAGE_NAME, repo: str = DOCKERHUB_REPO) -> bool:
    """Subir la etiqueta fija y latest. Una etiqueta que ya existe no se sobrescribe."""
    if is_published(tag, repo):
        print(f"{repo}:{tag} ya está publicada; no se sobrescribe.")
        return False
    for name in [tag, "latest"]:
        run(["docker", "tag", image, f"{repo}:{name}"])
        output = run(["docker", "push", f"{repo}:{name}"], check=False)
        if "denied" in output or "unauthorized" in output.lower() or "requested access" in output:
            print("No se pudo publicar: falta iniciar sesión con docker login.")
            return False
    print("Publicada: https://hub.docker.com/r/" + repo)
    return True


def main(auto_deploy: bool = AUTO_DEPLOY) -> None:
    """Construir, comprobar y, si auto_deploy es True, publicar la imagen del champion exportado."""
    if not docker_available():
        raise RuntimeError("Docker no está abierto. Abre Docker Desktop y vuelve a intentarlo.")
    model_version = json.loads((SERVING_DIR / "metadata.json").read_text())["model_version"]
    tag = image_tag(model_version)
    reference = pd.read_csv(SERVING_DIR / "reference_sample.csv", parse_dates=["Fecha"])

    build_image()
    try:
        start_container()
        summary, comparison = check_container(reference)
    finally:
        run(["docker", "rm", "-f", CONTAINER], check=False)
    REPRODUCIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPRODUCIBILITY_DIR / "docker_comparison.csv", index=False)
    (REPRODUCIBILITY_DIR / "docker_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not summary["reproducible"]:
        print("La imagen no reproduce la muestra de referencia; no se publica.")
    elif auto_deploy:
        publish(tag)
    else:
        state = "ya publicada" if is_published(tag) else "sin publicar"
        print(f"AUTO_DEPLOY = False: no se publica. {DOCKERHUB_REPO}:{tag} está {state}.")


if __name__ == "__main__":
    main()
