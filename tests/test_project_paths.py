"""Comprobar que las rutas no dependen del directorio de ejecución."""

from pathlib import Path

from mibici_mlops.config import DATA_DIR, MLFLOW_ARTIFACT_DIR, PROJECT_ROOT


def test_project_root_matches_package_location():
    package_directory = Path(__import__("mibici_mlops").__file__).resolve().parent
    assert PROJECT_ROOT == package_directory.parent


def test_project_paths_are_inside_repository():
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert MLFLOW_ARTIFACT_DIR == PROJECT_ROOT / "mlruns"
