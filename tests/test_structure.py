from pathlib import Path

MODULES = [
    "config.py", "ingestion.py", "preprocessing.py", "dataset.py", "external.py",
    "features.py", "checks.py", "eda.py", "plots.py", "reporting.py", "pipeline.py",
    "tracking.py", "versioning.py", "workflow.py",
    "modeling/train.py", "modeling/evaluate.py", "modeling/predict.py",
]


def test_source_modules_exist():
    root = Path(__file__).resolve().parents[1] / "mibici_mlops"
    missing = [name for name in MODULES if not (root / name).is_file()]
    assert not missing, f"Faltan módulos: {missing}"


def test_modules_import():
    import importlib

    for name in MODULES:
        importlib.import_module("mibici_mlops." + name[:-3].replace("/", "."))