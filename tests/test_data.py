import pandas as pd
import pytest

from mibici_mlops.checks import check_integration
from mibici_mlops.dataset import cache_is_current, save_cache_manifest
from mibici_mlops.versioning import version_csv_sources


def test_cache_detects_changed_file(tmp_path):
    data_path = tmp_path / "sample.parquet"
    data_path.write_bytes(b"first version")
    signature = {"period": "2020-2026", "schema": 1}
    save_cache_manifest(data_path, signature)
    assert cache_is_current(data_path, signature)
    assert not cache_is_current(data_path, {"period": "2021-2026", "schema": 1})
    data_path.write_bytes(b"changed version")
    assert not cache_is_current(data_path, signature)


def test_csv_versioning_keeps_history(tmp_path):
    source, versions = tmp_path / "sources", tmp_path / "versions"
    source.mkdir()
    csv_path = source / "sample.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")
    first = version_csv_sources(source, versions)
    assert len(first) == 1
    assert first.iloc[0]["version"] == 1
    assert version_csv_sources(source, versions).iloc[0]["known_versions"] == 1
    csv_path.write_text("value\n2\n", encoding="utf-8")
    second = version_csv_sources(source, versions)
    assert second.iloc[0]["version"] == 2
    assert second.iloc[0]["known_versions"] == 2
    assert csv_path.read_text(encoding="utf-8") == "value\n2\n"


def test_integration_rejects_changed_demand():
    daily = pd.DataFrame({
        "Origen_Id": [1, 1],
        "Fecha": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "target": [10, 20],
    })
    integrated = daily.copy()
    check_integration(daily, integrated)
    integrated.loc[0, "target"] = 11
    with pytest.raises(ValueError):
        check_integration(daily, integrated)
