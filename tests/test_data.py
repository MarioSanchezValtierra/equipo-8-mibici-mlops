import hashlib

from mibici_mlops.dataset import file_sha256


def test_file_sha256(tmp_path):
    # Crear un archivo temporal.
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("MiBici", encoding="utf-8")

    # Calcular el hash esperado.
    expected_hash = hashlib.sha256(
        b"MiBici"
    ).hexdigest()

    # Ejecutar nuestra función.
    actual_hash = file_sha256(file_path)

    # Verificar el resultado.
    assert actual_hash == expected_hash