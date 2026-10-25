# Equipo 8 - MiBici MLOps

Proyecto MLOps para analizar los viajes del sistema MiBici de Guadalajara y preparar los datos necesarios para desarrollar un modelo que permita predecir la demanda diaria por estación con tres días de anticipación.

El objetivo es que estas predicciones puedan ayudar a identificar diferencias de demanda entre estaciones y apoyar la planeación de la distribución de bicicletas.

Este repositorio utiliza como base la estructura de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), adaptada a las necesidades del proyecto.

## 1. Datos del proyecto

Para este proyecto utilizamos los datos abiertos de MiBici correspondientes al periodo de enero de 2020 a septiembre de 2026. En total, trabajamos con 81 archivos CSV mensuales y un catálogo de estaciones, con más de 27 millones de registros de viajes.

Los archivos originales se encuentran en una carpeta compartida de Google Drive, ya que por su tamaño no se incluyen en GitHub.

**[Consultar archivos CSV originales en Google Drive](https://drive.google.com/drive/folders/1n83lQ_nEH9u7KvKAYNRcnUhae5tjoAYB?usp=drive_link)**

Para ejecutar el procesamiento, los archivos deben descargarse y colocarse en la carpeta `data/raw/` del proyecto.

Durante el procesamiento se generan archivos Parquet que permiten trabajar con los datos procesados sin tener que leer nuevamente todos los CSV. Los archivos originales se conservan sin modificaciones.

## 2. Project Organization

El proyecto sigue la estructura de Cookiecutter Data Science, con algunas modificaciones para organizar el procesamiento, el análisis exploratorio, las pruebas y el seguimiento de ejecuciones.

```text
equipo-8-mibici-mlops/
├── data/
│   ├── raw/                         # Archivos CSV originales
│   ├── interim/                     # Datos intermedios
│   └── processed/                   # Datos procesados
├── docs/                            # Documentación
├── mibici_mlops/                    # Código Python del proyecto
│   ├── __init__.py
│   ├── config.py                    # Configuración y rutas
│   ├── ingestion.py                 # Lectura de archivos CSV
│   ├── checks.py                    # Validaciones de datos
│   ├── preprocessing.py            # Limpieza y transformación
│   ├── dataset.py                  # Preparación y partición de datos
│   ├── features.py                 # Construcción de variables
│   ├── eda.py                      # Análisis exploratorio
│   ├── plots.py                    # Gráficas
│   ├── pipeline.py                 # Organización del procesamiento
│   ├── main.py                     # Ejecución principal
│   ├── tracking.py                 # Registro en MLflow
│   ├── reporting.py                # Reportes
│   └── modeling/                   # Modelado en futuras fases
│       ├── __init__.py
│       ├── train.py
│       └── predict.py
├── notebooks/
│   ├── 01_MiBici_EDA_Functional.ipynb
│   ├── 02_MiBici_EDA_OOP.ipynb
│   ├── 03_MiBici_Refactored_V1.ipynb
│   └── 04_MiBici_Refactored_V2.ipynb
├── tests/                           # Pruebas automatizadas
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_splitting.py
│   ├── test_functional.py
│   ├── test_oop.py
│   └── test_behavior.py
├── models/                          # Modelos de futuras fases
├── references/                      # Material de referencia
├── reports/
│   └── validation/                  # Resultados de las ejecuciones
├── .gitignore
├── MLproject                        # Configuración de MLflow Projects
├── pyproject.toml                   # Configuración de Python
├── python_env.yaml                  # Entorno de MLflow Projects
├── requirements.txt                 # Dependencias
└── README.md                        # Documentación principal
```

Los archivos CSV, los Parquet y los resultados de las ejecuciones no se incluyen en GitHub debido a su tamaño. Algunos archivos de código pueden encontrarse todavía en ramas pendientes de integración a `main`.

## 3. Trabajo realizado en la Fase 1

Durante esta fase realizamos las siguientes actividades:

- Revisión de la estructura y calidad de los datos.
- Identificación de valores faltantes, duplicados e inconsistencias.
- Comparación de estrategias de limpieza estricta y selectiva.
- Procesamiento de los CSV por bloques.
- Construcción de variables relacionadas con los viajes.
- Cálculo de la demanda por estación y hora.
- Generación de archivos Parquet.
- Análisis exploratorio de la demanda.
- Comparación de implementaciones con funciones y programación orientada a objetos (OOP).
- Refactorización del código.
- Registro de ejecuciones y resultados mediante MLflow.

En esta primera fase no se entrenaron modelos predictivos.

## 4. Notebooks

| Notebook | Descripción |
|---|---|
| `01_MiBici_EDA_Functional.ipynb` | Procesamiento y análisis exploratorio mediante funciones. |
| `02_MiBici_EDA_OOP.ipynb` | Procesamiento y análisis exploratorio con OOP. |
| `03_MiBici_Refactored_V1.ipynb` | Versión funcional refactorizada. |
| `04_MiBici_Refactored_V2.ipynb` | Versión OOP refactorizada. |

## 5. Instalación

El proyecto utiliza Python y un entorno virtual para administrar las dependencias.

Desde la carpeta principal del repositorio, en Linux o WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Ejecución del procesamiento

Los archivos CSV originales deben encontrarse en `data/raw/`.

El procesamiento se organiza desde `mibici_mlops/main.py`, que coordina la lectura de los archivos, la preparación de los datos, la generación de Parquet, el análisis exploratorio y el registro en MLflow.

Para ejecutar el pipeline desde la carpeta principal del proyecto:

```bash
python -m mibici_mlops.main
```

Este comando requiere que `main.py` termine con la llamada a `main()` dentro del bloque `if __name__ == "__main__":`.

El procesamiento puede utilizar todos los archivos disponibles o solamente los primeros dos, dependiendo de la configuración establecida en `config.py`.

Los resultados se guardan en `reports/validation/`, dentro de una carpeta identificada por la fecha y hora de cada ejecución.

Entre los resultados generados se encuentran:

- `hourly_demand.parquet`: conteos de viajes por estación y hora.
- `hourly_complete.parquet`: demanda horaria incluyendo periodos sin salidas.
- `splits/`: particiones temporales de los datos.
- `eda_figures/`: gráficas del análisis exploratorio.

## 7. Seguimiento con MLflow

Utilizamos MLflow para registrar y consultar información de las ejecuciones del proyecto.

Entre los elementos registrados se encuentran:

- Parámetros de procesamiento.
- Métricas y resultados.
- Información de los datasets y sus particiones.
- Hashes SHA256 de los archivos originales.
- Gráficas y artefactos.
- Información del entorno de ejecución.

Para consultar los experimentos desde la carpeta principal del proyecto:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Después, abrir en el navegador:

`http://localhost:5000`

Los registros de MLflow se almacenan localmente. Los metadatos y hashes permiten revisar las ejecuciones, pero no sustituyen el almacenamiento de los archivos originales.

## 8. Pruebas

El proyecto incluye pruebas automatizadas con `pytest`.

Para ejecutarlas desde la carpeta principal del repositorio:

```bash
python -m pytest tests/
```

Estas pruebas permiten comprobar el comportamiento de distintas partes del proyecto después de realizar cambios en el código.

## 9. Siguientes etapas

Para las siguientes fases tenemos previsto:

- Obtener la demanda diaria por estación.
- Preparar las variables para el modelado.
- Desarrollar y comparar modelos predictivos.
- Reducir los tiempos de procesamiento y mejorar los notebooks.
- Seguir evaluando el uso de funciones y OOP.
- Investigar otras variables, como clima, calendario y características del entorno, si contamos con los datos y el tiempo necesarios.

## 10. Referencias

- [MiBici](https://www.mibici.net/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [MLflow](https://mlflow.org/)
- [pytest](https://docs.pytest.org/)