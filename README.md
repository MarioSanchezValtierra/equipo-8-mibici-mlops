# Equipo 8 — MiBici MLOps

Proyecto de la materia de MLOps para analizar los viajes del sistema MiBici de Guadalajara y predecir **cuántos viajes comenzarán en cada estación durante el día siguiente**. La idea es contar con una estimación de las salidas por estación que ayude a planear mejor la disponibilidad y distribución de bicicletas.

### Problema y variable objetivo

En MiBici no todas las estaciones tienen el mismo movimiento. Algunas concentran muchas salidas y otras registran pocas; además, la actividad cambia según el día de la semana, la época del año y el comportamiento reciente de cada estación. Esto complica anticipar dónde podría hacer falta disponer de más bicicletas.

Por eso definimos la variable objetivo **`target` como el número de viajes que comienzan en una estación durante un día**. Cada fila del conjunto de modelado representa una combinación **estación-fecha**. Para construirla, agrupamos los viajes por estación de origen (`Origen_Id`) y día de inicio del viaje; en el dataset diario, el conteo `Cantidad_viajes` se utiliza como `target`.

Elegimos las **salidas**, en lugar de las llegadas o el total de movimientos, porque queremos estimar cuántas bicicletas podrían necesitar los usuarios para iniciar sus recorridos en cada estación. También elegimos la **escala diaria** porque resulta útil para la planeación operativa y permite comparar estaciones y fechas sin depender de las variaciones de cada hora.

El modelo se plantea para **predecir la demanda del día siguiente** (`horizon_days = 1`), utilizando información histórica, variables de calendario y características de las estaciones. Las variables históricas deben calcularse únicamente con datos anteriores al día que se predice, para evitar utilizar información del futuro.

Esta predicción puede apoyar la identificación de estaciones con mayor demanda esperada y la planeación de la redistribución de bicicletas. **No indica directamente cuántas bicicletas habrá disponibles ni cuántos viajes no pudieron realizarse**: estima los viajes iniciados que aparecen en los registros. Para decidir movimientos concretos de bicicletas también harían falta datos de disponibilidad, capacidad y llegadas.

El repositorio utiliza como base [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), adaptado para las Fases 1 y 2.

## Inicio rápido

Hay dos formas de usar el proyecto, según lo que se necesite.

**Solo usar el modelo.** Basta con Docker; no hace falta clonar el repositorio ni descargar datos:

```bash
docker run -d -p 8000:8000 josecval/mibici-ml-service:1.2.0
```

Después abrir en el navegador:

- <http://localhost:8000>: página para elegir estación y día y ver la predicción.
- <http://localhost:8000/docs>: documentación de la API y prueba de `POST /predict`.
- <http://localhost:8000/health>: estado del servicio y versión del modelo.

Más detalle en las secciones 10.4 a 10.6.

**Reproducir todo el proyecto.** Desde Linux o WSL:

```bash
git clone https://github.com/MarioSanchezValtierra/equipo-8-mibici-mlops.git
cd equipo-8-mibici-mlops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. Descargar y acomodar los datos como indica la sección 3.
2. Correr las fases 1 y 2 (sección 6) o, si ya existe `mlflow.db`, exportar el champion: `make export-model`.
3. Correr las pruebas: `pytest -q` (sección 9).
4. Abrir MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` y entrar a <http://127.0.0.1:5000> (secciones 7 y 10.7).
5. Monitoreo de octubre y comparación de modelos: `make monitor`, `make challenger`, `make champion` y `make mlflow-catalog` (sección 10.7).
6. Servicio sin Docker: `make serve`; con Docker: `make docker-build` y `make docker-run` (sección 10).

## Correr desde cero

Pasos para alguien que clona el repositorio por primera vez. `mlflow.db`, `mlruns/`, `models/`, `reports/` y `data/` no están en GitHub: se generan al correr los pasos.

**1. Entorno.**

```bash
git clone -b fase-final https://github.com/MarioSanchezValtierra/equipo-8-mibici-mlops.git
cd equipo-8-mibici-mlops
python3 -m venv .venv
source .venv/bin/activate
make requirements          # pip install -r requirements.txt
pytest -q                  # 79 pasan y 3 se saltan hasta que exista models/serving
```

**2. Datos.** Descargar de la [carpeta compartida en Google Drive](https://drive.google.com/drive/folders/1n83lQ_nEH9u7KvKAYNRcnUhae5tjoAYB?usp=drive_link) y acomodar así (detalle en la sección 3):

| Archivo o carpeta | Ruta en el repositorio |
| --- | --- |
| Carpetas `2020` a `2026` (81 CSV mensuales) | `data/raw/2020/` ... `data/raw/2026/` |
| Carpeta `Manual_datos` | `data/raw/Manual_datos/` |
| `weather_daily_zones_2020_2026.csv` y `mibici_station_weather_zones.csv` | `data/raw/weather/` |
| `mibici_station_pois.csv` | `data/raw/pois/` |
| CSV de viajes de octubre de 2026 (solo para el monitoreo) | `data/monitoring/2026/datos_abiertos_2026_10.csv` |

**3. Correr en este orden.**

```bash
make data                       # Fase 1: limpieza y data/processed/mibici_integrated.parquet
make train                      # Fase 2: entrenamiento, prueba 2026 y registro en MLflow
make monitor                    # monitoreo de octubre 2026, escenarios y alertas
AUTO_PROMOTE=True make challenger   # challengers; registra el nuevo champion si gana
make champion                   # documenta el champion en MLflow
make export-model               # copia el champion a models/serving
pytest -q                       # ahora corren las 82 pruebas
make deploy                     # construye la imagen y compara local contra contenedor (no publica)
make mlflow-catalog             # corrida resumen MiBici_Entrega_Final
make docker-run                 # servicio en http://localhost:8000
```

`make train` también corre la Fase 1 si todavía no existe, así que `make data` es opcional. En una base de MLflow nueva las versiones empiezan en 1: el modelo de la Fase 2 queda como versión 1 (en la base del equipo es la 7) y el champion como versión 2 (en la base del equipo es la 10). `make monitor` detecta esto solo. Sin `AUTO_PROMOTE=True`, `make challenger` solo muestra la decisión y el servicio usa el modelo de la Fase 2.

**4. Revisar.** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` y abrir <http://127.0.0.1:5000>. El servicio queda en <http://localhost:8000/ui> y <http://localhost:8000/docs>.

Para usar solo el modelo publicado no hace falta nada de lo anterior: `docker run -d -p 8000:8000 josecval/mibici-ml-service:1.2.0`.

## 1. Alcance y resultados

- **Fase 1:** lectura de datos, validaciones, limpieza, generación de Parquet, particiones temporales, análisis exploratorio y registro de resultados.
- **Fase 2:** refactorización a módulos `.py`, construcción de variables, entrenamiento y comparación de modelos, validación temporal, prueba independiente y registro en MLflow Tracking y Model Registry.
- **Entrega final:** servicio de predicción con FastAPI, imagen de Docker publicada en DockerHub, pruebas unitarias y de integración, monitoreo de data drift con alertas y comparación champion contra challenger (sección 10).
- **Periodo de datos:** enero de 2020 a septiembre de 2026, inclusive. La Fase 1 trabaja con 81 CSV mensuales de viajes y un catálogo de estaciones; el conjunto original supera los 27 millones de registros de viajes.
- **Modelo final registrado:** Random Forest con 18 variables de calendario, estación e historia. La ejecución revisada utiliza 12 ventanas de validación en 2025 y prueba en 2026.

### Resultados de la ejecución revisada

| Métrica | Valor |
| --- | ---: |
| MAE promedio de validación (2025) | 5.4384 viajes/estación-día |
| RMSE promedio de validación (2025) | 8.7375 viajes/estación-día |
| MAE de prueba (2026) | 5.8964 viajes/estación-día |
| RMSE de prueba (2026) | 10.7824 viajes/estación-día |
| MAE de línea base de prueba | 7.9566 viajes/estación-día |
| Reducción relativa del MAE frente a la línea base | 25.89 % |
| Filas de prueba | 108,294 |

Estos valores corresponden a una ejecución concreta, no son resultados garantizados en otros entornos o con otros datos. HistGradientBoosting obtuvo un MAE de validación ligeramente menor que Random Forest en la comparación inicial; la elección del modelo final debe explicarse en el informe.

## 2. Organización del repositorio

La estructura utilizada para ejecutar desde la raíz del repositorio es:

```text
equipo-8-mibici-mlops/
├── data/
│   ├── raw/
│   │   ├── 2020/ ... 2026/       # CSV mensuales, separados por año
│   │   ├── Manual_datos/         # Catálogo de estaciones
│   │   ├── weather/              # Datos de clima y zonas
│   │   └── pois/                 # POI de estaciones y archivo OSM
│   ├── interim/                  # Archivos intermedios
│   └── processed/                # Parquet y conjuntos procesados
├── docs/                         # Documentación y análisis de requerimientos
├── mibici_mlops/
│   ├── __init__.py
│   ├── config.py                 # Rutas, fechas y opciones de ejecución
│   ├── ingestion.py              # Lectura de datos
│   ├── preprocessing.py          # Limpieza
│   ├── pipeline.py               # Procesamiento de Fase 1
│   ├── dataset.py                # Preparación y validación del dataset
│   ├── external.py               # Datos externos: clima y POI
│   ├── features.py               # Variables de calendario, historia y días festivos
│   ├── checks.py                 # Comprobaciones de calidad
│   ├── eda.py                    # Análisis exploratorio
│   ├── plots.py                  # Gráficas
│   ├── reporting.py              # Reportes y tiempos
│   ├── tracking.py               # MLflow Tracking y Registry
│   ├── versioning.py             # Inventarios, SHA-256 y procedencia
│   ├── workflow.py               # Orquestación de Fase 2
│   ├── main.py                   # Punto de entrada
│   ├── monitoring.py             # Data drift, escenarios, alertas y acción
│   ├── mlflow_catalog.py         # Descripción y etiquetas de experimentos, linaje de la Fase 3
│   ├── serving/
│   │   ├── __init__.py
│   │   └── app.py                # API de FastAPI y página /ui
│   └── modeling/
│       ├── __init__.py
│       ├── train.py              # Entrenamiento
│       ├── evaluate.py           # Validación y métricas
│       ├── predict.py            # Predicciones
│       ├── export_model.py       # Exporta el champion a models/serving
│       ├── challenger.py         # Champion contra challenger
│       └── deploy.py             # Construye, comprueba y publica la imagen
├── notebooks/                    # Desarrollo y exploración
├── tests/                        # Pruebas automatizadas
├── models/                       # Modelos serializados locales y models/serving para el servicio
├── references/                   # Referencias del proyecto
├── reports/
│   ├── validation/              # Resultados de Fase 1
│   ├── modeling/                # Comparación, validación y prueba
│   ├── figures/                 # Figuras y gráficas
│   ├── data_versioning/         # Versiones de archivos
│   ├── mlflow_provenance/       # Procedencia del dataset
│   ├── api/                     # Especificación OpenAPI
│   ├── reproducibility/         # Comparación local contra Docker
│   ├── drift/                   # Monitoreo de octubre de 2026
│   └── tests/                   # Salida de pytest
├── .dockerignore
├── Dockerfile                   # Imagen del servicio
├── .gitignore
├── MLproject                    # Configuración de MLflow Projects
├── Makefile
├── pyproject.toml
├── python_env.yaml
├── requirements.txt             # Entorno con versiones fijas
├── requirements-serving.txt     # Librerías de la imagen de Docker
└── README.md
```

Las carpetas de resultados se crean durante la ejecución. La lista muestra los módulos utilizados en la implementación revisada; **comprobar que estén integrados en la rama que se vaya a entregar**.

## 3. Descargar y organizar los datos

Los archivos grandes se mantienen fuera de GitHub. Se obtienen de la [carpeta compartida MiBici en Google Drive](https://drive.google.com/drive/folders/1n83lQ_nEH9u7KvKAYNRcnUhae5tjoAYB?usp=drive_link).

**No es necesario descargar todo el contenido de Drive si ya se tienen los archivos procesados y sus manifiestos válidos**, pero para reconstruir desde cero sí se requieren los archivos de origen.

### 3.1. Archivos necesarios y ubicación

| En Google Drive | Guardar en el repositorio | Uso |
| --- | --- | --- |
| Carpetas `2020`, `2021`, `2022`, `2023`, `2024`, `2025`, `2026` | `data/raw/2020/` ... `data/raw/2026/` | CSV mensuales de viajes (81 en total para el periodo del proyecto) |
| Carpeta `Manual_datos`, incluido `nomenclatura_2026_09.csv` | `data/raw/Manual_datos/` | Catálogo e identificadores de estaciones |
| `weather_daily_zones_2020_2026.csv` | `data/raw/weather/weather_daily_zones_2020_2026.csv` | Información meteorológica por zona |
| `mibici_station_weather_zones.csv` | `data/raw/weather/mibici_station_weather_zones.csv` | Relación entre estaciones y zonas meteorológicas |
| `mibici_station_pois.csv` | `data/raw/pois/mibici_station_pois.csv` | Puntos de interés cercanos a estaciones |
| `mexico-latest.osm.pbf` | `data/raw/pois/mexico-latest.osm.pbf` | Fuente OpenStreetMap para POI; archivo grande, aproximadamente 618 MB |
| `metadato_limites_municipales_jalisco_iieg.xlsx` | `data/raw/metadato_limites_municipales_jalisco_iieg.xlsx` | Material geográfico de referencia; no se ha confirmado como entrada obligatoria del entrenamiento |

**Importante:** no cambiar los nombres de los CSV originales ni colocarlos todos directamente en `data/raw/`. Mantener la estructura por año. La captura de Drive muestra los nombres anteriores, pero no permite comprobar el contenido de cada carpeta anual: verificar que estén los 81 archivos mensuales esperados.

Si se descarga una carpeta como ZIP, descomprimirla antes de ejecutar. Evitar una carpeta adicional como `data/raw/MiBici/2020/` si el código espera `data/raw/2020/`.

### 3.2. Fuentes externas y caché

El código también contempla Open-Meteo y OpenStreetMap. Algunas ejecuciones pueden reutilizar datos externos previamente descargados o generados. Para reproducir **exactamente** una ejecución se deben conservar los mismos archivos, no solo consultar otra vez una API, porque las fuentes externas pueden cambiar.

Los archivos originales de `data/raw/` no deben modificarse. Los resultados de limpieza y transformación se guardan en `data/interim/` y `data/processed/`.

## 4. Preparar el entorno

Las instrucciones siguientes están pensadas para **Linux o WSL Ubuntu**, desde la raíz del repositorio.

### 4.1. Clonar y entrar al proyecto

```bash
git clone https://github.com/MarioSanchezValtierra/equipo-8-mibici-mlops.git
cd equipo-8-mibici-mlops
```

### 4.2. Crear y activar el entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si `pyproject.toml` declara correctamente el paquete instalable, también puede utilizarse:

```bash
python -m pip install -e .
```

Este último comando instala el proyecto en modo editable; no sustituye necesariamente todas las dependencias de `requirements.txt` si no están declaradas en `pyproject.toml`.

Comprobar el entorno:

```bash
python --version
python -m pip check
```

Se utilizó Python 3.13 en el entorno de desarrollo revisado. Para reproducir resultados, conviene mantener la misma versión de Python y las mismas versiones de bibliotecas. Si se actualizan dependencias, registrar ese cambio en Git y en MLflow.

## 5. Configuración antes de ejecutar

Revisar `mibici_mlops/config.py`. Las opciones disponibles dependen de la versión del código integrada en la rama:

| Opción | Para qué sirve |
| --- | --- |
| `RUN_ALL_FILES` | `True`: procesamiento completo; `False`: prueba limitada de Fase 1, cuando esta opción está habilitada |
| `FORCE_REBUILD_DATA` | Fuerza la reconstrucción del dataset integrado en lugar de reutilizar el caché |
| `RUN_GRID_SEARCH` | Habilita la búsqueda de hiperparámetros; puede aumentar considerablemente el tiempo de ejecución |
| `RUN_WEATHER_REFERENCE` | Habilita el experimento de referencia con clima, si está implementado en la versión utilizada |
| `HORIZON_DAYS` | Horizonte de predicción: `1` significa estimar las salidas del día siguiente |
| `FIRST_CUTOFF`, `TEST_START`, `N_FOLDS` | Fechas y número de ventanas de validación temporal |
| `MLFLOW_TRACKING_URI` | Ubicación de la base de datos de seguimiento |

Para repetir la ejecución de referencia, **no cambiar las fechas, el horizonte, las variables ni los hiperparámetros**. Las opciones de prueba rápida no equivalen a una evaluación final.

Las rutas deben derivarse de la raíz del repositorio, sin depender de `/home/usuario/` ni del directorio desde el que se abrió VS Code. El proyecto debe poder ejecutarse después de clonarlo en otra computadora.

## 6. Orden de ejecución

### 6.1. Primera ejecución desde cero

1. Clonar el repositorio y seleccionar la rama o el commit que se quiere reproducir.
2. Crear el entorno e instalar las dependencias.
3. Descargar y organizar los archivos de la sección 3.
4. Revisar las opciones de `config.py`.
5. **Ejecutar primero el procesamiento de Fase 1** si todavía no existen los Parquet y una ejecución válida en `reports/validation/`. La preparación de Fase 2 depende de estos resultados.
6. Ejecutar la Fase 2 desde la raíz del repositorio.
7. Revisar los reportes, los resultados de prueba y los registros de MLflow.
8. Ejecutar las pruebas automatizadas.

Para correr **solo la Fase 1**: `make data`. Para la Fase 2: `make train` (equivale a `python -m mibici_mlops.main` y corre la Fase 1 si no existe una ejecución válida). El orden completo está en "Correr desde cero".

### 6.2. Ejecutar la Fase 2

Desde la raíz del repositorio, con el entorno activado:

```bash
python -m mibici_mlops.main
```

Este comando ejecuta la orquestación de Fase 2 que quedó integrada en el paquete `mibici_mlops/`: preparación del dataset, variables, validación, comparación, entrenamiento final, prueba, registro de artefactos y Model Registry.

Si aparece un error indicando que no hay una ejecución completa de Fase 1 en `reports/validation/`, primero deben generarse esos resultados. **No basta con tener los CSV en Drive**.

En una implementación alternativa con paquete bajo `src/` y `main.py` en la raíz, el punto de entrada es `python main.py`; **no mezclar esa estructura con la del repositorio actual**.

### 6.3. Ejecuciones posteriores

Si `data/processed/mibici_integrated.parquet` y su manifiesto siguen siendo válidos, el código puede reutilizar el dataset integrado. Si cambian los CSV, el catálogo, las fuentes externas o las reglas de limpieza, reconstruir los datos y registrar la nueva versión.

No borrar archivos de datos ni reportes anteriores sin revisar sus dependencias. No modificar el conjunto de prueba para mejorar los resultados de validación.

## 7. MLflow Tracking y Model Registry

La implementación registra el experimento `MiBici_Fase2_Modelado`, parámetros, métricas, datasets, artefactos y modelo final. La entrega final agrega dos experimentos con la misma estructura, `MiBici_Fase3_Monitoreo_Drift` y `MiBici_Fase3_Champion_Challenger` (sección 10.7).

### 7.1. Abrir la interfaz

Desde la raíz del repositorio:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir <http://127.0.0.1:5000>. La base `mlflow.db` debe ser la misma configurada en `MLFLOW_TRACKING_URI`; si no aparece el experimento esperado, comprobar primero esa coincidencia.

### 7.2. Qué revisar en cada ejecución

- **Overview:** nombre, descripción, estado, fuente de ejecución, parámetros y métricas.
- **Model metrics:** métricas de validación y prueba; las series exploratorias pueden utilizar `step` como índice de mes o categoría, no necesariamente como época de entrenamiento.
- **Artifacts:** gráficas, tablas, configuración, características, versiones de datos, reportes y tiempos.
- **Datasets:** identificadores de los conjuntos utilizados para entrenamiento, validación y prueba.
- **Models / Registry:** modelo `mibici_salidas_diarias`, versión registrada y alias `champion`.

La versión 7 es el modelo de la Fase 2 (el que guarda `models/mibici_final_model.joblib`). El alias `champion` apunta a la versión 10, que es la que se sirve (sección 10.1). Conviene comprobar directamente en el Registry que el alias apunte a la versión deseada.

### 7.3. Trazabilidad y reproducibilidad

Se registran, cuando están disponibles:

- Commit y rama de Git (`git_commit`, `git_branch`, `implementation_commit`).
- Estado del árbol de trabajo (`git_dirty`).
- Hashes SHA-256, inventarios y procedencia de los archivos.
- Configuración de entrenamiento y lista de características.
- Versiones de Python y bibliotecas.
- Tiempos por etapa y artefactos de evaluación.

**Si `git_dirty=true`, el commit por sí solo no permite reconstruir exactamente el código ejecutado.** Antes de generar la evidencia definitiva, confirmar los cambios que deben formar parte del repositorio. No se debe alterar una ejecución histórica para hacerla parecer limpia.

Los archivos grandes no se copian a MLflow; los manifiestos registran sus rutas relativas y hashes. Esto permite verificar los datos, pero no sustituye su almacenamiento en Drive.

## 8. Resultados generados

| Ruta | Contenido |
| --- | --- |
| `data/processed/` | Parquet procesados e integrados |
| `reports/validation/` | Validaciones y resultados de Fase 1 |
| `reports/modeling/` | Comparación de modelos, validación temporal y resultados de prueba |
| `reports/figures/modeling/` | Gráficas de comparación y predicciones |
| `reports/data_versioning/` | Versiones y hashes de archivos |
| `reports/mlflow_provenance/` | Procedencia de los datos |
| `models/` | Modelo final serializado localmente |
| `mlruns/` | Artefactos y registros locales de MLflow |
| `mlflow.db` | Base de datos local de MLflow |

En los artefactos de MLflow, `README.md` explica las carpetas; `features.json` documenta las 18 variables y `documentation/EDA_METRICS_README.md` describe las métricas exploratorias y sus unidades. También se incluyen archivos como `modeling/comparison/model_comparison.csv`, `modeling/validation/final_walk_forward.csv`, `modeling/test/test_results_2026.csv` y `performance/execution_times.csv`.

## 9. Pruebas automatizadas

Ejecutar desde la raíz:

```bash
pytest -q
```

Son 82 pruebas, todas con datos pequeños generados en el momento, así que no necesitan los CSV originales:

| Archivo | Tipo | Qué revisa |
| --- | --- | --- |
| `test_data.py`, `test_features.py`, `test_evaluate.py`, `test_modeling.py`, `test_structure.py`, `test_project_paths.py` | unitarias | procesamiento, variables, evaluación, estructura y rutas (Fase 2) |
| `test_metrics.py` | unitaria | MAE, RMSE, línea base ingenua y mejora contra ella |
| `test_api.py` | unitaria | esquema de entrada, `/predict`, errores 422 y 503, página `/ui` y documentación |
| `test_reproducibility.py` | unitaria | misma semilla, mismo resultado; el modelo de `models/serving` reproduce su muestra de referencia |
| `test_drift.py` | unitaria | PSI, tabla de drift, niveles de las alertas A1 a A5 y acción propuesta |
| `test_calendar.py` | unitaria | variable de días festivos |
| `test_integration_pipeline.py` | integración | del CSV crudo a las métricas: limpieza, demanda diaria, variables, entrenamiento y predicción directa y por la API |

Las pruebas del modelo desplegado leen `models/serving/`; si no existe (antes de `make export-model`), se saltan.

## 10. Entrega final: servicio de predicción, Docker y monitoreo

El modelo se expone con FastAPI y se empaqueta en una imagen de Docker publicada en DockerHub. Todos los comandos se corren desde la raíz del repositorio, con el entorno activado (sección 4).

### 10.1. Modelo que se sirve

| Uso | Modelo | URI en MLflow |
| --- | --- | --- |
| Reproducir la Fase 2 y monitorear octubre de 2026 | versión 7 (Random Forest de la Fase 2) | `models:/mibici_salidas_diarias/7` |
| Servicio (FastAPI y Docker) | versión 10, alias `champion` (Random Forest reentrenado hasta septiembre de 2026, con la variable de días festivos) | `models:/mibici_salidas_diarias/10` |

La versión 10 salió de la comparación champion contra challenger: se eligió en septiembre y se probó en octubre de 2026, donde bajó el MAE 3.3 % frente a la versión 7 (7.42 contra 7.68) sin empeorar el RMSE.

`make export-model` copia la versión con el alias `champion` a `models/serving/` junto con:

- `metadata.json`: versión, ejecución, variables, semilla, hash de los datos y versiones de las librerías.
- `reference_sample.csv`: 500 filas fijas de la prueba 2026 con la predicción del modelo, para comprobar que el servicio y la imagen responden igual.
- `ui_days.csv`: las variables por estación y día de septiembre y octubre de 2026, que usa la página `/ui`.

Esta carpeta no se sube a Git; se genera con el comando y va dentro de la imagen.

En la interfaz de MLflow el champion está en Models (menú de la izquierda), `mibici_salidas_diarias`: la versión con el alias `champion` tiene una descripción y la etiqueta `documentation_run`, que lleva a la ejecución "Modelo champion" del experimento `MiBici_Fase3_Champion_Challenger`. Ahí están sus métricas de septiembre y octubre, el error diario en Model metrics, las gráficas en `modeling/test`, `modeling/comparison` y `modeling/interpretation`, y la configuración en `documentation/`.

### 10.2. Instalar, exportar el modelo y probar

```bash
pip install -r requirements.txt   # versiones fijas del entorno de entrenamiento
make export-model                 # copia el champion a models/serving
pytest -q                         # pruebas unitarias y de integración (82)
```

### 10.3. Correr el servicio sin Docker

```bash
make serve
```

### 10.4. Construir y correr la imagen

Requiere Docker Desktop abierto. En Windows con WSL, activar una sola vez la integración con Ubuntu en *Settings > Resources > WSL integration*, y opcionalmente *Settings > General > Start Docker Desktop when you sign in*. Si `docker version` no muestra la sección `Server`, Docker no está abierto.

```bash
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest
```

Para usar la imagen publicada, sin construirla:

```bash
docker pull josecval/mibici-ml-service:1.2.0
docker run -p 8000:8000 josecval/mibici-ml-service:1.2.0
```

### 10.5. Direcciones del servicio

Con el servicio en marcha (local o en Docker):

| Dirección | Para qué |
| --- | --- |
| <http://localhost:8000> | lleva a la página `/ui` |
| <http://localhost:8000/ui> | página para elegir una estación y un día y ver la predicción |
| <http://localhost:8000/docs> | documentación interactiva (Swagger): probar `POST /predict` con *Try it out* |
| <http://localhost:8000/redoc> | la misma documentación en formato de lectura |
| <http://localhost:8000/health> | estado del servicio y versión del modelo cargado |

La página tiene los viajes reales del 1 de septiembre al 31 de octubre de 2026 y para esos días predice con las mismas variables del pipeline. Para un día posterior, o un día en que la estación no tuvo viajes, usa la última historia conocida de la estación con el calendario del día pedido, y la respuesta lo marca como estimación (`estimated` y `history_end`). La ruta que usa la página también se puede llamar directo: `GET /predict/estacion?origen_id=15&fecha=2026-11-19`.

`POST /predict` recibe una lista `instances` con las 18 variables de cada estación y día, y devuelve los viajes predichos. Pydantic valida tipos y rangos (respuesta 422 si algo no cumple) y el servicio responde 503 si no pudo cargar el modelo. Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "instances": [{"Origen_Id": 35, "day_of_week": 2, "month": 10, "day_of_month": 15, "weather_zone": 3,
    "education_count": 4, "transport_count": 1, "commerce_count": 2, "work_count": 6, "health_count": 0,
    "recreation_count": 3, "lag_1": 41, "lag_7": 38, "lag_14": 36, "roll_mean_7": 39.4,
    "active_days_7": 7, "roll_mean_28": 37.9, "active_days_28": 28}]}'
```

La especificación OpenAPI completa está en `reports/api/openapi.json`.

### 10.6. Imágenes publicadas

Repositorio: <https://hub.docker.com/r/josecval/mibici-ml-service>

| Etiqueta | Contenido |
| --- | --- |
| `1.0.0` | versión 7 (Fase 2) |
| `1.1.0` | versión 10, sin la página `/ui` |
| `1.2.0` y `latest` | versión 10 con la página `/ui` |

Una etiqueta publicada no se sobrescribe. Para publicar una versión nueva (por ejemplo, después de promover otro champion):

```bash
docker login -u josecval            # una sola vez
make export-model
docker build -t ml-service:latest .
docker tag ml-service:latest josecval/mibici-ml-service:<versión>
docker tag ml-service:latest josecval/mibici-ml-service:latest
docker push josecval/mibici-ml-service:<versión>
docker push josecval/mibici-ml-service:latest
```

`make deploy` hace lo mismo de forma automática: construye la imagen, comprueba dentro del contenedor la muestra de referencia y solo publica si coincide y `AUTO_DEPLOY = True` en `config.py`. La etiqueta de cada champion está en `RELEASES`; uno nuevo recibe la siguiente (1.3.0).

### 10.7. Monitoreo de data drift y champion contra challenger

```bash
make monitor      # octubre de 2026 con la versión 7: drift, escenarios, alertas y acción
make challenger   # reentrena los challengers y aplica la regla de promoción
make champion     # documenta al champion en MLflow con métricas y gráficas
make mlflow-catalog  # resumen de la entrega final y descripción de todos los experimentos
```

Todas las fases usan la misma base `mlflow.db` y la carpeta `mlruns/` de la raíz. Cada experimento tiene una descripción y las etiquetas `phase`, `content` y `status`:

| Experimento | Fase | Contenido | Estado |
| --- | --- | --- | --- |
| `MiBici` | 1 | carga, limpieza e integración de los datos | vigente |
| `MiBici_Fase2_Modelado` | 2 | líneas base, validación, búsqueda y modelo final (v7) | vigente |
| `MiBici_Fase3_Monitoreo_Drift` | 3 | monitoreo de octubre, escenarios y alertas | vigente |
| `MiBici_Fase3_Champion_Challenger` | 3 | comparación de modelos y documentación del champion | vigente |
| `MiBici_Monitoreo_Drift`, `MiBici_Champion_Challenger` | 3 | corridas de desarrollo del notebook 07, donde se registró la v10 | desarrollo |
| `MiBici_Entrega_Final` | 1, 2 y 3 | una ejecución resumen que junta todo, con enlaces a cada ejecución de origen | vigente |

Las ejecuciones de la Fase 3 siguen la convención de la Fase 2: nombre "Tipo | Modelo | detalle", la columna Source con el archivo que las genera, el dataset en la columna Dataset, `README.md`, `artifact_catalog.json`, `environment/runtime.json`, `documentation/run_configuration.json` y las carpetas `data/inventory`, `data/versions` y `data/lineage`. Si se vuelve a correr, no se borra nada: la ejecución anterior con el mismo nombre queda con la etiqueta `run_status = anterior` y la nueva con `vigente`, así que el filtro `tags.run_status = "vigente"` muestra una por tipo. Todas guardan en sus etiquetas el commit (`git_commit`), el entorno (`python_version`, `sklearn_version`) y el SHA-256 de los datos (`data_sha256`, `dataset_sha256`).

| Ejecución | Carpetas propias | Etiquetas para filtrar |
| --- | --- | --- |
| Monitoreo \| Random Forest v7 \| octubre 2026 | `monitoring/performance`, `monitoring/drift`, `monitoring/scenarios`, `monitoring/alerts` | `phase`, `stage`, `model_version` |
| Escenario \| Random Forest v7 \| nombre (hijas) | métricas del escenario | `scenario`, `scenario_type`, `action`, `alert_A1` a `alert_A5` |
| Champion contra challenger \| Random Forest \| octubre 2026 | `modeling/comparison`, `modeling/test` | `model_role`, `decision`, `candidate`, `reason` |
| Modelo champion \| Random Forest \| v10 | `modeling/comparison`, `modeling/test`, `modeling/interpretation` | `model_role`, `model_version`, `registered_model_alias` |

La ejecución "Entrega final | Random Forest | resumen de las tres fases" del experimento `MiBici_Entrega_Final` junta todo en un solo lugar: copia completa de los artefactos de la ejecución de datos de la Fase 1, del modelo final v7, del entrenamiento de la v10, del monitoreo, de la comparación y del champion, cada una en su carpeta; todas sus métricas con el prefijo de su origen; `resumen/` con la tabla de métricas, la lista de ejecuciones con enlace, SHA-256 y commit, las alertas por escenario y las revisiones del notebook 08; y un `README.md` con los enlaces. Se genera con `make mlflow-catalog` o desde el notebook 08.

Para ver las etiquetas como columnas: en la vista de ejecuciones, Columns, sección Tags. Para filtrar, por ejemplo: `tags.alert_A3 = "crítica"` o `tags.model_role = "champion"`.

`make monitor` espera el CSV de octubre en `data/monitoring/2026/datos_abiertos_2026_10.csv`. Guarda las tablas y los reportes de Evidently en `reports/drift/` y registra el resultado en el experimento `MiBici_Fase3_Monitoreo_Drift` de MLflow.

Las alertas tienen dos niveles, advertencia y crítica:

| Alerta | Qué mide | Advertencia | Crítica |
| --- | --- | --- | --- |
| A1 | `lag_1` vacío por arriba de la línea base | 5 puntos | 15 puntos |
| A2 | sesgo frente a la demanda media | 10 % | 20 % |
| A3 | MAE y mejora frente al ingenuo | MAE +20 % o mejora < 20 % | las dos |
| A4 | PSI máximo o proporción de variables con drift | PSI > 0.10 o más de 30 % | PSI > 0.25 |
| A5 | cambio en estaciones activas o viajes por día | 15 % | 30 % |

La acción sigue este orden: problemas de datos primero (revisar el pipeline, no reentrenar); reentrenar solo si A3 es crítica dos meses seguidos; revisar con operación si A5 es crítica; vigilar ante cualquier advertencia.

`make challenger` registra la comparación en el experimento `MiBici_Fase3_Champion_Challenger`. El challenger gana si baja el MAE al menos 2 % y no empeora el RMSE. Solo se promueve, es decir, se registra la versión nueva y se le pasa el alias `champion`, si además `AUTO_PROMOTE = True` en `config.py`; con `False` solo muestra la decisión.

### 10.8. Requisitos de la entrega final

| Requisito | Dónde se cumple |
| --- | --- |
| Pruebas unitarias y de integración con `pytest -q` | `tests/` (82 pruebas). `test_integration_pipeline.py` corre el flujo completo: carga, limpieza, variables, entrenamiento, predicción, métricas y API |
| `POST /predict` con Pydantic y OpenAPI | `mibici_mlops/serving/app.py`; esquema en `/docs` y `/redoc`; artefacto `models:/mibici_salidas_diarias/10` (sección 10.1) |
| Reproducibilidad en otro entorno | `requirements.txt` con versiones fijas, `SEED = 42`, muestra de referencia comparada dentro del contenedor (`reports/reproducibility/`), modelo y datos versionados en MLflow con su SHA-256 |
| Docker | `Dockerfile`, `docker build -t ml-service:latest .`, `docker run -p 8000:8000 ml-service:latest`, imagen `josecval/mibici-ml-service` con etiquetas 1.0.0, 1.1.0, 1.2.0 y `latest` |
| Data drift y pérdida de desempeño | `mibici_mlops/monitoring.py`: octubre real y tres escenarios (rezagos faltantes, medias desplazadas, cambio estacional) comparados contra la prueba 2026, alertas A1 a A5 con umbrales y acción propuesta, registrados en `MiBici_Fase3_Monitoreo_Drift` |

## 11. Buenas prácticas de MLOps utilizadas

- **Separación de responsabilidades:** cada módulo atiende una tarea específica; `workflow.py` coordina el orden sin duplicar el procesamiento.
- **Código reutilizable:** los notebooks pueden servir para explorar, pero la implementación se ejecuta desde archivos `.py`.
- **Dependencias explícitas:** entorno virtual y archivos de dependencias.
- **Rutas portables:** archivos localizados desde la raíz del proyecto y no mediante rutas absolutas de un integrante.
- **Datos originales intactos:** las transformaciones generan archivos nuevos.
- **Separación temporal:** entrenamiento, validación y prueba se dividen por fechas; la prueba independiente no se utiliza para seleccionar hiperparámetros.
- **Comparación contra líneas base:** los resultados se interpretan respecto a métodos sencillos, no solo mediante una métrica aislada.
- **Seguimiento de experimentos:** parámetros, métricas, gráficas, versiones y modelos se registran en MLflow.
- **Trazabilidad:** Git, hashes y manifiestos permiten identificar las entradas y el código asociado a una ejecución.
- **Pruebas:** `pytest` permite detectar cambios inesperados en el comportamiento.
- **Servicio reproducible:** la imagen de Docker usa las mismas versiones de librerías que el entrenamiento y se comprueba contra una muestra de referencia antes de publicarla.
- **Monitoreo:** el drift y el desempeño se revisan cada mes con alertas de dos niveles, y un cambio de modelo pasa por una regla fija antes de promoverse.

## 12. Referencias

- [MiBici](https://www.mibici.net/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [MLflow](https://mlflow.org/docs/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pytest](https://docs.pytest.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Evidently](https://docs.evidentlyai.com/)
- [Open-Meteo](https://open-meteo.com/)
- [OpenStreetMap](https://www.openstreetmap.org/)
