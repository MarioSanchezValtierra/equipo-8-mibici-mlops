# Equipo 8 — MiBici MLOps

Proyecto de la materia de MLOps para analizar los viajes del sistema MiBici de Guadalajara y predecir **cuántos viajes comenzarán en cada estación durante el día siguiente**. La idea es contar con una estimación de las salidas por estación que ayude a planear mejor la disponibilidad y distribución de bicicletas.

### Problema y variable objetivo

En MiBici no todas las estaciones tienen el mismo movimiento. Algunas concentran muchas salidas y otras registran pocas; además, la actividad cambia según el día de la semana, la época del año y el comportamiento reciente de cada estación. Esto complica anticipar dónde podría hacer falta disponer de más bicicletas.

Por eso definimos la variable objetivo **`target` como el número de viajes que comienzan en una estación durante un día**. Cada fila del conjunto de modelado representa una combinación **estación-fecha**. Para construirla, agrupamos los viajes por estación de origen (`Origen_Id`) y día de inicio del viaje; en el dataset diario, el conteo `Cantidad_viajes` se utiliza como `target`.

Elegimos las **salidas**, en lugar de las llegadas o el total de movimientos, porque queremos estimar cuántas bicicletas podrían necesitar los usuarios para iniciar sus recorridos en cada estación. También elegimos la **escala diaria** porque resulta útil para la planeación operativa y permite comparar estaciones y fechas sin depender de las variaciones de cada hora.

El modelo se plantea para **predecir la demanda del día siguiente** (`horizon_days = 1`), utilizando información histórica, variables de calendario y características de las estaciones. Las variables históricas deben calcularse únicamente con datos anteriores al día que se predice, para evitar utilizar información del futuro.

Esta predicción puede apoyar la identificación de estaciones con mayor demanda esperada y la planeación de la redistribución de bicicletas. **No indica directamente cuántas bicicletas habrá disponibles ni cuántos viajes no pudieron realizarse**: estima los viajes iniciados que aparecen en los registros. Para decidir movimientos concretos de bicicletas también harían falta datos de disponibilidad, capacidad y llegadas.

El repositorio utiliza como base [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/), adaptado para las Fases 1 y 2.

## 1. Alcance y resultados

- **Fase 1:** lectura de datos, validaciones, limpieza, generación de Parquet, particiones temporales, análisis exploratorio y registro de resultados.
- **Fase 2:** refactorización a módulos `.py`, construcción de variables, entrenamiento y comparación de modelos, validación temporal, prueba independiente y registro en MLflow Tracking y Model Registry.
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
│   ├── features.py               # Variables de calendario e historia
│   ├── checks.py                 # Comprobaciones de calidad
│   ├── eda.py                    # Análisis exploratorio
│   ├── plots.py                  # Gráficas
│   ├── reporting.py              # Reportes y tiempos
│   ├── tracking.py               # MLflow Tracking y Registry
│   ├── versioning.py             # Inventarios, SHA-256 y procedencia
│   ├── workflow.py               # Orquestación de Fase 2
│   ├── main.py                   # Punto de entrada
│   └── modeling/
│       ├── __init__.py
│       ├── train.py              # Entrenamiento
│       ├── evaluate.py           # Validación y métricas
│       └── predict.py            # Predicciones
├── notebooks/                    # Desarrollo y exploración
├── tests/                        # Pruebas automatizadas
├── models/                       # Modelos serializados locales
├── references/                   # Referencias del proyecto
├── reports/
│   ├── validation/              # Resultados de Fase 1
│   ├── modeling/                # Comparación, validación y prueba
│   ├── figures/                 # Figuras y gráficas
│   ├── data_versioning/         # Versiones de archivos
│   └── mlflow_provenance/       # Procedencia del dataset
├── .gitignore
├── MLproject                    # Configuración de MLflow Projects
├── Makefile
├── pyproject.toml
├── python_env.yaml
├── requirements.txt
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
git clone <URL_DEL_REPOSITORIO_DEL_EQUIPO>
cd equipo-8-mibici-mlops
```

Sustituir `<URL_DEL_REPOSITORIO_DEL_EQUIPO>` por la URL real de GitHub. No se incluye una URL inventada.

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

**Punto a verificar:** el comando exacto para lanzar **solo la Fase 1** depende de cómo quedó integrado el punto de entrada de `MiBiciPipeline` en la rama final. No se documenta un comando no confirmado. Antes de publicar, agregar aquí ese comando verificado; `python -m mibici_mlops.main` corresponde al flujo de Fase 2 que se ejecutó.

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

La implementación registra el experimento `MiBici_Fase2_Modelado`, parámetros, métricas, datasets, artefactos y modelo final.

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

En la ejecución revisada se observa **la versión 4 registrada**. El alias `champion` aparece en la configuración, pero conviene comprobar directamente en el Registry que apunte a la versión deseada.

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
python -m pytest tests/
```

Las pruebas verifican componentes del procesamiento, las variables, la evaluación y otras funciones del proyecto según la rama integrada. Una ejecución anterior reportó **15 pruebas aprobadas**, pero ese resultado no sustituye correrlas después de los últimos cambios.

## 10. Buenas prácticas de MLOps utilizadas

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

## 11. Referencias

- [MiBici](https://www.mibici.net/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [MLflow](https://mlflow.org/docs/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pytest](https://docs.pytest.org/)
- [Open-Meteo](https://open-meteo.com/)
- [OpenStreetMap](https://www.openstreetmap.org/)
