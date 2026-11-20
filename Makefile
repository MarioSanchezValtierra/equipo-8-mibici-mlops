#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = equipo-8-mibici-mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run unit and integration tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Fase 1: procesar los CSV de data/raw y generar data/processed/mibici_integrated.parquet
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -c "from mibici_mlops.dataset import prepare_model_data; prepare_model_data()"


## Fase 2: variables, validación, entrenamiento, prueba y registro en MLflow (corre la Fase 1 si falta)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici_mlops.main


## Export the champion model from MLflow to models/serving
.PHONY: export-model
export-model:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.export_model


## Serve the API locally on http://localhost:8000/docs
.PHONY: serve
serve:
	MODEL_URI=models/serving/model uvicorn mibici_mlops.serving.app:app --host 0.0.0.0 --port 8000


## Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t ml-service:latest .


## Run the Docker image on port 8000
.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 ml-service:latest


## Build, check against the reference sample and publish if AUTO_DEPLOY is True
.PHONY: deploy
deploy:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.deploy


## Monitor October 2026: drift, scenarios, alerts and action
.PHONY: monitor
monitor:
	$(PYTHON_INTERPRETER) -m mibici_mlops.monitoring


## Champion vs challenger with the promotion rule
.PHONY: challenger
challenger:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.challenger


## Log the champion evidence (metrics, figures, configuration) to MLflow
.PHONY: champion
champion:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.champion_report


## Final summary run and descriptions and tags for every MLflow experiment
.PHONY: mlflow-catalog
mlflow-catalog:
	$(PYTHON_INTERPRETER) -m mibici_mlops.mlflow_catalog


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
