#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
EXPERIMENT ?= wine-quality
# Local MLflow tracking server (start it with `make mlflow-server`)
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000
ALPHA ?= 0.5

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mlops
	isort --check --diff mlops
	black --check mlops

## Format source code with black
.PHONY: format
format:
	isort mlops
	black mlops





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> New virtual environment created in .venv. Activate with:"
	@echo "    source .venv/bin/activate      (Linux/macOS)"
	@echo "    .venv\\Scripts\\activate         (Windows)"



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Build the train/test split from the cleaned dataset (notebook 1.4)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) scripts/prepare_data.py

## Start the local MLflow tracking server (keep it running in its own terminal)
.PHONY: mlflow-server
mlflow-server:
	mlflow server --host 127.0.0.1 --port 5000

## Train the model without MLflow Projects (logs a run to the MLflow server)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mlops.modeling.train

## Run the MLflow project's main entry point (ALPHA=0.5 by default)
.PHONY: mlflow-run
mlflow-run:
	mlflow run . --env-manager local --experiment-name $(EXPERIMENT) -P alpha=$(ALPHA)

## Run the multi-step MLflow workflow: load -> features -> train
.PHONY: workflow
workflow:
	mlflow run . -e workflow --env-manager local --experiment-name $(EXPERIMENT)

## Run the unit tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q


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
