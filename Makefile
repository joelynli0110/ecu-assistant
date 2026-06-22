PYTHON ?= python

.PHONY: install install-dev test lint evaluate build log-model clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -B -m pytest -p no:cacheprovider

lint:
	$(PYTHON) -m pylint src/ecu_assistant

evaluate:
	$(PYTHON) -m ecu_assistant.evaluation.run_eval

build:
	$(PYTHON) -m build

log-model:
	$(PYTHON) -m ecu_assistant.mlflow_model.log_model

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', '.pytest_cache')]"
