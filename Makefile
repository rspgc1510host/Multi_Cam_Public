.PHONY: install lint format test check

PYTHON ?= python3
VENV ?= .venv
VENV_BIN := $(VENV)/bin

$(VENV_BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -e ".[dev]"

install: $(VENV_BIN)/python
	@echo "Virtual environment ready at $(VENV)"

lint: $(VENV_BIN)/python
	$(VENV_BIN)/flake8 lunar_rover tests

format: $(VENV_BIN)/python
	$(VENV_BIN)/isort lunar_rover tests
	$(VENV_BIN)/black lunar_rover tests

test: $(VENV_BIN)/python
	$(VENV_BIN)/pytest

check: lint test
