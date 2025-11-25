# Simple Makefile for tennis_prediction project

PYTHON := python3
VENV ?= venv

.PHONY: help venv install test lint clean

help:
	@echo "Targets:"
	@echo "  make venv       - create virtualenv"
	@echo "  make install    - install package in editable mode"
	@echo "  make test       - run pytest"
	@echo "  make clean      - remove build artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt
	pip3 install -e .

test:
	pytest -v

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
