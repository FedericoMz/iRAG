PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.DEFAULT_GOAL := help

.PHONY: help install run test lint validate smoke sample plot docker-build up down logs

help:
	@echo "Available commands:"
	@echo "  make install       Create the virtual environment and install dependencies"
	@echo "  make run           Run the FastAPI development server"
	@echo "  make test          Run the deterministic test suite"
	@echo "  make lint          Run Ruff checks"
	@echo "  make validate      Validate the complete SalesX dataset"
	@echo "  make smoke         Run the live Ollama smoke test"
	@echo "  make sample        Run a 20-ticket-per-quarter experiment with configured Ollama models"
	@echo "  make plot RESULT=outputs/<folder>/<result>.json  Plot FEA and cumulative error rate"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make up            Build and start the app with Docker Compose"
	@echo "  make down          Stop the Docker Compose services"
	@echo "  make logs          Follow the API container logs"

install:
	python3 -m venv .venv
	$(PIP) install -r requirements-dev.txt

run:
	$(PYTHON) -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

validate:
	$(PYTHON) "experiment data/validate_dataset.py"

smoke:
	$(PYTHON) ollama_smoke_test.py

sample:
	$(PYTHON) -m scripts.run_sample_experiment --tickets-per-quarter 20

plot:
	@test -n "$(RESULT)" || (echo "Usage: make plot RESULT=outputs/<folder>/<result>.json"; exit 2)
	$(PYTHON) -m scripts.plot_experiment_result "$(RESULT)"

docker-build:
	docker compose build

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api
