PYTHON := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := $(CURDIR)/src

.DEFAULT_GOAL := help

.PHONY: help install run test lint validate validate-10 validate-40 smoke sample plot legend analyze docker-build up down logs

help:
	@echo "Available commands:"
	@echo "  make install       Create the virtual environment and install dependencies"
	@echo "  make run           Run the FastAPI development server"
	@echo "  make test          Run the deterministic test suite"
	@echo "  make lint          Run Ruff checks"
	@echo "  make validate      Validate the default 10% drift dataset"
	@echo "  make validate-10   Validate the 10% quarterly-drift dataset"
	@echo "  make validate-40   Validate the 40% quarterly-drift dataset"
	@echo "  make smoke         Run the live Ollama smoke test"
	@echo "  make sample        Run a 20-ticket-per-quarter experiment with configured Ollama models"
	@echo "  make plot RESULT=outputs/<folder>/<result>.json  Plot FEA and cumulative error rate"
	@echo "  make legend        Render the horizontal plot legend as a separate image"
	@echo "  make analyze JOB=<job-id>  Plot repetition averages and export abstention/drift statistics"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make up            Build and start the app with Docker Compose"
	@echo "  make down          Stop the Docker Compose services"
	@echo "  make logs          Follow the API container logs"

install:
	python3 -m venv .venv
	$(PIP) install -r requirements-dev.txt
	$(PIP) install --no-deps -e .

run:
	$(PYTHON) -m uvicorn irag.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

validate: validate-10

validate-10:
	$(PYTHON) "experiment data/validate_dataset.py"

validate-40:
	$(PYTHON) "experiment data/validate_drift_40_dataset.py"

smoke:
	$(PYTHON) -m irag.tools.ollama_smoke_test

sample:
	$(PYTHON) -m irag.tools.run_sample_experiment --tickets-per-quarter 20

plot:
	@test -n "$(RESULT)" || (echo "Usage: make plot RESULT=outputs/<folder>/<result>.json"; exit 2)
	$(PYTHON) -m irag.tools.plot_experiment_result "$(RESULT)"

legend:
	$(PYTHON) -m irag.tools.plot_legend --output "$(or $(OUTPUT),outputs/plot-legend.png)"

analyze:
	@test -n "$(JOB)" || (echo "Usage: make analyze JOB=<job-id>"; exit 2)
	$(PYTHON) -m irag.tools.analyze_experiment "$(JOB)"

docker-build:
	docker compose build

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api
