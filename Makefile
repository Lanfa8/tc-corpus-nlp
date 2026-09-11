.PHONY: install lint lint-fix test train export-onnx benchmark run-api \
        docker-build docker-run up down airflow-up airflow-down

install:
	poetry install

lint:
	poetry run ruff check src tests scripts
	poetry run ruff format --check src tests scripts

lint-fix:
	poetry run ruff check --fix src tests scripts
	poetry run ruff format src tests scripts

test:
	poetry run pytest tests --cov=src/triage --cov-report=term-missing

train:
	poetry run python scripts/train.py --compare

export-onnx:
	poetry run python scripts/export_onnx.py

benchmark:
	poetry run python scripts/benchmark_latency.py

run-api:
	poetry run uvicorn triage.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t triage-api:local .

docker-run: docker-build
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models triage-api:local

up:
	docker compose up -d --build

down:
	docker compose down -v

airflow-up:
	docker compose -f docker-compose.airflow.yml up -d

airflow-down:
	docker compose -f docker-compose.airflow.yml down -v
