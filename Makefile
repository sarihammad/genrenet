.PHONY: setup data sanity train eval export api docker-train docker-api compose-up compose-down test lint all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r api/app/requirements.txt && pip install -e .

data:
	python scripts/download_gztan.py --out data

sanity:
	python scripts/sanity_check.py

train:
	python -m core.train.train --config core/config.yaml

eval:
	python -m core.train.eval --config core/config.yaml --ckpt saved_models/best_model.pt

export:
	python -m core.train.export --config core/config.yaml --ckpt saved_models/best_model.pt --out saved_models

api:
	uvicorn api.app.main:app --host 0.0.0.0 --port 8089 --reload

docker-train:
	docker build -f docker/train.Dockerfile -t gtzan-train .

docker-api:
	docker build -f docker/api.Dockerfile -t gtzan-api .

compose-up:
	docker compose -f docker/docker-compose.yml up -d

compose-down:
	docker compose -f docker/docker-compose.yml down

test:
	pytest -q

lint:
	ruff check . && black --check .

all: make data sanity train eval export
