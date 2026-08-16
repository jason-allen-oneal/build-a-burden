.PHONY: setup lint typecheck test security fixture-corpus tokenizer-dev train-dev evaluate-dev reproduce-dev node-build

setup:
	uv sync --all-extras --locked
	cd tools/typescript && npm ci --ignore-scripts

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy
	cd tools/typescript && npm run typecheck

test:
	uv run pytest

security:
	uv run bandit -c pyproject.toml -r src
	uv run pip-audit

node-build:
	cd tools/typescript && npm run build

fixture-corpus:
	./scripts/build_fixture_corpus.sh

tokenizer-dev:
	./scripts/train_dev_tokenizer.sh

train-dev:
	./scripts/train_dev_model.sh

evaluate-dev:
	./scripts/evaluate_dev_model.sh

reproduce-dev: fixture-corpus tokenizer-dev train-dev evaluate-dev
