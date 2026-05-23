.PHONY: install test format lint run-tsx60 clean

install:
	pip install -e ".[dev]"

test:
	pytest

format:
	black src tests scripts

lint:
	ruff check src tests scripts

run-tsx60:
	python scripts/run_tsx60_pipeline.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
