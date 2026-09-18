.PHONY: install test format lint check run-tsx60 scan-tsx60 scan-sp500 walk-tsx60 walk-sp500 paper-tsx60 paper-sp500 clean

install:
	pip install -e ".[dev]"

test:
	pytest

format:
	black src tests scripts

lint:
	ruff check src tests scripts

check:
	python scripts/check_setup.py

run-tsx60:
	python scripts/run_tsx60_pipeline.py

scan-tsx60:
	python scripts/scan_market.py --universe tsx60

scan-sp500:
	python scripts/scan_market.py --universe sp500

walk-tsx60:
	python scripts/run_walk_forward.py --universe tsx60

walk-sp500:
	python scripts/run_walk_forward.py --universe sp500

paper-tsx60:
	python scripts/paper_trade.py --universe tsx60

paper-sp500:
	python scripts/paper_trade.py --universe sp500

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
