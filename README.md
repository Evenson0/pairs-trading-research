# Pairs Trading Research

A reproducible framework for studying cointegration-based statistical arbitrage.

## Overview

**Pairs Trading Research** is an open-source quantitative finance project focused on designing, testing, and comparing market-neutral strategies across equity universes such as the S&P/TSX 60 and the S&P 500.

The project combines cointegration analysis, rolling z-score signals, PCA-based factor models, ARMA residual forecasting, and walk-forward backtesting to evaluate whether mean-reverting relationships between equities can generate robust paper-trading signals.

## Project Motivation

This project started from an academic case study on two S&P/TSX 60 equities. It is being extended into a modular research platform that supports multiple universes, systematic pair selection, model comparison, and out-of-sample performance evaluation.

## Main Features

- Equity universe support for S&P/TSX 60 and S&P 500
- Historical price data collection and cleaning
- Cointegration-based pair selection
- Rolling z-score signal generation
- Naive paper-trading strategy
- PCA-based factor modeling
- ARMA residual modeling
- Walk-forward analysis with rolling study windows
- Out-of-sample performance evaluation
- Modular Python codebase for future extensions

## Repository Structure

```text
pairs-trading-research/
│
├── config/       # Configuration files for universes, dates, and strategy parameters
├── data/         # Raw and processed data, usually ignored if too large
├── docs/         # Methodology notes, roadmap, and project documentation
├── notebooks/    # Exploratory research notebooks and case studies
├── reports/      # Generated reports, figures, and results
├── scripts/      # Executable scripts for running pipelines
├── src/          # Reusable Python source code
├── tests/        # Unit tests
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Evenson0/pairs-trading-research.git
cd pairs-trading-research
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

Run the baseline S&P/TSX 60 pipeline:

```bash
python scripts/run_tsx60_pipeline.py
```

The script will:

1. Load the S&P/TSX 60 universe.
2. Download adjusted price data.
3. Clean and align the price series.
4. Search for cointegrated pairs.
5. Select the best candidate pair.
6. Generate rolling z-score trading signals.
7. Run a baseline paper-trading backtest.
8. Print a performance summary.

## Running Tests

Run the unit tests with:

```bash
pytest
```

## Research Pipeline

The project is designed around the following workflow:

1. Load an equity universe.
2. Download and clean historical price data.
3. Identify candidate pairs using cointegration tests.
4. Estimate hedge ratios and construct spreads.
5. Generate rolling z-score trading signals.
6. Backtest a baseline strategy.
7. Build PCA-based factor models for selected equities.
8. Model residuals using ARMA processes.
9. Compare naive and model-based approaches.
10. Evaluate robustness through walk-forward analysis.

## Current Status

This project is under active development.

The first version focuses on reproducing and generalizing the original S&P/TSX 60 case study. Future versions will extend the framework to the S&P 500 and custom equity universes.

## Disclaimer

This project is for research and educational purposes only. It does not constitute financial advice, investment advice, or a recommendation to buy or sell any security.

The strategies implemented in this repository are designed for paper trading and research evaluation only.

## License

This project is released under the MIT License.
