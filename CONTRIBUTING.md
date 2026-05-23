# Contributing

Thank you for your interest in contributing to **Pairs Trading Research**.

This project is an open-source research framework for studying cointegration-based statistical arbitrage, pairs trading strategies, factor models, residual forecasting, and walk-forward backtesting.

## How to Contribute

You can contribute by:

- Improving documentation
- Fixing bugs
- Adding tests
- Improving data loading utilities
- Adding new equity universes
- Improving backtesting logic
- Adding new signal generation methods
- Extending the PCA or ARMA modeling components
- Improving visualizations and reports

## Development Principles

Contributions should follow these principles:

- Keep the code modular and readable.
- Prefer transparent methods over unnecessary complexity.
- Avoid look-ahead bias in all trading logic.
- Document assumptions clearly.
- Include tests when adding core functionality.
- Keep research results reproducible.

## Code Style

The project uses:

```text
black
ruff
pytest
```

Before submitting changes, contributors should run:

```bash
black src tests
ruff check src tests
pytest
```

## Financial Disclaimer

This project is for research and educational purposes only.

Nothing in this repository should be interpreted as financial advice, investment advice, or a recommendation to buy or sell any security.

## Pull Requests

When opening a pull request, please include:

- A short description of the change
- The motivation behind the change
- Any assumptions or limitations
- Screenshots or figures if the change affects visual outputs
- Tests if the change affects core functionality

## Issues

When opening an issue, please describe:

- The problem or proposed improvement
- Steps to reproduce, if relevant
- Expected behavior
- Actual behavior
- Your Python version and operating system, if relevant
