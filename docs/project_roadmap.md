# Project Roadmap

This roadmap outlines the planned development of the Pairs Trading Research project.

## Phase 1 — Baseline Research Pipeline

Goal: build a complete baseline pipeline for cointegration-based pairs trading on the S&P/TSX 60.

Planned tasks:

- Define project structure
- Add configuration files
- Add ticker universe definitions
- Add historical price data loader
- Add price preprocessing utilities
- Add cointegration analysis tools
- Add rolling z-score signal generation
- Add simple spread-based backtest
- Add performance metrics
- Add visualization utilities
- Add TSX60 baseline pipeline script
- Add TSX60 baseline report template

Status: in progress

## Phase 2 — Research Notebook

Goal: create a clear notebook that reproduces the TSX60 baseline case study.

Planned tasks:

- Convert the Markdown notebook outline into a Jupyter notebook
- Add step-by-step explanations
- Add figures for prices, spread, z-score, positions, and portfolio value
- Add interpretation of selected pair
- Add performance discussion

Status: planned

## Phase 3 — Factor Model Extension

Goal: extend the baseline strategy using PCA-based factor modeling.

Planned tasks:

- Add PCA factor model module
- Compute factor loadings
- Compare profiles of selected equities
- Extract residual returns
- Interpret residual behavior
- Document the factor model methodology

Status: planned

## Phase 4 — ARMA Residual Modeling

Goal: model residual dynamics and compare the ARMA-based approach with the naive z-score strategy.

Planned tasks:

- Add ARMA residual modeling module
- Add residual diagnostics
- Add short-horizon residual forecasts
- Generate model-based position signals
- Compare baseline and ARMA-enhanced strategies

Status: planned

## Phase 5 — Walk-Forward Evaluation

Goal: evaluate the robustness of the strategy through rolling study windows.

Planned tasks:

- Add walk-forward analysis module
- Re-estimate pairs and hedge ratios through time
- Update signals every few days
- Track position changes
- Compare stability of results

Status: planned

## Phase 6 — S&P 500 Extension

Goal: extend the project from the S&P/TSX 60 to the S&P 500.

Planned tasks:

- Add full S&P 500 universe loader
- Add sector-aware pair filtering
- Handle larger-scale pair search
- Add multiple-testing warnings or corrections
- Compare TSX60 and S&P500 results

Status: planned

## Phase 7 — Reporting and Automation

Goal: make the project easier to run and easier to present.

Planned tasks:

- Add automated report generation
- Save selected pairs and backtest summaries to CSV
- Export figures to the reports directory
- Add command-line arguments
- Add tests for core modules

Status: planned

## Long-Term Ideas

Potential future extensions:

- ETF pairs trading
- Sector-specific statistical arbitrage
- Multi-pair portfolio construction
- Dollar-neutral and beta-neutral portfolio sizing
- Slippage and borrow-cost modeling
- Interactive dashboard
- Blog article integration
