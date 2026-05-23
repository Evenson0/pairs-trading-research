# TSX60 Baseline Case Study

This notebook reproduces and generalizes the original TSX60 pairs trading case study.

## Objective

The objective is to build a baseline pairs trading strategy using:

- S&P/TSX 60 equity prices
- Cointegration-based pair selection
- Rolling z-score signal generation
- Paper-trading backtest
- Out-of-sample performance evaluation

## Research Workflow

1. Load the TSX60 universe.
2. Download historical adjusted prices.
3. Clean and align price series.
4. Search for cointegrated pairs.
5. Select the best candidate pair.
6. Estimate the hedge ratio.
7. Construct the spread.
8. Compute rolling z-scores.
9. Generate long, short, and flat positions.
10. Run the baseline backtest.
11. Evaluate performance metrics.
12. Visualize prices, spread, z-score, positions, and portfolio value.

## Notes

This is the baseline version of the project. It focuses on the naive cointegration and rolling z-score strategy.

Future notebooks will add:

- PCA factor modeling
- ARMA residual forecasting
- Walk-forward analysis
- S&P 500 extension
