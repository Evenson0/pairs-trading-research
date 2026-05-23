# Research Notes

This document records methodological notes, assumptions, and known limitations for the **Pairs Trading Research** project.

## 1. Cointegration and Pairs Trading

Pairs trading is based on the idea that two assets may share a long-term equilibrium relationship. If the spread between them is mean-reverting, temporary deviations from equilibrium may be used to generate trading signals.

Correlation alone is not sufficient. Two assets can be highly correlated without having a stable mean-reverting spread. This project therefore focuses on cointegration-based pair selection.

## 2. Baseline Spread Construction

For a candidate pair, the hedge ratio is estimated using an ordinary least squares regression:

```text
Y_t = alpha + beta X_t + epsilon_t
```

The spread is then constructed as:

```text
spread_t = Y_t - beta X_t
```

A stationary spread is interpreted as evidence of a potential mean-reverting relationship.

## 3. Rolling Z-Score

The baseline strategy uses a rolling z-score:

```text
z_t = (spread_t - rolling_mean_t) / rolling_std_t
```

Using a rolling window is important because computing the z-score over the full sample would introduce look-ahead bias.

## 4. Look-Ahead Bias

Look-ahead bias occurs when a backtest uses information that would not have been available at the time of trading.

Examples:

- Selecting a pair using the full historical period and then backtesting on the same period
- Computing z-scores using full-sample means and standard deviations
- Using future prices to estimate hedge ratios
- Evaluating signals before the data would have been known

The project aims to reduce this bias by using rolling windows, lagged positions, and out-of-sample evaluation.

## 5. Survivorship Bias

The initial implementation may use current index constituents. This creates survivorship bias because the historical backtest ignores companies that were previously in the index but later removed.

This limitation should be clearly disclosed in reports. A future version may include historical constituent data.

## 6. Multiple Testing

When many pairs are tested, some may appear statistically significant by chance.

This is especially important for large universes such as the S&P 500, where the number of possible pairs is very large.

Future versions should consider:

- Sector filtering
- Minimum correlation filters
- False discovery rate controls
- Out-of-sample confirmation
- Walk-forward validation

## 7. Transaction Costs and Slippage

Backtests without transaction costs are often overly optimistic.

The project includes configurable transaction costs in basis points. Future versions may add:

- Slippage
- Bid-ask spread assumptions
- Borrow costs for short positions
- Market impact assumptions

## 8. Dollar Neutrality

A more realistic pairs trading strategy should be dollar-neutral or beta-neutral.

The baseline implementation is simplified and spread-based. Future versions should model both legs of the trade explicitly:

```text
Long leg value ≈ Short leg value
```

or

```text
Portfolio beta ≈ 0
```

## 9. PCA Factor Models

PCA can be used to identify common sources of variation across assets. In this project, PCA-based factor models are planned as an extension to separate common market or sector movements from residual behavior.

The residuals can then be studied to determine whether additional mean-reverting structure remains.

## 10. ARMA Residual Modeling

ARMA models may be used to model short-term residual dynamics. However, they should not be treated as guaranteed forecasting tools.

Their usefulness must be evaluated out-of-sample and compared against the simpler rolling z-score baseline.

## 11. Research Philosophy

The project prioritizes:

- Reproducibility
- Transparent assumptions
- Modular implementation
- Out-of-sample testing
- Honest discussion of limitations

The goal is not to claim that pairs trading is automatically profitable, but to build a rigorous framework for testing when, where, and how such strategies may fail or succeed.
