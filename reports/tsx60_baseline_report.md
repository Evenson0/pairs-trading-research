# TSX60 Baseline Pairs Trading Report

## 1. Executive Summary

This report presents a baseline pairs trading strategy applied to equities from the S&P/TSX 60 universe.

The strategy uses cointegration analysis to identify a candidate pair, constructs a price spread using an estimated hedge ratio, generates rolling z-score signals, and evaluates a simple paper-trading strategy out-of-sample.

At this stage, the report is a template. Numerical results will be added after running the full pipeline.

## 2. Research Objective

The objective is to evaluate whether a pair of equities from the S&P/TSX 60 exhibits a mean-reverting relationship that can be used to generate systematic trading signals.

The baseline strategy focuses on:

- Cointegration-based pair selection
- Hedge ratio estimation
- Spread construction
- Rolling z-score signal generation
- Paper-trading backtest
- Performance evaluation

## 3. Data

The initial universe is the S&P/TSX 60.

The project uses daily adjusted closing prices downloaded from Yahoo Finance.

Key data parameters:

```text
Universe: S&P/TSX 60
Frequency: Daily
Price field: Adjusted close
Data source: Yahoo Finance
```

## 4. Methodology

### 4.1 Pair Selection

Candidate pairs are evaluated using the Engle-Granger cointegration test.

A pair is retained if:

```text
cointegration p-value <= configured threshold
absolute correlation >= configured threshold
```

### 4.2 Hedge Ratio

For each selected pair, the hedge ratio is estimated using an ordinary least squares regression:

```text
Y_t = alpha + beta X_t + epsilon_t
```

The estimated beta is used to construct the spread:

```text
spread_t = Y_t - beta X_t
```

### 4.3 Signal Generation

A rolling z-score is computed from the spread.

Trading rules:

```text
z-score <= -2.0    long spread
z-score >=  2.0    short spread
|z-score| <= 0.5   close position
```

### 4.4 Backtesting

The baseline backtest uses lagged positions to avoid look-ahead bias.

Transaction costs are included using a basis-point assumption from the configuration file.

## 5. Results

To be completed after running:

```bash
python scripts/run_tsx60_pipeline.py
```

Expected outputs:

```text
Selected pair:
Hedge ratio:
Cointegration p-value:
Spread ADF p-value:
Total return:
Annualized return:
Annualized volatility:
Sharpe ratio:
Maximum drawdown:
Win rate:
```

## 6. Discussion

This section will discuss:

- Whether the selected pair is economically meaningful
- Whether the spread appears mean-reverting
- Whether the z-score signals produce stable trading behavior
- Whether performance survives transaction costs
- Whether the results are robust enough for further modeling

## 7. Limitations

The baseline strategy has several limitations:

- The first version uses current index constituents.
- The backtest is simplified.
- The strategy does not yet include a full dollar-neutral position sizing model.
- Transaction costs are approximate.
- Pair selection may be affected by multiple testing.
- Results may be sensitive to the chosen historical window.

## 8. Next Steps

Future versions will add:

- PCA-based factor modeling
- ARMA residual forecasting
- Walk-forward analysis
- S&P 500 universe support
- More realistic portfolio construction
- Automated report generation

## 9. Disclaimer

This report is for research and educational purposes only. It does not constitute financial advice, investment advice, or a recommendation to buy or sell any security.
