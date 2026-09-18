# Project Roadmap

## v1.0 — Research and Paper-Trading System

The v1.0 objective is a complete end-to-end pairs-trading workflow that is scientifically defensible and practically usable for research and paper trading.

### Completed scope

- Chronological train/test separation
- Training-only pair selection and hedge-ratio estimation
- Engle-Granger cointegration testing
- Benjamini-Hochberg false-discovery-rate correction
- Spread ADF diagnostics
- Mean-reversion half-life estimation
- Rolling hedge-ratio stability score
- Multi-criterion candidate ranking
- Rolling z-score signals
- Entry, hold, exit, stop, watch, and extreme-no-entry states
- Explicit two-leg portfolio backtest
- Dollar-neutral or hedge-ratio-weighted portfolio modes
- Turnover-based transaction costs and slippage
- Current TSX and S&P 500 scanning workflows
- Sector-aware S&P 500 pair filtering
- Walk-forward pair reselection and re-estimation
- Portfolio risk limits
- Persistent local paper broker
- Trade, position, cost, PnL, and equity logs
- Automated baseline research report
- Unit tests for the core research and trading logic

## Research rule

New models are not automatically promoted into the trading system.

A candidate model must be compared against the simple baseline using held-out and walk-forward results after costs. A more complex model should be rejected when it does not improve robustness or risk-adjusted performance.

## Post-v1 experiments

### PCA residual model

Research whether common-factor removal produces more stable mean-reverting residual relationships.

### ARMA residual forecasts

Test whether short-horizon residual forecasts improve decisions relative to the rolling z-score baseline.

### Historical index membership

Replace current-constituent historical studies with point-in-time membership data to reduce survivorship bias.

### Better execution model

Potential additions:

- bid-ask spread estimates
- borrow fees
- borrow availability
- volume/liquidity filters
- market impact
- fractional-share constraints

### Portfolio optimization

Potential additions:

- beta-neutral sizing
- volatility targeting
- covariance-aware multi-pair allocation
- sector and factor exposure controls

### Broker integration

A real broker adapter should remain a separate, explicitly enabled layer. Paper trading is the default and v1.0 boundary.
