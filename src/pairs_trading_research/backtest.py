"""Backtesting utilities for pairs trading strategies."""

from __future__ import annotations

import pandas as pd


def compute_spread_returns(spread: pd.Series) -> pd.Series:
    """Compute spread returns from a spread series.

    Parameters
    ----------
    spread:
        Spread time series.

    Returns
    -------
    pd.Series
        Spread changes.
    """
    spread_returns = spread.diff()
    spread_returns.name = "spread_return"

    return spread_returns


def run_spread_backtest(
    spread: pd.Series,
    positions: pd.Series,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Run a simple spread-based pairs trading backtest.

    Parameters
    ----------
    spread:
        Spread time series.
    positions:
        Trading positions with values -1, 0, or 1.
    initial_capital:
        Initial portfolio value.
    transaction_cost_bps:
        Transaction cost in basis points applied when the position changes.

    Returns
    -------
    pd.DataFrame
        Backtest results containing spread, position, strategy returns,
        transaction costs, and portfolio value.
    """
    data = pd.concat(
        [
            spread.rename("spread"),
            positions.rename("position"),
        ],
        axis=1,
    ).dropna()

    data["spread_return"] = compute_spread_returns(data["spread"])

    data["lagged_position"] = data["position"].shift(1).fillna(0)

    data["gross_pnl"] = data["lagged_position"] * data["spread_return"]

    data["position_change"] = data["position"].diff().abs().fillna(0)

    transaction_cost_rate = transaction_cost_bps / 10_000

    data["transaction_cost"] = data["position_change"] * transaction_cost_rate

    scale = data["spread"].abs().median()

    if scale == 0 or pd.isna(scale):
        scale = 1.0

    data["strategy_return"] = (data["gross_pnl"] / scale) - data["transaction_cost"]

    data["portfolio_value"] = initial_capital * (1 + data["strategy_return"]).cumprod()

    return data


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    """Summarize a backtest result.

    Parameters
    ----------
    results:
        Backtest results produced by run_spread_backtest.

    Returns
    -------
    dict[str, float]
        Summary statistics.
    """
    if results.empty:
        raise ValueError("Backtest results are empty.")

    initial_value = results["portfolio_value"].iloc[0]
    final_value = results["portfolio_value"].iloc[-1]

    total_return = final_value / initial_value - 1

    daily_returns = results["portfolio_value"].pct_change().dropna()

    volatility = daily_returns.std()

    if volatility == 0 or pd.isna(volatility):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / volatility) * (252 ** 0.5)

    running_max = results["portfolio_value"].cummax()
    drawdown = results["portfolio_value"] / running_max - 1
    max_drawdown = drawdown.min()

    number_of_trades = int((results["position_change"] > 0).sum())

    return {
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "annualized_volatility": float(volatility * (252 ** 0.5)),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "number_of_trades": float(number_of_trades),
    }
