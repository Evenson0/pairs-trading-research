"""Visualization utilities for pairs trading research."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_pair_prices(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> None:
    """Plot the price series of a selected pair."""
    prices[[ticker_y, ticker_x]].plot(figsize=(12, 6))
    plt.title(f"Price Series: {ticker_y} and {ticker_x}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_spread(spread: pd.Series) -> None:
    """Plot the spread series."""
    spread.plot(figsize=(12, 5))
    plt.title("Pair Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.tight_layout()
    plt.show()


def plot_zscore(zscore: pd.Series) -> None:
    """Plot the rolling z-score with common trading thresholds."""
    zscore.plot(figsize=(12, 5))

    plt.axhline(2.0, linestyle="--", linewidth=1)
    plt.axhline(-2.0, linestyle="--", linewidth=1)
    plt.axhline(0.5, linestyle=":", linewidth=1)
    plt.axhline(-0.5, linestyle=":", linewidth=1)
    plt.axhline(0.0, linewidth=1)

    plt.title("Rolling Z-Score")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.tight_layout()
    plt.show()


def plot_positions(positions: pd.Series) -> None:
    """Plot trading positions over time."""
    positions.plot(figsize=(12, 4))
    plt.title("Trading Positions")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.show()


def plot_portfolio_value(results: pd.DataFrame) -> None:
    """Plot portfolio value from backtest results."""
    results["portfolio_value"].plot(figsize=(12, 5))
    plt.title("Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.show()


def plot_drawdown(portfolio_value: pd.Series) -> None:
    """Plot portfolio drawdown."""
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1

    drawdown.plot(figsize=(12, 5))
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def plot_backtest_summary(results: pd.DataFrame) -> None:
    """Plot portfolio value and drawdown from backtest results."""
    plot_portfolio_value(results)
    plot_drawdown(results["portfolio_value"])
