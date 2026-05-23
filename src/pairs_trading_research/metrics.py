"""Performance metrics for trading strategy evaluation."""

from __future__ import annotations

import pandas as pd


def compute_total_return(portfolio_value: pd.Series) -> float:
    """Compute total return from a portfolio value series."""
    if portfolio_value.empty:
        raise ValueError("Portfolio value series is empty.")

    return float(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1)


def compute_annualized_return(
    portfolio_value: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return from a portfolio value series."""
    if portfolio_value.empty:
        raise ValueError("Portfolio value series is empty.")

    n_periods = len(portfolio_value)

    if n_periods <= 1:
        return 0.0

    total_return = compute_total_return(portfolio_value)

    return float((1 + total_return) ** (periods_per_year / n_periods) - 1)


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility from a return series."""
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return 0.0

    return float(clean_returns.std() * (periods_per_year**0.5))


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic strategy returns.
    risk_free_rate:
        Annual risk-free rate expressed as a decimal.
    periods_per_year:
        Number of return periods per year.
    """
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return 0.0

    periodic_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = clean_returns - periodic_risk_free_rate

    volatility = excess_returns.std()

    if volatility == 0 or pd.isna(volatility):
        return 0.0

    return float((excess_returns.mean() / volatility) * (periods_per_year**0.5))


def compute_drawdown(portfolio_value: pd.Series) -> pd.Series:
    """Compute drawdown series from portfolio value."""
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1
    drawdown.name = "drawdown"

    return drawdown


def compute_max_drawdown(portfolio_value: pd.Series) -> float:
    """Compute maximum drawdown from portfolio value."""
    drawdown = compute_drawdown(portfolio_value)

    return float(drawdown.min())


def compute_win_rate(returns: pd.Series) -> float:
    """Compute the fraction of positive returns."""
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return 0.0

    return float((clean_returns > 0).mean())


def compute_performance_summary(
    portfolio_value: pd.Series,
    returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute a complete performance summary.

    Parameters
    ----------
    portfolio_value:
        Portfolio value series.
    returns:
        Optional return series. If None, returns are computed from portfolio value.
    risk_free_rate:
        Annual risk-free rate.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    dict[str, float]
        Performance summary.
    """
    if returns is None:
        returns = portfolio_value.pct_change().dropna()

    return {
        "total_return": compute_total_return(portfolio_value),
        "annualized_return": compute_annualized_return(
            portfolio_value,
            periods_per_year=periods_per_year,
        ),
        "annualized_volatility": compute_annualized_volatility(
            returns,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": compute_sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": compute_max_drawdown(portfolio_value),
        "win_rate": compute_win_rate(returns),
    }
