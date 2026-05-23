"""Tests for performance metrics."""

import pandas as pd

from pairs_trading_research.metrics import (
    compute_drawdown,
    compute_total_return,
    compute_win_rate,
)


def test_compute_total_return() -> None:
    """Test total return calculation."""
    portfolio_value = pd.Series([100, 110, 121], dtype=float)

    total_return = compute_total_return(portfolio_value)

    assert round(total_return, 2) == 0.21


def test_compute_drawdown() -> None:
    """Test drawdown calculation."""
    portfolio_value = pd.Series([100, 120, 90, 130], dtype=float)

    drawdown = compute_drawdown(portfolio_value)

    assert drawdown.iloc[0] == 0
    assert drawdown.iloc[1] == 0
    assert round(drawdown.iloc[2], 2) == -0.25
    assert drawdown.iloc[3] == 0


def test_compute_win_rate() -> None:
    """Test win rate calculation."""
    returns = pd.Series([0.01, -0.02, 0.03, 0.00], dtype=float)

    win_rate = compute_win_rate(returns)

    assert win_rate == 0.5
