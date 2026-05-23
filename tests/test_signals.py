"""Tests for signal generation utilities."""

import pandas as pd

from pairs_trading_research.signals import (
    compute_rolling_zscore,
    generate_zscore_positions,
)


def test_compute_rolling_zscore_returns_series() -> None:
    """Test that rolling z-score returns a pandas Series."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)

    zscore = compute_rolling_zscore(series, window=3)

    assert isinstance(zscore, pd.Series)
    assert not zscore.empty
    assert zscore.name == "zscore"


def test_generate_zscore_positions_enters_long_spread() -> None:
    """Test that a negative z-score triggers a long spread position."""
    zscore = pd.Series([-2.5, -1.5, -0.4], dtype=float)

    positions = generate_zscore_positions(
        zscore,
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    assert positions.iloc[0] == 1
    assert positions.iloc[1] == 1
    assert positions.iloc[2] == 0


def test_generate_zscore_positions_enters_short_spread() -> None:
    """Test that a positive z-score triggers a short spread position."""
    zscore = pd.Series([2.5, 1.5, 0.4], dtype=float)

    positions = generate_zscore_positions(
        zscore,
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    assert positions.iloc[0] == -1
    assert positions.iloc[1] == -1
    assert positions.iloc[2] == 0
