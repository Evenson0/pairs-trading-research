"""Signal generation utilities for pairs trading strategies."""

from __future__ import annotations

import pandas as pd


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute a rolling z-score for a time series.

    Parameters
    ----------
    series:
        Input time series, usually a price spread.
    window:
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling z-score series.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    zscore = (series - rolling_mean) / rolling_std
    zscore.name = "zscore"

    return zscore.dropna()


def generate_zscore_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """Generate trading positions from a z-score signal.

    Position convention
    -------------------
    1:
        Long spread.
    -1:
        Short spread.
    0:
        No position.

    Trading rules
    -------------
    - If z-score <= -entry_threshold: long spread.
    - If z-score >= entry_threshold: short spread.
    - If abs(z-score) <= exit_threshold: close position.
    - Otherwise: keep previous position.

    Parameters
    ----------
    zscore:
        Rolling z-score series.
    entry_threshold:
        Absolute z-score level used to enter trades.
    exit_threshold:
        Absolute z-score level used to exit trades.

    Returns
    -------
    pd.Series
        Position series with values -1, 0, or 1.
    """
    positions = pd.Series(index=zscore.index, data=0, dtype="int64")
    current_position = 0

    for date, value in zscore.items():
        if current_position == 0:
            if value <= -entry_threshold:
                current_position = 1
            elif value >= entry_threshold:
                current_position = -1

        elif current_position == 1:
            if abs(value) <= exit_threshold:
                current_position = 0

        elif current_position == -1:
            if abs(value) <= exit_threshold:
                current_position = 0

        positions.loc[date] = current_position

    positions.name = "position"

    return positions


def generate_pair_signals(
    spread: pd.Series,
    zscore_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate z-score and position signals for a spread.

    Parameters
    ----------
    spread:
        Spread time series.
    zscore_window:
        Rolling window used to compute the z-score.
    entry_threshold:
        Entry threshold for the z-score strategy.
    exit_threshold:
        Exit threshold for the z-score strategy.

    Returns
    -------
    pd.DataFrame
        DataFrame containing spread, z-score, and position.
    """
    zscore = compute_rolling_zscore(spread, window=zscore_window)
    positions = generate_zscore_positions(
        zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )

    signals = pd.concat(
        [
            spread.rename("spread"),
            zscore,
            positions,
        ],
        axis=1,
    ).dropna()

    return signals
