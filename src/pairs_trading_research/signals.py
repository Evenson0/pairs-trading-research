"""Signal generation utilities for pairs trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute a rolling z-score using current and past observations."""
    if window < 2:
        raise ValueError(
            "window must be at least 2."
        )

    rolling_mean = series.rolling(
        window=window,
        min_periods=window,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        min_periods=window,
    ).std(
        ddof=1
    )

    rolling_std = rolling_std.replace(
        0.0,
        np.nan,
    )

    zscore = (
        series
        - rolling_mean
    ) / rolling_std

    zscore.name = "zscore"

    return zscore.dropna()


def generate_zscore_actions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_zscore: float | None = 3.5,
) -> pd.DataFrame:
    """Generate positions and explicit trading actions."""
    if (
        exit_threshold < 0
        or entry_threshold
        <= exit_threshold
    ):
        raise ValueError(
            "Require 0 <= exit_threshold < entry_threshold."
        )

    if (
        stop_loss_zscore is not None
        and stop_loss_zscore
        <= entry_threshold
    ):
        raise ValueError(
            "stop_loss_zscore must exceed entry_threshold."
        )

    position_values: list[int] = []
    actions: list[str] = []

    current_position = 0

    for value in zscore.astype(float):
        action = "NONE"

        if current_position == 0:
            if (
                stop_loss_zscore
                is not None
                and abs(value)
                >= stop_loss_zscore
            ):
                action = (
                    "NO_ENTRY_EXTREME"
                )

            elif (
                value
                <= -entry_threshold
            ):
                current_position = 1
                action = "ENTRY_LONG"

            elif (
                value
                >= entry_threshold
            ):
                current_position = -1
                action = "ENTRY_SHORT"

        elif current_position == 1:
            if (
                stop_loss_zscore
                is not None
                and value
                <= -stop_loss_zscore
            ):
                current_position = 0
                action = "STOP_LONG"

            elif (
                abs(value)
                <= exit_threshold
                or value > 0
            ):
                current_position = 0
                action = "EXIT_LONG"

            else:
                action = "HOLD_LONG"

        elif current_position == -1:
            if (
                stop_loss_zscore
                is not None
                and value
                >= stop_loss_zscore
            ):
                current_position = 0
                action = "STOP_SHORT"

            elif (
                abs(value)
                <= exit_threshold
                or value < 0
            ):
                current_position = 0
                action = "EXIT_SHORT"

            else:
                action = "HOLD_SHORT"

        position_values.append(
            current_position
        )

        actions.append(
            action
        )

    return pd.DataFrame(
        {
            "position": pd.Series(
                position_values,
                index=zscore.index,
                dtype="int64",
            ),
            "action": pd.Series(
                actions,
                index=zscore.index,
                dtype="object",
            ),
        }
    )


def generate_zscore_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_zscore: float | None = None,
) -> pd.Series:
    """Return only the position series."""
    result = generate_zscore_actions(
        zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_zscore=stop_loss_zscore,
    )

    positions = (
        result["position"]
        .copy()
    )

    positions.name = "position"

    return positions


def generate_pair_signals(
    spread: pd.Series,
    zscore_window: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_zscore: float | None = 3.5,
) -> pd.DataFrame:
    """Generate spread, z-score, position and action."""
    zscore = compute_rolling_zscore(
        spread,
        window=zscore_window,
    )

    actions = generate_zscore_actions(
        zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_zscore=stop_loss_zscore,
    )

    return pd.concat(
        [
            spread.rename("spread"),
            zscore,
            actions,
        ],
        axis=1,
    ).dropna()


def classify_current_signal(
    zscore: float,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_zscore: float = 3.5,
    watch_threshold: float = 1.5,
) -> str:
    """Classify a current flat-book market opportunity."""
    if not np.isfinite(zscore):
        return "NO DATA"

    if (
        abs(zscore)
        >= stop_loss_zscore
    ):
        return (
            "EXTREME / NO ENTRY"
        )

    if (
        zscore
        <= -entry_threshold
    ):
        return "ENTRY LONG"

    if (
        zscore
        >= entry_threshold
    ):
        return "ENTRY SHORT"

    if (
        abs(zscore)
        >= watch_threshold
    ):
        return "WATCH"

    if (
        abs(zscore)
        <= exit_threshold
    ):
        return "NEUTRAL"

    return "NONE"
