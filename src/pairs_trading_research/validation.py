"""Chronological validation helpers."""

from __future__ import annotations

import pandas as pd


def split_train_test(
    prices: pd.DataFrame,
    train_ratio: float = 0.70,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Chronologically split prices into training and test data."""
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError(
            "train_ratio must be in [0.5, 1.0)."
        )

    if len(prices) < 20:
        raise ValueError(
            "At least 20 observations are required."
        )

    split_index = int(
        len(prices)
        * train_ratio
    )

    split_index = min(
        max(
            split_index,
            10,
        ),
        len(prices) - 5,
    )

    train = prices.iloc[
        :split_index
    ].copy()

    test = prices.iloc[
        split_index:
    ].copy()

    return train, test


def with_training_warmup(
    train_series: pd.Series,
    test_series: pd.Series,
    warmup_periods: int,
) -> pd.Series:
    """Prepend training history for OOS rolling statistics."""
    if warmup_periods < 1:
        raise ValueError(
            "warmup_periods must be positive."
        )

    history = train_series.iloc[
        -warmup_periods:
    ]

    combined = pd.concat(
        [
            history,
            test_series,
        ]
    )

    return combined[
        ~combined.index.duplicated(
            keep="last"
        )
    ]
