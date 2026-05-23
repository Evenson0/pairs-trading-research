"""Preprocessing utilities for price and return data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_assets_by_missing_ratio(
    prices: pd.DataFrame,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """Remove assets with too many missing price observations."""
    missing_ratio = prices.isna().mean()
    selected_columns = missing_ratio[missing_ratio <= max_missing_ratio].index

    return prices.loc[:, selected_columns]


def filter_assets_by_min_observations(
    prices: pd.DataFrame,
    min_observations: int = 90,
) -> pd.DataFrame:
    """Remove assets with fewer than a minimum number of observations."""
    observation_count = prices.notna().sum()
    selected_columns = observation_count[observation_count >= min_observations].index

    return prices.loc[:, selected_columns]


def clean_price_data(
    prices: pd.DataFrame,
    max_missing_ratio: float = 0.05,
    min_observations: int = 90,
    fill_method: str | None = "ffill",
) -> pd.DataFrame:
    """Clean historical price data.

    The function removes assets with too many missing values, removes assets
    with too few observations, optionally forward-fills missing prices, and
    drops remaining missing rows.
    """
    cleaned = prices.copy()
    cleaned = filter_assets_by_missing_ratio(cleaned, max_missing_ratio)
    cleaned = filter_assets_by_min_observations(cleaned, min_observations)

    if fill_method == "ffill":
        cleaned = cleaned.ffill()
    elif fill_method is not None:
        raise ValueError("fill_method must be either 'ffill' or None.")

    cleaned = cleaned.dropna(axis=0, how="any")

    return cleaned


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns from price data."""
    return prices.pct_change().dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    return np.log(prices / prices.shift(1)).dropna(how="all")
