"""Data loading utilities for historical market prices."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


def download_adjusted_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    price_field: str = "Adj Close",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download adjusted price data from Yahoo Finance.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start:
        Start date in YYYY-MM-DD format.
    end:
        End date in YYYY-MM-DD format. If None, yfinance uses the latest
        available data.
    price_field:
        Price field to extract from the downloaded data. Default is
        "Adj Close".
    auto_adjust:
        Whether yfinance should automatically adjust OHLC prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of prices with dates as index and tickers as columns.

    Raises
    ------
    ValueError
        If no tickers are provided or if the requested price field is missing.
    """
    ticker_list = list(tickers)

    if not ticker_list:
        raise ValueError("At least one ticker is required.")

    data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data was downloaded. Check tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        if price_field not in data.columns.get_level_values(0):
            raise ValueError(
                f"Price field '{price_field}' not found in downloaded data."
            )
        prices = data[price_field].copy()
    else:
        if price_field not in data.columns:
            raise ValueError(
                f"Price field '{price_field}' not found in downloaded data."
            )
        prices = data[[price_field]].copy()
        prices.columns = ticker_list

    prices = prices.sort_index()
    prices = prices.dropna(axis=1, how="all")

    return prices


def save_prices(prices: pd.DataFrame, output_path: str) -> None:
    """Save price data to a CSV file.

    Parameters
    ----------
    prices:
        Price DataFrame to save.
    output_path:
        Destination CSV file path.
    """
    prices.to_csv(output_path, index=True)


def load_prices(input_path: str) -> pd.DataFrame:
    """Load price data from a CSV file.

    Parameters
    ----------
    input_path:
        Path to a CSV file containing price data.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by date.
    """
    prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    return prices
