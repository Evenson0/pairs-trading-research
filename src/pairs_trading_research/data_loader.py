"""Data loading utilities for historical market prices."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _extract_prices(
    data: pd.DataFrame,
    tickers: list[str],
    price_field: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        first_level = (
            data.columns
            .get_level_values(0)
        )

        if (
            price_field
            not in first_level
        ):
            raise ValueError(
                f"Price field '{price_field}' not found in downloaded data."
            )

        prices = (
            data[
                price_field
            ].copy()
        )

        if isinstance(
            prices,
            pd.Series,
        ):
            prices = (
                prices
                .to_frame(
                    name=tickers[0]
                )
            )

        return prices

    if (
        price_field
        not in data.columns
    ):
        raise ValueError(
            f"Price field '{price_field}' not found in downloaded data."
        )

    prices = data[
        [
            price_field
        ]
    ].copy()

    prices.columns = [
        tickers[0]
    ]

    return prices


def download_adjusted_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    price_field: str = "Adj Close",
    auto_adjust: bool = False,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download Yahoo Finance prices in batches."""
    ticker_list = list(
        dict.fromkeys(
            tickers
        )
    )

    if not ticker_list:
        raise ValueError(
            "At least one ticker is required."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    frames: list[
        pd.DataFrame
    ] = []

    for start_idx in range(
        0,
        len(ticker_list),
        batch_size,
    ):
        batch = ticker_list[
            start_idx:
            start_idx + batch_size
        ]

        data = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=True,
        )

        prices = _extract_prices(
            data,
            batch,
            price_field,
        )

        if not prices.empty:
            frames.append(
                prices
            )

    if not frames:
        raise ValueError(
            "No data was downloaded. Check tickers and date range."
        )

    prices = pd.concat(
        frames,
        axis=1,
    )

    prices = prices.loc[
        :,
        ~prices.columns.duplicated(),
    ]

    prices = (
        prices
        .sort_index()
        .dropna(
            axis=1,
            how="all",
        )
    )

    return prices


def save_prices(
    prices: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_csv(
        path,
        index=True,
    )


def load_prices(
    input_path: str | Path,
) -> pd.DataFrame:
    prices = pd.read_csv(
        input_path,
        index_col=0,
        parse_dates=True,
    )

    return (
        prices
        .sort_index()
    )
