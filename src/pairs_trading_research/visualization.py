"""Visualization utilities for research outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _finish_figure(
    output_path: str | Path | None,
    show: bool,
) -> None:
    plt.tight_layout()

    if output_path is not None:
        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    else:
        plt.close()


def plot_pair_prices(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    prices[
        [
            ticker_y,
            ticker_x,
        ]
    ].plot(
        figsize=(
            12,
            6,
        )
    )

    plt.title(
        f"Price Series: {ticker_y} and {ticker_x}"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price"
    )

    _finish_figure(
        output_path,
        show,
    )


def plot_spread(
    spread: pd.Series,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    spread.plot(
        figsize=(
            12,
            5,
        )
    )

    plt.title(
        "Pair Spread"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Spread"
    )

    _finish_figure(
        output_path,
        show,
    )


def plot_zscore(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    zscore.plot(
        figsize=(
            12,
            5,
        )
    )

    plt.axhline(
        entry_threshold,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        -entry_threshold,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        exit_threshold,
        linestyle=":",
        linewidth=1,
    )

    plt.axhline(
        -exit_threshold,
        linestyle=":",
        linewidth=1,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.title(
        "Rolling Z-Score"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Z-Score"
    )

    _finish_figure(
        output_path,
        show,
    )


def plot_portfolio_value(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    results[
        "portfolio_value"
    ].plot(
        figsize=(
            12,
            5,
        )
    )

    plt.title(
        "Portfolio Value"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Portfolio Value"
    )

    _finish_figure(
        output_path,
        show,
    )


def plot_drawdown(
    portfolio_value: pd.Series,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    running_max = (
        portfolio_value
        .cummax()
    )

    drawdown = (
        portfolio_value
        / running_max
        - 1
    )

    drawdown.plot(
        figsize=(
            12,
            5,
        )
    )

    plt.title(
        "Drawdown"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Drawdown"
    )

    _finish_figure(
        output_path,
        show,
    )
