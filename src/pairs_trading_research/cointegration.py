"""Cointegration utilities for pairs trading research."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def _align_series(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [
            y.rename("y"),
            x.rename("x"),
        ],
        axis=1,
    ).dropna()

    if len(aligned) < 3:
        raise ValueError("At least three aligned observations are required.")

    return aligned


def estimate_hedge_parameters(
    y: pd.Series,
    x: pd.Series,
    add_constant: bool = True,
) -> dict[str, float]:
    """Estimate OLS intercept and hedge ratio.

    Model
    -----
    y_t = alpha + beta * x_t + epsilon_t
    """
    aligned = _align_series(y, x)

    y_aligned = aligned["y"]
    x_aligned = aligned["x"]

    if add_constant:
        design = sm.add_constant(x_aligned)
        model = sm.OLS(y_aligned, design).fit()

        intercept = float(model.params.iloc[0])
        hedge_ratio = float(model.params.iloc[1])
    else:
        model = sm.OLS(y_aligned, x_aligned).fit()

        intercept = 0.0
        hedge_ratio = float(model.params.iloc[0])

    return {
        "intercept": intercept,
        "hedge_ratio": hedge_ratio,
        "r_squared": float(model.rsquared),
    }


def estimate_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    add_constant: bool = True,
) -> float:
    """Return only the OLS hedge ratio."""
    return estimate_hedge_parameters(
        y,
        x,
        add_constant=add_constant,
    )["hedge_ratio"]


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
) -> pd.Series:
    """Compute the OLS residual spread.

    spread_t = y_t - alpha - beta*x_t
    """
    aligned = _align_series(y, x)

    spread = (
        aligned["y"]
        - intercept
        - hedge_ratio * aligned["x"]
    )

    spread.name = "spread"

    return spread


def adf_test(series: pd.Series) -> dict[str, Any]:
    """Run an Augmented Dickey-Fuller test."""
    clean = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(clean) < 10:
        raise ValueError(
            "At least ten observations are required for the ADF test."
        )

    result = adfuller(clean)

    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": int(result[2]),
        "n_observations": int(result[3]),
        "critical_values": result[4],
    }


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
) -> dict[str, Any]:
    """Run the Engle-Granger cointegration test."""
    aligned = _align_series(y, x)

    statistic, p_value, critical_values = coint(
        aligned["y"],
        aligned["x"],
    )

    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "critical_values": critical_values,
    }


def analyze_pair(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> dict[str, Any]:
    """Return statistical diagnostics for one candidate pair."""
    y = prices[ticker_y]
    x = prices[ticker_x]

    parameters = estimate_hedge_parameters(y, x)

    spread = compute_spread(
        y,
        x,
        hedge_ratio=parameters["hedge_ratio"],
        intercept=parameters["intercept"],
    )

    adf_result = adf_test(spread)
    coint_result = engle_granger_test(y, x)

    correlation = float(
        pd.concat([y, x], axis=1)
        .dropna()
        .corr()
        .iloc[0, 1]
    )

    return {
        "ticker_y": ticker_y,
        "ticker_x": ticker_x,
        "intercept": parameters["intercept"],
        "hedge_ratio": parameters["hedge_ratio"],
        "r_squared": parameters["r_squared"],
        "spread_adf_p_value": adf_result["p_value"],
        "coint_p_value": coint_result["p_value"],
        "correlation": correlation,
        "n_observations": len(spread),
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    max_p_value: float = 0.05,
    min_correlation: float = 0.50,
) -> pd.DataFrame:
    """Legacy pair finder.

    Prefer pair_selection.select_pairs() for production research.
    """
    results: list[dict[str, Any]] = []

    for ticker_y, ticker_x in combinations(
        prices.columns,
        2,
    ):
        try:
            pair_result = analyze_pair(
                prices,
                ticker_y,
                ticker_x,
            )
        except (
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue

        if (
            pair_result["coint_p_value"] <= max_p_value
            and abs(pair_result["correlation"]) >= min_correlation
        ):
            results.append(pair_result)

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("coint_p_value")
        .reset_index(drop=True)
    )
