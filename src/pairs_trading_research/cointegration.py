"""Cointegration utilities for pairs trading research."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def estimate_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    add_constant: bool = True,
) -> float:
    """Estimate the hedge ratio between two price series using OLS.

    The model is:

        y_t = alpha + beta x_t + epsilon_t

    The hedge ratio is beta.

    Parameters
    ----------
    y:
        Dependent price series.
    x:
        Independent price series.
    add_constant:
        Whether to include an intercept in the OLS regression.

    Returns
    -------
    float
        Estimated hedge ratio.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_aligned = aligned.iloc[:, 0]
    x_aligned = aligned.iloc[:, 1]

    if add_constant:
        x_model = sm.add_constant(x_aligned)
    else:
        x_model = x_aligned

    model = sm.OLS(y_aligned, x_model).fit()

    if add_constant:
        return float(model.params.iloc[1])

    return float(model.params.iloc[0])


def compute_spread(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """Compute the price spread between two assets.

    The spread is defined as:

        spread_t = y_t - beta x_t

    Parameters
    ----------
    y:
        First price series.
    x:
        Second price series.
    hedge_ratio:
        Estimated hedge ratio beta.

    Returns
    -------
    pd.Series
        Spread series.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_aligned = aligned.iloc[:, 0]
    x_aligned = aligned.iloc[:, 1]

    spread = y_aligned - hedge_ratio * x_aligned
    spread.name = "spread"

    return spread


def adf_test(series: pd.Series) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller test on a time series.

    Parameters
    ----------
    series:
        Time series to test.

    Returns
    -------
    dict[str, Any]
        Test statistic, p-value, lags used, number of observations,
        and critical values.
    """
    clean_series = series.dropna()

    result = adfuller(clean_series)

    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
    }


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
) -> dict[str, Any]:
    """Run the Engle-Granger cointegration test on two price series.

    Parameters
    ----------
    y:
        First price series.
    x:
        Second price series.

    Returns
    -------
    dict[str, Any]
        Cointegration test statistic, p-value, and critical values.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_aligned = aligned.iloc[:, 0]
    x_aligned = aligned.iloc[:, 1]

    statistic, p_value, critical_values = coint(y_aligned, x_aligned)

    return {
        "test_statistic": statistic,
        "p_value": p_value,
        "critical_values": critical_values,
    }


def analyze_pair(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> dict[str, Any]:
    """Analyze a candidate pair for cointegration.

    Parameters
    ----------
    prices:
        Price DataFrame with tickers as columns.
    ticker_y:
        First ticker.
    ticker_x:
        Second ticker.

    Returns
    -------
    dict[str, Any]
        Pair diagnostics including hedge ratio, Engle-Granger p-value,
        ADF p-value on the spread, and correlation.
    """
    y = prices[ticker_y]
    x = prices[ticker_x]

    hedge_ratio = estimate_hedge_ratio(y, x)
    spread = compute_spread(y, x, hedge_ratio)
    adf_result = adf_test(spread)
    coint_result = engle_granger_test(y, x)
    correlation = y.corr(x)

    return {
        "ticker_y": ticker_y,
        "ticker_x": ticker_x,
        "hedge_ratio": hedge_ratio,
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
    """Find cointegrated pairs in a price DataFrame.

    Parameters
    ----------
    prices:
        Price DataFrame with tickers as columns.
    max_p_value:
        Maximum Engle-Granger p-value required to keep a pair.
    min_correlation:
        Minimum absolute correlation required to keep a pair.

    Returns
    -------
    pd.DataFrame
        DataFrame of selected pairs sorted by cointegration p-value.
    """
    results: list[dict[str, Any]] = []

    for ticker_y, ticker_x in combinations(prices.columns, 2):
        try:
            pair_result = analyze_pair(prices, ticker_y, ticker_x)
        except Exception:
            continue

        if (
            pair_result["coint_p_value"] <= max_p_value
            and abs(pair_result["correlation"]) >= min_correlation
        ):
            results.append(pair_result)

    if not results:
        return pd.DataFrame(
            columns=[
                "ticker_y",
                "ticker_x",
                "hedge_ratio",
                "spread_adf_p_value",
                "coint_p_value",
                "correlation",
                "n_observations",
            ]
        )

    return pd.DataFrame(results).sort_values("coint_p_value").reset_index(drop=True)
