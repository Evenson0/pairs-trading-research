"""Multiple-testing-aware pair selection and ranking."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .cointegration import (
    analyze_pair,
    compute_spread,
)
from .mean_reversion import (
    estimate_half_life,
    hedge_ratio_stability_score,
)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker_y",
            "ticker_x",
            "intercept",
            "hedge_ratio",
            "r_squared",
            "spread_adf_p_value",
            "coint_p_value",
            "adjusted_p_value",
            "correlation",
            "half_life",
            "hedge_ratio_stability",
            "pair_score",
            "n_observations",
        ]
    )


def _pair_score(
    row: pd.Series,
    alpha: float,
    max_half_life: float,
) -> float:
    adjusted_p = max(
        float(
            row["adjusted_p_value"]
        ),
        1e-12,
    )

    adf_p = max(
        float(
            row["spread_adf_p_value"]
        ),
        1e-12,
    )

    p_score = min(
        -np.log10(adjusted_p) / 6.0,
        1.0,
    )

    adf_score = min(
        -np.log10(adf_p) / 6.0,
        1.0,
    )

    correlation_score = min(
        abs(
            float(
                row["correlation"]
            )
        ),
        1.0,
    )

    half_life = float(
        row["half_life"]
    )

    if np.isfinite(half_life):
        half_life_score = float(
            np.exp(
                -half_life
                / max_half_life
            )
        )
    else:
        half_life_score = 0.0

    stability_score = float(
        row[
            "hedge_ratio_stability"
        ]
    )

    score = (
        0.35 * p_score
        + 0.20 * correlation_score
        + 0.20 * half_life_score
        + 0.15 * stability_score
        + 0.10 * adf_score
    )

    if (
        float(
            row["adjusted_p_value"]
        )
        > alpha
    ):
        return 0.0

    return float(score)


def select_pairs(
    prices: pd.DataFrame,
    alpha: float = 0.05,
    min_correlation: float = 0.50,
    min_half_life: float = 2.0,
    max_half_life: float = 90.0,
    max_pairs_to_keep: int = 20,
    require_positive_hedge_ratio: bool = True,
    stability_window: int = 60,
    stability_step: int = 10,
    groups: dict[str, str] | None = None,
    require_same_group: bool = False,
) -> pd.DataFrame:
    """Test, adjust, filter, and rank candidate pairs.

    Correlation is first used as a computational pre-filter.

    Engle-Granger p-values are then corrected across all pairs
    actually tested using Benjamini-Hochberg FDR control.
    """
    if prices.shape[1] < 2:
        return _empty_result()

    preliminary: list[
        dict[
            str,
            float | str | int,
        ]
    ] = []

    for ticker_y, ticker_x in combinations(
        prices.columns,
        2,
    ):
        if (
            require_same_group
            and groups is not None
        ):
            group_y = groups.get(
                str(ticker_y)
            )

            group_x = groups.get(
                str(ticker_x)
            )

            if (
                group_y is None
                or group_x is None
                or group_y != group_x
            ):
                continue

        aligned = prices[
            [
                ticker_y,
                ticker_x,
            ]
        ].dropna()

        if len(aligned) < 30:
            continue

        correlation = float(
            aligned.corr().iloc[0, 1]
        )

        if (
            not np.isfinite(correlation)
            or abs(correlation)
            < min_correlation
        ):
            continue

        try:
            result = analyze_pair(
                aligned,
                ticker_y,
                ticker_x,
            )
        except (
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue

        if (
            require_positive_hedge_ratio
            and result["hedge_ratio"] <= 0
        ):
            continue

        preliminary.append(result)

    if not preliminary:
        return _empty_result()

    tested = pd.DataFrame(
        preliminary
    )

    _, adjusted, _, _ = multipletests(
        tested[
            "coint_p_value"
        ].to_numpy(),
        alpha=alpha,
        method="fdr_bh",
    )

    tested[
        "adjusted_p_value"
    ] = adjusted

    tested = tested[
        tested[
            "adjusted_p_value"
        ]
        <= alpha
    ].copy()

    if tested.empty:
        return _empty_result()

    half_lives: list[float] = []
    stability_scores: list[float] = []

    for row in tested.itertuples(
        index=False
    ):
        spread = compute_spread(
            prices[row.ticker_y],
            prices[row.ticker_x],
            hedge_ratio=float(
                row.hedge_ratio
            ),
            intercept=float(
                row.intercept
            ),
        )

        half_lives.append(
            estimate_half_life(
                spread
            )
        )

        stability_scores.append(
            hedge_ratio_stability_score(
                prices[row.ticker_y],
                prices[row.ticker_x],
                window=stability_window,
                step=stability_step,
            )
        )

    tested[
        "half_life"
    ] = half_lives

    tested[
        "hedge_ratio_stability"
    ] = stability_scores

    tested = tested[
        tested[
            "half_life"
        ].between(
            min_half_life,
            max_half_life,
            inclusive="both",
        )
    ].copy()

    if tested.empty:
        return _empty_result()

    tested[
        "pair_score"
    ] = tested.apply(
        _pair_score,
        axis=1,
        alpha=alpha,
        max_half_life=max_half_life,
    )

    columns = (
        _empty_result()
        .columns
        .tolist()
    )

    return (
        tested
        .sort_values(
            [
                "pair_score",
                "adjusted_p_value",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            max_pairs_to_keep
        )
        .reset_index(
            drop=True
        )[columns]
    )
