"""Mean-reversion diagnostics used by the pair-selection engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .cointegration import estimate_hedge_ratio


def estimate_half_life(
    spread: pd.Series,
) -> float:
    """Estimate the half-life of mean reversion.

    The result is expressed in trading periods.

    Infinity is returned when the estimated process does not
    exhibit mean reversion.
    """
    clean = (
        spread
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if len(clean) < 10:
        return float("inf")

    lagged = clean.shift(1)
    delta = clean.diff()

    regression_data = pd.concat(
        [
            delta.rename("delta"),
            lagged.rename("lagged"),
        ],
        axis=1,
    ).dropna()

    if len(regression_data) < 8:
        return float("inf")

    design = sm.add_constant(
        regression_data["lagged"]
    )

    model = sm.OLS(
        regression_data["delta"],
        design,
    ).fit()

    slope = float(model.params.iloc[1])

    if slope >= 0 or not np.isfinite(slope):
        return float("inf")

    half_life = -math.log(2.0) / slope

    if half_life <= 0 or not np.isfinite(half_life):
        return float("inf")

    return float(half_life)


def rolling_hedge_ratios(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
    step: int = 10,
) -> pd.Series:
    """Estimate hedge ratios on historical rolling windows."""
    aligned = pd.concat(
        [
            y.rename("y"),
            x.rename("x"),
        ],
        axis=1,
    ).dropna()

    if window < 20:
        raise ValueError(
            "window must be at least 20 observations."
        )

    if step < 1:
        raise ValueError(
            "step must be positive."
        )

    values: dict[
        pd.Timestamp | int,
        float,
    ] = {}

    for end in range(
        window,
        len(aligned) + 1,
        step,
    ):
        chunk = aligned.iloc[
            end - window:end
        ]

        try:
            beta = estimate_hedge_ratio(
                chunk["y"],
                chunk["x"],
            )
        except (
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue

        values[
            chunk.index[-1]
        ] = beta

    return pd.Series(
        values,
        dtype=float,
        name="rolling_hedge_ratio",
    )


def hedge_ratio_stability_score(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
    step: int = 10,
) -> float:
    """Return a 0-1 hedge-ratio stability score.

    Higher values represent more stable historical hedge ratios.
    """
    betas = rolling_hedge_ratios(
        y,
        x,
        window=window,
        step=step,
    ).dropna()

    if len(betas) < 2:
        return 0.0

    scale = float(
        np.median(
            np.abs(betas)
        )
    )

    if scale < 1e-12:
        scale = 1.0

    relative_dispersion = float(
        betas.std(ddof=1)
        / scale
    )

    return float(
        np.exp(
            -max(
                relative_dispersion,
                0.0,
            )
        )
    )
