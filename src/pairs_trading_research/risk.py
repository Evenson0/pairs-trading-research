"""Portfolio risk controls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(
    frozen=True
)
class RiskLimits:
    max_open_pairs: int = 5
    max_pair_gross_exposure: float = 0.20
    max_portfolio_gross_exposure: float = 1.00
    max_sector_gross_exposure: float = 0.40
    max_drawdown: float = 0.10


def current_drawdown(
    equity: pd.Series,
) -> float:
    """Return the current drawdown from the historical equity peak."""
    clean = equity.dropna()

    if clean.empty:
        return 0.0

    peak = float(
        clean
        .cummax()
        .iloc[-1]
    )

    if peak <= 0:
        return 0.0

    return float(
        clean.iloc[-1]
        / peak
        - 1.0
    )


def approve_new_pair(
    open_pair_count: int,
    current_gross_exposure: float,
    proposed_pair_gross_exposure: float,
    limits: RiskLimits,
    current_drawdown_value: float = 0.0,
    current_sector_gross_exposure: float | None = None,
    proposed_sector_gross_exposure: float | None = None,
) -> dict[str, object]:
    """Apply portfolio constraints before opening a pair."""
    reasons: list[str] = []

    if (
        open_pair_count
        >= limits.max_open_pairs
    ):
        reasons.append(
            "maximum number of open pairs reached"
        )

    if (
        proposed_pair_gross_exposure
        > limits.max_pair_gross_exposure
    ):
        reasons.append(
            "pair gross exposure exceeds limit"
        )

    if (
        current_gross_exposure
        + proposed_pair_gross_exposure
        > limits.max_portfolio_gross_exposure
    ):
        reasons.append(
            "portfolio gross exposure would exceed limit"
        )

    if (
        current_drawdown_value
        <= -abs(
            limits.max_drawdown
        )
    ):
        reasons.append(
            "portfolio drawdown limit reached"
        )

    if (
        current_sector_gross_exposure
        is not None
        and proposed_sector_gross_exposure
        is not None
        and (
            current_sector_gross_exposure
            + proposed_sector_gross_exposure
            > limits.max_sector_gross_exposure
        )
    ):
        reasons.append(
            "sector gross exposure would exceed limit"
        )

    return {
        "approved": not reasons,
        "reasons": reasons,
    }
