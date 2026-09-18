"""Performance metrics for trading strategy evaluation."""

from __future__ import annotations

import pandas as pd


def compute_total_return(
    portfolio_value: pd.Series,
) -> float:
    if portfolio_value.empty:
        raise ValueError(
            "Portfolio value series is empty."
        )

    return float(
        portfolio_value.iloc[-1]
        / portfolio_value.iloc[0]
        - 1
    )


def compute_annualized_return(
    portfolio_value: pd.Series,
    periods_per_year: int = 252,
) -> float:
    if portfolio_value.empty:
        raise ValueError(
            "Portfolio value series is empty."
        )

    n_periods = len(
        portfolio_value
    )

    if n_periods <= 1:
        return 0.0

    total_return = (
        compute_total_return(
            portfolio_value
        )
    )

    if (
        1
        + total_return
        <= 0
    ):
        return -1.0

    return float(
        (
            1
            + total_return
        )
        ** (
            periods_per_year
            / n_periods
        )
        - 1
    )


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    clean = returns.dropna()

    if clean.empty:
        return 0.0

    return float(
        clean.std()
        * (
            periods_per_year
            ** 0.5
        )
    )


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    clean = returns.dropna()

    if clean.empty:
        return 0.0

    excess = (
        clean
        - risk_free_rate
        / periods_per_year
    )

    volatility = excess.std()

    if (
        volatility == 0
        or pd.isna(
            volatility
        )
    ):
        return 0.0

    return float(
        (
            excess.mean()
            / volatility
        )
        * (
            periods_per_year
            ** 0.5
        )
    )


def compute_drawdown(
    portfolio_value: pd.Series,
) -> pd.Series:
    running_max = (
        portfolio_value
        .cummax()
    )

    result = (
        portfolio_value
        / running_max
        - 1
    )

    result.name = "drawdown"

    return result


def compute_max_drawdown(
    portfolio_value: pd.Series,
) -> float:
    if portfolio_value.empty:
        return 0.0

    return float(
        compute_drawdown(
            portfolio_value
        ).min()
    )


def compute_win_rate(
    returns: pd.Series,
) -> float:
    clean = returns.dropna()

    if clean.empty:
        return 0.0

    return float(
        (
            clean
            > 0
        ).mean()
    )


def compute_trade_count(
    positions: pd.Series,
) -> int:
    """Count entries from flat into a position."""
    clean = (
        positions
        .fillna(0)
        .astype(int)
    )

    previous = (
        clean
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return int(
        (
            (clean != 0)
            & (
                previous == 0
            )
        ).sum()
    )


def compute_average_holding_period(
    positions: pd.Series,
) -> float:
    """Average trade length in periods."""
    clean = (
        positions
        .fillna(0)
        .astype(int)
    )

    lengths: list[int] = []

    current = 0

    for value in clean:
        if value != 0:
            current += 1

        elif current > 0:
            lengths.append(
                current
            )

            current = 0

    if current > 0:
        lengths.append(
            current
        )

    if not lengths:
        return 0.0

    return float(
        sum(lengths)
        / len(lengths)
    )


def compute_performance_summary(
    portfolio_value: pd.Series,
    returns: pd.Series | None = None,
    positions: pd.Series | None = None,
    turnover: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    if returns is None:
        returns = (
            portfolio_value
            .pct_change()
            .dropna()
        )

    summary = {
        "total_return": (
            compute_total_return(
                portfolio_value
            )
        ),
        "annualized_return": (
            compute_annualized_return(
                portfolio_value,
                periods_per_year,
            )
        ),
        "annualized_volatility": (
            compute_annualized_volatility(
                returns,
                periods_per_year,
            )
        ),
        "sharpe_ratio": (
            compute_sharpe_ratio(
                returns,
                risk_free_rate,
                periods_per_year,
            )
        ),
        "max_drawdown": (
            compute_max_drawdown(
                portfolio_value
            )
        ),
        "win_rate": (
            compute_win_rate(
                returns
            )
        ),
    }

    if positions is not None:
        summary[
            "number_of_trades"
        ] = float(
            compute_trade_count(
                positions
            )
        )

        summary[
            "average_holding_period"
        ] = (
            compute_average_holding_period(
                positions
            )
        )

    if turnover is not None:
        summary[
            "total_turnover"
        ] = float(
            turnover
            .fillna(0)
            .sum()
        )

    return summary
