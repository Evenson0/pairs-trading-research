"""Two-leg portfolio construction and backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_pair_data(
    price_y: pd.Series,
    price_x: pd.Series,
    positions: pd.Series,
) -> pd.DataFrame:
    data = pd.concat(
        [
            price_y.rename(
                "price_y"
            ),
            price_x.rename(
                "price_x"
            ),
            positions.rename(
                "position"
            ),
        ],
        axis=1,
    ).dropna()

    if data.empty:
        raise ValueError(
            "No aligned price and position observations."
        )

    return data


def compute_target_weights(
    previous_price_y: pd.Series,
    previous_price_x: pd.Series,
    lagged_position: pd.Series,
    hedge_ratio: float,
    gross_exposure: float = 1.0,
    position_type: str = "dollar_neutral",
) -> tuple[
    pd.Series,
    pd.Series,
]:
    """Compute lagged target weights for both equity legs."""
    if gross_exposure <= 0:
        raise ValueError(
            "gross_exposure must be positive."
        )

    if (
        position_type
        == "dollar_neutral"
    ):
        weight_y = (
            0.5
            * gross_exposure
            * lagged_position
        )

        weight_x = (
            -0.5
            * gross_exposure
            * lagged_position
        )

        return (
            weight_y,
            weight_x,
        )

    if (
        position_type
        == "hedge_ratio"
    ):
        y_notional = (
            previous_price_y
            .abs()
        )

        x_notional = (
            abs(hedge_ratio)
            * previous_price_x.abs()
        )

        total = (
            y_notional
            + x_notional
        ).replace(
            0.0,
            np.nan,
        )

        base_y = (
            y_notional
            / total
        ).fillna(
            0.0
        )

        base_x = (
            -np.sign(
                hedge_ratio
            )
            * x_notional
            / total
        ).fillna(
            0.0
        )

        return (
            gross_exposure
            * lagged_position
            * base_y,
            gross_exposure
            * lagged_position
            * base_x,
        )

    raise ValueError(
        "position_type must be 'dollar_neutral' or 'hedge_ratio'."
    )


def run_pair_backtest(
    price_y: pd.Series,
    price_x: pd.Series,
    positions: pd.Series,
    hedge_ratio: float,
    initial_capital: float = 100_000.0,
    gross_exposure: float = 1.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 0.0,
    position_type: str = "dollar_neutral",
) -> pd.DataFrame:
    """Backtest explicit long/short legs.

    Signals are executed with a one-period lag.
    """
    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive."
        )

    data = _aligned_pair_data(
        price_y,
        price_x,
        positions,
    )

    data["return_y"] = (
        data["price_y"]
        .pct_change()
        .fillna(0.0)
    )

    data["return_x"] = (
        data["price_x"]
        .pct_change()
        .fillna(0.0)
    )

    data[
        "lagged_position"
    ] = (
        data["position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    previous_y = (
        data["price_y"]
        .shift(1)
        .bfill()
    )

    previous_x = (
        data["price_x"]
        .shift(1)
        .bfill()
    )

    weight_y, weight_x = (
        compute_target_weights(
            previous_y,
            previous_x,
            data[
                "lagged_position"
            ],
            hedge_ratio=hedge_ratio,
            gross_exposure=gross_exposure,
            position_type=position_type,
        )
    )

    data[
        "weight_y"
    ] = weight_y

    data[
        "weight_x"
    ] = weight_x

    data[
        "gross_exposure"
    ] = (
        data[
            "weight_y"
        ].abs()
        + data[
            "weight_x"
        ].abs()
    )

    data[
        "net_exposure"
    ] = (
        data[
            "weight_y"
        ]
        + data[
            "weight_x"
        ]
    )

    data[
        "gross_return"
    ] = (
        data[
            "weight_y"
        ]
        * data[
            "return_y"
        ]
        + data[
            "weight_x"
        ]
        * data[
            "return_x"
        ]
    )

    data[
        "turnover"
    ] = (
        data[
            "weight_y"
        ]
        .diff()
        .abs()
        .fillna(
            data[
                "weight_y"
            ].abs()
        )
        + data[
            "weight_x"
        ]
        .diff()
        .abs()
        .fillna(
            data[
                "weight_x"
            ].abs()
        )
    )

    cost_rate = (
        transaction_cost_bps
        + slippage_bps
    ) / 10_000.0

    data[
        "trading_cost"
    ] = (
        data[
            "turnover"
        ]
        * cost_rate
    )

    data[
        "strategy_return"
    ] = (
        data[
            "gross_return"
        ]
        - data[
            "trading_cost"
        ]
    )

    data[
        "portfolio_value"
    ] = (
        initial_capital
        * (
            1.0
            + data[
                "strategy_return"
            ]
        ).cumprod()
    )

    return data


def position_notional_plan(
    capital: float,
    price_y: float,
    price_x: float,
    direction: int,
    gross_exposure_fraction: float = 0.20,
) -> dict[str, float]:
    """Create a dollar-neutral fractional-share trade plan."""
    if (
        capital <= 0
        or price_y <= 0
        or price_x <= 0
    ):
        raise ValueError(
            "capital and prices must be positive."
        )

    if direction not in {
        -1,
        1,
    }:
        raise ValueError(
            "direction must be 1 or -1."
        )

    if not (
        0
        < gross_exposure_fraction
        <= 2.0
    ):
        raise ValueError(
            "gross_exposure_fraction must be in (0, 2]."
        )

    gross_notional = (
        capital
        * gross_exposure_fraction
    )

    leg_notional = (
        gross_notional
        / 2.0
    )

    notional_y = (
        direction
        * leg_notional
    )

    notional_x = (
        -direction
        * leg_notional
    )

    return {
        "gross_notional": gross_notional,
        "notional_y": notional_y,
        "notional_x": notional_x,
        "shares_y": (
            notional_y
            / price_y
        ),
        "shares_x": (
            notional_x
            / price_x
        ),
        "net_notional": (
            notional_y
            + notional_x
        ),
    }
