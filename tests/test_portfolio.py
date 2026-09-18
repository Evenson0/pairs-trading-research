import pandas as pd

from pairs_trading_research.portfolio import (
    position_notional_plan,
    run_pair_backtest,
)


def test_dollar_neutral_identical_assets_have_zero_gross_return() -> None:
    idx = pd.date_range(
        "2024-01-01",
        periods=6,
        freq="D",
    )

    y = pd.Series(
        [
            100,
            101,
            102,
            103,
            104,
            105,
        ],
        index=idx,
        dtype=float,
    )

    x = pd.Series(
        [
            50,
            50.5,
            51,
            51.5,
            52,
            52.5,
        ],
        index=idx,
        dtype=float,
    )

    positions = pd.Series(
        [
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        index=idx,
    )

    result = run_pair_backtest(
        y,
        x,
        positions,
        hedge_ratio=2.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        position_type="dollar_neutral",
    )

    assert (
        result[
            "gross_return"
        ]
        .abs()
        .max()
        < 1e-12
    )


def test_position_notional_plan_is_dollar_neutral() -> None:
    plan = (
        position_notional_plan(
            10_000,
            100,
            50,
            direction=1,
            gross_exposure_fraction=0.2,
        )
    )

    assert (
        plan[
            "gross_notional"
        ]
        == 2000
    )

    assert (
        abs(
            plan[
                "net_notional"
            ]
        )
        < 1e-12
    )

    assert (
        plan[
            "shares_y"
        ]
        > 0
    )

    assert (
        plan[
            "shares_x"
        ]
        < 0
    )
