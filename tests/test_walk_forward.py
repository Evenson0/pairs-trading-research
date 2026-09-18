import numpy as np
import pandas as pd

from pairs_trading_research.walk_forward import (
    run_walk_forward,
)


def test_walk_forward_runs_without_lookahead_crash() -> None:
    rng = (
        np.random.default_rng(
            123
        )
    )

    n = 360

    x = (
        100
        + np.cumsum(
            rng.normal(
                size=n
            )
        )
    )

    e = np.zeros(
        n
    )

    for i in range(
        1,
        n,
    ):
        e[i] = (
            0.8
            * e[
                i - 1
            ]
            + rng.normal(
                scale=0.4
            )
        )

    y = (
        3
        + 1.2 * x
        + e
    )

    z = (
        70
        + np.cumsum(
            rng.normal(
                size=n
            )
        )
    )

    prices = pd.DataFrame(
        {
            "Y": y,
            "X": x,
            "Z": z,
        },
        index=pd.date_range(
            "2024-01-01",
            periods=n,
            freq="B",
        ),
    )

    result = run_walk_forward(
        prices,
        training_window=180,
        test_window=30,
        step_size=30,
        selection_kwargs={
            "alpha": 0.10,
            "min_correlation": 0.30,
            "min_half_life": 0.5,
            "max_half_life": 100.0,
            "stability_window": 60,
            "stability_step": 15,
        },
        strategy_kwargs={
            "zscore_window": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss_zscore": 3.5,
        },
        backtest_kwargs={
            "initial_capital": 10_000.0,
            "gross_exposure": 1.0,
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
            "position_type": "dollar_neutral",
        },
    )

    assert not result[
        "windows"
    ].empty

    assert not result[
        "daily_results"
    ].empty

    assert (
        "total_return"
        in result[
            "summary"
        ]
    )
