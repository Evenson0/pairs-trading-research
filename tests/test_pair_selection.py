import numpy as np
import pandas as pd

from pairs_trading_research.pair_selection import (
    select_pairs,
)


def test_select_pairs_finds_synthetic_cointegrated_pair() -> None:
    rng = (
        np.random.default_rng(
            7
        )
    )

    n = 600

    x = (
        100
        + np.cumsum(
            rng.normal(
                size=n
            )
        )
    )

    noise = np.zeros(
        n
    )

    for i in range(
        1,
        n,
    ):
        noise[i] = (
            0.85
            * noise[
                i - 1
            ]
            + rng.normal(
                scale=0.5
            )
        )

    y = (
        5.0
        + 1.4 * x
        + noise
    )

    z = (
        50
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
        }
    )

    result = select_pairs(
        prices,
        alpha=0.10,
        min_correlation=0.30,
        min_half_life=0.5,
        max_half_life=100.0,
        max_pairs_to_keep=10,
        stability_window=100,
        stability_step=20,
    )

    pairs = {
        frozenset(
            (
                row.ticker_y,
                row.ticker_x,
            )
        )
        for row
        in result.itertuples()
    }

    assert (
        frozenset(
            (
                "Y",
                "X",
            )
        )
        in pairs
    )

    assert (
        result[
            "adjusted_p_value"
        ]
        <= 0.10
    ).all()
