import numpy as np
import pandas as pd

from pairs_trading_research.mean_reversion import (
    estimate_half_life,
)


def test_estimate_half_life_for_ar1_process() -> None:
    rng = (
        np.random.default_rng(
            42
        )
    )

    values = [
        0.0
    ]

    phi = 0.90

    for _ in range(
        1500
    ):
        values.append(
            phi
            * values[-1]
            + rng.normal(
                scale=1.0
            )
        )

    half_life = (
        estimate_half_life(
            pd.Series(
                values[
                    200:
                ]
            )
        )
    )

    assert (
        3.0
        < half_life
        < 12.0
    )
