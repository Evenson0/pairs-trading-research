"""Current-market pair scanner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .cointegration import (
    compute_spread,
)
from .pair_selection import (
    select_pairs,
)
from .signals import (
    classify_current_signal,
    compute_rolling_zscore,
)


def build_market_scan(
    prices: pd.DataFrame,
    config: dict[str, Any],
    groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rank current candidate pairs and attach current signals."""
    live_cfg = config.get(
        "live",
        {},
    )

    pair_cfg = config[
        "pair_selection"
    ]

    strategy_cfg = config[
        "strategy"
    ]

    lookback = int(
        live_cfg.get(
            "selection_lookback_days",
            504,
        )
    )

    research_prices = (
        prices
        .iloc[
            -lookback:
        ]
        .copy()
    )

    candidates = select_pairs(
        research_prices,
        alpha=float(
            pair_cfg[
                "fdr_alpha"
            ]
        ),
        min_correlation=float(
            pair_cfg[
                "min_correlation"
            ]
        ),
        min_half_life=float(
            pair_cfg[
                "min_half_life"
            ]
        ),
        max_half_life=float(
            pair_cfg[
                "max_half_life"
            ]
        ),
        max_pairs_to_keep=int(
            pair_cfg[
                "max_pairs_to_keep"
            ]
        ),
        require_positive_hedge_ratio=bool(
            pair_cfg.get(
                "require_positive_hedge_ratio",
                True,
            )
        ),
        stability_window=int(
            pair_cfg.get(
                "stability_window",
                60,
            )
        ),
        stability_step=int(
            pair_cfg.get(
                "stability_step",
                10,
            )
        ),
        groups=groups,
        require_same_group=bool(
            pair_cfg.get(
                "sector_filter",
                False,
            )
        ),
    )

    if candidates.empty:
        return candidates

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for row in candidates.itertuples(
        index=False
    ):
        spread = compute_spread(
            research_prices[
                row.ticker_y
            ],
            research_prices[
                row.ticker_x
            ],
            hedge_ratio=float(
                row.hedge_ratio
            ),
            intercept=float(
                row.intercept
            ),
        )

        zscore = (
            compute_rolling_zscore(
                spread,
                window=int(
                    strategy_cfg[
                        "zscore_window"
                    ]
                ),
            )
        )

        if zscore.empty:
            latest_z = float(
                "nan"
            )
        else:
            latest_z = float(
                zscore.iloc[-1]
            )

        status = classify_current_signal(
            latest_z,
            entry_threshold=float(
                strategy_cfg[
                    "entry_threshold"
                ]
            ),
            exit_threshold=float(
                strategy_cfg[
                    "exit_threshold"
                ]
            ),
            stop_loss_zscore=float(
                strategy_cfg[
                    "stop_loss_zscore"
                ]
            ),
            watch_threshold=float(
                strategy_cfg.get(
                    "watch_threshold",
                    1.5,
                )
            ),
        )

        result = row._asdict()

        result[
            "latest_zscore"
        ] = latest_z

        result[
            "status"
        ] = status

        result[
            "as_of"
        ] = (
            research_prices
            .index[-1]
        )

        rows.append(
            result
        )

    return (
        pd.DataFrame(
            rows
        )
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
        .reset_index(
            drop=True
        )
    )
