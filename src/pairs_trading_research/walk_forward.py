"""Walk-forward evaluation for pair selection and trading."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .cointegration import (
    compute_spread,
)
from .metrics import (
    compute_performance_summary,
)
from .pair_selection import (
    select_pairs,
)
from .portfolio import (
    run_pair_backtest,
)
from .signals import (
    generate_pair_signals,
)


def run_walk_forward(
    prices: pd.DataFrame,
    training_window: int = 252,
    test_window: int = 20,
    step_size: int | None = None,
    selection_kwargs: dict[
        str,
        Any,
    ] | None = None,
    strategy_kwargs: dict[
        str,
        Any,
    ] | None = None,
    backtest_kwargs: dict[
        str,
        Any,
    ] | None = None,
) -> dict[
    str,
    pd.DataFrame
    | dict[str, float],
]:
    """Run non-overlapping walk-forward evaluation."""
    selection_kwargs = (
        selection_kwargs
        or {}
    )

    strategy_kwargs = (
        strategy_kwargs
        or {}
    )

    backtest_kwargs = (
        backtest_kwargs
        or {}
    )

    if step_size is None:
        step_size = test_window

    if training_window < 60:
        raise ValueError(
            "training_window must be at least 60 observations."
        )

    if test_window < 2:
        raise ValueError(
            "test_window must be at least 2 observations."
        )

    if step_size < test_window:
        raise ValueError(
            "step_size must be >= test_window to avoid overlapping OOS periods."
        )

    if (
        len(prices)
        < training_window
        + test_window
    ):
        raise ValueError(
            "Not enough observations for one walk-forward window."
        )

    zscore_window = int(
        strategy_kwargs.get(
            "zscore_window",
            20,
        )
    )

    initial_capital = float(
        backtest_kwargs.get(
            "initial_capital",
            100_000.0,
        )
    )

    current_capital = (
        initial_capital
    )

    window_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    daily_frames: list[
        pd.DataFrame
    ] = []

    window_id = 0

    for train_start in range(
        0,
        len(prices)
        - training_window
        - test_window
        + 1,
        step_size,
    ):
        train_end = (
            train_start
            + training_window
        )

        test_end = (
            train_end
            + test_window
        )

        train = prices.iloc[
            train_start:
            train_end
        ]

        test = prices.iloc[
            train_end:
            test_end
        ]

        if test.empty:
            continue

        window_id += 1

        candidates = select_pairs(
            train,
            **selection_kwargs,
        )

        if candidates.empty:
            flat = pd.DataFrame(
                index=test.index
            )

            flat[
                "strategy_return"
            ] = 0.0

            flat[
                "portfolio_value"
            ] = current_capital

            flat[
                "window_id"
            ] = window_id

            flat[
                "ticker_y"
            ] = ""

            flat[
                "ticker_x"
            ] = ""

            daily_frames.append(
                flat
            )

            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "ticker_y": "",
                    "ticker_x": "",
                    "pair_score": 0.0,
                    "adjusted_p_value": float("nan"),
                    "half_life": float("nan"),
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }
            )

            continue

        best = (
            candidates
            .iloc[0]
        )

        ticker_y = str(
            best[
                "ticker_y"
            ]
        )

        ticker_x = str(
            best[
                "ticker_x"
            ]
        )

        warmup = train[
            [
                ticker_y,
                ticker_x,
            ]
        ].iloc[
            -max(
                2 * zscore_window,
                zscore_window + 5,
            ):
        ]

        combined = pd.concat(
            [
                warmup,
                test[
                    [
                        ticker_y,
                        ticker_x,
                    ]
                ],
            ]
        )

        combined = combined[
            ~combined.index.duplicated(
                keep="last"
            )
        ]

        spread = compute_spread(
            combined[
                ticker_y
            ],
            combined[
                ticker_x
            ],
            hedge_ratio=float(
                best[
                    "hedge_ratio"
                ]
            ),
            intercept=float(
                best[
                    "intercept"
                ]
            ),
        )

        signals = generate_pair_signals(
            spread,
            **strategy_kwargs,
        )

        test_positions = (
            signals[
                "position"
            ]
            .reindex(
                test.index
            )
            .fillna(0)
            .astype(int)
        )

        bt_kwargs = dict(
            backtest_kwargs
        )

        bt_kwargs[
            "initial_capital"
        ] = current_capital

        results = run_pair_backtest(
            test[
                ticker_y
            ],
            test[
                ticker_x
            ],
            positions=test_positions,
            hedge_ratio=float(
                best[
                    "hedge_ratio"
                ]
            ),
            **bt_kwargs,
        )

        current_capital = float(
            results[
                "portfolio_value"
            ].iloc[-1]
        )

        summary = (
            compute_performance_summary(
                results[
                    "portfolio_value"
                ],
                returns=results[
                    "strategy_return"
                ],
            )
        )

        daily = results.copy()

        daily[
            "window_id"
        ] = window_id

        daily[
            "ticker_y"
        ] = ticker_y

        daily[
            "ticker_x"
        ] = ticker_x

        daily_frames.append(
            daily
        )

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "pair_score": float(
                    best[
                        "pair_score"
                    ]
                ),
                "adjusted_p_value": float(
                    best[
                        "adjusted_p_value"
                    ]
                ),
                "half_life": float(
                    best[
                        "half_life"
                    ]
                ),
                "total_return": float(
                    summary[
                        "total_return"
                    ]
                ),
                "sharpe_ratio": float(
                    summary[
                        "sharpe_ratio"
                    ]
                ),
                "max_drawdown": float(
                    summary[
                        "max_drawdown"
                    ]
                ),
            }
        )

    windows = pd.DataFrame(
        window_rows
    )

    if daily_frames:
        daily_results = (
            pd.concat(
                daily_frames
            )
            .sort_index()
        )
    else:
        daily_results = (
            pd.DataFrame()
        )

    if daily_results.empty:
        aggregate_summary = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    else:
        aggregate_summary = (
            compute_performance_summary(
                daily_results[
                    "portfolio_value"
                ],
                returns=daily_results[
                    "strategy_return"
                ],
            )
        )

    return {
        "windows": windows,
        "daily_results": daily_results,
        "summary": aggregate_summary,
    }
