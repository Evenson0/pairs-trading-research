"""Run walk-forward validation."""

from __future__ import annotations

import argparse
import json

from pairs_trading_research.config import (
    get_project_root,
    load_named_config,
)
from pairs_trading_research.data_loader import (
    download_adjusted_prices,
)
from pairs_trading_research.preprocessing import (
    clean_price_data,
)
from pairs_trading_research.universe import (
    get_universe_metadata,
)
from pairs_trading_research.walk_forward import (
    run_walk_forward,
)


def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--universe",
        choices=[
            "tsx60",
            "sp500",
        ],
        default="tsx60",
    )

    args = parser.parse_args()

    config = load_named_config(
        args.universe
    )

    metadata = (
        get_universe_metadata(
            args.universe
        )
    )

    tickers = metadata[
        "ticker"
    ].tolist()

    if (
        "sector"
        in metadata.columns
    ):
        groups = (
            metadata
            .dropna(
                subset=[
                    "sector"
                ]
            )
            .set_index(
                "ticker"
            )[
                "sector"
            ]
            .to_dict()
        )
    else:
        groups = None

    prices = (
        download_adjusted_prices(
            tickers,
            start=config[
                "data"
            ][
                "start_date"
            ],
            end=config[
                "data"
            ][
                "end_date"
            ],
            price_field=config[
                "data"
            ][
                "price_field"
            ],
            batch_size=int(
                config[
                    "data"
                ].get(
                    "batch_size",
                    100,
                )
            ),
        )
    )

    prices = clean_price_data(
        prices,
        max_missing_ratio=float(
            config[
                "filters"
            ][
                "max_missing_ratio"
            ]
        ),
        min_observations=int(
            config[
                "filters"
            ][
                "min_observations"
            ]
        ),
    )

    pair_cfg = config[
        "pair_selection"
    ]

    strategy_cfg = config[
        "strategy"
    ]

    backtest_cfg = config[
        "backtest"
    ]

    wf_cfg = config[
        "walk_forward"
    ]

    result = run_walk_forward(
        prices,
        training_window=int(
            wf_cfg[
                "training_window"
            ]
        ),
        test_window=int(
            wf_cfg[
                "test_window"
            ]
        ),
        step_size=int(
            wf_cfg[
                "step_size"
            ]
        ),
        selection_kwargs={
            "alpha": float(
                pair_cfg[
                    "fdr_alpha"
                ]
            ),
            "min_correlation": float(
                pair_cfg[
                    "min_correlation"
                ]
            ),
            "min_half_life": float(
                pair_cfg[
                    "min_half_life"
                ]
            ),
            "max_half_life": float(
                pair_cfg[
                    "max_half_life"
                ]
            ),
            "max_pairs_to_keep": int(
                pair_cfg[
                    "max_pairs_to_keep"
                ]
            ),
            "require_positive_hedge_ratio": bool(
                pair_cfg[
                    "require_positive_hedge_ratio"
                ]
            ),
            "stability_window": int(
                pair_cfg[
                    "stability_window"
                ]
            ),
            "stability_step": int(
                pair_cfg[
                    "stability_step"
                ]
            ),
            "groups": groups,
            "require_same_group": bool(
                pair_cfg.get(
                    "sector_filter",
                    False,
                )
            ),
        },
        strategy_kwargs={
            "zscore_window": int(
                strategy_cfg[
                    "zscore_window"
                ]
            ),
            "entry_threshold": float(
                strategy_cfg[
                    "entry_threshold"
                ]
            ),
            "exit_threshold": float(
                strategy_cfg[
                    "exit_threshold"
                ]
            ),
            "stop_loss_zscore": float(
                strategy_cfg[
                    "stop_loss_zscore"
                ]
            ),
        },
        backtest_kwargs={
            "initial_capital": float(
                backtest_cfg[
                    "initial_capital"
                ]
            ),
            "gross_exposure": float(
                backtest_cfg[
                    "gross_exposure"
                ]
            ),
            "transaction_cost_bps": float(
                backtest_cfg[
                    "transaction_cost_bps"
                ]
            ),
            "slippage_bps": float(
                backtest_cfg[
                    "slippage_bps"
                ]
            ),
            "position_type": str(
                backtest_cfg[
                    "position_type"
                ]
            ),
        },
    )

    reports = (
        get_project_root()
        / "reports"
    )

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    result[
        "windows"
    ].to_csv(
        reports
        / (
            f"{args.universe}"
            "_walk_forward_windows.csv"
        ),
        index=False,
    )

    result[
        "daily_results"
    ].to_csv(
        reports
        / (
            f"{args.universe}"
            "_walk_forward_daily.csv"
        )
    )

    summary_path = (
        reports
        / (
            f"{args.universe}"
            "_walk_forward_summary.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            result[
                "summary"
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result[
                "summary"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
