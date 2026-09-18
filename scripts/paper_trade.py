"""Run one end-of-day paper-trading cycle."""

from __future__ import annotations

import argparse

import pandas as pd

from pairs_trading_research.cointegration import (
    compute_spread,
    engle_granger_test,
)
from pairs_trading_research.config import (
    get_project_root,
    load_named_config,
)
from pairs_trading_research.data_loader import (
    download_adjusted_prices,
)
from pairs_trading_research.paper_broker import (
    PaperBroker,
)
from pairs_trading_research.preprocessing import (
    clean_price_data,
)
from pairs_trading_research.risk import (
    RiskLimits,
    approve_new_pair,
    current_drawdown,
)
from pairs_trading_research.scanner import (
    build_market_scan,
)
from pairs_trading_research.signals import (
    compute_rolling_zscore,
)
from pairs_trading_research.universe import (
    get_universe_metadata,
)


def _latest_zscore(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    intercept: float,
    hedge_ratio: float,
    zscore_window: int,
) -> float:
    spread = compute_spread(
        prices[
            ticker_y
        ],
        prices[
            ticker_x
        ],
        hedge_ratio=hedge_ratio,
        intercept=intercept,
    )

    zscore = compute_rolling_zscore(
        spread,
        window=zscore_window,
    )

    if zscore.empty:
        return float(
            "nan"
        )

    return float(
        zscore.iloc[-1]
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

    latest_date = (
        prices.index[-1]
    )

    latest_prices = (
        prices
        .iloc[-1]
        .astype(float)
        .to_dict()
    )

    state_dir = (
        get_project_root()
        / config[
            "paper"
        ][
            "state_dir"
        ]
    )

    broker = PaperBroker(
        state_dir=state_dir,
        initial_capital=float(
            config[
                "backtest"
            ][
                "initial_capital"
            ]
        ),
    )

    strategy = config[
        "strategy"
    ]

    live = config[
        "live"
    ]

    effective_cost_bps = (
        float(
            config[
                "backtest"
            ][
                "transaction_cost_bps"
            ]
        )
        + float(
            config[
                "backtest"
            ][
                "slippage_bps"
            ]
        )
    )

    for row in (
        broker
        .open_positions()
        .itertuples(
            index=False
        )
    ):
        ticker_y = str(
            row.ticker_y
        )

        ticker_x = str(
            row.ticker_x
        )

        if (
            ticker_y
            not in prices.columns
            or ticker_x
            not in prices.columns
        ):
            continue

        recent = prices[
            [
                ticker_y,
                ticker_x,
            ]
        ].iloc[
            -int(
                live.get(
                    "invalidation_lookback_days",
                    252,
                )
            ):
        ]

        z = _latest_zscore(
            recent,
            ticker_y,
            ticker_x,
            intercept=float(
                row.intercept
            ),
            hedge_ratio=float(
                row.hedge_ratio
            ),
            zscore_window=int(
                strategy[
                    "zscore_window"
                ]
            ),
        )

        invalidated = False

        try:
            coint_p = float(
                engle_granger_test(
                    recent[
                        ticker_y
                    ],
                    recent[
                        ticker_x
                    ],
                )[
                    "p_value"
                ]
            )

            invalidated = (
                coint_p
                > float(
                    live.get(
                        "invalidation_pvalue",
                        0.10,
                    )
                )
            )

        except ValueError:
            invalidated = True

        direction = int(
            row.direction
        )

        exit_reason: (
            str | None
        ) = None

        if invalidated:
            exit_reason = (
                "RELATIONSHIP_INVALIDATED"
            )

        elif (
            direction == 1
            and z
            <= -float(
                strategy[
                    "stop_loss_zscore"
                ]
            )
        ):
            exit_reason = (
                "STOP_LONG"
            )

        elif (
            direction == -1
            and z
            >= float(
                strategy[
                    "stop_loss_zscore"
                ]
            )
        ):
            exit_reason = (
                "STOP_SHORT"
            )

        elif (
            direction == 1
            and (
                abs(z)
                <= float(
                    strategy[
                        "exit_threshold"
                    ]
                )
                or z > 0
            )
        ):
            exit_reason = (
                "EXIT_LONG"
            )

        elif (
            direction == -1
            and (
                abs(z)
                <= float(
                    strategy[
                        "exit_threshold"
                    ]
                )
                or z < 0
            )
        ):
            exit_reason = (
                "EXIT_SHORT"
            )

        if (
            exit_reason
            is not None
        ):
            result = (
                broker.close_pair(
                    pair=str(
                        row.pair
                    ),
                    date=str(
                        latest_date.date()
                    ),
                    price_y=float(
                        latest_prices[
                            ticker_y
                        ]
                    ),
                    price_x=float(
                        latest_prices[
                            ticker_x
                        ]
                    ),
                    transaction_cost_bps=(
                        effective_cost_bps
                    ),
                    reason=exit_reason,
                )
            )

            print(
                f"CLOSED {row.pair}: "
                f"{exit_reason}, "
                f"net PnL "
                f"{result['net_pnl']:.2f}"
            )

    scan = build_market_scan(
        prices,
        config,
        groups=groups,
    )

    reports_dir = (
        get_project_root()
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan.to_csv(
        reports_dir
        / (
            f"{args.universe}"
            "_current_scan.csv"
        ),
        index=False,
    )

    risk_cfg = config[
        "risk"
    ]

    limits = RiskLimits(
        max_open_pairs=int(
            risk_cfg[
                "max_open_pairs"
            ]
        ),
        max_pair_gross_exposure=float(
            risk_cfg[
                "max_pair_gross_exposure"
            ]
        ),
        max_portfolio_gross_exposure=float(
            risk_cfg[
                "max_portfolio_gross_exposure"
            ]
        ),
        max_sector_gross_exposure=float(
            risk_cfg[
                "max_sector_gross_exposure"
            ]
        ),
        max_drawdown=float(
            risk_cfg[
                "max_drawdown"
            ]
        ),
    )

    drawdown_value = 0.0

    if broker.equity_path.exists():
        equity_history = (
            pd.read_csv(
                broker.equity_path
            )
        )

        if (
            not equity_history.empty
            and "equity"
            in equity_history
        ):
            drawdown_value = (
                current_drawdown(
                    equity_history[
                        "equity"
                    ].astype(float)
                )
            )

    if not scan.empty:
        for candidate in scan.itertuples(
            index=False
        ):
            if candidate.status not in {
                "ENTRY LONG",
                "ENTRY SHORT",
            }:
                continue

            pair = (
                f"{candidate.ticker_y}"
                "/"
                f"{candidate.ticker_x}"
            )

            if broker.has_pair(
                pair
            ):
                continue

            snapshot = (
                broker
                .mark_to_market(
                    latest_prices
                )
            )

            open_count = len(
                broker
                .open_positions()
            )

            proposed = (
                limits
                .max_pair_gross_exposure
            )

            decision = approve_new_pair(
                open_pair_count=open_count,
                current_gross_exposure=float(
                    snapshot[
                        "gross_exposure_fraction"
                    ]
                ),
                proposed_pair_gross_exposure=(
                    proposed
                ),
                limits=limits,
                current_drawdown_value=(
                    drawdown_value
                ),
            )

            if not bool(
                decision[
                    "approved"
                ]
            ):
                continue

            if (
                candidate.status
                == "ENTRY LONG"
            ):
                direction = 1
            else:
                direction = -1

            broker.open_pair(
                pair=pair,
                ticker_y=str(
                    candidate.ticker_y
                ),
                ticker_x=str(
                    candidate.ticker_x
                ),
                direction=direction,
                date=str(
                    latest_date.date()
                ),
                price_y=float(
                    latest_prices[
                        candidate.ticker_y
                    ]
                ),
                price_x=float(
                    latest_prices[
                        candidate.ticker_x
                    ]
                ),
                zscore=float(
                    candidate.latest_zscore
                ),
                intercept=float(
                    candidate.intercept
                ),
                hedge_ratio=float(
                    candidate.hedge_ratio
                ),
                capital_reference=float(
                    snapshot[
                        "equity"
                    ]
                ),
                gross_exposure_fraction=(
                    proposed
                ),
                transaction_cost_bps=(
                    effective_cost_bps
                ),
            )

            print(
                f"OPENED {pair}: "
                f"{candidate.status}, "
                f"z="
                f"{float(candidate.latest_zscore):.2f}"
            )

    final_snapshot = (
        broker.record_equity(
            str(
                latest_date.date()
            ),
            latest_prices,
        )
    )

    print(
        "\nPaper account"
    )

    for key, value in (
        final_snapshot.items()
    ):
        print(
            f"{key}: {value:.4f}"
        )

    print(
        "Open pairs: "
        f"{len(broker.open_positions())}"
    )

    print(
        f"State: {state_dir}"
    )


if __name__ == "__main__":
    main()
