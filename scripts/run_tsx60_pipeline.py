"""Run a strict train/test S&P/TSX 60 baseline experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pairs_trading_research.cointegration import (
    compute_spread,
)
from pairs_trading_research.config import (
    get_project_root,
    load_named_config,
)
from pairs_trading_research.data_loader import (
    download_adjusted_prices,
)
from pairs_trading_research.metrics import (
    compute_performance_summary,
)
from pairs_trading_research.pair_selection import (
    select_pairs,
)
from pairs_trading_research.portfolio import (
    run_pair_backtest,
)
from pairs_trading_research.preprocessing import (
    clean_price_data,
)
from pairs_trading_research.reporting import (
    save_summary_json,
    write_baseline_report,
)
from pairs_trading_research.signals import (
    generate_pair_signals,
)
from pairs_trading_research.universe import (
    get_universe_tickers,
)
from pairs_trading_research.validation import (
    split_train_test,
)
from pairs_trading_research.visualization import (
    plot_drawdown,
    plot_portfolio_value,
    plot_zscore,
)


def main() -> None:
    config = load_named_config(
        "tsx60"
    )

    root = get_project_root()

    reports_dir = (
        root
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tickers = (
        get_universe_tickers(
            "tsx60"
        )
    )

    prices = (
        download_adjusted_prices(
            tickers=tickers,
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

    train, test = split_train_test(
        prices,
        float(
            config[
                "backtest"
            ][
                "train_ratio"
            ]
        ),
    )

    pair_cfg = config[
        "pair_selection"
    ]

    candidates = select_pairs(
        train,
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
            pair_cfg[
                "require_positive_hedge_ratio"
            ]
        ),
        stability_window=int(
            pair_cfg[
                "stability_window"
            ]
        ),
        stability_step=int(
            pair_cfg[
                "stability_step"
            ]
        ),
    )

    candidates.to_csv(
        reports_dir
        / "tsx60_training_candidates.csv",
        index=False,
    )

    if candidates.empty:
        print(
            "No FDR-approved training-sample pair satisfied the configured filters."
        )
        return

    best = candidates.iloc[0]

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

    strategy_cfg = config[
        "strategy"
    ]

    z_window = int(
        strategy_cfg[
            "zscore_window"
        ]
    )

    warmup = train[
        [
            ticker_y,
            ticker_x,
        ]
    ].iloc[
        -max(
            2 * z_window,
            z_window + 5,
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
        zscore_window=z_window,
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

    backtest_cfg = config[
        "backtest"
    ]

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
        initial_capital=float(
            backtest_cfg[
                "initial_capital"
            ]
        ),
        gross_exposure=float(
            backtest_cfg[
                "gross_exposure"
            ]
        ),
        transaction_cost_bps=float(
            backtest_cfg[
                "transaction_cost_bps"
            ]
        ),
        slippage_bps=float(
            backtest_cfg[
                "slippage_bps"
            ]
        ),
        position_type=str(
            backtest_cfg[
                "position_type"
            ]
        ),
    )

    summary = (
        compute_performance_summary(
            results[
                "portfolio_value"
            ],
            returns=results[
                "strategy_return"
            ],
            positions=test_positions,
            turnover=results[
                "turnover"
            ],
        )
    )

    results.to_csv(
        reports_dir
        / "tsx60_oos_backtest.csv"
    )

    figures_dir = (
        reports_dir
        / "figures"
    )

    plot_portfolio_value(
        results,
        figures_dir
        / "tsx60_oos_equity.png",
        show=False,
    )

    plot_drawdown(
        results[
            "portfolio_value"
        ],
        figures_dir
        / "tsx60_oos_drawdown.png",
        show=False,
    )

    test_zscore = (
        signals[
            "zscore"
        ]
        .reindex(
            test.index
        )
        .dropna()
    )

    plot_zscore(
        test_zscore,
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
        output_path=(
            figures_dir
            / "tsx60_oos_zscore.png"
        ),
        show=False,
    )

    save_summary_json(
        summary,
        reports_dir
        / "tsx60_oos_summary.json",
    )

    write_baseline_report(
        reports_dir
        / "tsx60_baseline_report.md",
        universe_name=config[
            "universe"
        ][
            "name"
        ],
        train=train,
        test=test,
        candidates=candidates,
        selected_pair=best,
        summary=summary,
        transaction_cost_bps=float(
            backtest_cfg[
                "transaction_cost_bps"
            ]
        ),
        slippage_bps=float(
            backtest_cfg[
                "slippage_bps"
            ]
        ),
    )

    print(
        f"Training: {train.index[0].date()} -> {train.index[-1].date()}"
    )

    print(
        f"Testing : {test.index[0].date()} -> {test.index[-1].date()}"
    )

    print(
        f"Selected pair: {ticker_y} / {ticker_x}"
    )

    print(
        "Adjusted p-value: "
        f"{float(best['adjusted_p_value']):.6f}"
    )

    print(
        "Half-life: "
        f"{float(best['half_life']):.2f} trading days"
    )

    print(
        "\nOut-of-sample performance"
    )

    for key, value in summary.items():
        print(
            f"{key}: {value:.6f}"
        )

    print(
        "\nSaved outputs in: "
        f"{Path(reports_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
