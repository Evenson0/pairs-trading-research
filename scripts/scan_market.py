"""Scan an equity universe for current pairs-trading opportunities."""

from __future__ import annotations

import argparse

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
from pairs_trading_research.scanner import (
    build_market_scan,
)
from pairs_trading_research.universe import (
    get_universe_metadata,
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

    parser.add_argument(
        "--top",
        type=int,
        default=20,
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

    scan = build_market_scan(
        prices,
        config,
        groups=groups,
    )

    if scan.empty:
        print(
            "No current pair passed the configured statistical filters."
        )
        return

    output = (
        get_project_root()
        / "reports"
        / (
            f"{args.universe}"
            "_current_scan.csv"
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan.to_csv(
        output,
        index=False,
    )

    columns = [
        "ticker_y",
        "ticker_x",
        "latest_zscore",
        "half_life",
        "adjusted_p_value",
        "correlation",
        "pair_score",
        "status",
    ]

    print(
        scan[
            columns
        ]
        .head(
            args.top
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()
