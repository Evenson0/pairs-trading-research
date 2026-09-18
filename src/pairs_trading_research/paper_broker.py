"""Persistent local paper broker."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .portfolio import (
    position_notional_plan,
)


POSITION_COLUMNS = [
    "pair",
    "ticker_y",
    "ticker_x",
    "direction",
    "entry_date",
    "entry_price_y",
    "entry_price_x",
    "shares_y",
    "shares_x",
    "gross_notional",
    "entry_zscore",
    "intercept",
    "hedge_ratio",
    "entry_cost",
]


TRADE_COLUMNS = [
    "pair",
    "ticker_y",
    "ticker_x",
    "direction",
    "entry_date",
    "exit_date",
    "entry_price_y",
    "entry_price_x",
    "exit_price_y",
    "exit_price_x",
    "gross_pnl",
    "entry_cost",
    "exit_cost",
    "net_pnl",
    "exit_reason",
]


class PaperBroker:
    """Persistent CSV/JSON-backed paper broker."""

    def __init__(
        self,
        state_dir: str | Path,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.state_dir = Path(
            state_dir
        )

        self.state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.positions_path = (
            self.state_dir
            / "positions.csv"
        )

        self.trades_path = (
            self.state_dir
            / "trades.csv"
        )

        self.equity_path = (
            self.state_dir
            / "equity.csv"
        )

        self.account_path = (
            self.state_dir
            / "account.json"
        )

        if not self.account_path.exists():
            self._write_account(
                {
                    "initial_capital": float(
                        initial_capital
                    ),
                    "realized_pnl": 0.0,
                    "cumulative_costs": 0.0,
                }
            )

        if not self.positions_path.exists():
            pd.DataFrame(
                columns=POSITION_COLUMNS
            ).to_csv(
                self.positions_path,
                index=False,
            )

        if not self.trades_path.exists():
            pd.DataFrame(
                columns=TRADE_COLUMNS
            ).to_csv(
                self.trades_path,
                index=False,
            )

    def _read_account(
        self,
    ) -> dict[str, float]:
        return json.loads(
            self.account_path
            .read_text(
                encoding="utf-8"
            )
        )

    def _write_account(
        self,
        account: dict[str, float],
    ) -> None:
        self.account_path.write_text(
            json.dumps(
                account,
                indent=2,
            ),
            encoding="utf-8",
        )

    def open_positions(
        self,
    ) -> pd.DataFrame:
        positions = pd.read_csv(
            self.positions_path
        )

        if positions.empty:
            return pd.DataFrame(
                columns=POSITION_COLUMNS
            )

        return positions

    def trades(
        self,
    ) -> pd.DataFrame:
        return pd.read_csv(
            self.trades_path
        )

    def has_pair(
        self,
        pair: str,
    ) -> bool:
        positions = (
            self.open_positions()
        )

        return (
            not positions.empty
            and pair
            in set(
                positions[
                    "pair"
                ].astype(str)
            )
        )

    def mark_to_market(
        self,
        latest_prices: dict[
            str,
            float,
        ],
    ) -> dict[str, float]:
        account = (
            self._read_account()
        )

        positions = (
            self.open_positions()
        )

        unrealized = 0.0
        gross_exposure = 0.0

        for row in positions.itertuples(
            index=False
        ):
            if (
                row.ticker_y
                not in latest_prices
                or row.ticker_x
                not in latest_prices
            ):
                continue

            price_y = float(
                latest_prices[
                    row.ticker_y
                ]
            )

            price_x = float(
                latest_prices[
                    row.ticker_x
                ]
            )

            unrealized += (
                float(
                    row.shares_y
                )
                * (
                    price_y
                    - float(
                        row.entry_price_y
                    )
                )
            )

            unrealized += (
                float(
                    row.shares_x
                )
                * (
                    price_x
                    - float(
                        row.entry_price_x
                    )
                )
            )

            gross_exposure += abs(
                float(
                    row.shares_y
                )
                * price_y
            )

            gross_exposure += abs(
                float(
                    row.shares_x
                )
                * price_x
            )

        equity = (
            float(
                account[
                    "initial_capital"
                ]
            )
            + float(
                account[
                    "realized_pnl"
                ]
            )
            + unrealized
            - float(
                account[
                    "cumulative_costs"
                ]
            )
        )

        if equity > 0:
            gross_exposure_fraction = (
                gross_exposure
                / equity
            )
        else:
            gross_exposure_fraction = (
                float("inf")
            )

        return {
            "equity": float(
                equity
            ),
            "unrealized_pnl": float(
                unrealized
            ),
            "realized_pnl": float(
                account[
                    "realized_pnl"
                ]
            ),
            "cumulative_costs": float(
                account[
                    "cumulative_costs"
                ]
            ),
            "gross_exposure": float(
                gross_exposure
            ),
            "gross_exposure_fraction": float(
                gross_exposure_fraction
            ),
        }

    def record_equity(
        self,
        date: str,
        latest_prices: dict[
            str,
            float,
        ],
    ) -> dict[str, float]:
        snapshot = (
            self.mark_to_market(
                latest_prices
            )
        )

        row = pd.DataFrame(
            [
                {
                    "date": date,
                    **snapshot,
                }
            ]
        )

        if self.equity_path.exists():
            equity = pd.read_csv(
                self.equity_path
            )

            if not equity.empty:
                equity = equity[
                    equity[
                        "date"
                    ].astype(str)
                    != str(date)
                ]

            equity = pd.concat(
                [
                    equity,
                    row,
                ],
                ignore_index=True,
            )

        else:
            equity = row

        equity.to_csv(
            self.equity_path,
            index=False,
        )

        return snapshot

    def open_pair(
        self,
        pair: str,
        ticker_y: str,
        ticker_x: str,
        direction: int,
        date: str,
        price_y: float,
        price_x: float,
        zscore: float,
        intercept: float,
        hedge_ratio: float,
        capital_reference: float,
        gross_exposure_fraction: float,
        transaction_cost_bps: float,
    ) -> None:
        if self.has_pair(
            pair
        ):
            raise ValueError(
                f"Pair already open: {pair}"
            )

        plan = position_notional_plan(
            capital=capital_reference,
            price_y=price_y,
            price_x=price_x,
            direction=direction,
            gross_exposure_fraction=(
                gross_exposure_fraction
            ),
        )

        entry_cost = (
            plan[
                "gross_notional"
            ]
            * transaction_cost_bps
            / 10_000.0
        )

        positions = (
            self.open_positions()
        )

        row = {
            "pair": pair,
            "ticker_y": ticker_y,
            "ticker_x": ticker_x,
            "direction": int(
                direction
            ),
            "entry_date": date,
            "entry_price_y": float(
                price_y
            ),
            "entry_price_x": float(
                price_x
            ),
            "shares_y": float(
                plan[
                    "shares_y"
                ]
            ),
            "shares_x": float(
                plan[
                    "shares_x"
                ]
            ),
            "gross_notional": float(
                plan[
                    "gross_notional"
                ]
            ),
            "entry_zscore": float(
                zscore
            ),
            "intercept": float(
                intercept
            ),
            "hedge_ratio": float(
                hedge_ratio
            ),
            "entry_cost": float(
                entry_cost
            ),
        }

        if positions.empty:
            positions = (
                pd.DataFrame(
                    [row]
                )
            )
        else:
            positions = pd.concat(
                [
                    positions,
                    pd.DataFrame(
                        [row]
                    ),
                ],
                ignore_index=True,
            )

        positions.to_csv(
            self.positions_path,
            index=False,
        )

        account = (
            self._read_account()
        )

        account[
            "cumulative_costs"
        ] = (
            float(
                account[
                    "cumulative_costs"
                ]
            )
            + entry_cost
        )

        self._write_account(
            account
        )

    def close_pair(
        self,
        pair: str,
        date: str,
        price_y: float,
        price_x: float,
        transaction_cost_bps: float,
        reason: str,
    ) -> dict[str, float]:
        positions = (
            self.open_positions()
        )

        matches = positions[
            positions[
                "pair"
            ].astype(str)
            == pair
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one open position for {pair}."
            )

        row = matches.iloc[0]

        gross_pnl = (
            float(
                row[
                    "shares_y"
                ]
            )
            * (
                price_y
                - float(
                    row[
                        "entry_price_y"
                    ]
                )
            )
            + float(
                row[
                    "shares_x"
                ]
            )
            * (
                price_x
                - float(
                    row[
                        "entry_price_x"
                    ]
                )
            )
        )

        exit_notional = (
            abs(
                float(
                    row[
                        "shares_y"
                    ]
                )
                * price_y
            )
            + abs(
                float(
                    row[
                        "shares_x"
                    ]
                )
                * price_x
            )
        )

        exit_cost = (
            exit_notional
            * transaction_cost_bps
            / 10_000.0
        )

        net_pnl = (
            gross_pnl
            - float(
                row[
                    "entry_cost"
                ]
            )
            - exit_cost
        )

        trade = {
            "pair": pair,
            "ticker_y": row[
                "ticker_y"
            ],
            "ticker_x": row[
                "ticker_x"
            ],
            "direction": int(
                row[
                    "direction"
                ]
            ),
            "entry_date": row[
                "entry_date"
            ],
            "exit_date": date,
            "entry_price_y": float(
                row[
                    "entry_price_y"
                ]
            ),
            "entry_price_x": float(
                row[
                    "entry_price_x"
                ]
            ),
            "exit_price_y": float(
                price_y
            ),
            "exit_price_x": float(
                price_x
            ),
            "gross_pnl": float(
                gross_pnl
            ),
            "entry_cost": float(
                row[
                    "entry_cost"
                ]
            ),
            "exit_cost": float(
                exit_cost
            ),
            "net_pnl": float(
                net_pnl
            ),
            "exit_reason": reason,
        }

        trades = self.trades()

        if trades.empty:
            trades = pd.DataFrame(
                [trade]
            )

        else:
            trades = pd.concat(
                [
                    trades,
                    pd.DataFrame(
                        [trade]
                    ),
                ],
                ignore_index=True,
            )

        trades.to_csv(
            self.trades_path,
            index=False,
        )

        positions = positions[
            positions[
                "pair"
            ].astype(str)
            != pair
        ]

        positions.to_csv(
            self.positions_path,
            index=False,
        )

        account = (
            self._read_account()
        )

        account[
            "realized_pnl"
        ] = (
            float(
                account[
                    "realized_pnl"
                ]
            )
            + gross_pnl
        )

        account[
            "cumulative_costs"
        ] = (
            float(
                account[
                    "cumulative_costs"
                ]
            )
            + exit_cost
        )

        self._write_account(
            account
        )

        return {
            "gross_pnl": float(
                gross_pnl
            ),
            "net_pnl": float(
                net_pnl
            ),
            "exit_cost": float(
                exit_cost
            ),
        }
