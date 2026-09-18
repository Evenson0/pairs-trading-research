from pathlib import Path

from pairs_trading_research.paper_broker import (
    PaperBroker,
)


def test_paper_broker_open_mark_close(
    tmp_path: Path,
) -> None:
    broker = PaperBroker(
        tmp_path,
        initial_capital=10_000,
    )

    broker.open_pair(
        pair="Y/X",
        ticker_y="Y",
        ticker_x="X",
        direction=1,
        date="2026-01-01",
        price_y=100,
        price_x=50,
        zscore=-2.2,
        intercept=0.0,
        hedge_ratio=2.0,
        capital_reference=10_000,
        gross_exposure_fraction=0.20,
        transaction_cost_bps=0.0,
    )

    assert broker.has_pair(
        "Y/X"
    )

    mtm = (
        broker.mark_to_market(
            {
                "Y": 105,
                "X": 50,
            }
        )
    )

    assert (
        mtm[
            "unrealized_pnl"
        ]
        > 0
    )

    trade = (
        broker.close_pair(
            pair="Y/X",
            date="2026-01-10",
            price_y=105,
            price_x=50,
            transaction_cost_bps=0.0,
            reason="TEST",
        )
    )

    assert (
        trade[
            "net_pnl"
        ]
        > 0
    )

    assert not broker.has_pair(
        "Y/X"
    )
