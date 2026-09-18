from pairs_trading_research.risk import (
    RiskLimits,
    approve_new_pair,
)


def test_rejects_when_pair_limit_reached() -> None:
    limits = RiskLimits(
        max_open_pairs=2
    )

    result = (
        approve_new_pair(
            2,
            0.4,
            0.2,
            limits,
        )
    )

    assert (
        result[
            "approved"
        ]
        is False
    )

    assert result[
        "reasons"
    ]


def test_approves_within_limits() -> None:
    limits = (
        RiskLimits()
    )

    result = (
        approve_new_pair(
            1,
            0.2,
            0.2,
            limits,
        )
    )

    assert (
        result[
            "approved"
        ]
        is True
    )
