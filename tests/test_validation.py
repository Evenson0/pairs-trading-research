import pandas as pd

from pairs_trading_research.validation import (
    split_train_test,
)


def test_split_train_test_is_chronological() -> None:
    index = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="D",
    )

    prices = pd.DataFrame(
        {
            "A": range(100),
            "B": range(
                100,
                200,
            ),
        },
        index=index,
    )

    train, test = split_train_test(
        prices,
        0.70,
    )

    assert len(train) == 70
    assert len(test) == 30
    assert (
        train.index.max()
        < test.index.min()
    )
