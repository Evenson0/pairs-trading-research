"""Check that the project is correctly installed."""

from pairs_trading_research.config import load_named_config
from pairs_trading_research.universe import get_universe_tickers


def main() -> None:
    """Run a simple setup check."""
    config = load_named_config("tsx60")
    tickers = get_universe_tickers("tsx60")

    print("Project setup check")
    print("-------------------")
    print(f"Loaded universe: {config['universe']['name']}")
    print(f"Number of tickers: {len(tickers)}")
    print("Setup check completed successfully.")


if __name__ == "__main__":
    main()
