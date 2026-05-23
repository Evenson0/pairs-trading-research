"""Run the baseline TSX60 pairs trading pipeline."""

from __future__ import annotations

from pairs_trading_research.backtest import run_spread_backtest
from pairs_trading_research.cointegration import (
    compute_spread,
    find_cointegrated_pairs,
)
from pairs_trading_research.config import load_named_config
from pairs_trading_research.data_loader import download_adjusted_prices
from pairs_trading_research.metrics import compute_performance_summary
from pairs_trading_research.preprocessing import clean_price_data
from pairs_trading_research.signals import generate_pair_signals
from pairs_trading_research.universe import get_universe_tickers


def main() -> None:
    """Run the TSX60 baseline research pipeline."""
    config = load_named_config("tsx60")

    tickers = get_universe_tickers("tsx60")

    prices = download_adjusted_prices(
        tickers=tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        price_field=config["data"]["price_field"],
    )

    prices = clean_price_data(
        prices,
        max_missing_ratio=config["filters"]["max_missing_ratio"],
        min_observations=config["filters"]["min_observations"],
    )

    pairs = find_cointegrated_pairs(
        prices,
        max_p_value=config["pair_selection"]["adf_pvalue_threshold"],
        min_correlation=config["pair_selection"]["min_correlation"],
    )

    if pairs.empty:
        print("No cointegrated pairs found with the current parameters.")
        return

    best_pair = pairs.iloc[0]

    ticker_y = best_pair["ticker_y"]
    ticker_x = best_pair["ticker_x"]
    hedge_ratio = best_pair["hedge_ratio"]

    spread = compute_spread(
        prices[ticker_y],
        prices[ticker_x],
        hedge_ratio=hedge_ratio,
    )

    signals = generate_pair_signals(
        spread,
        zscore_window=config["strategy"]["zscore_window"],
        entry_threshold=config["strategy"]["entry_threshold"],
        exit_threshold=config["strategy"]["exit_threshold"],
    )

    results = run_spread_backtest(
        spread=signals["spread"],
        positions=signals["position"],
        initial_capital=config["backtest"]["initial_capital"],
        transaction_cost_bps=config["strategy"]["transaction_cost_bps"],
    )

    summary = compute_performance_summary(
        portfolio_value=results["portfolio_value"],
        returns=results["strategy_return"],
    )

    print("Selected pair")
    print("-------------")
    print(f"{ticker_y} / {ticker_x}")
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"Cointegration p-value: {best_pair['coint_p_value']:.4f}")
    print(f"Spread ADF p-value: {best_pair['spread_adf_p_value']:.4f}")
    print()

    print("Backtest summary")
    print("----------------")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
