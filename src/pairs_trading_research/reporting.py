"""Markdown reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_summary_json(
    summary: dict[str, float],
    output_path: str | Path,
) -> None:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_baseline_report(
    output_path: str | Path,
    universe_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_pair: pd.Series,
    summary: dict[str, float],
    transaction_cost_bps: float,
    slippage_bps: float,
) -> None:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top = (
        candidates
        .head(10)[
            [
                "ticker_y",
                "ticker_x",
                "coint_p_value",
                "adjusted_p_value",
                "correlation",
                "half_life",
                "hedge_ratio_stability",
                "pair_score",
            ]
        ]
        .copy()
    )

    top_table = top.to_markdown(
        index=False,
        floatfmt=".4f",
    )

    text = f"""# {universe_name} Pairs Trading Baseline Report

## Executive Summary

This report evaluates a cointegration-based pairs trading strategy using a strict chronological train/test split. Pair selection, multiple-testing correction, hedge-ratio estimation, and model diagnostics are performed on the training sample only. Trading performance is then measured on the held-out test sample.

## Data Split

- Training: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} observations)
- Test: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} observations)

## Selection Method

Candidate pairs are screened by correlation, tested with Engle-Granger cointegration, adjusted with Benjamini-Hochberg false-discovery-rate control, and ranked using cointegration strength, correlation, estimated half-life, hedge-ratio stability, and spread stationarity.

### Top Training-Sample Candidates

{top_table}

## Selected Pair

- Pair: {selected_pair['ticker_y']} / {selected_pair['ticker_x']}
- OLS intercept: {float(selected_pair['intercept']):.6f}
- Hedge ratio: {float(selected_pair['hedge_ratio']):.6f}
- Raw cointegration p-value: {float(selected_pair['coint_p_value']):.6f}
- FDR-adjusted p-value: {float(selected_pair['adjusted_p_value']):.6f}
- Spread ADF p-value: {float(selected_pair['spread_adf_p_value']):.6f}
- Correlation: {float(selected_pair['correlation']):.4f}
- Estimated half-life: {float(selected_pair['half_life']):.2f} trading days
- Hedge-ratio stability score: {float(selected_pair['hedge_ratio_stability']):.4f}

## Out-of-Sample Performance

- Total return: {summary['total_return']:.2%}
- Annualized return: {summary['annualized_return']:.2%}
- Annualized volatility: {summary['annualized_volatility']:.2%}
- Sharpe ratio: {summary['sharpe_ratio']:.3f}
- Maximum drawdown: {summary['max_drawdown']:.2%}
- Positive-return day fraction: {summary['win_rate']:.2%}
- Number of trades: {int(summary.get('number_of_trades', 0))}
- Average holding period: {summary.get('average_holding_period', 0.0):.2f} trading days
- Total turnover: {summary.get('total_turnover', 0.0):.2f}x portfolio notional

## Trading Assumptions

- Transaction cost: {transaction_cost_bps:.1f} bps of turnover
- Slippage: {slippage_bps:.1f} bps of turnover
- Signals are acted on with a one-period lag in the backtest.
- The primary backtest models both legs explicitly.

## Limitations

- Current index constituents can create survivorship bias in historical research.
- Yahoo Finance data is suitable for research and prototyping, not institutional execution quality.
- Borrow availability, borrow fees, taxes, market impact, and intraday execution are not modeled.
- Statistical relationships can break after the sample period.
- A profitable historical result is not evidence of guaranteed future profitability.

## Next Validation Layer

The walk-forward workflow repeatedly reselects pairs and re-estimates parameters using only information available at each historical decision date. Paper trading should be used before considering real capital.
"""

    path.write_text(
        text,
        encoding="utf-8",
    )
