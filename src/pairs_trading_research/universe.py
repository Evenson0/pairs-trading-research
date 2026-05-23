"""Equity universe definitions.

This module provides ticker lists for supported equity universes.
The first version uses manually curated lists to keep the project
stable and reproducible. Automated data sources can be added later.
"""

from __future__ import annotations


def get_tsx60_tickers() -> list[str]:
    """Return a manually curated list of S&P/TSX 60 tickers.

    The tickers use the Yahoo Finance Canadian suffix `.TO`.
    """
    return [
        "AEM.TO",
        "AQN.TO",
        "ATD.TO",
        "BAM.TO",
        "BCE.TO",
        "BIP-UN.TO",
        "BMO.TO",
        "BN.TO",
        "BNS.TO",
        "CAE.TO",
        "CAR-UN.TO",
        "CCL-B.TO",
        "CCO.TO",
        "CM.TO",
        "CNQ.TO",
        "CNR.TO",
        "CP.TO",
        "CSU.TO",
        "CTC-A.TO",
        "CU.TO",
        "CVE.TO",
        "DOL.TO",
        "EMA.TO",
        "ENB.TO",
        "FM.TO",
        "FNV.TO",
        "FSV.TO",
        "FTS.TO",
        "GIB-A.TO",
        "GIL.TO",
        "H.TO",
        "IFC.TO",
        "IMO.TO",
        "K.TO",
        "L.TO",
        "MFC.TO",
        "MG.TO",
        "MRU.TO",
        "NA.TO",
        "NTR.TO",
        "OTEX.TO",
        "POW.TO",
        "QSR.TO",
        "RCI-B.TO",
        "RY.TO",
        "SAP.TO",
        "SHOP.TO",
        "SLF.TO",
        "SU.TO",
        "T.TO",
        "TD.TO",
        "TECK-B.TO",
        "TOU.TO",
        "TRI.TO",
        "TRP.TO",
        "WCN.TO",
        "WN.TO",
        "WPM.TO",
        "WSP.TO",
    ]


def get_sp500_sample_tickers() -> list[str]:
    """Return a small sample of S&P 500 tickers for development.

    A full S&P 500 universe loader will be added later.
    """
    return [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "JPM",
        "BAC",
        "WFC",
        "XOM",
        "CVX",
        "KO",
        "PEP",
        "PG",
        "WMT",
        "COST",
        "HD",
        "UNH",
        "JNJ",
        "PFE",
    ]


def get_universe_tickers(universe: str) -> list[str]:
    """Return tickers for a supported universe.

    Parameters
    ----------
    universe:
        Universe identifier. Supported values are `"tsx60"` and `"sp500_sample"`.

    Returns
    -------
    list[str]
        List of ticker symbols.

    Raises
    ------
    ValueError
        If the universe is not supported.
    """
    universe = universe.lower().strip()

    if universe == "tsx60":
        return get_tsx60_tickers()

    if universe in {"sp500", "sp500_sample"}:
        return get_sp500_sample_tickers()

    raise ValueError(
        f"Unsupported universe: {universe}. "
        "Supported universes are: 'tsx60', 'sp500_sample'."
    )
