"""Equity universe definitions and metadata loaders."""

from __future__ import annotations

import pandas as pd


def get_tsx60_tickers() -> list[str]:
    """Return the curated Canadian large-cap universe."""
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


def get_sp500_constituents() -> pd.DataFrame:
    """Load current S&P 500 tickers and GICS sectors.

    This is appropriate for current scanning.

    Using current constituents for historical research introduces
    survivorship bias.
    """
    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    table = pd.read_html(
        url,
        match="Symbol",
    )[0]

    required = {
        "Symbol",
        "GICS Sector",
    }

    if not required.issubset(
        table.columns
    ):
        raise ValueError(
            "Could not identify S&P 500 symbol/sector columns."
        )

    result = table[
        [
            "Symbol",
            "GICS Sector",
        ]
    ].copy()

    result[
        "Symbol"
    ] = (
        result[
            "Symbol"
        ]
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    result = result.rename(
        columns={
            "Symbol": "ticker",
            "GICS Sector": "sector",
        }
    )

    return (
        result
        .drop_duplicates(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )


def get_sp500_sample_tickers() -> list[str]:
    """Return a small offline development universe."""
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


def get_universe_metadata(
    universe: str,
) -> pd.DataFrame:
    """Return ticker metadata."""
    name = (
        universe
        .lower()
        .strip()
    )

    if name == "tsx60":
        return pd.DataFrame(
            {
                "ticker": (
                    get_tsx60_tickers()
                ),
                "sector": None,
            }
        )

    if name == "sp500":
        return (
            get_sp500_constituents()
        )

    if name == "sp500_sample":
        return pd.DataFrame(
            {
                "ticker": (
                    get_sp500_sample_tickers()
                ),
                "sector": None,
            }
        )

    raise ValueError(
        f"Unsupported universe: {universe}"
    )


def get_universe_tickers(
    universe: str,
) -> list[str]:
    """Return ticker symbols."""
    return (
        get_universe_metadata(
            universe
        )["ticker"]
        .tolist()
    )
