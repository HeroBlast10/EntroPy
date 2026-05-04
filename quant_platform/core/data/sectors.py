"""Sector classification helpers for portfolio constraints and analytics."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.utils.io import load_parquet, resolve_data_path, save_parquet


def default_sector_map_path() -> Path:
    """Default cache location for the sector map."""
    return resolve_data_path("reference", "sector_map.parquet")


def fetch_sp500_sector_map() -> pd.DataFrame:
    """Fetch current S&P 500 sector classifications from Wikipedia."""
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = tables[0].rename(columns={
        "Symbol": "ticker",
        "GICS Sector": "sector",
    })
    if "ticker" not in table.columns or "sector" not in table.columns:
        raise ValueError("Wikipedia S&P 500 table did not contain Symbol/GICS Sector columns")

    sector_map = table[["ticker", "sector"]].copy()
    sector_map["ticker"] = sector_map["ticker"].str.replace(".", "-", regex=False)
    sector_map["sector"] = sector_map["sector"].fillna("Unknown")
    sector_map = sector_map.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    logger.info("Fetched sector map for {} tickers from Wikipedia", len(sector_map))
    return sector_map


def load_sector_map(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load a cached sector map, returning an empty frame when unavailable."""
    path = Path(path) if path is not None else default_sector_map_path()
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "sector"])
    df = load_parquet(path)
    return df[["ticker", "sector"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def ensure_sector_map(
    tickers: Optional[Iterable[str]] = None,
    path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load a cached sector map or fetch/build it when missing."""
    path = Path(path) if path is not None else default_sector_map_path()

    sector_map = load_sector_map(path)
    requested = set(tickers or [])
    if requested:
        cached = set(sector_map["ticker"]) if not sector_map.empty else set()
        missing = requested - cached
    else:
        missing = set()

    if sector_map.empty or missing:
        try:
            fetched = fetch_sp500_sector_map()
            if fetched is not None and not fetched.empty:
                sector_map = fetched
                save_parquet(sector_map, path)
        except Exception as exc:
            logger.warning("Sector map fetch failed: {}", exc)

    if requested and not sector_map.empty:
        sector_map = sector_map[sector_map["ticker"].isin(requested)].copy()

    if sector_map.empty:
        logger.warning("No sector map available; sector constraints will be skipped")

    return sector_map.reset_index(drop=True)
