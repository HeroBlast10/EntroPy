"""Independent universe/identity and financial-vintage admission gates."""
from __future__ import annotations

import pandas as pd

from .universe import require_point_in_time_universe


FUNDAMENTAL_FEATURES = frozenset({"EARNINGS_YIELD", "BOOK_TO_MARKET", "GROSS_PROFITABILITY",
                                 "ASSET_GROWTH", "ACCRUALS", "PIOTROSKI_F8"})


def require_wrds_identity(universe, *frames):
    require_point_in_time_universe(universe, context="WRDS research")
    for frame in (universe, *frames):
        if frame.empty or not {"date", "permno", "ticker"} <= set(frame):
            raise ValueError("WRDS research requires nonempty date/PERMNO/stable alias keys")
        ids = pd.to_numeric(frame.permno, errors="coerce")
        if ids.isna().any() or ids.mod(1).ne(0).any() or ids.le(0).any():
            raise ValueError("Invalid PERMNO identifier")
        aliases = "PERMNO:" + ids.astype("int64").astype(str)
        if not aliases.eq(frame.ticker).all():
            raise ValueError("WRDS ticker must be the stable PERMNO alias, never an exchange ticker")
        if pd.to_datetime(frame.date, errors="coerce").isna().any():
            raise ValueError("Invalid WRDS date key")
        if frame.duplicated(["date", "permno"]).any():
            raise ValueError("Duplicate WRDS date/PERMNO keys")
    allowed = pd.MultiIndex.from_frame(frames[-1][["date", "permno"]]) if frames else None
    if allowed is not None:
        keys = pd.MultiIndex.from_frame(universe[["date", "permno"]])
        if not keys.isin(allowed).all():
            raise ValueError("Universe contains keys absent from the price tape")


def require_financial_vintage(fundamentals, features, *, allow_restated=False):
    if not FUNDAMENTAL_FEATURES.intersection(features):
        return
    if fundamentals is None or fundamentals.empty:
        raise ValueError("Financial features require fundamental provenance")
    if not {"available_date", "is_point_in_time"} <= set(fundamentals):
        raise ValueError("Missing financial availability/vintage provenance")
    if pd.to_datetime(fundamentals.available_date, errors="coerce").isna().any():
        raise ValueError("Missing financial availability date")
    if not allow_restated and not fundamentals.is_point_in_time.eq(True).all():
        raise ValueError("Restated fundamentals require --allow-restated-fundamentals; universe PIT remains mandatory")
