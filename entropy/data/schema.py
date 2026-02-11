"""Parquet schema definitions for the three canonical tables.

Every table is keyed by (date, ticker).  Schemas are defined as
``pyarrow.Schema`` objects so we can enforce types at write-time and
catch drift early.
"""

from __future__ import annotations

import pyarrow as pa

# ===================================================================
# prices
# ===================================================================
# One row per (date, ticker).
# All price fields are UNADJUSTED (raw) plus a cumulative adjustment
# factor ``adj_factor`` so the user can choose when/how to adjust.

PRICES_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        # --- raw OHLCV ---
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("amount", pa.float64()),          # close * volume (turnover proxy)
        # --- adjustment ---
        pa.field("adj_factor", pa.float64()),       # cumulative split factor (point-in-time)
        pa.field("adj_close", pa.float64()),         # close * adj_factor (convenience)
        # --- trading status ---
        pa.field("is_tradable", pa.bool_()),         # False if halted / no trades
    ],
    metadata={
        b"table": b"prices",
        b"description": b"Daily OHLCV with point-in-time adjustment factor for US equities.",
        b"key": b"(date, ticker)",
    },
)


# ===================================================================
# universe
# ===================================================================
# One row per (date, ticker) that passes all filters on that date.

UNIVERSE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        # --- filter flags (for diagnostics) ---
        pa.field("days_since_ipo", pa.int32()),
        pa.field("market_cap", pa.float64()),        # USD, as-of date
        pa.field("close_price", pa.float64()),
        pa.field("in_index", pa.bool_()),             # member of target index
        pa.field("pass_all_filters", pa.bool_()),     # final tradable flag
    ],
    metadata={
        b"table": b"universe",
        b"description": b"Daily tradable universe after applying listing, price, and cap filters.",
        b"key": b"(date, ticker)",
    },
)


# ===================================================================
# fundamentals
# ===================================================================
# One row per (date, ticker) — *point-in-time* view.
# ``date`` is the TRADING date on which the data becomes "known"
# (report_date + publication_lag).

FUNDAMENTALS_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        # --- identifiers ---
        pa.field("report_date", pa.date32()),         # fiscal period end
        pa.field("publish_date", pa.date32()),        # SEC filing / press release
        # --- income statement ---
        pa.field("revenue", pa.float64()),
        pa.field("gross_profit", pa.float64()),
        pa.field("operating_income", pa.float64()),
        pa.field("net_income", pa.float64()),
        pa.field("eps_diluted", pa.float64()),
        # --- balance sheet ---
        pa.field("total_assets", pa.float64()),
        pa.field("total_liabilities", pa.float64()),
        pa.field("total_equity", pa.float64()),
        pa.field("cash_and_equivalents", pa.float64()),
        pa.field("total_debt", pa.float64()),
        # --- cash flow ---
        pa.field("cash_from_operations", pa.float64()),
        pa.field("capex", pa.float64()),
        pa.field("free_cash_flow", pa.float64()),
        # --- derived / market ---
        pa.field("market_cap", pa.float64()),
        pa.field("shares_outstanding", pa.float64()),
        pa.field("book_value_per_share", pa.float64()),
    ],
    metadata={
        b"table": b"fundamentals",
        b"description": (
            b"Point-in-time fundamental data (income, balance, cashflow) "
            b"with publication lag applied to avoid look-ahead bias."
        ),
        b"key": b"(date, ticker)",
    },
)


# ===================================================================
# Helpers
# ===================================================================

SCHEMA_MAP = {
    "prices": PRICES_SCHEMA,
    "universe": UNIVERSE_SCHEMA,
    "fundamentals": FUNDAMENTALS_SCHEMA,
}


def validate_dataframe(df, table_name: str) -> bool:
    """Check that *df* columns match the registered schema.

    Raises ``ValueError`` on mismatch.  Returns ``True`` on success.
    """
    schema = SCHEMA_MAP.get(table_name)
    if schema is None:
        raise KeyError(f"Unknown table name: {table_name!r}")
    expected_cols = set(schema.names)
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    msgs = []
    if missing:
        msgs.append(f"Missing columns: {sorted(missing)}")
    if extra:
        msgs.append(f"Extra columns: {sorted(extra)}")
    if msgs:
        raise ValueError(f"Schema validation failed for '{table_name}': " + "; ".join(msgs))
    return True
