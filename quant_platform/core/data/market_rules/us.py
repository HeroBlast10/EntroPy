"""US equity market rules and constraints."""

from __future__ import annotations

US_MARKET_RULES = {
    "exchange": "XNYS",
    "settlement_cycle": 2,  # T+2
    "lot_size": 1,
    "short_selling": True,
    "price_limits": None,  # no daily price limits
    "trading_hours": "09:30-16:00 ET",
    "currency": "USD",
}
