"""CN A-share market rules and constraints."""

from __future__ import annotations

CN_MARKET_RULES = {
    "exchange_sh": "XSHG",
    "exchange_sz": "XSHE",
    "settlement_cycle": 1,  # T+1
    "lot_size": 100,
    "short_selling": False,
    "price_limits": {
        "main_board": 0.10,       # 10% for SSE/SZSE main board
        "star_market": 0.20,      # 20% for STAR (688xxx) and ChiNext (300xxx/301xxx)
    },
    "stamp_tax_rate": 0.0005,     # sell-only, 5 bps (post-Aug 2023)
    "trading_hours": "09:30-11:30, 13:00-15:00 CST",
    "currency": "CNY",
}
