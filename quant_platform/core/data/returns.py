"""Calendar-aligned total returns and terminal cash settlement.

Returns dated t cover (t-1,t]. CIZ total_return already includes distributions
and delisting; it must not be combined with DLRET again. Ordinary missing
observations remain unknown. After a terminal return, proceeds earn zero in
cash, matching the daily execution engines' implicit cash convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def terminal_return_value(policy: str) -> float:
    if policy not in {"raise", "total_loss", "zero_return"}:
        raise ValueError("missing_terminal_return must be raise, total_loss, or zero_return")
    if policy == "raise":
        raise ValueError("Missing terminal return; choose an explicit research sensitivity policy")
    return -1.0 if policy == "total_loss" else 0.0


def daily_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Prefer direct returns; price fallback never pads or bridges a session."""
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/security return keys")
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"])
    col = "total_return" if "total_return" in work else (
        "adj_close" if "adj_close" in work else "close")
    values = work.pivot(index="date", columns="ticker", values=col).sort_index().astype("float64")
    ret = values if col == "total_return" else values.where(values.gt(0)).pct_change(fill_method=None)
    if (ret.lt(-1) | np.isinf(ret)).any().any():
        raise ValueError("Invalid total return")
    # Calendar days in CRSP duration flags include weekends (e.g. D3 on
    # Monday). They are not trading-session counts and must not filter D3.
    return ret


def _aligned(values: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(prices["date"]), prices["ticker"]])
    return pd.Series(values.stack(future_stack=True).reindex(keys).to_numpy(), index=prices.index)


def trailing_total_return(prices: pd.DataFrame, periods: int = 1) -> pd.Series:
    ret = daily_return_matrix(prices)
    if periods == 1:
        return _aligned(ret, prices)
    gross = 1 + ret
    log = np.log(gross.where(gross.gt(0), 1))
    compounded = np.expm1(log.rolling(periods, min_periods=periods).sum())
    compounded = compounded.mask(gross.eq(0).rolling(periods).sum().gt(0), -1)
    compounded = compounded.where(ret.notna().rolling(periods).sum().eq(periods))
    return _aligned(compounded, prices)


def forward_total_returns(prices: pd.DataFrame, horizon: int, *,
                          missing_terminal_return: str = "raise") -> pd.DataFrame:
    """Hold h calendar sessions, settle a terminal payoff once, then cash.

    A quote gap before settlement invalidates the label, even if an endpoint
    price exists. No horizon beyond the supplied tape is treated as observed.
    label_end remains the contractual horizon, also for early settlements.
    """
    if horizon < 1 or missing_terminal_return not in {"raise", "total_loss", "zero_return"}:
        raise ValueError("Invalid horizon or missing_terminal_return policy")
    ret = daily_return_matrix(prices)
    terminal = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    if "is_terminal" in prices:
        terminal = prices.pivot(index="date", columns="ticker", values="is_terminal").reindex_like(ret).eq(True).fillna(False).astype(bool)
    gross = np.ones(ret.shape)
    alive = np.ones(ret.shape, dtype=bool)
    valid = np.ones(ret.shape, dtype=bool)
    settled = np.zeros(ret.shape, dtype=bool)
    imputed = np.zeros(ret.shape, dtype=bool)
    for step in range(1, horizon + 1):
        r = ret.shift(-step).to_numpy()
        term = terminal.shift(-step, fill_value=False).to_numpy()
        missing = alive & term & np.isnan(r)
        if missing.any():
            if missing_terminal_return == "raise":
                terminal_return_value(missing_terminal_return)
            r[missing] = terminal_return_value(missing_terminal_return)
            imputed |= missing
        valid &= ~alive | np.isfinite(r)
        gross *= np.where(alive & np.isfinite(r), 1 + r, 1)
        settled |= alive & term
        alive &= ~term
    end = pd.Series(ret.index, index=ret.index).shift(-horizon)
    result = pd.DataFrame(gross - 1, index=ret.index, columns=ret.columns)
    result = result.where(valid).where(end.notna(), axis=0)
    out = prices[["date", "ticker"]].copy()
    out["forward_return"] = _aligned(result, prices)
    price_col = "close" if "close" in prices else "adj_close"
    if price_col in prices:
        out["forward_return"] = out.forward_return.where(prices[price_col].gt(0))
    if "is_terminal" in prices:
        out["forward_return"] = out.forward_return.where(~prices.is_terminal.eq(True))
    out["label_end"] = pd.to_datetime(out["date"]).map(end)
    out["label_terminal_settled"] = _aligned(pd.DataFrame(settled, index=ret.index, columns=ret.columns), prices)
    out["label_terminal_imputed"] = _aligned(pd.DataFrame(imputed, index=ret.index, columns=ret.columns), prices)
    return out


def overnight_total_return(prices: pd.DataFrame) -> pd.Series:
    """Distribution-inclusive overnight return, with split-neutral units.

    Factor total daily gross = overnight gross * (close/open). With direct
    CIZ returns this defines overnight=(1+RET)*open/close-1. Distributions are
    attributed to overnight; intraday is the raw open-to-close price return.
    Legacy data must supply an explicit adjustment factor. Never assume 1.
    """
    if "total_return" in prices:
        daily = trailing_total_return(prices)
        result = (1 + daily) * prices["open"] / prices["close"] - 1
    elif "adj_factor" in prices:
        adjusted = prices[["date", "ticker"]].copy()
        adjusted["adj_close"] = prices["close"] * prices["adj_factor"]
        daily = trailing_total_return(adjusted)
        result = (1 + daily) * prices["open"] / prices["close"] - 1
    else:
        raise ValueError("Overnight returns require total_return or explicit adj_factor")
    result = result.where(prices["open"].gt(0) & prices["close"].gt(0))
    if "is_tradable" in prices:
        result = result.where(prices["is_tradable"].eq(True))
    return result.replace([np.inf, -np.inf], np.nan)
