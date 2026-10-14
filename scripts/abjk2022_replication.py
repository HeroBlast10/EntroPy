"""Free-data reproduction of Akbas, Boehmer, Jiang & Koch (2022).

The module keeps the notebook short.  It snapshots current index constituents,
downloads split/dividend-adjusted daily OHLCV from Yahoo Finance, constructs the
paper's AB_NR signal, and provides small helpers for inference and trading-cost
analysis.  It deliberately does not pretend that current-constituent Yahoo data
are CRSP: the notebook reports the resulting survivorship and data-source gaps.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf


WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ABJK-academic-replication/1.0)"}
UNIVERSE_SOURCES = {
    "US": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "CN": "https://en.wikipedia.org/wiki/CSI_300_Index",
    "HK": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
}


def _read_html(url: str) -> list[pd.DataFrame]:
    response = requests.get(url, headers=WIKI_HEADERS, timeout=45)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def snapshot_universe(path: Path, refresh: bool = False) -> pd.DataFrame:
    """Snapshot current benchmark constituents and apply paper-like sector exclusions."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_csv(path)

    us = _read_html(UNIVERSE_SOURCES["US"])[0].copy()
    us = us.loc[~us["GICS Sector"].isin(["Financials", "Utilities"])]
    us_out = pd.DataFrame(
        {
            "market": "US",
            "ticker": us["Symbol"].str.replace(".", "-", regex=False),
            "name": us["Security"],
            "sector": us["GICS Sector"],
            "index": "S&P 500",
        }
    )

    cn = _read_html(UNIVERSE_SOURCES["CN"])[3].copy()
    cn = cn.loc[~cn["Segment"].isin(["Financials", "Utilities"])]
    cn_code = cn["Ticker"].astype(str).str.extract(r"(\d{6})", expand=False)
    cn_suffix = np.where(cn["Exchange"].eq("Shanghai"), ".SS", ".SZ")
    cn_out = pd.DataFrame(
        {
            "market": "CN",
            "ticker": cn_code + cn_suffix,
            "name": cn["Company"],
            "sector": cn["Segment"],
            "index": "CSI 300",
        }
    )

    hk_tables = _read_html(UNIVERSE_SOURCES["HK"])
    hk = next(t for t in hk_tables if set(["Ticker", "Name", "Sub-index"]).issubset(t.columns)).copy()
    hk = hk.loc[~hk["Sub-index"].isin(["Finance", "Utilities"])]
    hk_code = hk["Ticker"].astype(str).str.extract(r"(\d+)$", expand=False).str.zfill(4)
    hk_out = pd.DataFrame(
        {
            "market": "HK",
            "ticker": hk_code + ".HK",
            "name": hk["Name"],
            "sector": hk["Sub-index"],
            "index": "Hang Seng",
        }
    )

    universe = pd.concat([us_out, cn_out, hk_out], ignore_index=True)
    universe = universe.dropna(subset=["ticker"]).drop_duplicates(["market", "ticker"])
    universe["source_url"] = universe["market"].map(UNIVERSE_SOURCES)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(path, index=False, encoding="utf-8-sig")
    return universe


def _parse_yf_batch(raw: pd.DataFrame, tickers: list[str], market: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if raw.empty:
        return pd.DataFrame()
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                level1 = set(raw.columns.get_level_values(1))
                if ticker in level0:
                    one = raw[ticker].copy()
                elif ticker in level1:
                    one = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    continue
            else:
                if len(tickers) != 1:
                    continue
                one = raw.copy()
            one.columns = [str(c).lower().replace(" ", "_") for c in one.columns]
            required = ["open", "high", "low", "close", "volume"]
            if not set(required).issubset(one.columns):
                continue
            one = one[required].dropna(subset=["open", "close"], how="all")
            if one.empty:
                continue
            one = one.reset_index().rename(columns={one.index.name or "index": "date", "Date": "date"})
            if "date" not in one.columns:
                one = one.rename(columns={one.columns[0]: "date"})
            one["date"] = pd.to_datetime(one["date"]).dt.tz_localize(None)
            one["ticker"] = ticker
            one["market"] = market
            parts.append(one[["date", "market", "ticker", *required]])
        except (KeyError, ValueError):
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _download_batch(tickers: list[str], market: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        back_adjust=False,
        repair=True,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=45,
        multi_level_index=True,
    )
    return _parse_yf_batch(raw, tickers, market)


def download_market_data(
    universe: pd.DataFrame,
    out_dir: Path,
    start: str,
    end: str,
    refresh: bool = False,
    batch_size: int = 60,
) -> pd.DataFrame:
    """Download adjusted daily OHLCV with retry and a per-market parquet cache."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    market_frames: list[pd.DataFrame] = []
    for market, market_universe in universe.groupby("market", sort=True):
        cache = out_dir / f"daily_{market}_{start}_{pd.Timestamp(end) - pd.Timedelta(days=1):%Y-%m-%d}.parquet"
        if cache.exists() and not refresh:
            market_frames.append(pd.read_parquet(cache))
            continue

        tickers = market_universe["ticker"].astype(str).tolist()
        batches: list[pd.DataFrame] = []
        for idx in range(0, len(tickers), batch_size):
            batch = tickers[idx : idx + batch_size]
            for attempt in range(3):
                try:
                    got = _download_batch(batch, market, start, end)
                    if not got.empty:
                        batches.append(got)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)

        panel = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        received = set(panel["ticker"].unique()) if not panel.empty else set()
        missing = [t for t in tickers if t not in received]
        retry_parts: list[pd.DataFrame] = []
        for ticker in missing:
            try:
                one = _download_batch([ticker], market, start, end)
                if not one.empty:
                    retry_parts.append(one)
            except Exception:
                pass
        if retry_parts:
            panel = pd.concat([panel, *retry_parts], ignore_index=True)
        panel = panel.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
        panel.to_parquet(cache, index=False)
        market_frames.append(panel)

    all_data = pd.concat(market_frames, ignore_index=True)
    return all_data.sort_values(["market", "ticker", "date"]).reset_index(drop=True)


def data_quality_report(raw: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, u in universe.groupby("market"):
        x = raw.loc[raw["market"].eq(market)]
        rows.append(
            {
                "market": market,
                "requested": u["ticker"].nunique(),
                "received": x["ticker"].nunique(),
                "coverage": x["ticker"].nunique() / u["ticker"].nunique(),
                "rows": len(x),
                "first_date": x["date"].min(),
                "latest_date": x["date"].max(),
                "open_missing_pct": 100 * x["open"].isna().mean(),
                "close_missing_pct": 100 * x["close"].isna().mean(),
                "zero_volume_pct": 100 * x["volume"].fillna(0).eq(0).mean(),
                "duplicate_keys": int(x.duplicated(["ticker", "date"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("market").reset_index(drop=True)


def construct_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct close-to-open, open-to-close, and reversal indicators."""
    x = raw.copy().sort_values(["ticker", "date"])
    positive_prices = x[["open", "close"]].gt(0).all(axis=1)
    x = x.loc[positive_prices].copy()
    group = x.groupby("ticker", sort=False)
    x["prev_close"] = group["close"].shift(1)
    x["overnight"] = x["open"] / x["prev_close"] - 1.0
    x["daytime"] = x["close"] / x["open"] - 1.0
    x["close_to_close"] = x["close"] / x["prev_close"] - 1.0
    x["identity_error"] = (1 + x["overnight"]) * (1 + x["daytime"]) - 1 - x["close_to_close"]
    x["nr_day"] = (x["overnight"] > 0) & (x["daytime"] < 0)
    x["pr_day"] = (x["overnight"] < 0) & (x["daytime"] > 0)
    x["next_overnight"] = group["overnight"].shift(-1)
    next_date = group["date"].shift(-1)
    same_month_next_session = next_date.dt.to_period("M").eq(x["date"].dt.to_period("M"))
    x["dopr_day"] = (x["daytime"] < 0) & (x["next_overnight"] > 0) & same_month_next_session
    x["dollar_volume"] = x["close"] * x["volume"]
    x["month"] = x["date"].dt.to_period("M")
    return x


def _compound(values: pd.Series) -> float:
    values = values.dropna()
    return float(np.prod(1.0 + values) - 1.0) if len(values) else np.nan


def construct_monthly(daily: pd.DataFrame, min_days: int = 15) -> pd.DataFrame:
    """Aggregate daily reversals, construct lagged-normalized signals and next-month returns."""
    valid = daily.loc[daily["close_to_close"].notna()].copy()
    monthly = (
        valid.groupby(["market", "ticker", "month"], observed=True)
        .agg(
            trading_days=("date", "size"),
            nr=("nr_day", "mean"),
            pr=("pr_day", "mean"),
            dopr=("dopr_day", "mean"),
            ret=("close_to_close", _compound),
            overnight_ret=("overnight", _compound),
            daytime_ret=("daytime", _compound),
            volatility=("close_to_close", "std"),
            adv=("dollar_volume", "mean"),
            month_end_price=("close", "last"),
        )
        .reset_index()
        .sort_values(["ticker", "month"])
    )
    illiq = (
        valid.assign(illiq_day=valid["close_to_close"].abs() / valid["dollar_volume"].replace(0, np.nan))
        .groupby(["market", "ticker", "month"], observed=True)["illiq_day"]
        .mean()
        .mul(1e6)
        .rename("amihud_million")
        .reset_index()
    )
    monthly = monthly.merge(illiq, on=["market", "ticker", "month"], how="left")
    monthly.loc[monthly["trading_days"] < min_days, ["nr", "pr", "dopr"]] = np.nan

    def add_lags(one: pd.DataFrame) -> pd.DataFrame:
        one = one.sort_values("month").copy()
        for raw_name, abnormal_name in [("nr", "ab_nr"), ("pr", "ab_pr"), ("dopr", "ab_dopr")]:
            prior_mean = one[raw_name].shift(1).rolling(12, min_periods=10).mean()
            one[abnormal_name] = one[raw_name] / prior_mean.replace(0, np.nan)
        one["mom_6m"] = one["ret"].shift(1).rolling(6, min_periods=6).apply(_compound, raw=False)
        one["next_month"] = one["month"].shift(-1)
        consecutive = one["next_month"].eq(one["month"] + 1)
        for col in ["ret", "overnight_ret", "daytime_ret"]:
            one[f"next_{col}"] = one[col].shift(-1).where(consecutive)
        return one

    monthly = pd.concat(
        [add_lags(one) for _, one in monthly.groupby(["market", "ticker"], sort=False)],
        ignore_index=True,
    )
    return monthly.sort_values(["market", "month", "ticker"]).reset_index(drop=True)


def eligible_monthly(monthly: pd.DataFrame, last_complete_month: str | pd.Period) -> pd.DataFrame:
    cutoff = pd.Period(last_complete_month, freq="M")
    mask = (
        monthly["month"].le(cutoff - 1)
        & monthly["ab_nr"].notna()
        & monthly["next_ret"].notna()
        & monthly["month_end_price"].gt(1.0)
    )
    return monthly.loc[mask].copy()


def assign_groups(panel: pd.DataFrame, signal: str = "ab_nr", n_groups: int = 5) -> pd.DataFrame:
    """Assign robust within-market/month quantiles; rank-first handles discrete signal ties."""
    x = panel.loc[panel[signal].notna()].copy()

    def one_group(s: pd.Series) -> pd.Series:
        if s.notna().sum() < n_groups * 5:
            return pd.Series(np.nan, index=s.index)
        pct = s.rank(method="first", pct=True)
        return np.ceil(pct * n_groups).clip(1, n_groups)

    x["group"] = x.groupby(["market", "month"], observed=True)[signal].transform(one_group)
    return x.loc[x["group"].notna()].assign(group=lambda z: z["group"].astype(int))


def portfolio_returns(grouped: pd.DataFrame, n_groups: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = ["next_ret", "next_overnight_ret", "next_daytime_ret"]
    by_group = (
        grouped.groupby(["market", "month", "group"], observed=True)[fields]
        .mean()
        .reset_index()
    )
    wide = by_group.pivot(index=["market", "month"], columns="group", values=fields)
    spreads = pd.DataFrame(index=wide.index)
    for field in fields:
        spreads[field] = wide[(field, n_groups)] - wide[(field, 1)]
    spreads = spreads.reset_index()
    return by_group, spreads


def nw_mean_stats(values: pd.Series, maxlags: int = 12) -> dict[str, float]:
    y = pd.Series(values).dropna().astype(float)
    if len(y) < 12:
        return {"mean_pct": np.nan, "t_hac": np.nan, "ann_sharpe": np.nan, "months": len(y), "win_pct": np.nan}
    fit = sm.OLS(y.to_numpy(), np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "mean_pct": 100 * y.mean(),
        "t_hac": float(fit.tvalues[0]),
        "ann_sharpe": math.sqrt(12) * y.mean() / y.std(ddof=1),
        "months": len(y),
        "win_pct": 100 * y.gt(0).mean(),
    }


def spread_summary(spreads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, x in spreads.groupby("market"):
        for component in ["next_ret", "next_overnight_ret", "next_daytime_ret"]:
            rows.append({"market": market, "component": component, **nw_mean_stats(x[component])})
    return pd.DataFrame(rows)


def group_return_summary(by_group: pd.DataFrame) -> pd.DataFrame:
    out = (
        by_group.groupby(["market", "group"], observed=True)["next_ret"]
        .agg(mean="mean", median="median", months="count")
        .reset_index()
    )
    out[["mean", "median"]] *= 100
    return out


def _winsor_zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 10:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile([0.01, 0.99])
    s = s.clip(lo, hi)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else pd.Series(np.nan, index=s.index)


def fama_macbeth(
    panel: pd.DataFrame,
    predictors: Iterable[str],
    outcome: str = "next_ret",
    maxlags: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = list(predictors)
    work = panel.copy()
    work["log_adv"] = np.log(work["adv"].where(work["adv"] > 0))
    monthly_betas = []
    for (market, month), x in work.groupby(["market", "month"], observed=True):
        cols = [outcome, *predictors]
        one = x[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(one) < len(predictors) + 15:
            continue
        for col in predictors:
            one[col] = _winsor_zscore(one[col])
        one = one.dropna()
        if len(one) < len(predictors) + 15:
            continue
        fit = sm.OLS(100 * one[outcome], sm.add_constant(one[predictors]), missing="drop").fit()
        monthly_betas.append({"market": market, "month": month, **fit.params.to_dict()})
    betas = pd.DataFrame(monthly_betas)
    rows = []
    if not betas.empty:
        for market, x in betas.groupby("market"):
            for predictor in predictors:
                stats = nw_mean_stats(x[predictor] / 100, maxlags=maxlags)
                rows.append(
                    {
                        "market": market,
                        "predictor": predictor,
                        "coef_pct_per_1sd": stats["mean_pct"],
                        "t_hac": stats["t_hac"],
                        "months": stats["months"],
                    }
                )
    return pd.DataFrame(rows), betas


def strategy_turnover(grouped: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """Equal-weight one-way turnover for both legs, charged at each monthly rebalance."""
    rows = []
    for market, market_data in grouped.groupby("market"):
        previous: dict[int, dict[str, float]] = {1: {}, n_groups: {}}
        for month, x in market_data.sort_values("month").groupby("month", observed=True):
            record = {"market": market, "month": month}
            for leg, label in [(1, "short"), (n_groups, "long")]:
                members = x.loc[x["group"].eq(leg), "ticker"].tolist()
                current = {ticker: 1.0 / len(members) for ticker in members} if members else {}
                names = set(previous[leg]) | set(current)
                turnover = 0.5 * sum(abs(current.get(name, 0.0) - previous[leg].get(name, 0.0)) for name in names)
                record[f"{label}_turnover"] = turnover if previous[leg] else np.nan
                previous[leg] = current
            record["two_leg_turnover"] = record["long_turnover"] + record["short_turnover"]
            rows.append(record)
    return pd.DataFrame(rows)


def cost_scenarios(spreads: pd.DataFrame, turnover: pd.DataFrame, costs_bps=(0, 10, 25, 50)) -> pd.DataFrame:
    merged = spreads.merge(turnover, on=["market", "month"], how="left")
    rows = []
    for market, x in merged.groupby("market"):
        for bps in costs_bps:
            net = x["next_ret"] - (bps / 10_000.0) * x["two_leg_turnover"]
            rows.append(
                {
                    "market": market,
                    "roundtrip_cost_bps": bps,
                    "avg_two_leg_turnover": x["two_leg_turnover"].mean(),
                    **nw_mean_stats(net),
                }
            )
    return pd.DataFrame(rows)


def manual_checks(daily: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        one = daily.loc[daily["ticker"].eq(ticker) & daily["prev_close"].notna()].sort_values("date")
        if one.empty:
            continue
        r = one.iloc[-1]
        rows.append(
            {
                "market": r["market"],
                "ticker": ticker,
                "date": r["date"],
                "prev_adj_close": r["prev_close"],
                "adj_open": r["open"],
                "adj_close": r["close"],
                "overnight=open/prev_close-1": r["overnight"],
                "daytime=close/open-1": r["daytime"],
                "close_to_close": r["close_to_close"],
                "recombined": (1 + r["overnight"]) * (1 + r["daytime"]) - 1,
                "identity_error": r["identity_error"],
                "NR_flag": bool(r["nr_day"]),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(path: Path, raw: pd.DataFrame, universe_path: Path, config: dict) -> dict:
    path = Path(path)
    digest = hashlib.sha256(Path(universe_path).read_bytes()).hexdigest()
    manifest = {
        **config,
        "yfinance_version": yf.__version__,
        "universe_sha256": digest,
        "rows": int(len(raw)),
        "tickers": int(raw["ticker"].nunique()),
        "latest_dates": {k: str(v.date()) for k, v in raw.groupby("market")["date"].max().items()},
        "source_notes": "Yahoo Finance, auto_adjust=True; current benchmark constituent snapshot; no paid data.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
