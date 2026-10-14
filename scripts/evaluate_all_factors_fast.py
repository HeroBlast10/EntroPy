#!/usr/bin/env python
"""Vectorized evaluation of an already-built factor panel.

This is the recovery/large-panel path for ``build_factors.py --evaluate``.
It avoids recomputing factors and avoids the chart-heavy per-factor tearsheet
loop while retaining the ranking metrics needed for factor selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.signals.evaluation.router import evaluate_signal
from quant_platform.core.utils.io import load_parquet, resolve_data_path, set_project_root


HORIZONS = (1, 5, 10, 20)


def _group_corr(codes: np.ndarray, x: np.ndarray, y: np.ndarray, n_groups: int) -> np.ndarray:
    """Numerically stable, two-pass grouped Pearson correlation."""

    valid = np.isfinite(x) & np.isfinite(y)
    c = codes[valid]
    xv = x[valid]
    yv = y[valid]
    n = np.bincount(c, minlength=n_groups).astype(float)
    sx = np.bincount(c, weights=xv, minlength=n_groups)
    sy = np.bincount(c, weights=yv, minlength=n_groups)
    mean_x = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
    mean_y = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
    x_centered = xv - mean_x[c]
    y_centered = yv - mean_y[c]
    var_x = np.bincount(c, weights=x_centered * x_centered, minlength=n_groups)
    var_y = np.bincount(c, weights=y_centered * y_centered, minlength=n_groups)
    covariance = np.bincount(c, weights=x_centered * y_centered, minlength=n_groups)
    denominator = np.sqrt(var_x * var_y)
    out = np.full(n_groups, np.nan)
    tolerance = np.finfo(float).eps * np.maximum(n, 1.0)
    good = (n >= 10) & (var_x > tolerance) & (var_y > tolerance)
    out[good] = covariance[good] / denominator[good]
    out[good] = np.clip(out[good], -1.0, 1.0)
    return out


def _summary(values: np.ndarray, *, hac_lags: int = 0) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"mean": np.nan, "std": np.nan, "icir": np.nan, "t": np.nan,
                "hit": np.nan, "n": float(len(values))}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    demeaned = values - mean
    long_run_var = float(np.dot(demeaned, demeaned) / len(values))
    for lag in range(1, min(hac_lags, len(values) - 1) + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / len(values))
        long_run_var += 2.0 * (1.0 - lag / (hac_lags + 1.0)) * covariance
    long_run_std = np.sqrt(max(long_run_var, 0.0))
    return {
        "mean": mean,
        "std": std,
        "icir": mean / long_run_std * np.sqrt(252.0) if long_run_std > 0 else np.nan,
        "t": mean / (long_run_std / np.sqrt(len(values))) if long_run_std > 0 else np.nan,
        "hit": float((values > 0).mean()),
        "n": float(len(values)),
    }


def _rank_by_date(values: pd.Series, dates: pd.Series) -> np.ndarray:
    return values.groupby(dates, observed=True).rank(pct=True).to_numpy(dtype=float)


def _daily_nonconstant_fraction(
    codes: np.ndarray,
    values: np.ndarray,
    n_groups: int,
    *,
    min_count: int = 10,
) -> float:
    valid = np.isfinite(values)
    c = codes[valid]
    x = values[valid]
    n = np.bincount(c, minlength=n_groups).astype(float)
    sx = np.bincount(c, weights=x, minlength=n_groups)
    mean = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
    centered = x - mean[c]
    variance_sum = np.bincount(c, weights=centered * centered, minlength=n_groups)
    eligible = n >= min_count
    if not eligible.any():
        return 0.0
    nonconstant = variance_sum > np.finfo(float).eps * np.maximum(n, 1.0)
    return float(nonconstant[eligible].mean())


def main() -> None:
    set_project_root(ROOT)
    reg = FactorRegistry()
    reg.discover()

    factor_path = resolve_data_path("factors", "factors.parquet")
    price_path = resolve_data_path("prices", "prices.parquet")
    factors = load_parquet(factor_path)
    prices = load_parquet(price_path, columns=["date", "ticker", "adj_close", "close", "volume", "amount"])
    factors["date"] = pd.to_datetime(factors["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    factors = factors.sort_values(["date", "ticker"]).reset_index(drop=True)
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    same_keys = (
        len(factors) == len(prices)
        and factors[["date", "ticker"]].equals(prices[["date", "ticker"]])
    )
    if not same_keys:
        factors = factors.merge(prices, on=["date", "ticker"], how="inner")
    else:
        for col in ("adj_close", "close", "volume", "amount"):
            factors[col] = prices[col].to_numpy()

    factors = factors.sort_values(["date", "ticker"]).reset_index(drop=True)
    unique_dates = pd.Index(factors["date"].drop_duplicates())
    date_codes = pd.Categorical(factors["date"], categories=unique_dates, ordered=True).codes
    n_dates = len(unique_dates)
    oos_date_mask = unique_dates >= pd.Timestamp("2022-01-01")
    years = unique_dates.year.to_numpy()
    price_lookup = factors.set_index(["date", "ticker"])["adj_close"]
    if not price_lookup.index.is_unique:
        raise ValueError("Price panel contains duplicate date/ticker keys")
    for horizon in HORIZONS:
        future_map = pd.Series(unique_dates, index=unique_dates).shift(-horizon)
        future_dates = factors["date"].map(future_map)
        future_index = pd.MultiIndex.from_arrays(
            [future_dates, factors["ticker"]],
            names=["date", "ticker"],
        )
        future_price = price_lookup.reindex(future_index).to_numpy(dtype=float)
        factors[f"fwd_ret_{horizon}d"] = future_price / factors["adj_close"].to_numpy() - 1.0

    rows: list[dict[str, object]] = []
    typed: dict[str, dict[str, object]] = {}
    factor_cols = [c for c in factors.columns if c in reg]

    for idx, name in enumerate(factor_cols, 1):
        meta = reg.get(name).meta
        print(f"[{idx:02d}/{len(factor_cols)}] {name} ({meta.signal_type})", flush=True)
        if meta.signal_type != "cross_sectional":
            result = evaluate_signal(
                signal_df=factors[["date", "ticker", name]],
                signal_col=name,
                meta=meta,
                prices=prices,
            )
            typed[name] = {
                "signal_type": meta.signal_type,
                **{k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in result.items()},
            }
            continue

        x = factors[name].to_numpy(dtype=float) * float(meta.direction)
        x_rank = _rank_by_date(pd.Series(x), factors["date"])
        row: dict[str, object] = {
            "factor": name,
            "category": meta.category,
            "direction": meta.direction,
            "coverage_rows": int(np.isfinite(x).sum()),
            "coverage_pct": float(np.isfinite(x).mean()),
        }
        row["daily_nonconstant_fraction"] = _daily_nonconstant_fraction(
            date_codes, x, n_dates
        )
        if row["daily_nonconstant_fraction"] < 0.05:
            row["rejected_reason"] = "cross_sectionally_constant"
            rows.append(row)
            continue

        y1 = factors["fwd_ret_1d"].to_numpy(dtype=float)
        ic = _group_corr(date_codes, x, y1, n_dates)
        ic_s = _summary(ic, hac_lags=1)
        for key, value in ic_s.items():
            row[f"ic_{key}"] = value

        for horizon in HORIZONS:
            y = factors[f"fwd_ret_{horizon}d"].to_numpy(dtype=float)
            y_rank = _rank_by_date(pd.Series(y), factors["date"])
            ric = _group_corr(date_codes, x_rank, y_rank, n_dates)
            ric_s = _summary(ric, hac_lags=max(1, horizon - 1))
            for key, value in ric_s.items():
                row[f"ric_{horizon}d_{key}"] = value
            if horizon == 1:
                oos_s = _summary(ric[oos_date_mask], hac_lags=1)
                row["ric_oos_2022_mean"] = oos_s["mean"]
                row["ric_oos_2022_t"] = oos_s["t"]
                yearly = pd.Series(ric, index=unique_dates).groupby(years).mean()
                row["positive_year_fraction"] = float((yearly > 0).mean())
                row["yearly_ric_min"] = float(yearly.min())
                row["yearly_ric_max"] = float(yearly.max())

        valid = np.isfinite(x_rank) & np.isfinite(y1)
        top = valid & (x_rank >= 0.8)
        bottom = valid & (x_rank <= 0.2)
        top_sum = np.bincount(date_codes[top], weights=y1[top], minlength=n_dates)
        top_n = np.bincount(date_codes[top], minlength=n_dates)
        bot_sum = np.bincount(date_codes[bottom], weights=y1[bottom], minlength=n_dates)
        bot_n = np.bincount(date_codes[bottom], minlength=n_dates)
        spread = np.full(n_dates, np.nan)
        good = (top_n > 0) & (bot_n > 0)
        spread[good] = top_sum[good] / top_n[good] - bot_sum[good] / bot_n[good]
        spread_valid = spread[np.isfinite(spread)]
        row["ls_daily_mean"] = float(spread_valid.mean()) if len(spread_valid) else np.nan
        row["ls_ann_return"] = float(spread_valid.mean() * 252) if len(spread_valid) else np.nan
        row["ls_gross_sharpe"] = (
            float(spread_valid.mean() / spread_valid.std(ddof=1) * np.sqrt(252))
            if len(spread_valid) > 1 and spread_valid.std(ddof=1) > 0 else np.nan
        )

        turnover_frame = factors[["date", "ticker"]].copy()
        turnover_frame["rank"] = x_rank
        turnover_frame = turnover_frame.sort_values(["ticker", "date"])
        changes = turnover_frame.groupby("ticker", sort=False)["rank"].diff().abs()
        row["mean_rank_turnover"] = float(changes.mean())
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("factor")
    comparison["robust_score"] = (
        comparison["ric_1d_t"].clip(-5, 5)
        + comparison["ric_oos_2022_t"].clip(-5, 5)
        + 2.0 * (comparison["positive_year_fraction"] - 0.5)
        + comparison["ls_gross_sharpe"].clip(-3, 3)
        - comparison["mean_rank_turnover"].fillna(1.0)
    )
    comparison = comparison.sort_values("robust_score", ascending=False)

    out_path = resolve_data_path("factors", "factor_evaluation_20260721.csv")
    comparison.to_csv(out_path)
    typed_path = resolve_data_path("factors", "typed_factor_evaluation_20260721.json")
    typed_path.write_text(json.dumps(typed, indent=2), encoding="utf-8")
    print("\nTop factors:\n", comparison.head(15).to_string(), flush=True)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {typed_path}")


if __name__ == "__main__":
    main()
