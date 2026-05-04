"""Production-oriented multi-factor alpha combiners.

This module upgrades the alpha layer from a static z-score average to
trainable, regime-aware factor combination methods:

* rolling ICIR weighting
* factor-return covariance mean-variance weighting
* factor-return risk parity
* baseline-orthogonal incremental alpha weighting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.cross_sectional.evaluation import compute_rank_ic_series
from quant_platform.core.signals.effective import build_effective_signal
from quant_platform.core.signals.redundancy import factor_long_short_return_panel


@dataclass(frozen=True)
class RegimePolicy:
    """How regime signals modulate factor usage and portfolio controls."""

    regime_col: Optional[str] = "HMM_TURBULENCE_PROB"
    threshold: float = 0.60
    reduction_factor: float = 0.50
    high_regime_net_exposure: float = 0.50
    low_regime_net_exposure: float = 1.00
    high_regime_rebalance_threshold: float = 0.05
    low_regime_rebalance_threshold: float = 0.02
    disabled_categories_high: tuple[str, ...] = field(default_factory=tuple)
    category_multipliers_high: Dict[str, float] = field(default_factory=dict)
    category_multipliers_low: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiFactorConfig:
    """Configuration for factor combination."""

    method: str = "rolling_icir"
    lookback: int = 126
    min_periods: int = 40
    l2_reg: float = 1e-4
    long_only_factor_weights: bool = True
    baseline_factors: tuple[str, ...] = field(default_factory=tuple)
    output_col: str = "alpha_multi"
    regime_policy: Optional[RegimePolicy] = None


class MultiFactorCombiner:
    """Build one effective alpha column from several oriented factor columns."""

    def __init__(
        self,
        config: Optional[MultiFactorConfig] = None,
        *,
        direction_map: Optional[Dict[str, int]] = None,
        category_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config or MultiFactorConfig()
        self.direction_map = direction_map or {}
        self.category_map = category_map or {}
        self.factor_weights_: pd.DataFrame = pd.DataFrame()
        self.regime_controls_: pd.DataFrame = pd.DataFrame()
        self.effective_columns_: list[str] = []

    def fit_transform(
        self,
        factor_df: pd.DataFrame,
        factor_cols: Iterable[str],
        *,
        return_col: str = "fwd_ret_1d",
        neutralize_by: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return factor_df with ``config.output_col`` added."""
        factors = [c for c in factor_cols if c in factor_df.columns]
        if not factors:
            raise ValueError("No requested factor columns are present in factor_df")

        df = factor_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        effective = self._build_effective_columns(df, factors, neutralize_by=neutralize_by)

        method = self.config.method.lower()
        if method == "rolling_icir":
            weights = self._rolling_icir_weights(effective, factors, return_col)
            signal_frame = effective
        elif method == "mean_variance":
            weights = self._mean_variance_weights(effective, factors, return_col)
            signal_frame = effective
        elif method == "risk_parity":
            weights = self._risk_parity_weights(effective, factors, return_col)
            signal_frame = effective
        elif method in {"orthogonal_incremental", "baseline_orthogonal"}:
            signal_frame = self._orthogonalize_incremental_signals(effective, factors)
            weights = self._rolling_icir_weights(signal_frame, factors, return_col)
        else:
            raise ValueError(
                f"Unknown multi-factor method {self.config.method!r}. "
                "Use rolling_icir, mean_variance, risk_parity, or orthogonal_incremental."
            )

        weights = self._apply_regime_policy(df, factors, weights)
        self.factor_weights_ = weights

        out = df.copy()
        aligned = signal_frame.set_index(["date", "ticker"])
        out_index = pd.MultiIndex.from_frame(out[["date", "ticker"]])
        for col in self.effective_columns_:
            out[col] = aligned[col].reindex(out_index).values if col in aligned.columns else np.nan

        weight_lookup = weights.set_index("date")
        alpha_values = []
        eff_cols = [self._eff_col(f) for f in factors]
        for dt, group in out.groupby("date", sort=False):
            if dt in weight_lookup.index:
                w = weight_lookup.loc[dt, factors].astype(float)
            else:
                w = pd.Series(1.0 / len(factors), index=factors)
            x = group[eff_cols].to_numpy(dtype=float)
            alpha_values.append(pd.Series(np.nan_to_num(x, nan=0.0) @ w.values, index=group.index))

        alpha = pd.concat(alpha_values).sort_index() if alpha_values else pd.Series(dtype=float)
        out[self.config.output_col] = alpha.reindex(out.index)

        # Scaling alpha does not change quantile selection, but it matters for
        # optimizers and records the intended net-risk stance for downstream use.
        if not self.regime_controls_.empty:
            controls = self.regime_controls_.set_index("date")
            scalar = out["date"].map(controls["regime_scalar"]).fillna(1.0)
            out[self.config.output_col] = out[self.config.output_col] * scalar
            out["regime_net_exposure"] = out["date"].map(controls["net_exposure"]).fillna(1.0)
            out["rebalance_threshold"] = out["date"].map(controls["rebalance_threshold"]).fillna(0.0)

        return out

    def _build_effective_columns(
        self,
        df: pd.DataFrame,
        factors: list[str],
        *,
        neutralize_by: Optional[Iterable[str]],
    ) -> pd.DataFrame:
        effective = df.copy()
        self.effective_columns_ = []
        for factor in factors:
            eff_col = self._eff_col(factor)
            transformed = build_effective_signal(
                effective,
                factor,
                output_col=eff_col,
                direction=self.direction_map.get(factor, 1),
                neutralize_by=neutralize_by,
                rank=True,
            )
            eff_index = pd.MultiIndex.from_frame(transformed[["date", "ticker"]])
            base_index = pd.MultiIndex.from_frame(effective[["date", "ticker"]])
            effective[eff_col] = pd.Series(
                transformed[eff_col].values,
                index=eff_index,
            ).reindex(base_index).values
            self.effective_columns_.append(eff_col)
        return effective

    def _rolling_icir_weights(
        self,
        effective: pd.DataFrame,
        factors: list[str],
        return_col: str,
    ) -> pd.DataFrame:
        dates = pd.DatetimeIndex(sorted(effective["date"].dropna().unique()))
        if return_col not in effective.columns:
            return self._equal_weights(dates, factors)

        ic_panel = {}
        for factor in factors:
            eff_col = self._eff_col(factor)
            ic_panel[factor] = compute_rank_ic_series(effective, eff_col, return_col=return_col)
        ic = pd.concat(ic_panel, axis=1).reindex(dates)
        mean = ic.rolling(self.config.lookback, min_periods=self.config.min_periods).mean().shift(1)
        std = ic.rolling(self.config.lookback, min_periods=self.config.min_periods).std().shift(1)
        raw = mean.divide(std.replace(0, np.nan))
        return self._normalise_weight_frame(raw, factors, fallback_dates=dates)

    def _mean_variance_weights(
        self,
        effective: pd.DataFrame,
        factors: list[str],
        return_col: str,
    ) -> pd.DataFrame:
        returns = factor_long_short_return_panel(
            effective,
            [self._eff_col(f) for f in factors],
            return_col=return_col,
            direction_map={self._eff_col(f): 1 for f in factors},
        )
        returns = returns.rename(columns={self._eff_col(f): f for f in factors})
        dates = pd.DatetimeIndex(sorted(effective["date"].dropna().unique()))
        if returns.empty:
            return self._equal_weights(dates, factors)

        rows = []
        for dt in dates:
            hist = returns.loc[returns.index < dt].tail(self.config.lookback).dropna(how="all")
            if len(hist) < self.config.min_periods:
                rows.append(pd.Series(1.0 / len(factors), index=factors, name=dt))
                continue
            mu = hist.mean().reindex(factors).fillna(0.0).to_numpy(dtype=float)
            cov = hist.cov().reindex(index=factors, columns=factors).fillna(0.0).to_numpy(dtype=float)
            cov = cov + self.config.l2_reg * np.eye(len(factors))
            try:
                raw = np.linalg.solve(cov, mu)
            except np.linalg.LinAlgError:
                raw = np.linalg.pinv(cov) @ mu
            if self.config.long_only_factor_weights:
                raw = np.clip(raw, 0.0, None)
            rows.append(pd.Series(raw, index=factors, name=dt))
        return self._normalise_weight_frame(pd.DataFrame(rows), factors, fallback_dates=dates)

    def _risk_parity_weights(
        self,
        effective: pd.DataFrame,
        factors: list[str],
        return_col: str,
    ) -> pd.DataFrame:
        returns = factor_long_short_return_panel(
            effective,
            [self._eff_col(f) for f in factors],
            return_col=return_col,
            direction_map={self._eff_col(f): 1 for f in factors},
        )
        returns = returns.rename(columns={self._eff_col(f): f for f in factors})
        dates = pd.DatetimeIndex(sorted(effective["date"].dropna().unique()))
        if returns.empty:
            return self._equal_weights(dates, factors)

        vol = returns.rolling(self.config.lookback, min_periods=self.config.min_periods).std().shift(1)
        inv_vol = 1.0 / vol.replace(0, np.nan)
        return self._normalise_weight_frame(inv_vol.reindex(dates), factors, fallback_dates=dates)

    def _orthogonalize_incremental_signals(
        self,
        effective: pd.DataFrame,
        factors: list[str],
    ) -> pd.DataFrame:
        baseline = [f for f in self.config.baseline_factors if f in factors]
        if not baseline:
            logger.warning("No baseline factors supplied for orthogonal_incremental; using raw effective signals")
            return effective

        baseline_cols = [self._eff_col(f) for f in baseline]
        result = effective.copy()
        candidate_cols = [self._eff_col(f) for f in factors if f not in baseline]
        for _, group in result.groupby("date", sort=False):
            x = group[baseline_cols].to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=0.0)
            x = np.column_stack([np.ones(len(x)), x])
            for col in candidate_cols:
                y = group[col].to_numpy(dtype=float)
                mask = np.isfinite(y)
                if mask.sum() <= x.shape[1]:
                    continue
                beta = np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]
                residual = y - x @ beta
                result.loc[group.index, col] = residual
        return result

    def _apply_regime_policy(
        self,
        raw_df: pd.DataFrame,
        factors: list[str],
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        policy = self.config.regime_policy
        dates = pd.DatetimeIndex(sorted(raw_df["date"].dropna().unique()))
        if policy is None or not policy.regime_col or policy.regime_col not in raw_df.columns:
            self.regime_controls_ = pd.DataFrame({
                "date": dates,
                "regime_score": np.nan,
                "regime_state": "normal",
                "regime_scalar": 1.0,
                "net_exposure": 1.0,
                "rebalance_threshold": 0.0,
            })
            return weights

        score = raw_df.groupby("date")[policy.regime_col].mean().reindex(dates)
        high = score > policy.threshold
        controls = pd.DataFrame({
            "date": dates,
            "regime_score": score.values,
            "regime_state": np.where(high, "turbulent", "normal"),
            "regime_scalar": np.where(high, policy.reduction_factor, 1.0),
            "net_exposure": np.where(high, policy.high_regime_net_exposure, policy.low_regime_net_exposure),
            "rebalance_threshold": np.where(
                high,
                policy.high_regime_rebalance_threshold,
                policy.low_regime_rebalance_threshold,
            ),
        })
        self.regime_controls_ = controls

        adjusted = weights.copy()
        adjusted = adjusted.set_index("date").reindex(dates).fillna(1.0 / len(factors))
        for dt in dates:
            row_high = bool(high.loc[dt]) if dt in high.index and pd.notna(high.loc[dt]) else False
            disabled = set(policy.disabled_categories_high if row_high else ())
            multipliers = policy.category_multipliers_high if row_high else policy.category_multipliers_low
            for factor in factors:
                category = self.category_map.get(factor, "unknown")
                if category in disabled:
                    adjusted.loc[dt, factor] = 0.0
                adjusted.loc[dt, factor] = adjusted.loc[dt, factor] * multipliers.get(category, 1.0)

        adjusted = self._normalise_weight_frame(adjusted, factors, fallback_dates=dates)
        adjusted["date"] = adjusted.index
        return adjusted.reset_index(drop=True)

    def _normalise_weight_frame(
        self,
        raw: pd.DataFrame,
        factors: list[str],
        *,
        fallback_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        if raw is None or raw.empty:
            return self._equal_weights(fallback_dates, factors)

        frame = raw.reindex(fallback_dates)
        frame = frame.reindex(columns=factors)
        if self.config.long_only_factor_weights:
            frame = frame.clip(lower=0.0)

        rows = []
        for dt, row in frame.iterrows():
            vals = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if vals.notna().sum() == 0 or np.isclose(vals.fillna(0.0).abs().sum(), 0.0):
                vals = pd.Series(1.0 / len(factors), index=factors)
            else:
                vals = vals.fillna(0.0)
                denom = vals.sum() if self.config.long_only_factor_weights else vals.abs().sum()
                if np.isclose(denom, 0.0):
                    vals = pd.Series(1.0 / len(factors), index=factors)
                else:
                    vals = vals / denom
            vals.name = dt
            rows.append(vals)
        out = pd.DataFrame(rows)
        out["date"] = out.index
        return out.reset_index(drop=True)

    @staticmethod
    def _equal_weights(dates: pd.DatetimeIndex, factors: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(1.0 / len(factors), index=dates, columns=factors)
        frame["date"] = frame.index
        return frame.reset_index(drop=True)

    @staticmethod
    def _eff_col(factor: str) -> str:
        return f"__eff_{factor}"


def infer_factor_metadata(factor_cols: Iterable[str]) -> tuple[Dict[str, int], Dict[str, str]]:
    """Best-effort direction/category lookup from the factor registry."""
    directions: Dict[str, int] = {}
    categories: Dict[str, str] = {}
    try:
        from quant_platform.core.signals.registry import FactorRegistry

        registry = FactorRegistry()
        registry.discover()
        for factor in factor_cols:
            if factor in registry:
                meta = registry.get(factor).meta
                directions[factor] = -1 if int(meta.direction) < 0 else 1
                categories[factor] = meta.category
    except Exception as exc:
        logger.debug("Could not infer factor metadata from registry: {}", exc)
    return directions, categories

