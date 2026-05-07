"""Tests for production-grade factor research upgrades.

Covers:
- Newey-West HAC IC t-stat
- Alpha half-life estimator
- Long/short leg attribution
- Cross-sectional stability across universe slices
- Hierarchical-clustering redundancy pruning
- Full combinatorial CSCV
- Hansen SPA test
- Residual Momentum, Overnight Return, Accruals, Piotroski F-Score factors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# Shared helpers
# ===================================================================

def _toy_panel(n_dates: int = 80, n_tickers: int = 6, seed: int = 1) -> pd.DataFrame:
    """Synthetic price panel for factor and IC tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for tk in tickers:
        base = rng.uniform(50, 200)
        ret = rng.normal(0.0005, 0.02, n_dates)
        closes = base * np.cumprod(1.0 + ret)
        opens = closes * rng.uniform(0.99, 1.01, n_dates)
        for i, dt in enumerate(dates):
            rows.append({
                "date": dt, "ticker": tk,
                "open": opens[i],
                "high": closes[i] * 1.02, "low": closes[i] * 0.98,
                "close": closes[i],
                "adj_close": closes[i],
                "adj_factor": 1.0,
                "volume": int(rng.uniform(1e6, 1e7)),
                "amount": closes[i] * rng.uniform(1e6, 1e7),
                "is_tradable": True,
                "market_cap": closes[i] * 1e8 * rng.uniform(0.5, 5.0),
            })
    return pd.DataFrame(rows)


def _ic_dataframe(n: int = 250, n_tickers: int = 30, signal_corr_with_ret: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Build a (date, ticker, factor, fwd_ret_1d) DataFrame for IC tests.

    Defaults to 30 tickers so every cross-section has enough names for IC
    computation (which requires >= 5 valid observations) and for bucketed
    sub-universe stability tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        z = rng.standard_normal(len(tickers))
        ret = signal_corr_with_ret * z + np.sqrt(1 - signal_corr_with_ret ** 2) * rng.standard_normal(len(tickers))
        for i, tk in enumerate(tickers):
            rows.append({"date": d, "ticker": tk, "factor": z[i], "fwd_ret_1d": ret[i] * 0.02})
    return pd.DataFrame(rows)


# ===================================================================
# Newey-West HAC IC + alpha half-life
# ===================================================================

class TestNeweyWestIC:
    def test_newey_west_se_returns_finite(self):
        from quant_platform.core.signals.cross_sectional.evaluation import newey_west_se
        rng = np.random.default_rng(0)
        x = pd.Series(rng.standard_normal(200))
        se = newey_west_se(x)
        assert np.isfinite(se) and se > 0

    def test_newey_west_se_inflates_under_autocorrelation(self):
        """HAC SE should be >= i.i.d. SE for a positively autocorrelated series."""
        from quant_platform.core.signals.cross_sectional.evaluation import newey_west_se
        rng = np.random.default_rng(7)
        n = 500
        eps = rng.standard_normal(n)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.5 * ar[t - 1] + eps[t]
        s = pd.Series(ar)
        nw = newey_west_se(s, lag=10)
        iid = float(s.std(ddof=1) / np.sqrt(n))
        assert nw >= iid * 1.05  # at least 5% inflation under rho=0.5

    def test_ic_summary_has_nw_keys(self):
        from quant_platform.core.signals.cross_sectional.evaluation import (
            compute_rank_ic_series, ic_summary,
        )
        df = _ic_dataframe(n=200, signal_corr_with_ret=0.08)
        ic = compute_rank_ic_series(df, "factor")
        stats = ic_summary(ic)
        for key in ("nw_t_stat", "nw_p_value", "nw_lag", "t_stat", "p_value"):
            assert key in stats
        assert np.isfinite(stats["nw_t_stat"])
        # NW p-value should be in [0, 1]
        assert 0.0 <= stats["nw_p_value"] <= 1.0


class TestAlphaHalfLife:
    def test_synthetic_decay_recovers_half_life(self):
        from quant_platform.core.signals.cross_sectional.evaluation import estimate_alpha_half_life
        # IC(h) = 0.05 * exp(-h / 10) → half-life = 10 * ln 2 ≈ 6.93
        h = np.arange(1, 21)
        ic = 0.05 * np.exp(-h / 10.0)
        df = pd.DataFrame({"horizon": h, "mean_ic": ic})
        result = estimate_alpha_half_life(df)
        assert abs(result["half_life_days"] - 10.0 * np.log(2)) < 0.5
        assert result["r_squared"] > 0.99

    def test_flat_ic_returns_inf_half_life(self):
        from quant_platform.core.signals.cross_sectional.evaluation import estimate_alpha_half_life
        df = pd.DataFrame({"horizon": np.arange(1, 11), "mean_ic": np.full(10, 0.03)})
        result = estimate_alpha_half_life(df)
        assert result["half_life_days"] == np.inf or np.isnan(result["half_life_days"])

    def test_negative_ic_excluded_returns_nan(self):
        from quant_platform.core.signals.cross_sectional.evaluation import estimate_alpha_half_life
        df = pd.DataFrame({"horizon": np.arange(1, 11), "mean_ic": np.full(10, -0.02)})
        result = estimate_alpha_half_life(df)
        assert np.isnan(result["half_life_days"])


# ===================================================================
# Long/short attribution + cross-sectional stability
# ===================================================================

class TestLongShortAttribution:
    def test_attribution_returns_required_keys(self):
        from quant_platform.core.signals.cross_sectional.evaluation import long_short_leg_attribution
        df = _ic_dataframe(n=120, signal_corr_with_ret=0.10)
        result = long_short_leg_attribution(df, "factor", n_quantiles=4)
        for key in ("long_leg_sharpe", "short_leg_sharpe", "long_share_of_ls",
                    "long_only_compatible"):
            assert key in result

    def test_long_share_plus_short_share_close_to_one(self):
        from quant_platform.core.signals.cross_sectional.evaluation import long_short_leg_attribution
        df = _ic_dataframe(n=200, signal_corr_with_ret=0.10)
        result = long_short_leg_attribution(df, "factor", n_quantiles=4)
        if pd.notna(result["long_share_of_ls"]) and pd.notna(result["short_share_of_ls"]):
            assert abs(result["long_share_of_ls"] + result["short_share_of_ls"] - 1.0) < 1e-6

    def test_no_returns_returns_nans(self):
        from quant_platform.core.signals.cross_sectional.evaluation import long_short_leg_attribution
        df = _ic_dataframe(n=50).drop(columns=["fwd_ret_1d"])
        result = long_short_leg_attribution(df, "factor", "fwd_ret_1d")
        assert pd.isna(result["long_leg_sharpe"])


class TestCrossSectionalStability:
    def test_stability_runs_with_market_cap(self):
        from quant_platform.core.signals.cross_sectional.evaluation import cross_sectional_stability
        df = _ic_dataframe(n=120)
        rng = np.random.default_rng(2)
        df["market_cap"] = rng.uniform(1e8, 1e11, len(df))
        df["amount"] = rng.uniform(1e5, 1e8, len(df))
        result = cross_sectional_stability(df, "factor", n_buckets=3)
        assert not result.empty
        assert {"segment", "bucket", "rank_ic_mean", "rank_ic_t", "rank_ic_nw_t", "n_dates"} <= set(result.columns)

    def test_stability_empty_when_no_segment(self):
        from quant_platform.core.signals.cross_sectional.evaluation import cross_sectional_stability
        df = _ic_dataframe(n=50)
        result = cross_sectional_stability(df, "factor", segment_cols=("nonexistent_col",))
        assert result.empty


# ===================================================================
# Hierarchical-clustering redundancy pruning
# ===================================================================

class TestHierarchicalSelection:
    def _redundancy_report(self, n_factors: int = 6, seed: int = 3):
        rng = np.random.default_rng(seed)
        names = [f"F{i}" for i in range(n_factors)]
        # Two highly correlated pairs, two independent
        base1 = rng.standard_normal(200)
        base2 = rng.standard_normal(200)
        cols = {
            "F0": base1,
            "F1": base1 + 0.05 * rng.standard_normal(200),  # corr ~ 0.99 with F0
            "F2": base2,
            "F3": base2 + 0.05 * rng.standard_normal(200),  # corr ~ 0.99 with F2
            "F4": rng.standard_normal(200),
            "F5": rng.standard_normal(200),
        }
        sig_corr = pd.DataFrame(cols).corr()
        return {
            "signal_correlation": sig_corr,
            "factor_return_correlation": sig_corr,
            "exposure_similarity": sig_corr.abs(),
        }, names

    def test_clusters_separate_correlated_pairs(self):
        from quant_platform.core.signals.redundancy import cluster_based_factor_selection
        report, names = self._redundancy_report()
        score = pd.DataFrame({
            "factor": names,
            "selection_score": [1.0, 0.9, 1.5, 0.8, 0.5, 2.0],
        }).set_index("factor", drop=False)
        result = cluster_based_factor_selection(score, report, n_clusters=4)
        # 4 clusters → exactly 4 factors marked selected
        assert int(result["selected"].sum()) == 4
        # Within each cluster, the chosen factor must have the max score
        for cid, grp in result.groupby("cluster_id"):
            in_cluster = grp.copy()
            best_idx = in_cluster["selection_score"].idxmax()
            chosen = in_cluster[in_cluster["selected"]].index
            assert best_idx in chosen

    def test_single_factor_universe(self):
        from quant_platform.core.signals.redundancy import cluster_based_factor_selection
        report = {"signal_correlation": pd.DataFrame([[1.0]], index=["F0"], columns=["F0"])}
        score = pd.DataFrame({"factor": ["F0"], "selection_score": [0.5]}).set_index("factor", drop=False)
        result = cluster_based_factor_selection(score, report)
        assert result["selected"].iloc[0]
        assert result["cluster_id"].iloc[0] == 1

    def test_distance_threshold_path(self):
        from quant_platform.core.signals.redundancy import cluster_based_factor_selection
        report, names = self._redundancy_report()
        score = pd.DataFrame({
            "factor": names,
            "selection_score": np.arange(len(names))[::-1].astype(float),
        }).set_index("factor", drop=False)
        # Threshold 0.30 corresponds to merging anything with |corr| > 0.70
        result = cluster_based_factor_selection(score, report, distance_threshold=0.30)
        assert int(result["selected"].sum()) >= 1
        assert int(result["selected"].sum()) <= len(names)


# ===================================================================
# Full CSCV
# ===================================================================

class TestFullCSCV:
    def test_pure_noise_pbo_centered_around_half_on_average(self):
        """For pure noise the average PBO over many seeds must approach 0.5.

        A single CSCV run is noisy: with n_var=40 noise variants and 252
        combinations, PBO has finite-sample std ~0.20.  Averaging across
        seeds removes that variance and the population value (~0.5) emerges.
        """
        from quant_platform.core.evaluation.overfit import probability_of_backtest_overfitting
        pbos = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            returns = pd.DataFrame(rng.standard_normal((500, 40)) * 0.01)
            result = probability_of_backtest_overfitting(returns, n_splits=10, return_logits=False)
            pbos.append(result["pbo"])
        avg = float(np.mean(pbos))
        assert 0.40 <= avg <= 0.60, f"average PBO over noise seeds = {avg:.3f}, expected ~ 0.5"

    def test_full_combinatorial_count(self):
        """Verify CSCV truly enumerates all C(S, S/2) combinations, not just 2."""
        from quant_platform.core.evaluation.overfit import probability_of_backtest_overfitting
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.standard_normal((400, 10)) * 0.01)
        result = probability_of_backtest_overfitting(returns, n_splits=8, return_logits=False)
        assert result["n_combinations"] == 70   # C(8, 4)
        assert result["n_variants"] == 10
        assert result["n_splits"] == 8

    def test_strong_signal_pbo_near_zero(self):
        """With a clearly best variant, IS-best == OOS-best almost always."""
        from quant_platform.core.evaluation.overfit import probability_of_backtest_overfitting
        rng = np.random.default_rng(22)
        n_obs, n_var = 500, 20
        noise = rng.standard_normal((n_obs, n_var)) * 0.01
        noise[:, 0] += 0.005  # winner with persistent edge
        returns = pd.DataFrame(noise)
        result = probability_of_backtest_overfitting(returns, n_splits=8, return_logits=False)
        assert result["pbo"] < 0.10

    def test_returns_logits_optional(self):
        from quant_platform.core.evaluation.overfit import probability_of_backtest_overfitting
        rng = np.random.default_rng(33)
        returns = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01)
        result = probability_of_backtest_overfitting(returns, n_splits=6, return_logits=True)
        assert "logits" in result
        assert len(result["logits"]) == result["n_combinations"]
        # Without logits returned the dict still works
        result2 = probability_of_backtest_overfitting(returns, n_splits=6, return_logits=False)
        assert "logits" not in result2


# ===================================================================
# Hansen SPA test
# ===================================================================

class TestHansenSPA:
    def test_pure_noise_high_pvalue(self):
        from quant_platform.core.signals.factor_selection import hansen_spa_test
        rng = np.random.default_rng(44)
        n_obs = 400
        ls = {f"S{i}": pd.Series(rng.standard_normal(n_obs) * 0.01) for i in range(15)}
        result = hansen_spa_test(ls, n_boot=300)
        assert 0.0 <= result["spa_pvalue_c"] <= 1.0
        # Pure noise — SPA p-value should not be tiny
        assert result["spa_pvalue_c"] > 0.05

    def test_real_alpha_low_pvalue(self):
        from quant_platform.core.signals.factor_selection import hansen_spa_test
        rng = np.random.default_rng(55)
        n_obs = 600
        # 4 noise + 1 with strong, persistent drift
        signals = {f"NOISE_{i}": pd.Series(rng.standard_normal(n_obs) * 0.01) for i in range(4)}
        signals["WINNER"] = pd.Series(rng.standard_normal(n_obs) * 0.01 + 0.004)
        result = hansen_spa_test(signals, n_boot=300)
        assert result["best_strategy"] == "WINNER"
        # With ~Sharpe=6 the SPA p-value should be small
        assert result["spa_pvalue_c"] < 0.10

    def test_pvalue_ordering(self):
        """Hansen (2005) Theorem 1: p_l <= p_c <= p_u asymptotically."""
        from quant_platform.core.signals.factor_selection import hansen_spa_test
        rng = np.random.default_rng(66)
        signals = {f"S{i}": pd.Series(rng.standard_normal(400) * 0.01) for i in range(12)}
        result = hansen_spa_test(signals, n_boot=400)
        # Asymptotic ordering with finite-sample / Monte-Carlo tolerance
        tol = 0.10
        assert result["spa_pvalue_l"] <= result["spa_pvalue_c"] + tol
        assert result["spa_pvalue_c"] <= result["spa_pvalue_u"] + tol


# ===================================================================
# New factors
# ===================================================================

class TestResidualMomentum:
    def test_residual_momentum_returns_series(self):
        from quant_platform.core.signals.cross_sectional.momentum import ResidualMomentum
        # Need ~ 36 months of history; use a short panel for smoke test
        # and disable strict beta_window via param override
        panel = _toy_panel(n_dates=300, n_tickers=8)
        factor = ResidualMomentum(beta_window=120, mom_window=84, skip=10)
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)
        assert len(result) == len(panel)
        # Some observations near the end should be filled
        assert result.notna().sum() > 0

    def test_meta_direction(self):
        from quant_platform.core.signals.cross_sectional.momentum import ResidualMomentum
        assert ResidualMomentum.meta.direction == 1
        assert ResidualMomentum.meta.signal_type == "cross_sectional"


class TestOvernightReturn:
    def test_overnight_returns_series(self):
        from quant_platform.core.signals.cross_sectional.momentum import OvernightReturn1M
        panel = _toy_panel(n_dates=80, n_tickers=4)
        factor = OvernightReturn1M()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)
        assert len(result) == len(panel)
        assert result.notna().sum() > 0

    def test_no_open_column_returns_nans(self):
        from quant_platform.core.signals.cross_sectional.momentum import OvernightReturn1M
        panel = _toy_panel(n_dates=40).drop(columns=["open"])
        factor = OvernightReturn1M()
        result = factor._compute(panel)
        assert result.isna().all()


class TestAccrualsAndPiotroski:
    @pytest.fixture
    def fundamentals(self) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        report_dates = pd.date_range("2020-03-31", periods=12, freq="QE")
        tickers = ["AAA", "BBB"]
        rows = []
        for tk in tickers:
            for i, rd in enumerate(report_dates):
                d = rd + pd.Timedelta(days=45)
                rows.append({
                    "date": d,
                    "ticker": tk,
                    "report_date": rd,
                    "publish_date": d,
                    "net_income": 1000.0 + 50 * i + rng.normal(0, 10),
                    "cash_from_operations": 800.0 + 60 * i + rng.normal(0, 10),
                    "total_assets": 50000.0 + 200 * i,
                    "total_debt": 12000.0 - 50 * i,
                    "total_equity": 30000.0 + 200 * i,
                    "shares_outstanding": 1000.0,
                    "gross_profit": 2000.0 + 80 * i,
                    "revenue": 5000.0 + 100 * i,
                    "market_cap": 100000.0,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def prices(self) -> pd.DataFrame:
        dates = pd.bdate_range("2020-04-01", periods=900, freq="B")
        rows = []
        for tk in ("AAA", "BBB"):
            for d in dates:
                rows.append({"date": d, "ticker": tk, "adj_close": 100.0})
        return pd.DataFrame(rows)

    def test_accruals_runs(self, prices, fundamentals):
        from quant_platform.core.signals.cross_sectional.value_quality import Accruals
        result = Accruals()._compute(prices, fundamentals)
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0

    def test_accruals_direction_negative(self):
        from quant_platform.core.signals.cross_sectional.value_quality import Accruals
        assert Accruals.meta.direction == -1

    def test_piotroski_runs(self, prices, fundamentals):
        from quant_platform.core.signals.cross_sectional.value_quality import PiotroskiFScore
        result = PiotroskiFScore()._compute(prices, fundamentals)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert len(valid) > 0
        # Score must be in [0, 8]
        assert valid.min() >= 0
        assert valid.max() <= 8

    def test_piotroski_direction_positive(self):
        from quant_platform.core.signals.cross_sectional.value_quality import PiotroskiFScore
        assert PiotroskiFScore.meta.direction == 1


# ===================================================================
# Registry must auto-discover the four new factors
# ===================================================================

class TestRegistryAutoDiscovery:
    def test_new_factors_registered(self):
        from quant_platform.core.signals.registry import FactorRegistry
        reg = FactorRegistry()
        reg.discover()
        for name in ("RESID_MOM_12_1M", "OVERNIGHT_RET_21D", "ACCRUALS", "PIOTROSKI_F8"):
            assert name in reg, f"{name} missing from registry"
