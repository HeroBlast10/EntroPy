"""Tests for new signal wrappers (time-series, regime, relative-value)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(n_dates=120, n_tickers=5, seed=42):
    """Generate synthetic price panel for signal testing."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_dates, freq="B")
    tickers = [f"STK{i}" for i in range(n_tickers)]
    rows = []
    for tkr in tickers:
        base = np.random.uniform(50, 200)
        cumret = np.cumprod(1 + np.random.normal(0.0005, 0.02, n_dates))
        closes = base * cumret
        for i, dt in enumerate(dates):
            rows.append({
                "date": dt, "ticker": tkr,
                "adj_close": closes[i],
                "close": closes[i],
                "open": closes[i] * 1.001,
                "high": closes[i] * 1.02,
                "low": closes[i] * 0.98,
                "volume": int(1e6),
                "amount": closes[i] * 1e6,
                "adj_factor": 1.0,
                "is_tradable": True,
            })
    return pd.DataFrame(rows)


class TestFactorMeta:
    def test_signal_type_default(self):
        from quant_platform.core.signals.base import FactorMeta
        meta = FactorMeta(name="TEST", category="test")
        assert meta.signal_type == "cross_sectional"

    def test_signal_type_custom(self):
        from quant_platform.core.signals.base import FactorMeta
        meta = FactorMeta(name="TEST", category="test", signal_type="time_series")
        assert meta.signal_type == "time_series"


class TestKalmanFactors:
    def test_kalman_velocity_returns_series(self):
        from quant_platform.core.signals.time_series.kalman_state_space import KalmanVelocity
        panel = _make_panel()
        factor = KalmanVelocity()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)
        assert len(result) == len(panel)

    def test_kalman_trend_strength_returns_series(self):
        from quant_platform.core.signals.time_series.kalman_state_space import KalmanTrendStrength
        panel = _make_panel()
        factor = KalmanTrendStrength()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_kalman_meta_signal_type(self):
        from quant_platform.core.signals.time_series.kalman_state_space import KalmanVelocity
        assert KalmanVelocity.meta.signal_type == "time_series"


class TestEntropyHurstFactors:
    def test_spectral_entropy_returns_series(self):
        from quant_platform.core.signals.time_series.entropy_hurst import SpectralEntropy60D
        panel = _make_panel()
        factor = SpectralEntropy60D()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_hurst_returns_series(self):
        from quant_platform.core.signals.time_series.entropy_hurst import HurstExponent60D
        panel = _make_panel()
        factor = HurstExponent60D()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_entropy_meta_signal_type(self):
        from quant_platform.core.signals.time_series.entropy_hurst import SpectralEntropy60D
        assert SpectralEntropy60D.meta.signal_type == "time_series"


class TestHigherMoments:
    def test_skew_returns_series(self):
        from quant_platform.core.signals.time_series.higher_moments import RollingSkew60D
        panel = _make_panel()
        factor = RollingSkew60D()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_kurt_returns_series(self):
        from quant_platform.core.signals.time_series.higher_moments import RollingKurt60D
        panel = _make_panel()
        factor = RollingKurt60D()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)


class TestHMMRegime:
    def test_regime_returns_series(self):
        from quant_platform.core.signals.regime.hmm_regime import HMMTurbulenceProbability
        panel = _make_panel()
        factor = HMMTurbulenceProbability()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_regime_meta_signal_type(self):
        from quant_platform.core.signals.regime.hmm_regime import HMMTurbulenceProbability
        assert HMMTurbulenceProbability.meta.signal_type == "regime"


class TestOUPairs:
    def test_ou_zscore_returns_series(self):
        from quant_platform.core.signals.relative_value.ou_pairs import OUZScore
        panel = _make_panel()
        factor = OUZScore()
        result = factor._compute(panel)
        assert isinstance(result, pd.Series)

    def test_ou_meta_signal_type(self):
        from quant_platform.core.signals.relative_value.ou_pairs import OUZScore
        assert OUZScore.meta.signal_type == "relative_value"


class TestRegistry:
    def test_discover_all_categories(self):
        from quant_platform.core.signals.registry import FactorRegistry
        reg = FactorRegistry()
        reg.discover()
        # Should have at least: 24 CS + 3 Kalman + 2 entropy/hurst + 3 moments + 1 HMM + 1 OU = 34
        assert len(reg) >= 34

    def test_filter_by_signal_type(self):
        from quant_platform.core.signals.registry import FactorRegistry
        reg = FactorRegistry()
        reg.discover()
        ts_factors = reg.list_factors(signal_type="time_series")
        assert len(ts_factors) >= 8  # 3 Kalman + 2 entropy/hurst + 3 moments

    def test_filter_by_category(self):
        from quant_platform.core.signals.registry import FactorRegistry
        reg = FactorRegistry()
        reg.discover()
        mom = reg.list_factors(category="momentum")
        assert len(mom) == 7


class TestAlphaModels:
    def test_cross_sectional_ranker(self):
        from quant_platform.core.alpha_models.cross_sectional_ranker import CrossSectionalRanker
        df = pd.DataFrame({
            "date": pd.Timestamp("2023-01-03"),
            "ticker": ["A", "B", "C"],
            "MOM": [0.1, 0.2, 0.3],
            "VOL": [-0.3, -0.2, -0.1],
        })
        ranker = CrossSectionalRanker(["MOM", "VOL"])
        scored = ranker.score(df)
        assert "alpha_cs" in scored.columns
        assert len(scored) == 3

    def test_ensemble_alpha(self):
        from quant_platform.core.alpha_models.ensemble import EnsembleAlpha
        from quant_platform.core.alpha_models.cross_sectional_ranker import CrossSectionalRanker
        model = EnsembleAlpha(
            cs_ranker=CrossSectionalRanker(["MOM"]),
            w_cs=1.0, w_ts=0.0,
        )
        df = pd.DataFrame({
            "date": pd.Timestamp("2023-01-03"),
            "ticker": ["A", "B", "C"],
            "MOM": [0.1, 0.2, 0.3],
        })
        result = model.score(df)
        assert "alpha_ensemble" in result.columns


class TestCNAShareCostModel:
    def test_cost_positive_for_trade(self):
        from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel
        model = AShareCostModel()
        w_prev = np.array([0.5, 0.5])
        w_new = np.array([0.3, 0.7])
        cost = model.compute_total_cost(w_prev, w_new)
        assert cost > 0

    def test_zero_cost_for_no_trade(self):
        from quant_platform.core.execution.cost_models.cn_a_share import AShareCostModel
        model = AShareCostModel()
        w = np.array([0.5, 0.5])
        cost = model.compute_total_cost(w, w)
        assert cost == 0.0
