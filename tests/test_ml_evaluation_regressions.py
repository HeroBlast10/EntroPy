"""Regression tests for ML/evaluation leakage and configuration fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.alpha_models.cross_sectional_ranker import CrossSectionalRanker
from quant_platform.core.alpha_models.regime_overlay import RegimeOverlay
from quant_platform.core.alpha_models.ts_forecaster import TSForecaster
from scripts.evaluate_all_factors_fast import _group_corr


def test_cross_sectional_ranker_maps_weights_by_factor_name() -> None:
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame({
        "date": [date] * 3,
        "ticker": ["A", "B", "C"],
        "f1": [1.0, 2.0, 3.0],
        "f2": [3.0, 2.0, 1.0],
    })
    result = CrossSectionalRanker(
        ["missing", "f1", "f2"],
        weights=[0.8, 0.1, 0.1],
        directions={"f1": 1, "f2": 1},
    ).score(frame)
    # f1 and f2 retain their intended equal weights after "missing" is absent.
    assert result["alpha_cs"].nunique() == 1


def test_ts_forecaster_renormalizes_available_row_weights() -> None:
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame({
        "date": [date] * 3,
        "ticker": ["A", "B", "C"],
        "f1": [1.0, 2.0, 3.0],
        "f2": [3.0, 2.0, np.nan],
    })
    result = TSForecaster(
        ["f1", "f2"],
        weights=[0.8, 0.2],
    ).score(frame).set_index("ticker")
    # C has only f1, so its score is the f1 effective rank itself (1.0).
    assert result.loc["C", "alpha_ts"] == pytest.approx(1.0)


def test_regime_overlay_reduces_long_only_gross_exposure() -> None:
    date = pd.Timestamp("2024-01-02")
    weights = pd.DataFrame({
        "date": [date, date],
        "ticker": ["A", "B"],
        "weight": [0.6, 0.4],
    })
    factors = pd.DataFrame({
        "date": [date, date],
        "ticker": ["A", "B"],
        "HMM_TURBULENCE_PROB": [0.9, 0.9],
    })
    result = RegimeOverlay(reduction_factor=0.5).apply(weights, factors)
    assert result["weight"].abs().sum() == pytest.approx(0.5)


def test_fast_group_corr_rejects_constant_cross_section() -> None:
    codes = np.repeat(np.arange(2), 10)
    constant = np.ones(20)
    varying = np.tile(np.arange(10, dtype=float), 2)
    result = _group_corr(codes, constant, varying, 2)
    assert np.isnan(result).all()
