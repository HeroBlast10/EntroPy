"""Tests for the transaction cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.execution.cost_models.us_equity import (
    CostModel,
    daily_borrow_cost,
    estimate_batch_costs,
    estimate_trade_cost,
    summarise_costs,
)


class TestEstimateTradeCost:
    def setup_method(self):
        self.model = CostModel()

    def test_buy_has_no_sec_fee(self):
        cost = estimate_trade_cost("buy", 1000, 100.0, 1e6, 0.02, self.model)
        assert cost["sec_fee"] == 0.0
        assert cost["finra_taf"] == 0.0

    def test_sell_has_sec_fee(self):
        cost = estimate_trade_cost("sell", 1000, 100.0, 1e6, 0.02, self.model)
        assert cost["sec_fee"] > 0.0
        assert cost["finra_taf"] > 0.0

    def test_total_cost_positive(self):
        cost = estimate_trade_cost("buy", 500, 50.0, 5e5, 0.015, self.model)
        assert cost["total_cost"] > 0.0
        assert cost["total_cost_bps"] > 0.0

    def test_zero_shares_zero_cost(self):
        cost = estimate_trade_cost("buy", 0, 100.0, 1e6, 0.02, self.model)
        assert cost["total_cost"] == 0.0

    def test_slippage_proportional_to_notional(self):
        cost_small = estimate_trade_cost("buy", 100, 100.0, 1e6, 0.02, self.model)
        cost_large = estimate_trade_cost("buy", 1000, 100.0, 1e6, 0.02, self.model)
        assert cost_large["slippage"] > cost_small["slippage"]

    def test_impact_increases_with_participation(self):
        # Higher participation rate → higher impact
        cost_low = estimate_trade_cost("buy", 1000, 100.0, 1e7, 0.02, self.model)   # 0.01%
        cost_high = estimate_trade_cost("buy", 1000, 100.0, 1e4, 0.02, self.model)  # 10%
        assert cost_high["impact"] > cost_low["impact"]

    def test_commission_minimum(self):
        # 1 share × $0.005 = $0.005, but min is $1
        cost = estimate_trade_cost("buy", 1, 100.0, 1e6, 0.02, self.model)
        assert cost["commission"] >= self.model.commission_min

    def test_zero_cost_model(self):
        zero = CostModel(
            commission_per_share=0, commission_pct=0, commission_min=0,
            slippage_bps=0, impact_coeff=0, sec_fee_rate=0,
            finra_taf_per_share=0, stamp_duty_pct=0,
        )
        cost = estimate_trade_cost("sell", 1000, 100.0, 1e6, 0.02, zero)
        assert cost["total_cost"] == 0.0


class TestDailyBorrowCost:
    def test_positive_for_short(self):
        model = CostModel(borrow_rate_annual=0.01)
        cost = daily_borrow_cost(-100_000, model)
        expected = 100_000 * 0.01 / 252
        assert abs(cost - expected) < 0.01

    def test_zero_for_no_short(self):
        model = CostModel()
        assert daily_borrow_cost(0, model) == 0.0


class TestBatchCosts:
    def test_batch_adds_columns(self):
        trades = pd.DataFrame({
            "side": ["buy", "sell"],
            "shares": [1000, 500],
            "price": [100.0, 200.0],
            "adv_shares": [1e6, 1e6],
            "daily_vol": [0.02, 0.02],
        })
        result = estimate_batch_costs(trades)
        assert "total_cost" in result.columns
        assert "slippage" in result.columns
        assert len(result) == 2

    def test_summarise(self):
        trades = pd.DataFrame({
            "side": ["buy", "sell"],
            "shares": [1000, 500],
            "price": [100.0, 200.0],
            "adv_shares": [1e6, 1e6],
            "daily_vol": [0.02, 0.02],
        })
        result = estimate_batch_costs(trades)
        summary = summarise_costs(result)
        assert summary["total_notional"] > 0
        assert summary["total_cost"] > 0
