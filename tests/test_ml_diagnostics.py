import numpy as np
import pandas as pd
from quant_platform.core.evaluation.ml_diagnostics import adjacent_rank_correlation, threshold_rank_update, coefficient_stability


def test_adjacent_correlation_does_not_bridge_a_missing_security_session():
    dates=pd.bdate_range("2020-01-01",periods=3)
    rows=pd.DataFrame({"date":dates.repeat(6),"ticker":list("ABCDEF")*3,"score":list(range(6))*3})
    rows=rows.loc[~(rows.date.eq(dates[1])&rows.ticker.eq("A"))]
    actual=adjacent_rank_correlation(rows,"score")
    assert np.isnan(actual.iloc[0]) and np.allclose(actual.iloc[1:],1)
    # Fewer than five shared names cannot produce a correlation.
    sparse=rows.loc[~(rows.date.eq(dates[1])&rows.ticker.eq("B"))]
    assert adjacent_rank_correlation(sparse,"score").iloc[1:].isna().all()


def test_rank_deadband_is_prefix_invariant_and_resets_absent_names():
    dates=pd.bdate_range("2020-01-01",periods=4)
    rows=pd.DataFrame({"date":dates.repeat(6),"ticker":list("ABCDEF")*4,"score":np.arange(24)%7})
    prefix=rows.loc[rows.date.lt(dates[-1])]
    np.testing.assert_allclose(threshold_rank_update(rows,"score").loc[prefix.index],threshold_rank_update(prefix,"score"))


def test_sign_stability_ignores_near_zero_coefficients_without_deleting_features():
    rows=pd.DataFrame({"fold":[1,2,3,4],"feature":"x","coefficient":[.1,1e-8,-.2,.3]})
    stats=coefficient_stability(rows).iloc[0]
    assert stats.material_folds==3 and stats.sign_transitions==2
