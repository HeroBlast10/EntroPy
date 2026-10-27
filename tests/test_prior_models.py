import numpy as np
import pandas as pd
import pytest
from quant_platform.core.alpha_models.prior_models import PriorRidge,full_date_sample,prior_coefficients,training_ic_selection


def test_prior_ridge_solves_centered_penalty_with_unpenalized_intercept():
    rng=np.random.default_rng(8);x=rng.normal(size=(300,4));y=x@np.array([.8,-.4,.2,.1])+2+rng.normal(0,.1,300)
    prior=np.array([.25,-.25,0,0]);lam=.3
    model=PriorRidge(lam).fit(x,y,prior)
    xc=x-x.mean(0);yc=y-y.mean()
    exact=np.linalg.solve(xc.T@xc+len(x)*lam*np.eye(4),xc.T@yc+len(x)*lam*prior)
    np.testing.assert_allclose(model.coef_,exact,atol=1e-7)
    assert model.intercept_==pytest.approx(y.mean()-x.mean(0)@exact)
    strong=PriorRidge(1e8).fit(x,y,prior)
    np.testing.assert_allclose(strong.coef_,prior,atol=1e-7)


def test_date_sample_keeps_whole_cross_sections_and_no_lookahead_selection():
    dates=pd.bdate_range("2020-01-01",periods=50)
    rows=pd.DataFrame({"date":dates.repeat(6),"ticker":list("ABCDEF")*50})
    subset=rows.loc[full_date_sample(rows,stride=5)]
    assert len(subset)==60 and subset.groupby("date").size().eq(6).all()
    ic=pd.DataFrame({"prior":0.,"good":np.linspace(.1,.2,50),"dup":.2},index=dates)
    features,audit=training_ic_selection(ic,dates[:30],cutoff=dates[30],horizon=20,
        prior_features=["prior"],directions={},duplicates={"dup":"good"})
    assert "prior" in features and "dup" not in features
    with pytest.raises(ValueError,match="cutoff"):
        training_ic_selection(ic,dates,cutoff=dates[30],horizon=20,prior_features=[],directions={})
    np.testing.assert_allclose(prior_coefficients(["a","b","c"],{"a":1,"b":-1}),[.5,-.5,0])
