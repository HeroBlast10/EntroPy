import numpy as np
import pandas as pd
from scripts.replay_wrds_extensions import positive_icir_forecast
from quant_platform.core.portfolio.low_turnover import TurnoverPolicy,build_turnover_targets


def test_zero_training_weights_clear_prior_ema_selection():
    forecast=positive_icir_forecast([[1,3],[8,9],[2,4]],[[1,3],[-1,0],[0,2]])
    assert forecast[0]==2.5 and np.isnan(forecast[1]) and forecast[2]==4
    dates=pd.bdate_range("2020-01-02",periods=2)
    rows=pd.DataFrame({"date":dates.repeat(5),"ticker":list("ABCDE")*2,
        "permno":[1,2,3,4,5]*2,"permco":[1,2,3,4,5]*2,"prediction":[-2,-1,0,1,2,*([np.nan]*5)]})
    execution={"gross_target":1,"min_names":5,"max_stock_weight":1,"max_issuer_gross_weight":1}
    weights,_=build_turnover_targets(rows,"prediction",execution,TurnoverPolicy(exit_fraction=.3,ema_halflife=5))
    assert weights.loc[weights.date.eq(dates[0]),"weight"].abs().sum()>0
    assert weights.loc[weights.date.eq(dates[1]),"weight"].eq(0).all()
