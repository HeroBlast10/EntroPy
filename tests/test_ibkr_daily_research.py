import datetime as dt

import numpy as np
import pandas as pd
import pytest

from scripts.ibkr_daily_research import prepare_panel


def vendor_frame(n=300):
    close = np.arange(n, dtype=float) + 100
    return pd.DataFrame({'date':pd.bdate_range('2024-01-01',periods=n),
        'symbol':'TEST','open':close,'high':close+1,'low':close-1,
        'close':close,'volume':10000})


def test_excludes_incomplete_session_and_uses_skipped_month():
    frame = vendor_frame()
    cutoff = frame.date.iloc[-1].date()
    panel = prepare_panel(frame,frame,cutoff)
    assert len(panel) == 299 and panel.date.max().date() < cutoff
    close = panel.adjusted_close
    expected = close.iloc[-22]/close.iloc[-253]-1
    assert panel.mom_12_1.iloc[-1] == pytest.approx(expected)


def test_recent_month_cannot_change_momentum():
    frame = vendor_frame()
    before = prepare_panel(frame,frame,dt.date(2030,1,1))
    changed = frame.copy()
    changed.loc[changed.index[-21:],['open','high','low','close']] *= 10
    after = prepare_panel(frame,changed,dt.date(2030,1,1))
    assert before.mom_12_1.iloc[-1] == after.mom_12_1.iloc[-1]


def test_duplicate_or_missing_adjusted_bar_is_rejected():
    frame = vendor_frame()
    with pytest.raises(ValueError,match='Duplicate'):
        prepare_panel(pd.concat([frame,frame.iloc[:1]]),frame,dt.date(2030,1,1))
    with pytest.raises(ValueError,match='Missing corresponding'):
        prepare_panel(frame,frame.iloc[:-1],dt.date(2030,1,1))
