"""Unit tests for the WRDS aggregation and point-in-time mapping helpers.

These cover the parts of the WRDS pull that are easy to get wrong and impossible
to notice afterwards -- none of them require network access or credentials.

Each test encodes a specific failure mode observed in practice:

* reading a cumulative year-to-date flow as a quarterly flow;
* letting ``rolling(4)`` emit partial warm-up sums that are silently half-scale;
* comparing a quarter against the previous row instead of the same quarter last
  year;
* joining identifiers by ticker, which splices two companies after a ticker is
  recycled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.wrds_ingest import fundamentals as fb
from scripts.wrds_ingest import universe as uv


# ---------------------------------------------------------------------------
# Cumulative -> single quarter
# ---------------------------------------------------------------------------


@pytest.fixture
def one_company_cumulative() -> pd.DataFrame:
    """Four quarters of year-to-date revenue totalling 500."""
    return pd.DataFrame(
        {
            "gvkey": ["001234"] * 4,
            "datadate": pd.to_datetime(
                ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]
            ),
            "rdq": pd.to_datetime(
                ["2020-05-01", "2020-08-01", "2020-11-01", "2021-02-01"]
            ),
            "fyearq": [2020] * 4,
            "fqtr": [1, 2, 3, 4],
            "saleq": [100.0, 250.0, 380.0, 500.0],   # cumulative within FY2020
            "atq": [1000.0, 1050.0, 1100.0, 1150.0],  # stock, not cumulative
        }
    )


def test_cumulative_flow_is_differenced(one_company_cumulative):
    quarterly, stats = fb.to_single_quarter(one_company_cumulative, ["saleq", "atq"])

    # 100, 150, 130, 120 -- not 100, 250, 380, 500.
    assert quarterly["saleq_q"].round(6).tolist() == [100.0, 150.0, 130.0, 120.0]
    assert stats["n_flow_items"] == 1
    assert stats["n_flow_values_left_as_nan"] == 0


def test_stock_items_are_never_differenced(one_company_cumulative):
    quarterly, _ = fb.to_single_quarter(one_company_cumulative, ["saleq", "atq"])
    assert quarterly["atq_q"].tolist() == [1000.0, 1050.0, 1100.0, 1150.0]


def test_missing_prior_quarter_yields_nan_not_year_to_date():
    """Q3 without a Q2 must not fall back to the accumulated figure.

    Falling back would report 380 instead of ~130: a plausible-looking number
    three times too large, which no downstream range check would catch.
    """
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 2,
            "datadate": pd.to_datetime(["2020-03-31", "2020-09-30"]),
            "fyearq": [2020, 2020],
            "fqtr": [1, 3],
            "saleq": [100.0, 380.0],
        }
    )
    quarterly, stats = fb.to_single_quarter(frame, ["saleq"])

    assert quarterly["saleq_q"].tolist()[0] == 100.0
    assert np.isnan(quarterly["saleq_q"].tolist()[1])
    assert stats["n_flow_values_left_as_nan"] == 1


def test_fiscal_year_boundary_does_not_difference_across_years():
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 3,
            "datadate": pd.to_datetime(["2020-12-31", "2021-03-31", "2021-06-30"]),
            "fyearq": [2020, 2021, 2021],
            "fqtr": [4, 1, 2],
            "saleq": [500.0, 90.0, 200.0],
        }
    )
    quarterly, stats = fb.to_single_quarter(frame, ["saleq"])

    # 2021-Q1 restarts accumulation, so no prior quarter exists -> NaN, not 90-500.
    assert np.isnan(quarterly["saleq_q"].tolist()[0])   # FY2020 Q4 with no Q3 either
    assert quarterly["saleq_q"].tolist()[1] == 90.0     # Q1 is already single-quarter
    assert quarterly["saleq_q"].tolist()[2] == 110.0    # 200 - 90
    assert stats["n_flow_values_left_as_nan"] == 1


# ---------------------------------------------------------------------------
# TTM
# ---------------------------------------------------------------------------


def test_ttm_requires_a_full_four_quarters():
    """Warm-up periods must be NaN, not half-scale sums."""
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 6,
            "datadate": pd.to_datetime(pd.to_datetime(["2020-03-31","2020-06-30","2020-09-30","2020-12-31","2021-03-31","2021-06-30"])),
            "saleq_q": [100.0, 150.0, 130.0, 120.0, 110.0, 140.0],
        }
    )
    out = fb.to_ttm(frame, ["saleq"])

    ttm = out["saleq_ttm"].tolist()
    assert all(np.isnan(v) for v in ttm[:3]), "first three quarters have no full window"
    assert ttm[3] == pytest.approx(500.0)
    assert ttm[4] == pytest.approx(510.0)
    assert ttm[5] == pytest.approx(500.0)


def test_ttm_does_not_leak_across_gvkeys():
    frame = pd.DataFrame(
        {
            "gvkey": ["A"] * 4 + ["B"] * 4,
            "datadate": pd.to_datetime(
                pd.to_datetime(["2020-03-31","2020-06-30","2020-09-30","2020-12-31"]).tolist() * 2
            ),
            "saleq_q": [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    out = fb.to_ttm(frame, ["saleq"])
    by_key = out.groupby("gvkey")["saleq_ttm"].last()
    assert by_key["A"] == pytest.approx(4.0)
    assert by_key["B"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# YoY
# ---------------------------------------------------------------------------


def test_yoy_matches_same_quarter_prior_year():
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 8,
            "datadate": pd.to_datetime(pd.to_datetime(["2020-03-31","2020-06-30","2020-09-30","2020-12-31","2021-03-31","2021-06-30","2021-09-30","2021-12-31"])),
            "fyearq": [2020] * 4 + [2021] * 4,
            "fqtr": [1, 2, 3, 4] * 2,
            "saleq_q": [100.0, 150.0, 130.0, 120.0, 120.0, 120.0, 130.0, 120.0],
        }
    )
    out = fb.to_yoy(frame, ["saleq"])

    pct = out.set_index("datadate")["saleq_yoy_pct"]
    assert np.isnan(pct.loc["2020-03-31"])
    assert pct.loc["2021-03-31"] == pytest.approx(0.20)   # 120/100 - 1
    assert pct.loc["2021-06-30"] == pytest.approx(-0.20)  # 120/150 - 1


def test_yoy_ratio_is_nan_on_near_zero_base():
    """A near-zero denominator must produce NaN, not an infinity."""
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 5,
            "datadate": pd.to_datetime(pd.to_datetime(["2020-03-31","2020-06-30","2020-09-30","2020-12-31","2021-03-31"])),
            "fyearq": [2020] * 4 + [2021],
            "fqtr": [1, 2, 3, 4, 1],
            "saleq_q": [0.0, 150.0, 130.0, 120.0, 10.0],
        }
    )
    out = fb.to_yoy(frame, ["saleq"])
    assert np.isnan(out["saleq_yoy_pct"].iloc[-1])
    assert out["saleq_yoy_diff"].iloc[-1] == pytest.approx(10.0)


def test_yoy_is_nan_when_no_same_quarter_exists_in_the_prior_year():
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 2,
            "datadate": pd.to_datetime(["2023-03-31", "2023-06-30"]),
            "fyearq": [2023, 2023],
            "fqtr": [1, 2],
            "saleq_q": [10.0, 20.0],
        }
    )
    out = fb.to_yoy(frame, ["saleq"])
    assert out["saleq_yoy_pct"].notna().sum() == 0
    assert out["saleq_yoy_diff"].notna().sum() == 0
    assert len(out) == 2


def test_yoy_skips_the_year_gap_when_only_one_season_overlaps():
    """2024-Q1 must compare against 2023-Q1 even though other seasons are absent."""
    frame = pd.DataFrame(
        {
            "gvkey": ["001234"] * 3,
            "datadate": pd.to_datetime(["2023-03-31", "2023-06-30", "2024-03-31"]),
            "fyearq": [2023, 2023, 2024],
            "fqtr": [1, 2, 1],
            "saleq_q": [10.0, 20.0, 30.0],
        }
    )
    out = fb.to_yoy(frame, ["saleq"])
    pct = out["saleq_yoy_pct"].tolist()
    assert np.isnan(pct[0]) and np.isnan(pct[1])
    assert pct[2] == pytest.approx(2.0)   # 30 / 10 - 1


# ---------------------------------------------------------------------------
# Availability dates
# ---------------------------------------------------------------------------


def test_availability_prefers_rdq_and_counts_imputation():
    frame = pd.DataFrame(
        {
            "datadate": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "rdq": pd.to_datetime(["2020-05-01", pd.NaT]),
        }
    )
    out, stats = fb.availability_dates(frame)

    assert out["available_date"].iloc[0] == pd.Timestamp("2020-05-01")
    assert out["available_date"].iloc[1] == pd.Timestamp("2020-06-30") + pd.Timedelta(days=90)
    assert stats["n_available_date_imputed"] == 1


def test_availability_never_precedes_period_end():
    frame = pd.DataFrame(
        {
            "datadate": pd.to_datetime(["2020-12-31"]),
            "rdq": pd.to_datetime(["2019-01-01"]),   # clearly a data error
        }
    )
    out, stats = fb.availability_dates(frame)
    assert out["available_date"].iloc[0] == pd.Timestamp("2020-12-31")
    assert stats["n_available_date_clipped"] == 1


# ---------------------------------------------------------------------------
# PERMNO-keyed identifiers
# ---------------------------------------------------------------------------


def test_ticker_is_resolved_as_of_the_observation_date():
    """Before and after a rename the same security carries different tickers."""
    names = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "effective_from": pd.to_datetime(["2010-01-01", "2015-06-01"]),
            "effective_to": pd.to_datetime(["2015-05-31", None]),
            "ticker": ["OLDCO", "NEWCO"],
        }
    )
    targets = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2012-01-03", "2015-05-29", "2016-01-04"]),
        }
    )
    out = uv.attach_ticker(targets, names)
    assert out["ticker"].tolist() == ["OLDCO", "OLDCO", "NEWCO"]


def test_audit_flags_simultaneous_ticker_reuse():
    """Two different companies trading under the same ticker at once.

    This is the case that silently splices two return series into one, and the
    reason PERMNO rather than ticker has to be the join key.
    """
    names = pd.DataFrame(
        {
            "permno": [10001, 10002, 10003],
            "effective_from": pd.to_datetime(["2010-01-01", "2020-01-01", "2021-01-01"]),
            "effective_to": pd.to_datetime([None, None, None]),
            "ticker": ["ABC", "ABC", "XYZ"],
        }
    )
    verdict = uv.audit_ticker_ambiguity(names)

    assert verdict["n_tickers_bound_to_multiple_permnos"] == 1
    assert verdict["n_simultaneous_ticker_collisions"] == 1
    assert verdict["verdict"].startswith("A ticker-keyed pipeline is UNSOUND")


def test_audit_distinguishes_rename_from_reuse():
    names = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "effective_from": pd.to_datetime(["2010-01-01", "2015-01-01"]),
            "effective_to": pd.to_datetime(["2014-12-31", None]),
            "ticker": ["OLD", "NEW"],
        }
    )
    verdict = uv.audit_ticker_ambiguity(names)
    assert verdict["n_permnos_with_multiple_tickers"] == 1
    assert verdict["n_simultaneous_ticker_collisions"] == 0


# ---------------------------------------------------------------------------
# Membership interval merging
# ---------------------------------------------------------------------------


def test_membership_intervals_are_merged_when_touching():
    frame = pd.DataFrame(
        {
            "permno": [10001, 10001, 10002],
            "membership_start": pd.to_datetime(["2010-01-01", "2012-01-01", "2011-01-01"]),
            "membership_end": pd.to_datetime(["2011-12-31", "2013-12-31", None]),
            "is_current_member": [False, False, True],
        }
    )
    merged = uv._merge_intervals(frame)

    assert len(merged) == 2
    first = merged[merged["permno"] == 10001].iloc[0]
    assert first["membership_start"] == pd.Timestamp("2010-01-01")
    assert first["membership_end"] == pd.Timestamp("2013-12-31")


def test_membership_intervals_stay_separate_across_a_gap():
    frame = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "membership_start": pd.to_datetime(["2010-01-01", "2015-01-01"]),
            "membership_end": pd.to_datetime(["2011-12-31", "2016-12-31"]),
            "is_current_member": [False, False],
        }
    )
    merged = uv._merge_intervals(frame)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Point-in-time GVKEY -> PERMNO linkage
# ---------------------------------------------------------------------------


def test_permno_linkage_uses_the_link_in_force_on_the_disclosure_date():
    """Using today's primary link for a historical filing is look-ahead."""
    fundamentals = pd.DataFrame(
        {
            "gvkey": ["001234", "001234"],
            "datadate": pd.to_datetime(["2011-03-31", "2019-03-31"]),
            "available_date": pd.to_datetime(["2011-05-01", "2019-05-01"]),
            "saleq_q": [1.0, 2.0],
        }
    )
    link = pd.DataFrame(
        {
            "gvkey": ["001234", "001234"],
            "permno": [10001, 10002],
            "link_start": pd.to_datetime(["2010-01-01", "2015-01-01"]),
            "link_end": pd.to_datetime(["2014-12-31", None]),
        }
    )
    out, stats = fb.attach_permno_at_disclosure(fundamentals, link)

    assert out["permno"].tolist() == [10001, 10002]
    assert stats["n_rows_unmatched_drop"] == 0


def test_permno_linkage_drops_rows_outside_every_link_window():
    fundamentals = pd.DataFrame(
        {
            "gvkey": ["001234"],
            "available_date": pd.to_datetime(["2005-05-01"]),
            "saleq_q": [1.0],
        }
    )
    link = pd.DataFrame(
        {
            "gvkey": ["001234"],
            "permno": [10001],
            "link_start": pd.to_datetime(["2010-01-01"]),
            "link_end": pd.to_datetime(["2020-12-31"]),
        }
    )
    out, stats = fb.attach_permno_at_disclosure(fundamentals, link)
    assert len(out) == 0
    assert stats["n_rows_unmatched_drop"] == 1
