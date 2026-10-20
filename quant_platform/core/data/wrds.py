"""CRSP CIZ adapter and version-aware research factors.

``ticker`` is a compatibility column containing a stable PERMNO alias, never
an exchange ticker. Joins and saved artifacts also retain numeric PERMNO.
Dollar market caps/shares use CIZ's thousands conversion; volume is already
in shares. Embedded CIZ delisting returns must never be merged again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def company_market_caps(classes, links):
    """Daily company common-equity value in USD and dated CCM attribution.

    Include all CRSP common share classes, including non-index classes. Missing
    nonterminal class values invalidate the company denominator. Conflicting
    active GVKEY links suppress financial attribution, never choose an issuer
    by row order. No future PERMCO/class relationship is carried backwards.
    """
    if classes.duplicated(["date", "permno"]).any():
        raise ValueError("Duplicate historical share class")
    common = classes[classes.securitytype.eq("EQTY") & classes.securitysubtype.eq("COM")].copy()
    common["_cap"] = common.market_cap.where(common.market_cap.gt(0)) * 1000
    common["_unknown"] = common._cap.isna() & ~common.delisting_flag.eq("Y")
    groups = common.groupby(["date", "permco"], observed=True)
    out = groups.agg(n_share_classes=("permno", "size"), _unknown=("_unknown", "any"))
    out["company_market_cap"] = groups["_cap"].sum(min_count=1)
    out = out.reset_index()
    out.loc[out.pop("_unknown"), "company_market_cap"] = np.nan
    link = links.rename(columns={"permco": "ccm_permco"})
    candidates = common[["date", "permco", "permno"]].merge(link, on="permno")
    candidates = candidates[candidates.date.ge(candidates.link_start) &
                            (candidates.link_end.isna() | candidates.date.le(candidates.link_end))]
    if "linktype" in candidates:
        candidates = candidates[candidates.linktype.isin(["LC", "LU"])]
    if "linkprim" in candidates:
        candidates = candidates[candidates.linkprim.isin(["P", "C"])]
    if "ccm_permco" in candidates:
        candidates = candidates[candidates.ccm_permco.isna() | candidates.permco.eq(candidates.ccm_permco)]
    issuers = candidates.groupby(["date", "permco"]).gvkey.agg(["nunique", "first"]).reset_index()
    issuers["gvkey"] = issuers["first"].where(issuers["nunique"].eq(1))
    issuers["issuer_ambiguous"] = issuers["nunique"].gt(1)
    out = out.merge(issuers[["date", "permco", "gvkey", "issuer_ambiguous"]],
                    on=["date", "permco"], how="left", validate="one_to_one")
    out["issuer_ambiguous"] = out.issuer_ambiguous.eq(True)
    return out


def canonical_prices(raw, supplement=None):
    px = raw.copy().sort_values(["permno", "date"]).reset_index(drop=True)
    px["date"] = pd.to_datetime(px["date"])
    if px.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate CRSP PERMNO/date")
    if supplement is not None:
        extra = ["permno", "date", *[c for c in supplement if c not in px]]
        px = px.merge(supplement[extra], on=["permno", "date"], validate="one_to_one", how="left")
    px["ticker"] = "PERMNO:" + px["permno"].astype("int64").astype(str)
    px["close"] = px["price"].where(px["price"].gt(0))
    px["total_return"] = pd.to_numeric(px["ret"], errors="coerce")
    if (px["total_return"].dropna() < -1).any():
        raise ValueError("CRSP return below -100%")
    px["is_terminal"] = px["delisting_flag"].eq("Y")
    px["shares_outstanding"] *= 1000
    px["market_cap"] *= 1000
    # An index is for return-based signals, never share sizing. Missing return
    # observations remain NaN; factor windows below explicitly reject them.
    px["adj_close"] = (1 + px["total_return"]).groupby(px["permno"]).cumprod() * 100
    px["is_tradable"] = px["close"].gt(0) & px["volume"].gt(0) & ~px["is_terminal"]
    if "tradingstatusflg" in px:
        px["is_tradable"] &= px["tradingstatusflg"].eq("A")
    if "conditionaltype" in px:
        px["is_tradable"] &= px["conditionaltype"].eq("RW")
    px["data_source"] = "crsp_ciz"
    return px


def membership_panel(prices, membership, names=None, min_market_cap=50_000_000):
    spine = prices[["date", "permno", "ticker", "is_tradable", "market_cap"]].copy()
    intervals = membership.copy()
    joined = spine.merge(intervals[["permno", "membership_start", "membership_end"]], on="permno")
    joined = joined[joined["date"].ge(joined["membership_start"]) &
                    (joined["membership_end"].isna() | joined["date"].le(joined["membership_end"]))]
    if joined.duplicated(["date", "permno"]).any():
        raise ValueError("Overlapping membership intervals")
    joined["pass_all_filters"] = joined["is_tradable"] & joined["market_cap"].ge(min_market_cap)
    if names is not None:
        starts = names.groupby("permno")["effective_from"].min()
        joined["days_since_ipo"] = (joined["date"] - joined["permno"].map(starts)).dt.days
        joined["pass_all_filters"] &= joined["days_since_ipo"].ge(60)
    joined["membership_is_point_in_time"] = True
    joined["membership_source"] = "crsp_a_indexes.dsp500list_v2"
    return joined.sort_values(["date", "permno"]).reset_index(drop=True)


def fundamental_factors(events):
    """Ratios from already projected quarters/TTMs; amounts are USD millions.

    Price-scaled factors retain a numerator here, divided by contemporaneous
    CRSP market cap during daily projection. Missing F8 legs invalidate the
    score instead of being counted as failed tests.
    """
    out = events[["available_date", "permno", "datadate", "is_point_in_time", "link_end"]].copy()
    if "gvkey" in events:
        out["gvkey"] = events["gvkey"]
    def s(c):
        return pd.to_numeric(events.get(c, pd.Series(np.nan, index=events.index)), errors="coerce")
    def ratio(a, b):
        return a / b.where(b.abs().gt(1e-8))
    assets, ni, cfo = s("atq_q"), s("niq_ttm"), s("oancfy_ttm")
    sales, gp = s("saleq_ttm"), s("saleq_ttm") - s("cogsq_ttm")
    equity = s("ceqq_q").fillna(s("seqq_q") - s("pstkq_q").fillna(0))
    usd = events.get("curcdq", pd.Series(None, index=events.index)).eq("USD")
    out["EARNINGS_YIELD"] = (ni * 1e6).where(usd)
    out["BOOK_TO_MARKET"] = (equity * 1e6).where(usd)
    out["GROSS_PROFITABILITY"] = ratio(gp, assets)
    out["ASSET_GROWTH"] = s("atq_yoy_pct")
    out["ACCRUALS"] = ratio(ni - cfo, assets)
    old_assets = assets - s("atq_yoy_diff")
    old_ni = s("niq_ttm_prior_year")
    debt = s("dlcq_q") + s("dlttq_q")
    old_debt = debt - s("dlcq_yoy_diff") - s("dlttq_yoy_diff")
    old_sales = s("saleq_ttm_prior_year")
    old_gp = old_sales - s("cogsq_ttm_prior_year")
    # Compustat shares are unadjusted: CSHOQ * AJEXQ removes stock splits.
    # Compare both periods using adjustment factors eligible at this event.
    adjusted_shares = s("cshoq_q") * s("ajexq_q")
    old_adjusted_shares = ((s("cshoq_q") - s("cshoq_yoy_diff")) *
                           (s("ajexq_q") - s("ajexq_yoy_diff")))
    legs = pd.concat([
        ni, cfo, ratio(ni, assets) - ratio(old_ni, old_assets), cfo - ni,
        ratio(old_debt, old_assets) - ratio(debt, assets), old_adjusted_shares - adjusted_shares,
        ratio(gp, sales) - ratio(old_gp, old_sales),
        ratio(sales, assets) - ratio(old_sales, old_assets),
    ], axis=1)
    positive = legs.gt(0)
    positive.iloc[:, 5] = legs.iloc[:, 5].ge(0)
    out["PIOTROSKI_F8"] = positive.sum(axis=1).where(legs.notna().all(axis=1))
    return out


FACTOR_DIRECTIONS = {"MOM_12_1M": 1, "EARNINGS_YIELD": 1, "BOOK_TO_MARKET": 1,
                     "GROSS_PROFITABILITY": 1, "ASSET_GROWTH": -1,
                     "ACCRUALS": -1, "PIOTROSKI_F8": 1}


def research_factors(prices, events, max_report_age_days=550):
    """As-of join event features to prices, with one session of decision lag.

    A vintage expiry event carrying NaN clears that feature; no per-field ffill
    can resurrect an expired value. Restated research provenance stays false.
    """
    px = prices.sort_values(["date", "permno"]).reset_index(drop=True)
    feature_events = fundamental_factors(events).sort_values(["available_date", "permno"])
    if feature_events.duplicated(["available_date", "permno"]).any():
        raise ValueError("Multiple issuer events map to one PERMNO/date")
    company_mode = {"company_market_cap", "gvkey", "permco"} <= set(px)
    if company_mode:
        # Active issuer mapping is independently validated for EACH date,
        # rather than carrying a disclosure-day company relation forward.
        feature_events = feature_events.drop(columns=["permno", "link_end"]).drop_duplicates()
        if feature_events.duplicated(["available_date", "gvkey"]).any():
            raise ValueError("Conflicting issuer financial events")
        px["gvkey"] = px.gvkey.fillna("__NO_ISSUER__").astype(str)
        feature_events["gvkey"] = feature_events.gvkey.astype(str)
        out = pd.merge_asof(px, feature_events.sort_values("available_date"),
                            left_on="date", right_on="available_date", by="gvkey", direction="backward")
        out["link_end"] = pd.NaT  # dated issuer validity is enforced above
    else:
        if "permco" in px:
            raise ValueError("PERMCO data requires company_market_cap and dated GVKEY attribution")
        out = pd.merge_asof(px, feature_events, left_on="date", right_on="available_date",
                            by="permno", direction="backward")
    valid = ((out["date"] - out["datadate"]).dt.days.le(max_report_age_days) &
             (out["link_end"].isna() | out["date"].le(out["link_end"])))
    for col in FACTOR_DIRECTIONS:
        if col in out:
            out.loc[~valid, col] = np.nan
    for col in ("EARNINGS_YIELD", "BOOK_TO_MARKET"):
        denominator = out["company_market_cap"] if company_mode else out["market_cap"]
        out[col] /= denominator.where(denominator.gt(0))
    out = out.sort_values(["permno", "date"])
    g = out.groupby("permno", sort=False)
    out["MOM_12_1M"] = g["adj_close"].shift(21) / g["adj_close"].shift(252) - 1
    missing = out["total_return"].isna().groupby(out["permno"]).transform(
        lambda s: s.rolling(252, min_periods=252).sum()).gt(0)
    out.loc[missing, "MOM_12_1M"] = np.nan
    # All decisions use the preceding session's observable values.
    out[list(FACTOR_DIRECTIONS)] = out.groupby("permno")[list(FACTOR_DIRECTIONS)].shift(1)
    if company_mode:
        prior = out.groupby("permno")[["permco", "gvkey"]].shift(1)
        changed = prior.permco.ne(out.permco) | prior.gvkey.ne(out.gvkey)
        financial = [c for c in FACTOR_DIRECTIONS if c != "MOM_12_1M"]
        out.loc[changed, financial] = np.nan
    out["fundamentals_is_point_in_time"] = out.groupby("permno")["is_point_in_time"].shift(1).eq(True)
    identity = [c for c in ("permco", "gvkey", "company_market_cap", "n_share_classes", "issuer_ambiguous") if c in out]
    return out[["date", "permno", "ticker", *identity, "fundamentals_is_point_in_time", *FACTOR_DIRECTIONS]].sort_values(["date", "permno"]).reset_index(drop=True)
