"""CN A-share data adapter.

Migrated from TradeX: AShareDataLoader + AShareDataCleaner + DataValidator,
unified under ``CNAShareAdapter`` with ``ticker`` column convention.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from quant_platform.core.data.calendar import trading_dates

logger = logging.getLogger(__name__)


# ======================================================================
# A-share market config
# ======================================================================

@dataclass
class AShareConfig:
    """CN A-share data pipeline configuration."""

    data_root: Path = Path(os.environ.get("TRADEX_DATA_ROOT", "./data"))
    parquet_dir: str = "parquet"
    tushare_token: Optional[str] = os.environ.get("TUSHARE_TOKEN", None)
    data_source: str = "baostock"
    lookback_years: int = 5
    price_limit_main: float = 0.10
    price_limit_star: float = 0.20
    settlement_cycle: int = 1
    lot_size: int = 100
    max_suspension_days: int = 20
    min_listing_days: int = 60

    @property
    def parquet_path(self) -> Path:
        return self.data_root / self.parquet_dir

    def ensure_dirs(self) -> None:
        self.parquet_path.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Data loader
# ======================================================================

class AShareDataLoader:
    """Fetch A-share daily bar data from Baostock / Tushare."""

    _BAO_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,"
        "turn,tradestatus,pctChg,isST"
    )

    def __init__(self, config: Optional[AShareConfig] = None, source: Optional[str] = None):
        self.cfg = config or AShareConfig()
        self.cfg.ensure_dirs()
        self.source = source or self.cfg.data_source
        self._bao_logged_in = False

    def fetch_daily(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Literal["hfq", "qfq", "none"] = "hfq",
    ) -> pd.DataFrame:
        start_date, end_date = self._default_dates(start_date, end_date)
        if self.source == "baostock":
            return self._bao_fetch_daily(ts_code, start_date, end_date, adjust)
        elif self.source == "tushare":
            return self._ts_fetch_daily(ts_code, start_date, end_date, adjust)
        raise ValueError(f"Unknown data source: {self.source}")

    def fetch_stock_list(self) -> pd.DataFrame:
        if self.source == "baostock":
            return self._bao_stock_list()
        elif self.source == "tushare":
            return self._ts_stock_list()
        raise ValueError(f"Unknown data source: {self.source}")

    def load_parquet(self, ts_code: str) -> pd.DataFrame:
        p = self.cfg.parquet_path / f"{ts_code}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        return pd.read_parquet(p)

    def load_all_parquet(self) -> pd.DataFrame:
        files = sorted(self.cfg.parquet_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {self.cfg.parquet_path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # ---- Baostock ----

    def _ensure_baostock(self):
        if not self._bao_logged_in:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
            self._bao_logged_in = True

    def _logout_baostock(self):
        if self._bao_logged_in:
            import baostock as bs
            bs.logout()
            self._bao_logged_in = False

    @staticmethod
    def _ts_to_bao(ts_code: str) -> str:
        code, exch = ts_code.split(".")
        return f"{exch.lower()}.{code}"

    @staticmethod
    def _bao_to_ts(bao_code: str) -> str:
        exch, code = bao_code.split(".")
        return f"{code}.{exch.upper()}"

    def _bao_stock_list(self) -> pd.DataFrame:
        import baostock as bs
        self._ensure_baostock()
        rs = bs.query_stock_industry()
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if df.empty:
            return pd.DataFrame(columns=["ts_code", "name", "exchange", "list_date"])
        df = df.drop_duplicates(subset=["code"])
        df["ts_code"] = df["code"].apply(self._bao_to_ts)
        df = df.rename(columns={"code_name": "name"})
        df["exchange"] = df["code"].apply(lambda c: "SSE" if c.startswith("sh") else "SZSE")
        df["list_date"] = ""
        return df[["ts_code", "name", "exchange", "list_date"]].reset_index(drop=True)

    def _bao_fetch_daily(self, ts_code, start, end, adjust) -> pd.DataFrame:
        import baostock as bs
        self._ensure_baostock()
        bao_code = self._ts_to_bao(ts_code)
        adj_map = {"hfq": "1", "qfq": "2", "none": "3"}
        rs = bs.query_history_k_data_plus(
            bao_code, self._BAO_FIELDS,
            start_date=start, end_date=end,
            frequency="d", adjustflag=adj_map.get(adjust, "1"),
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields)
        return self._standardize_baostock(df, ts_code)

    @staticmethod
    def _standardize_baostock(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        col_map = {
            "date": "date", "open": "open", "high": "high", "low": "low",
            "close": "close", "preclose": "pre_close", "volume": "volume",
            "amount": "amount", "turn": "turnover", "tradestatus": "trade_status",
            "pctChg": "pct_change", "isST": "is_st",
        }
        df = df.rename(columns=col_map)
        for c in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "pct_change"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["ts_code"] = ts_code
        df["trade_status"] = df["trade_status"].astype(str)
        df["is_suspended"] = df["trade_status"] != "1"
        df["is_st"] = df["is_st"].astype(str) == "1"
        df["adj_factor"] = 1.0
        desired = [
            "date", "ts_code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "pct_change",
            "trade_status", "is_suspended", "is_st", "adj_factor",
        ]
        return df[[c for c in desired if c in df.columns]].reset_index(drop=True)

    # ---- Tushare ----

    def _get_tushare_api(self):
        import tushare as ts
        if not self.cfg.tushare_token:
            raise ValueError("Tushare token not set.")
        return ts.pro_api(self.cfg.tushare_token)

    def _ts_stock_list(self) -> pd.DataFrame:
        pro = self._get_tushare_api()
        return pro.stock_basic(exchange="", list_status="L",
                               fields="ts_code,name,exchange,list_date")

    def _ts_fetch_daily(self, ts_code, start, end, adjust) -> pd.DataFrame:
        pro = self._get_tushare_api()
        start_fmt, end_fmt = start.replace("-", ""), end.replace("-", "")
        df = pro.daily(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt)
        if df is None or df.empty:
            return pd.DataFrame()
        adj = pro.adj_factor(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt)
        if adj is not None and not adj.empty:
            df = df.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
            df["adj_factor"] = df["adj_factor"].ffill().bfill()
            if adjust == "hfq":
                base = df["adj_factor"].iloc[-1]
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c] * df["adj_factor"] / base
            elif adjust == "qfq":
                base = df["adj_factor"].iloc[0]
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c] * df["adj_factor"] / base
        else:
            df["adj_factor"] = 1.0
        df = df.rename(columns={"trade_date": "date", "vol": "volume", "pct_chg": "pct_change"})
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["ts_code"] = ts_code
        df["turnover"] = 0.0
        df["is_suspended"] = False
        df["is_st"] = False
        df = df.sort_values("date").reset_index(drop=True)
        desired = [
            "date", "ts_code", "open", "high", "low", "close", "pre_close",
            "volume", "amount", "turnover", "pct_change",
            "is_suspended", "is_st", "adj_factor",
        ]
        return df[[c for c in desired if c in df.columns]]

    def _default_dates(self, start, end):
        today = dt.date.today()
        if end is None:
            end = today.strftime("%Y-%m-%d")
        if start is None:
            start = (today - dt.timedelta(days=365 * self.cfg.lookback_years)).strftime("%Y-%m-%d")
        return start.replace("/", "-"), end.replace("/", "-")


# ======================================================================
# Data cleaner
# ======================================================================

class AShareDataCleaner:
    """Clean and enrich A-share daily OHLCV DataFrames."""

    def __init__(self, config: Optional[AShareConfig] = None):
        self.cfg = config or AShareConfig()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy().sort_values("date").reset_index(drop=True)
        df = self._ensure_types(df)
        df = self._flag_suspensions(df)
        df = self._forward_fill_suspended(df)
        df = self._flag_price_limit(df)
        df = self._flag_new_listing(df)
        df = self._compute_returns(df)
        return df

    def clean_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return panel
        ticker_col = "ts_code" if "ts_code" in panel.columns else "ticker"
        return pd.concat(
            [self.clean(grp) for _, grp in panel.groupby(ticker_col)],
            ignore_index=True,
        )

    @staticmethod
    def _ensure_types(df):
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "pre_close", "volume", "amount", "turnover", "pct_change", "adj_factor"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _flag_suspensions(self, df):
        if "is_suspended" not in df.columns:
            df["is_suspended"] = df["volume"].isna() | (df["volume"] == 0)
        suspended = df["is_suspended"].astype(int)
        groups = (suspended != suspended.shift()).cumsum()
        df["suspension_streak"] = suspended.groupby(groups).cumsum()
        df["long_suspension"] = df["suspension_streak"] > self.cfg.max_suspension_days
        return df

    @staticmethod
    def _forward_fill_suspended(df):
        mask = df["is_suspended"]
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan).ffill()
        if "volume" in df.columns:
            df.loc[mask, "volume"] = 0
        if "amount" in df.columns:
            df.loc[mask, "amount"] = 0
        return df

    def _flag_price_limit(self, df):
        if "pct_change" not in df.columns:
            df["limit_up"] = False
            df["limit_down"] = False
            return df
        ticker_col = "ts_code" if "ts_code" in df.columns else "ticker"
        ts_code = df[ticker_col].iloc[0] if len(df) > 0 else ""
        code_num = ts_code.split(".")[0] if "." in ts_code else ts_code
        if code_num.startswith("688") or code_num.startswith("30"):
            limit = self.cfg.price_limit_star * 100
        else:
            limit = self.cfg.price_limit_main * 100
        df["limit_up"] = df["pct_change"] >= (limit - 0.5)
        df["limit_down"] = df["pct_change"] <= -(limit - 0.5)
        return df

    def _flag_new_listing(self, df):
        if len(df) == 0:
            df["is_new_listing"] = False
            return df
        first_date = df["date"].iloc[0]
        cutoff = first_date + pd.Timedelta(days=self.cfg.min_listing_days)
        df["is_new_listing"] = df["date"] < cutoff
        return df

    @staticmethod
    def _compute_returns(df):
        if "close" in df.columns:
            df["ret"] = df["close"].pct_change()
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            if "is_suspended" in df.columns:
                mask = df["is_suspended"]
                df.loc[mask, "ret"] = 0.0
                df.loc[mask, "log_ret"] = 0.0
        return df


# ======================================================================
# Data validator
# ======================================================================

class DataValidator:
    """Run data quality checks on cleaned DataFrames."""

    @staticmethod
    def check_single(df: pd.DataFrame, ts_code: str = "") -> Dict[str, any]:
        if df.empty:
            return {"ts_code": ts_code, "rows": 0, "valid": False}
        total = len(df)
        ticker_col = "ts_code" if "ts_code" in df.columns else "ticker"
        report = {
            "ts_code": ts_code or df[ticker_col].iloc[0] if ticker_col in df.columns else "?",
            "rows": total,
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}",
            "null_close_pct": df["close"].isna().mean() * 100,
            "zero_volume_pct": (df["volume"] == 0).mean() * 100 if "volume" in df.columns else None,
            "suspension_pct": df["is_suspended"].mean() * 100 if "is_suspended" in df.columns else None,
            "valid": True,
        }
        if "ret" in df.columns:
            report["extreme_return_days"] = int((df["ret"].abs() > 0.22).sum())
        return report

    @staticmethod
    def check_panel(panel: pd.DataFrame) -> pd.DataFrame:
        validator = DataValidator()
        ticker_col = "ts_code" if "ts_code" in panel.columns else "ticker"
        return pd.DataFrame([
            validator.check_single(grp, code)
            for code, grp in panel.groupby(ticker_col)
        ])


# ======================================================================
# Unified adapter
# ======================================================================

class CNAShareAdapter:
    """Unified data adapter for CN A-share market.

    Normalizes ``ts_code`` to ``ticker`` column for platform consistency.
    """

    EXCHANGE_SH = "XSHG"
    EXCHANGE_SZ = "XSHE"
    TICKER_COL = "ticker"

    def __init__(self, config: Optional[AShareConfig] = None):
        self.cfg = config or AShareConfig()
        self.loader = AShareDataLoader(config=self.cfg)
        self.cleaner = AShareDataCleaner(config=self.cfg)
        self.validator = DataValidator()

    @staticmethod
    def normalize_ticker(df: pd.DataFrame) -> pd.DataFrame:
        """Rename ``ts_code`` to ``ticker`` for platform consistency."""
        if "ts_code" in df.columns and "ticker" not in df.columns:
            df = df.rename(columns={"ts_code": "ticker"})
        return df

    def load_and_clean(self, ts_code: str) -> pd.DataFrame:
        raw = self.loader.load_parquet(ts_code)
        cleaned = self.cleaner.clean(raw)
        return self.normalize_ticker(cleaned)

    def load_panel(self) -> pd.DataFrame:
        raw = self.loader.load_all_parquet()
        cleaned = self.cleaner.clean_panel(raw)
        return self.normalize_ticker(cleaned)

    def get_trading_dates(self, start=None, end=None) -> pd.DatetimeIndex:
        return trading_dates(start, end, exchange=self.EXCHANGE_SH)
