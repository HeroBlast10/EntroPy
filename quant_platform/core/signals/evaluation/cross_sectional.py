"""Cross-sectional signal scorecard.

Metrics for traditional cross-sectional factors (momentum, volatility, liquidity):
- IC (Information Coefficient): Pearson correlation with forward returns
- RankIC: Spearman rank correlation with forward returns
- Monotonicity: Quintile return spread (Q5 - Q1)
- Turnover: Portfolio turnover rate
- Hit rate: % of periods with positive IC
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorMeta
from quant_platform.core.signals.effective import build_effective_signal


class CrossSectionalScorecard:
    """Scorecard for cross-sectional factors."""
    
    def evaluate(
        self,
        signal_df: pd.DataFrame,
        signal_col: str,
        meta: FactorMeta,
        prices: pd.DataFrame,
        forward_periods: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate cross-sectional signal.
        
        Parameters
        ----------
        signal_df : DataFrame [date, ticker, signal_col]
        signal_col : str
        meta : FactorMeta
        prices : DataFrame [date, ticker, close, ...]
        forward_periods : int
            Number of days for forward returns (default 1)
        
        Returns
        -------
        Dict with:
            - mean_ic: Average IC
            - mean_rank_ic: Average RankIC
            - ic_ir: IC information ratio (mean/std)
            - rank_ic_ir: RankIC information ratio
            - hit_rate: % of periods with positive IC
            - monotonicity: Q5-Q1 return spread
            - turnover: Average turnover
        """
        # Add forward returns
        prices = prices.copy()
        prices = prices.sort_values(["ticker", "date"])
        prices["fwd_ret"] = prices.groupby("ticker")["close"].pct_change(forward_periods).shift(-forward_periods)
        
        # Merge signal with forward returns
        eval_df = signal_df.merge(
            prices[["date", "ticker", "fwd_ret"]],
            on=["date", "ticker"],
            how="inner",
        )
        eval_df = build_effective_signal(eval_df, signal_col, direction=meta.direction)
        eval_df = eval_df.dropna(subset=[signal_col, "fwd_ret"])
        
        if len(eval_df) == 0:
            logger.warning("No valid data for cross-sectional evaluation")
            return self._empty_result()
        
        # Compute IC and RankIC by date
        ic_series = []
        rank_ic_series = []
        
        for date, group in eval_df.groupby("date"):
            if len(group) < 5:  # Need minimum stocks
                continue
            
            # IC (Pearson)
            ic = group[signal_col].corr(group["fwd_ret"], method="pearson")
            ic_series.append(ic)
            
            # RankIC (Spearman)
            rank_ic = group[signal_col].corr(group["fwd_ret"], method="spearman")
            rank_ic_series.append(rank_ic)
        
        if not ic_series:
            return self._empty_result()
        
        ic_series = pd.Series(ic_series)
        rank_ic_series = pd.Series(rank_ic_series)
        
        # Monotonicity: quintile analysis
        eval_df["quintile"] = eval_df.groupby("date")[signal_col].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
        )
        
        quintile_rets = eval_df.groupby("quintile")["fwd_ret"].mean()
        if len(quintile_rets) >= 2:
            monotonicity = quintile_rets.iloc[-1] - quintile_rets.iloc[0]
        else:
            monotonicity = np.nan
        
        # Turnover (simplified: % of stocks changed between periods)
        turnover = self._compute_turnover(eval_df, signal_col)
        
        return {
            "mean_ic": float(ic_series.mean()),
            "std_ic": float(ic_series.std()),
            "mean_rank_ic": float(rank_ic_series.mean()),
            "std_rank_ic": float(rank_ic_series.std()),
            "ic_ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0,
            "rank_ic_ir": float(rank_ic_series.mean() / rank_ic_series.std()) if rank_ic_series.std() > 0 else 0.0,
            "hit_rate": float((ic_series > 0).mean()),
            "monotonicity": float(monotonicity) if not np.isnan(monotonicity) else 0.0,
            "turnover": float(turnover),
            "n_periods": len(ic_series),
        }
    
    def _compute_turnover(self, eval_df: pd.DataFrame, signal_col: str) -> float:
        """Compute average turnover (% of stocks changed in top quintile)."""
        # Get top quintile stocks for each date
        top_quintile_by_date = {}
        
        for date, group in eval_df.groupby("date"):
            # Top 20% by signal
            threshold = group[signal_col].quantile(0.8)
            top_stocks = set(group[group[signal_col] >= threshold]["ticker"])
            top_quintile_by_date[date] = top_stocks
        
        # Compute turnover between consecutive dates
        dates = sorted(top_quintile_by_date.keys())
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_stocks = top_quintile_by_date[dates[i-1]]
            curr_stocks = top_quintile_by_date[dates[i]]
            
            if len(prev_stocks) > 0:
                # Turnover = stocks that changed / total stocks
                changed = len(prev_stocks.symmetric_difference(curr_stocks))
                turnover = changed / len(prev_stocks)
                turnovers.append(turnover)
        
        return np.mean(turnovers) if turnovers else 0.0
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "ic_ir": 0.0,
            "rank_ic_ir": 0.0,
            "hit_rate": 0.0,
            "monotonicity": 0.0,
            "turnover": 0.0,
            "n_periods": 0,
        }
