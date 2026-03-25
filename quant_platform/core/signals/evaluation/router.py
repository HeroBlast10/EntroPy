"""Evaluation router that dispatches signals to appropriate scorecards."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from loguru import logger

from quant_platform.core.signals.base import FactorMeta


def get_scorecard(signal_type: str):
    """Get the appropriate scorecard class for a signal type.
    
    Parameters
    ----------
    signal_type : str
        One of: "cross_sectional", "time_series", "regime", "relative_value"
    
    Returns
    -------
    Scorecard class
    """
    from .cross_sectional import CrossSectionalScorecard
    from .time_series import TimeSeriesScorecard
    from .regime import RegimeScorecard
    from .relative_value import RelativeValueScorecard
    
    scorecards = {
        "cross_sectional": CrossSectionalScorecard,
        "time_series": TimeSeriesScorecard,
        "regime": RegimeScorecard,
        "relative_value": RelativeValueScorecard,
    }
    
    if signal_type not in scorecards:
        logger.warning(
            "Unknown signal_type '{}', defaulting to cross_sectional scorecard",
            signal_type,
        )
        return CrossSectionalScorecard
    
    return scorecards[signal_type]


def evaluate_signal(
    signal_df: pd.DataFrame,
    signal_col: str,
    meta: FactorMeta,
    prices: pd.DataFrame,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a signal using the appropriate scorecard based on its type.
    
    Parameters
    ----------
    signal_df : DataFrame
        Signal values with [date, ticker, signal_col]
    signal_col : str
        Name of the signal column
    meta : FactorMeta
        Signal metadata (contains signal_type)
    prices : DataFrame
        Price data for computing returns/spreads
    **kwargs
        Additional arguments passed to scorecard
    
    Returns
    -------
    Dict with evaluation metrics appropriate for the signal type
    """
    scorecard_class = get_scorecard(meta.signal_type)
    scorecard = scorecard_class()
    
    logger.info(
        "Evaluating signal '{}' (type: {}) with {}",
        signal_col,
        meta.signal_type,
        scorecard_class.__name__,
    )
    
    return scorecard.evaluate(
        signal_df=signal_df,
        signal_col=signal_col,
        meta=meta,
        prices=prices,
        **kwargs,
    )
