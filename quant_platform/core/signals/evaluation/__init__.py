"""Signal evaluation module with type-specific scorecards.

Routes signals to appropriate evaluation metrics based on signal_type:
- cross_sectional: IC/RankIC/monotonicity/turnover
- time_series: hit rate/directional accuracy/Sharpe
- regime: overlay before/after comparison
- relative_value: cointegration/half-life/spread Sharpe
"""

from .router import evaluate_signal, get_scorecard
from .cross_sectional import CrossSectionalScorecard
from .time_series import TimeSeriesScorecard
from .regime import RegimeScorecard
from .relative_value import RelativeValueScorecard

__all__ = [
    "evaluate_signal",
    "get_scorecard",
    "CrossSectionalScorecard",
    "TimeSeriesScorecard",
    "RegimeScorecard",
    "RelativeValueScorecard",
]
