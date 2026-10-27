"""Training-only, contemporaneous cross-sectional labels and causal volatility.

Future returns are labels, never predictors. Scaling uses returns strictly
before each decision. Neutralization is applied AFTER per-stock scaling so
heterogeneous denominators cannot undo industry/size neutrality.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.core.data.industry import neutralize_industry_size


@dataclass(frozen=True)
class TargetConfig:
    mode: str = "industry_size_vol"
    volatility_window: int = 60
    volatility_min_periods: int = 40
    daily_volatility_floor: float = .005
    clip: float = 3.0
    min_industry_names: int = 5

    def __post_init__(self):
        if self.mode not in ("cross_sectional", "industry_size_vol"):
            raise ValueError("Unknown target mode")
        if not 2 <= self.volatility_min_periods <= self.volatility_window:
            raise ValueError("Invalid volatility history")
        if not np.isfinite(self.daily_volatility_floor) or self.daily_volatility_floor <= 0:
            raise ValueError("Volatility floor must be finite and positive")
        if not np.isfinite(self.clip) or self.clip <= 0 or self.min_industry_names < 3:
            raise ValueError("Invalid target bounds")


def decision_volatility(prices, config=TargetConfig()):
    """Shared-calendar rolling daily standard deviation, shifted one session.

    Missing ordinary returns remain missing observations; there is no forward
    fill over an absence. Market holidays use the shared observed calendar.
    """
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate price keys")
    wide = prices.pivot(index="date", columns="ticker", values="total_return").sort_index()
    wide = wide.replace([np.inf, -np.inf], np.nan)
    volatility = wide.rolling(config.volatility_window, min_periods=config.volatility_min_periods).std().shift(1)
    result = volatility.rename_axis(index="date", columns="ticker").stack(dropna=False).rename("label_exante_vol").reset_index()
    return result


def neutral_scaled_targets(panel, context, *, horizon, config=TargetConfig()):
    """Bounded industry/size-neutral targets, preserving all inference rows.

    Winsorization itself can reintroduce exposure. Project once more after
    clipping and apply only a common, non-increasing daily bound adjustment.
    Final targets are orthogonal to industry dummies and lagged log size,
    including after robustification; their variance need not be exactly one.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if panel.duplicated(["date", "ticker"]).any() or context.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate target/context keys")
    frame = panel[["date", "ticker", "forward_return"]].merge(
        context[["date", "ticker", "decision_industry", "decision_log_size", "label_exante_vol"]],
        on=["date", "ticker"], how="left", validate="one_to_one")
    if config.mode == "cross_sectional":
        value = frame.forward_return
        groups = value.groupby(frame.date)
        target = ((value - groups.transform("mean")) / groups.transform("std", ddof=0).replace(0, np.nan)).clip(-config.clip, config.clip)
    else:
        volatility = frame.label_exante_vol.clip(lower=config.daily_volatility_floor) * np.sqrt(horizon)
        value = frame.forward_return / volatility

        def project(values):
            return neutralize_industry_size(values, frame.date, frame.decision_industry,
                                            frame.decision_log_size, min_group=config.min_industry_names)

        residual = project(value)
        sigma = residual.groupby(frame.date).transform("std", ddof=0).replace(0, np.nan)
        robust = (residual / sigma).clip(-config.clip, config.clip)
        target = project(robust)
        maximum = target.abs().groupby(frame.date).transform("max")
        target *= np.minimum(1, config.clip / maximum.replace(0, np.nan)).fillna(1)
    return pd.DataFrame({"date": frame.date, "ticker": frame.ticker,
                         "label_neutral_target": target.astype("float32"),
                         "label_exante_vol": frame.label_exante_vol})
