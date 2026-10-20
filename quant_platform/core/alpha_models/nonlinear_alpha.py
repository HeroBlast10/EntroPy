"""Non-linear cross-sectional alpha models with purged walk-forward OOF.

The implementation deliberately separates model fitting from research
evaluation.  Both LightGBM and the small PyTorch MLP consume the same
point-in-time, cross-sectionally ranked features and are evaluated on the same
out-of-fold rows.

The main leakage guard is date-based purging: a training label at date ``t``
is only admitted when its complete forward-return horizon precedes the first
validation date.  Splitting flattened security rows is intentionally avoided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Literal, Optional

import numpy as np
import pandas as pd

from .feature_schema import BUILTIN_FEATURES, is_metadata
from quant_platform.core.data.returns import forward_total_returns


ModelKind = Literal["lightgbm", "mlp"]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration shared by all non-linear alpha backends."""

    horizon: int = 1
    min_train_days: int = 756
    validation_days: int = 252
    n_splits: int = 4
    purge_days: Optional[int] = None
    target_clip: float = 3.0
    seed: int = 20260726
    min_cross_section: int = 20
    max_train_rows: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("horizon", "min_train_days", "validation_days", "n_splits"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.purge_days is not None and self.purge_days < self.horizon:
            raise ValueError("purge_days must be at least the forward-return horizon")
        if self.min_cross_section < 3:
            raise ValueError("min_cross_section must be at least 3")

    @property
    def effective_purge_days(self) -> int:
        return self.horizon if self.purge_days is None else self.purge_days

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["effective_purge_days"] = self.effective_purge_days
        return result


@dataclass(frozen=True)
class LightGBMConfig:
    """Conservative LightGBM defaults inspired by the EG take-home model."""

    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_child_samples: int = 1000
    n_estimators: int = 120
    reg_alpha: float = 1.0
    reg_lambda: float = 20.0
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    n_jobs: int = -1


@dataclass(frozen=True)
class MLPConfig:
    """A compact tabular MLP suitable for factor panels."""

    hidden_width: int = 64
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 3e-3
    epochs: int = 3
    batch_size: int = 8192
    inference_batch_size: int = 32768


def select_usable_features(
    factors: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    *,
    min_coverage: float = 0.80,
    min_daily_nonconstant_fraction: float = 0.05,
    min_daily_observations: int = 3,
    max_dominant_value_fraction: float = 0.95,
    return_diagnostics: bool = False,
) -> list[str] | tuple[list[str], pd.DataFrame]:
    """Select numeric features with adequate, cross-sectionally useful history.

    A feature that changes only through time but is constant within every
    daily cross-section cannot rank securities.  The daily non-constant gate
    prevents such columns from entering a cross-sectional learner merely
    because they have global variance.
    """

    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    if not 0.0 <= min_daily_nonconstant_fraction <= 1.0:
        raise ValueError("min_daily_nonconstant_fraction must be between 0 and 1")
    if min_daily_observations < 2:
        raise ValueError("min_daily_observations must be at least 2")
    if not 0.0 < max_dominant_value_fraction <= 1.0:
        raise ValueError("max_dominant_value_fraction must be in (0, 1]")

    # Custom features require an explicit declaration. Numeric metadata does
    # not become predictive data just because it passes variance checks.
    candidates = list(feature_cols) if feature_cols is not None else [
        c for c in factors.columns if c in BUILTIN_FEATURES
    ]
    forbidden = [c for c in candidates if is_metadata(c)]
    if forbidden:
        raise ValueError(f"Identifiers/provenance cannot be prediction features: {forbidden}")
    selected: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for col in candidates:
        reason: Optional[str] = None
        if col not in factors.columns:
            reason = "missing_column"
        elif not pd.api.types.is_numeric_dtype(factors[col]):
            reason = "non_numeric"
        if reason is not None:
            diagnostics.append({"feature": col, "selected": False, "reason": reason})
            continue
        values = factors[col]
        coverage = float(values.notna().mean())
        if coverage < min_coverage:
            diagnostics.append({
                "feature": col,
                "selected": False,
                "reason": "insufficient_coverage",
                "coverage": coverage,
            })
            continue
        if values.nunique(dropna=True) <= 1:
            diagnostics.append({
                "feature": col,
                "selected": False,
                "reason": "globally_constant",
                "coverage": coverage,
            })
            continue
        nonmissing = values.dropna()
        dominant_fraction = float(
            nonmissing.value_counts(normalize=True, dropna=False).iloc[0]
        )
        if dominant_fraction > max_dominant_value_fraction:
            diagnostics.append({
                "feature": col,
                "selected": False,
                "reason": "dominant_value",
                "coverage": coverage,
                "dominant_value_fraction": dominant_fraction,
            })
            continue
        useful_fraction = np.nan
        if "date" in factors.columns:
            daily = factors[["date", col]].groupby("date", sort=False, observed=True)[col]
            daily_count = daily.count()
            eligible = daily_count >= min_daily_observations
            if not bool(eligible.any()):
                diagnostics.append({
                    "feature": col,
                    "selected": False,
                    "reason": "insufficient_daily_observations",
                    "coverage": coverage,
                    "dominant_value_fraction": dominant_fraction,
                })
                continue
            daily_nonconstant = daily.nunique(dropna=True).reindex(daily_count.index) > 1
            useful_fraction = float(daily_nonconstant.loc[eligible].mean())
            if useful_fraction < min_daily_nonconstant_fraction:
                diagnostics.append({
                    "feature": col,
                    "selected": False,
                    "reason": "cross_sectionally_constant",
                    "coverage": coverage,
                    "dominant_value_fraction": dominant_fraction,
                    "daily_nonconstant_fraction": useful_fraction,
                })
                continue
        selected.append(col)
        diagnostics.append({
            "feature": col,
            "selected": True,
            "reason": "selected",
            "coverage": coverage,
            "dominant_value_fraction": dominant_fraction,
            "daily_nonconstant_fraction": useful_fraction,
        })
    if not selected:
        raise ValueError("No usable feature columns passed coverage/variance checks")
    if return_diagnostics:
        return selected, pd.DataFrame(diagnostics)
    return selected


def _daily_standardize(values: pd.Series, dates: pd.Series, clip: float) -> pd.Series:
    grouped = values.groupby(dates, sort=False, observed=True)
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0)).replace(0.0, np.nan)
    return ((values - mean) / std).clip(-clip, clip)


def _daily_rank_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Map each feature to [-1, 1] using only its contemporaneous cross-section."""

    ranked = frame.groupby("date", sort=False, observed=True)[feature_cols].rank(
        pct=True,
        method="average",
    )
    return (2.0 * ranked - 1.0).astype("float32")


def filter_tradable_universe(
    rows: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    context: str = "factor rows",
    eligibility_lag: int = 1,
) -> pd.DataFrame:
    """Keep only securities eligible using previously known information.

    Universe rows are end-of-day observations, so the default shifts them one
    trading session before joining the full ``(date, ticker)`` key.  This
    filter must be applied before cross-sectional feature ranking; otherwise
    an untradable security can change every tradable security's rank even if
    it is removed later.
    """

    keys = ["date", "ticker"]
    missing_rows = set(keys) - set(rows.columns)
    missing_universe = {"date", "ticker", "pass_all_filters"} - set(universe.columns)
    if missing_rows:
        raise ValueError(f"{context} is missing columns: {sorted(missing_rows)}")
    if missing_universe:
        raise ValueError(
            "universe is missing eligibility columns: "
            f"{sorted(missing_universe)}"
        )

    work = rows.copy()
    from quant_platform.core.data.universe import lag_universe_eligibility

    eligibility = lag_universe_eligibility(
        universe[["date", "ticker", "pass_all_filters"]],
        periods=eligibility_lag,
    )
    work["date"] = pd.to_datetime(work["date"])
    eligibility["date"] = pd.to_datetime(eligibility["date"])
    if work.duplicated(keys).any():
        raise ValueError(f"{context} contains duplicate date/ticker keys")
    eligible_keys = eligibility.loc[
        eligibility["pass_all_filters"].eq(True), keys  # noqa: E712
    ]
    return work.merge(
        eligible_keys,
        on=keys,
        how="inner",
        validate="one_to_one",
    )


def build_learning_panel(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    feature_cols: Optional[Iterable[str]] = None,
    horizon: int = 1,
    target_clip: float = 3.0,
    min_cross_section: int = 20,
    min_feature_coverage: float = 0.80,
    missing_terminal_return: str = "raise",
) -> tuple[pd.DataFrame, list[str]]:
    """Build point-in-time ranked features and forward-return labels.

    Returns a frame with ``date``, ``ticker``, selected feature columns,
    ``forward_return`` and ``target``.  Every point-in-time tradable factor row
    is retained so that a future suspension or missing quote cannot alter the
    contemporaneous prediction universe.  Missing feature ranks are represented
    by zero (the daily cross-sectional midpoint), while labels are never
    imputed.  Rows without an eligible label therefore have null
    ``forward_return``/``target`` values and remain available for inference.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required_keys = {"date", "ticker"}
    if not required_keys.issubset(factors.columns) or not required_keys.issubset(prices.columns):
        raise ValueError("factors and prices must contain date and ticker")

    factors = filter_tradable_universe(
        factors,
        universe,
        context="factors",
    )
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("prices contains duplicate date/ticker keys")
    features = select_usable_features(
        factors,
        feature_cols,
        min_coverage=min_feature_coverage,
    )

    # Rank features on the complete point-in-time signal universe.  Label
    # availability is future information and must not change contemporaneous
    # feature values for the other securities.
    ranked_factors = factors[["date", "ticker", *features]].copy()
    ranked_factors[features] = _daily_rank_features(ranked_factors, features).fillna(0.0)

    # Define horizon on the shared trading calendar, not as "the next observed
    # row for this ticker".  A suspension or missing quote therefore produces
    # a missing label instead of silently extending the holding period.
    price_panel = forward_total_returns(prices, horizon,
                                        missing_terminal_return=missing_terminal_return)

    panel = ranked_factors.merge(
        price_panel,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    counts = panel.groupby("date", observed=True)["forward_return"].transform("count")
    # ``min_cross_section`` determines whether labels from a date are suitable
    # for training/evaluation; it must not determine which securities receive
    # a prediction.  Mask labels on undersized dates instead of deleting rows.
    panel["forward_return"] = panel["forward_return"].where(
        counts >= min_cross_section
    )
    panel["target"] = _daily_standardize(
        panel["forward_return"],
        panel["date"],
        target_clip,
    ).astype("float32")
    return panel, features


def expanding_date_folds(
    dates: Iterable[pd.Timestamp],
    config: WalkForwardConfig,
) -> Iterator[dict[str, Any]]:
    """Yield contiguous, purged expanding folds keyed by unique trading date."""

    unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates).unique())).sort_values()
    required = (
        config.min_train_days
        + config.effective_purge_days
        + config.n_splits * config.validation_days
    )
    if len(unique_dates) < required:
        raise ValueError(
            f"Need at least {required} unique dates for this walk-forward setup; "
            f"received {len(unique_dates)}"
        )

    first_validation = len(unique_dates) - config.n_splits * config.validation_days
    for fold_number in range(config.n_splits):
        validation_start = first_validation + fold_number * config.validation_days
        validation_end = validation_start + config.validation_days
        train_end = validation_start - config.effective_purge_days
        train_dates = unique_dates[:train_end]
        validation_dates = unique_dates[validation_start:validation_end]
        if len(train_dates) < config.min_train_days:
            raise ValueError("Insufficient training history after purging")
        yield {
            "fold": fold_number + 1,
            "train_dates": train_dates,
            "validation_dates": validation_dates,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "validation_start": validation_dates[0],
            "validation_end": validation_dates[-1],
            "purge_days": config.effective_purge_days,
        }


def _sample_training_rows(indices: np.ndarray, limit: Optional[int], seed: int) -> np.ndarray:
    if limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return indices[np.sort(rng.choice(len(indices), size=limit, replace=False))]


def _fit_predict_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: LightGBMConfig,
    seed: int,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.Series]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("LightGBM alpha requires `pip install lightgbm`") from exc

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="l2",
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        n_estimators=config.n_estimators,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        colsample_bytree=config.feature_fraction,
        subsample=config.bagging_fraction,
        subsample_freq=config.bagging_freq,
        max_bin=63,
        random_state=seed,
        n_jobs=config.n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_validation).astype("float32")
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=feature_names,
        dtype=float,
    )
    return prediction, importance


def _fit_predict_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: MLPConfig,
    seed: int,
    feature_names: list[str],
) -> tuple[np.ndarray, pd.Series]:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("MLP alpha requires `pip install torch`") from exc

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _TabularMLP(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_features, config.hidden_width),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_width, config.hidden_width),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_width, 1),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values).squeeze(-1)

    model = _TabularMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(x_train.astype("float32", copy=False)),
        torch.from_numpy(y_train.astype("float32", copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    loss_function = nn.MSELoss()
    model.train()
    for _ in range(config.epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            loss = loss_function(model(x_batch), y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    model.eval()
    prediction_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_validation), config.inference_batch_size):
            batch = torch.from_numpy(
                x_validation[start : start + config.inference_batch_size].astype(
                    "float32",
                    copy=False,
                )
            ).to(device)
            prediction_parts.append(model(batch).cpu().numpy())
    first_layer = model.network[0].weight.detach().abs().mean(dim=0).cpu().numpy()
    importance = pd.Series(first_layer, index=feature_names, dtype=float)
    return np.concatenate(prediction_parts).astype("float32"), importance


def _daily_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    # Float32 reductions depend on security ordering at roughly 1e-7.
    # Use a sorted float64 cross-section so renumbering IDs changes neither
    # normalization nor the model's corresponding predictions.
    values = values.astype("float64")
    grouped = values.groupby(dates, sort=False, observed=True)
    mean = grouped.transform(lambda x: np.sort(x.to_numpy()).mean())
    std = grouped.transform(lambda x: np.sort(x.to_numpy()).std(ddof=0)).replace(0.0, np.nan)
    return ((values - mean) / std).fillna(0.0)


def run_walk_forward_oof(
    panel: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    model: ModelKind,
    walk_forward: Optional[WalkForwardConfig] = None,
    lightgbm: Optional[LightGBMConfig] = None,
    mlp: Optional[MLPConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Train a model per fold and return strictly out-of-fold alpha scores."""

    wf = walk_forward or WalkForwardConfig()
    feature_names = list(feature_cols)
    if any(is_metadata(c) for c in feature_names):
        raise ValueError("Identifiers/provenance cannot be prediction features")
    missing = set(["date", "ticker", "forward_return", "target", *feature_names]) - set(panel)
    if missing:
        raise ValueError(f"Learning panel is missing columns: {sorted(missing)}")
    if model not in {"lightgbm", "mlp"}:
        raise ValueError("model must be 'lightgbm' or 'mlp'")

    date_values = pd.to_datetime(panel["date"])
    target_is_valid = panel["target"].notna().to_numpy()
    parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for fold in expanding_date_folds(date_values, wf):
        train_universe_mask = date_values.isin(fold["train_dates"]).to_numpy()
        # Labels are required for fitting only.  Validation rows deliberately
        # remain independent of future-return availability.
        train_mask = train_universe_mask & target_is_valid
        if "label_end" in panel:
            train_mask &= pd.to_datetime(panel.label_end).lt(fold["validation_start"]).to_numpy()
        validation_mask = date_values.isin(fold["validation_dates"]).to_numpy()
        # Bagging and capped sampling depend on row order. Canonicalize using
        # training observations, never arbitrary security numbering, so an ID
        # permutation cannot change random samples or fitted predictions.
        candidates = np.flatnonzero(train_mask)
        ordering = panel.iloc[candidates][["date", *feature_names, "target"]].reset_index(drop=True)
        canonical = ordering.sort_values(["date", *feature_names, "target"], kind="mergesort").index
        train_indices = _sample_training_rows(
            candidates[canonical],
            wf.max_train_rows,
            wf.seed + fold["fold"],
        )
        validation_indices = np.flatnonzero(validation_mask)
        prediction_order = panel.iloc[validation_indices][["date", *feature_names]].reset_index(drop=True)
        canonical_validation = prediction_order.sort_values(["date", *feature_names], kind="mergesort").index
        validation_indices = validation_indices[canonical_validation]
        if len(train_indices) == 0:
            raise ValueError(
                f"Fold {fold['fold']} has no non-null training targets"
            )
        x_train = panel.iloc[train_indices][feature_names].to_numpy(dtype=np.float32)
        y_train = panel.iloc[train_indices]["target"].to_numpy(dtype=np.float32)
        x_validation = panel.iloc[validation_indices][feature_names].to_numpy(dtype=np.float32)
        if model == "lightgbm":
            prediction, importance = _fit_predict_lightgbm(
                x_train,
                y_train,
                x_validation,
                lightgbm or LightGBMConfig(),
                wf.seed + fold["fold"],
                feature_names,
            )
        else:
            prediction, importance = _fit_predict_mlp(
                x_train,
                y_train,
                x_validation,
                mlp or MLPConfig(),
                wf.seed + fold["fold"],
                feature_names,
            )

        fold_result = panel.iloc[validation_indices][
            ["date", "ticker", "forward_return", "target", *[
                c for c in ("label_end", "label_terminal_settled", "label_terminal_imputed") if c in panel]]
        ].copy()
        fold_result["prediction_raw"] = prediction
        fold_result["prediction"] = _daily_zscore(
            fold_result["prediction_raw"],
            fold_result["date"],
        ).astype("float32")
        fold_result["fold"] = fold["fold"]
        fold_result["model"] = model
        parts.append(fold_result)
        importance_parts.append(
            importance.rename("importance").rename_axis("feature").reset_index().assign(
                fold=fold["fold"],
                model=model,
            )
        )
        metadata.append({
            "fold": fold["fold"],
            "model": model,
            "train_start": str(fold["train_start"].date()),
            "train_end": str(fold["train_end"].date()),
            "validation_start": str(fold["validation_start"].date()),
            "validation_end": str(fold["validation_end"].date()),
            "purge_days": fold["purge_days"],
            "train_rows_available": int(train_mask.sum()),
            "train_universe_rows": int(train_universe_mask.sum()),
            "train_rows_used": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "validation_rows_labeled": int(
                panel.iloc[validation_indices]["target"].notna().sum()
            ),
        })

    oof = pd.concat(parts, ignore_index=True)
    importance_frame = pd.concat(importance_parts, ignore_index=True)
    return oof, importance_frame, metadata


def daily_rank_ic(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    return_col: str = "forward_return",
) -> pd.Series:
    """Compute daily Spearman IC with a minimum of five valid securities.

    Spearman correlation is Pearson correlation of ranks.  Computing the group
    moments with ``bincount`` is materially faster than invoking scipy once per
    date and naturally treats a constant daily cross-section as undefined.
    """

    work = frame[["date", signal_col, return_col]].dropna().copy()
    if work.empty:
        return pd.Series(dtype=float, name=signal_col)
    work["_x_rank"] = work.groupby("date", sort=False, observed=True)[signal_col].rank()
    work["_y_rank"] = work.groupby("date", sort=False, observed=True)[return_col].rank()
    codes, unique_dates = pd.factorize(work["date"], sort=True)
    n_groups = len(unique_dates)
    x = work["_x_rank"].to_numpy(dtype=float)
    y = work["_y_rank"].to_numpy(dtype=float)
    n = np.bincount(codes, minlength=n_groups).astype(float)
    sx = np.bincount(codes, weights=x, minlength=n_groups)
    sy = np.bincount(codes, weights=y, minlength=n_groups)
    sxx = np.bincount(codes, weights=x * x, minlength=n_groups)
    syy = np.bincount(codes, weights=y * y, minlength=n_groups)
    sxy = np.bincount(codes, weights=x * y, minlength=n_groups)
    numerator = n * sxy - sx * sy
    denominator = np.sqrt(
        np.maximum(n * sxx - sx * sx, 0.0)
        * np.maximum(n * syy - sy * sy, 0.0)
    )
    values = np.full(n_groups, np.nan)
    valid = (n >= 5) & (denominator > 0)
    values[valid] = numerator[valid] / denominator[valid]
    return pd.Series(values, index=pd.DatetimeIndex(unique_dates), name=signal_col)


def evaluate_alpha(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    return_col: str = "forward_return",
    horizon: int = 1,
) -> tuple[dict[str, float], pd.Series]:
    """Evaluate IC and an equal-weight top-minus-bottom quintile spread."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"date", "ticker", signal_col, return_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation frame is missing {sorted(required - set(frame.columns))}")
    work = frame[list(required)].dropna(subset=[signal_col, return_col]).copy()
    ic = daily_rank_ic(work, signal_col, return_col=return_col).dropna()
    rank = work.groupby("date", observed=True)[signal_col].rank(pct=True)
    work["_top"] = rank >= 0.8
    work["_bottom"] = rank <= 0.2
    top = work.loc[work["_top"]].groupby("date", observed=True)[return_col].mean()
    bottom = work.loc[work["_bottom"]].groupby("date", observed=True)[return_col].mean()
    spread = (top - bottom).dropna()
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    spread_std = float(spread.std(ddof=1)) if len(spread) > 1 else np.nan
    ranked = work.assign(_rank=rank).sort_values(["ticker", "date"])
    rank_turnover = ranked.groupby("ticker", sort=False)["_rank"].diff().abs().mean()
    portfolio_turnover = _mean_quintile_portfolio_turnover(work, rank)
    iid_t_stat = (
        float(ic.mean() / (std_ic / np.sqrt(len(ic))))
        if np.isfinite(std_ic) and std_ic > 0
        else np.nan
    )
    hac_t_stat = _newey_west_mean_t_stat(ic, min_lag=horizon - 1)
    metrics = {
        "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic_std": std_ic,
        "rank_ic_ir_ann": (
            float(ic.mean() / std_ic * np.sqrt(252 / horizon))
            if np.isfinite(std_ic) and std_ic > 0
            else np.nan
        ),
        "rank_ic_t_stat": hac_t_stat,
        "rank_ic_t_stat_hac": hac_t_stat,
        "rank_ic_t_stat_iid": iid_t_stat,
        "ic_hit_rate": float((ic > 0).mean()) if len(ic) else np.nan,
        "ls_daily_mean": float(spread.mean()) if len(spread) else np.nan,
        "ls_ann_return": (
            float(spread.mean() * 252 / horizon)
            if len(spread)
            else np.nan
        ),
        "ls_gross_sharpe": (
            float(spread.mean() / spread_std * np.sqrt(252 / horizon))
            if np.isfinite(spread_std) and spread_std > 0
            else np.nan
        ),
        "mean_rank_turnover": float(rank_turnover),
        "mean_portfolio_turnover": portfolio_turnover,
        "return_horizon_days": float(horizon),
        "overlapping_forward_returns": bool(horizon > 1),
        "n_days": float(len(ic)),
        "n_rows": float(len(work)),
    }
    return metrics, ic


def _newey_west_mean_t_stat(
    values: pd.Series,
    max_lag: Optional[int] = None,
    *,
    min_lag: int = 0,
) -> float:
    """Newey-West t-statistic for the mean of a serially correlated series."""

    clean = pd.Series(values, dtype=float).dropna().to_numpy()
    n_obs = len(clean)
    if n_obs < 2:
        return np.nan
    automatic_lag = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    requested_lag = automatic_lag if max_lag is None else max(0, int(max_lag))
    lag = min(n_obs - 1, max(min_lag, requested_lag))
    demeaned = clean - clean.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / n_obs)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(demeaned[offset:], demeaned[:-offset]) / n_obs)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance, 0.0) / n_obs
    if variance_of_mean <= 0.0:
        return np.nan
    return float(clean.mean() / np.sqrt(variance_of_mean))


def _mean_quintile_portfolio_turnover(work: pd.DataFrame, rank: pd.Series) -> float:
    """Actual one-way turnover of daily equal-weight long/short quintiles."""

    weighted = work[["date", "ticker"]].copy()
    weighted["_rank"] = rank
    values: list[float] = []
    previous = pd.Series(dtype=float)
    for _, cross_section in weighted.groupby("date", sort=True, observed=True):
        top = cross_section.loc[cross_section["_rank"] >= 0.8, "ticker"]
        bottom = cross_section.loc[cross_section["_rank"] <= 0.2, "ticker"]
        current = pd.Series(dtype=float)
        if len(top):
            current = pd.concat(
                [current, pd.Series(1.0 / len(top), index=top.to_numpy(), dtype=float)]
            )
        if len(bottom):
            current = pd.concat(
                [current, pd.Series(-1.0 / len(bottom), index=bottom.to_numpy(), dtype=float)]
            )
        if not previous.empty:
            names = previous.index.union(current.index)
            values.append(
                float(
                    (
                        current.reindex(names, fill_value=0.0)
                        - previous.reindex(names, fill_value=0.0)
                    ).abs().sum()
                    / 2.0
                )
            )
        previous = current
    return float(np.mean(values)) if values else np.nan


def block_bootstrap_ic_delta(
    challenger_ic: pd.Series,
    baseline_ic: pd.Series,
    *,
    block_size: int = 20,
    n_boot: int = 2000,
    seed: int = 20260726,
) -> dict[str, float]:
    """Paired circular moving-block bootstrap for mean daily IC improvement."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    joined = pd.concat(
        [challenger_ic.rename("challenger"), baseline_ic.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    delta = (joined["challenger"] - joined["baseline"]).to_numpy(dtype=float)
    if len(delta) == 0:
        raise ValueError("No overlapping daily IC observations")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(delta) / block_size))
    offsets = np.arange(block_size)
    means = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        starts = rng.integers(0, len(delta), size=n_blocks)
        sample = ((starts[:, None] + offsets) % len(delta)).ravel()[: len(delta)]
        means[iteration] = delta[sample].mean()
    return {
        "mean_delta_ic": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_delta_le_zero": float((means <= 0).mean()),
        "n_days": float(len(delta)),
        "block_size": float(block_size),
    }
