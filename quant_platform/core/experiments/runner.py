"""YAML-driven experiment runner.

Experiments configure factor sets, redundancy pruning, multi-factor combiners,
portfolio construction, costs, benchmark analytics, and capacity analysis in
one file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from quant_platform.core.alpha_models.multi_factor import infer_factor_metadata
from quant_platform.core.execution.cost_models.us_equity import CostModel
from quant_platform.core.execution.backtest.pipeline import run_trading_pipeline
from quant_platform.core.portfolio.construction import PortfolioConfig, PortfolioMode, WeightScheme
from quant_platform.core.portfolio.pipeline import run_portfolio_pipeline
from quant_platform.core.signals.cross_sectional.evaluation import add_forward_returns
from quant_platform.core.signals.redundancy import (
    RedundancyConfig,
    build_redundancy_report,
    select_complementary_factors,
)
from quant_platform.core.utils.io import load_parquet, resolve_data_path


@dataclass
class ExperimentResult:
    """Paths and headline outputs from a completed experiment."""

    name: str
    output_dir: Path
    selected_factors: list[str]
    portfolio_path: Optional[Path]
    backtest_dir: Optional[Path]
    performance: Dict[str, Any]


def load_experiment_config(path: Path | str) -> Dict[str, Any]:
    """Load an experiment YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return cfg


def list_experiments(experiment_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """List available YAML experiments."""
    if experiment_dir is None:
        experiment_dir = Path(__file__).resolve().parents[2] / "experiments"
    rows = []
    for path in sorted(Path(experiment_dir).glob("*.yaml")):
        cfg = load_experiment_config(path)
        meta = cfg.get("experiment", {})
        rows.append({
            "name": meta.get("name", path.stem),
            "description": meta.get("description", ""),
            "path": str(path),
        })
    return pd.DataFrame(rows)


class ExperimentRunner:
    """Execute EntroPy research experiments from YAML."""

    def __init__(self, config_path: Path | str, *, output_dir: Optional[Path | str] = None) -> None:
        self.config_path = Path(config_path)
        self.config = load_experiment_config(config_path)
        name = self.config.get("experiment", {}).get("name", self.config_path.stem)
        self.name = str(name)
        self._validate_config_contract()
        self.output_dir = Path(output_dir) if output_dir else resolve_data_path("experiments", self.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config_contract(self) -> None:
        """Fail closed instead of silently accepting unsupported YAML keys."""

        supported_sections = {
            "experiment", "data", "data_contract", "factors", "baseline",
            "advanced_features", "alpha_model", "multi_factor", "redundancy",
            "portfolio", "backtest", "costs", "evaluation", "execution",
        }
        unknown = set(self.config) - supported_sections
        if unknown:
            raise ValueError(f"Unsupported experiment sections: {sorted(unknown)}")

        allowed_keys = {
            "experiment": {"name", "description"},
            "data": {
                "start", "end", "universe", "exchange", "factors_path",
                "prices_path", "universe_path", "fundamentals_path",
            },
            "data_contract": {"mode", "allow_non_pit_research"},
            "factors": {
                "signal_type", "names", "winsorize", "zscore", "neutralize",
            },
            "baseline": {"factors", "portfolio"},
            "advanced_features": {
                "kalman", "noise", "higher_moments", "regime_overlay",
            },
            "alpha_model": {"combiner"},
            "multi_factor": {
                "method", "lookback", "return_col", "regime_col",
                "regime_threshold",
            },
            "redundancy": {
                "enabled", "min_factors", "max_factors", "max_signal_corr",
                "max_return_corr", "max_exposure_similarity",
                "min_incremental_sharpe",
                "min_incremental_alpha_tstat",
            },
            "backtest": {
                "initial_capital", "benchmark_market", "benchmark_path",
                "risk_free_rate", "capacity_capitals",
            },
            "costs": {
                "model", "slippage_bps", "impact_coeff", "impact_exponent",
                "commission_per_share", "commission_pct", "borrow_rate",
                "borrow_rate_annual", "commission_rate", "stamp_tax_rate",
                "transfer_fee_rate", "slippage_model", "slippage_k",
                "min_commission", "commission_min", "sec_fee_rate",
                "finra_taf_per_share", "finra_taf_max", "stamp_duty_pct",
                "adv_lookback", "max_participation_rate", "min_adv_shares",
            },
            "evaluation": {
                "walkforward", "ablation_scenarios", "cost_stress_bps",
                "regime_split",
            },
        }
        for section, allowed in allowed_keys.items():
            value = self.config.get(section)
            if isinstance(value, dict):
                unknown_keys = set(value) - allowed
                if unknown_keys:
                    raise ValueError(
                        f"Unsupported {section} fields: {sorted(unknown_keys)}"
                    )

        allowed_portfolio = {
            "method", "mode", "weight_scheme", "weight", "rebalance_freq",
            "n_quantiles", "long_quantile", "short_quantile", "top_n",
            "max_stock_weight", "max_sector_weight", "max_turnover",
            "sector_neutral",
        }
        for label, portfolio_cfg in (
            ("portfolio", self.config.get("portfolio")),
            ("baseline.portfolio", self.config.get("baseline", {}).get("portfolio")),
        ):
            if isinstance(portfolio_cfg, dict):
                unknown_portfolio = set(portfolio_cfg) - allowed_portfolio
                if unknown_portfolio:
                    raise ValueError(
                        f"Unsupported {label} fields: {sorted(unknown_portfolio)}"
                    )

        portfolio = self._portfolio_section()
        if portfolio.get("sector_neutral"):
            raise ValueError(
                "portfolio.sector_neutral is not implemented; configure explicit "
                "sector exposure constraints instead"
            )

        alpha_model = self.config.get("alpha_model", {})
        unsupported_alpha = set(alpha_model) - {"combiner"}
        if unsupported_alpha:
            raise ValueError(
                "Unsupported alpha_model fields would otherwise be ignored: "
                f"{sorted(unsupported_alpha)}"
            )
        if (
            alpha_model.get("combiner")
            and self.config.get("multi_factor", {}).get("method")
            and alpha_model["combiner"] != self.config["multi_factor"]["method"]
        ):
            raise ValueError("alpha_model.combiner and multi_factor.method disagree")

        factors_cfg = self.config.get("factors", {})
        if factors_cfg.get("signal_type", "cross_sectional") != "cross_sectional":
            raise ValueError("ExperimentRunner currently supports cross_sectional factors only")
        if factors_cfg.get("winsorize", [0.01, 0.99]) != [0.01, 0.99]:
            raise ValueError("Custom factors.winsorize is not wired into the pipeline")
        if factors_cfg.get("zscore", True) is not True:
            raise ValueError("Disabling factors.zscore is not wired into the pipeline")
        if factors_cfg.get("neutralize") not in (None, []):
            raise ValueError("Custom factors.neutralize is not wired into the pipeline")

        execution = self.config.get("execution", {})
        if execution:
            raise ValueError(
                "The configured execution section is not wired into the trading "
                "pipeline yet; refusing to run it silently"
            )

        evaluation = self.config.get("evaluation", {})
        supported_evaluation = {
            "walkforward", "ablation_scenarios", "cost_stress_bps", "regime_split"
        }
        unknown_evaluation = set(evaluation) - supported_evaluation
        if unknown_evaluation:
            raise ValueError(
                f"Unsupported evaluation fields: {sorted(unknown_evaluation)}"
            )
        walkforward = evaluation.get("walkforward", {})
        allowed_walkforward = {
            "train_months", "test_months", "step_months", "min_train_obs",
            "factor_select_top_k", "purge_days",
        }
        if isinstance(walkforward, dict):
            unknown_walkforward = set(walkforward) - allowed_walkforward
            if unknown_walkforward:
                raise ValueError(
                    "Unsupported evaluation.walkforward fields: "
                    f"{sorted(unknown_walkforward)}"
                )

    def run(self, *, run_backtest: bool = True) -> ExperimentResult:
        """Run the configured experiment end-to-end."""
        factors = self._load_factors()
        prices = self._load_prices()
        universe = self._load_universe()

        factors = self._filter_dates(factors)
        prices = self._filter_dates(prices)
        universe = self._filter_dates(universe) if universe is not None else None

        factor_cols = [f for f in collect_factor_names(self.config) if f in factors.columns]
        if not factor_cols:
            raise ValueError("Experiment has no factor columns available in factors.parquet")
        self._validate_point_in_time_inputs(universe, factor_cols)

        regime_col = collect_regime_col(self.config)
        if regime_col and regime_col not in factors.columns:
            logger.warning("Regime column {} not found; regime controls disabled", regime_col)
            regime_col = None

        factors = self._ensure_returns(factors, prices)
        selected_factors = self._select_factors(factors, factor_cols)
        if not selected_factors:
            selected_factors = factor_cols[:5]

        portfolio_cfg = self._portfolio_config()
        portfolio_method = self._portfolio_method()
        combiner_method = self._combiner_method()

        portfolio_path = self.output_dir / f"weights_{self.name}.parquet"
        portfolio_result = run_portfolio_pipeline(
            method=portfolio_method,
            factor_cols=selected_factors,
            config=portfolio_cfg,
            output_path=portfolio_path,
            factors_override=factors,
            prices_override=prices,
            universe_override=universe,
            multi_factor_method=combiner_method,
            baseline_factors=self.config.get("baseline", {}).get("factors", []),
            regime_col=regime_col,
            multi_factor_lookback=int(self.config.get("multi_factor", {}).get("lookback", 126)),
        )

        performance: Dict[str, Any] = {}
        backtest_dir = None
        backtest_result: Optional[Dict[str, Any]] = None
        if run_backtest:
            backtest_dir = self.output_dir / "backtest"
            backtest_result = run_trading_pipeline(
                weights_path=portfolio_result["output_path"],
                prices_override=prices,
                cost_model=self._cost_model(),
                initial_capital=float(self.config.get("backtest", {}).get("initial_capital", 1_000_000.0)),
                output_dir=backtest_dir,
                benchmark_market=self._benchmark_market(),
                benchmark_path=self.config.get("backtest", {}).get("benchmark_path"),
                risk_free_rate=float(self.config.get("backtest", {}).get("risk_free_rate", 0.0)),
                capacity_capitals=self.config.get("backtest", {}).get("capacity_capitals"),
            )
            performance = {
                "performance_scope": "diagnostic_in_sample_selected",
                **{
                    f"diagnostic_insample_{key}": value
                    for key, value in backtest_result["performance"].items()
                },
            }

        walkforward_headline = self._run_walkforward_evaluation(
            factors,
            prices,
            universe,
            factor_cols,
            selected_factors,
        )
        performance.update(walkforward_headline)
        if run_backtest and backtest_result is not None:
            self._run_cost_stress(portfolio_result["output_path"], prices)
            self._run_regime_split(backtest_result, factors, regime_col)
            self._run_ablation_scenarios(
                factors,
                prices,
                universe,
                portfolio_cfg,
                regime_col,
            )

        result = ExperimentResult(
            name=self.name,
            output_dir=self.output_dir,
            selected_factors=selected_factors,
            portfolio_path=portfolio_result["output_path"],
            backtest_dir=backtest_dir,
            performance=performance,
        )
        self._save_summary(result, portfolio_result)
        return result

    def _run_walkforward_evaluation(
        self,
        factors: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame],
        factor_cols: list[str],
        selected_factors: list[str],
    ) -> Dict[str, float]:
        settings = self.config.get("evaluation", {}).get("walkforward")
        if not settings:
            return {}
        from quant_platform.core.evaluation.walkforward import (
            WalkForwardConfig as EvaluationWalkForwardConfig,
            run_walk_forward,
        )

        if universe is None:
            raise ValueError(
                "Walk-forward evaluation requires an explicit tradable universe"
            )
        from quant_platform.core.alpha_models.nonlinear_alpha import (
            filter_tradable_universe,
        )

        candidate_frame = filter_tradable_universe(
            factors[["date", "ticker", *factor_cols]].copy(),
            universe,
            context="walk-forward factor candidates",
        )
        max_factors = int(
            self.config.get("redundancy", {}).get("max_factors", len(factor_cols))
        )
        config = EvaluationWalkForwardConfig(
            train_months=int(settings.get("train_months", 36)),
            test_months=int(settings.get("test_months", 12)),
            step_months=int(settings.get("step_months", 12)),
            min_train_obs=int(settings.get("min_train_obs", 100)),
            factor_select_top_k=int(
                settings.get("factor_select_top_k", min(max_factors, len(factor_cols)))
            ),
            purge_days=int(settings.get("purge_days", 1)),
        )
        primary = selected_factors[0] if selected_factors else factor_cols[0]
        result = run_walk_forward(
            candidate_frame,
            prices,
            primary,
            config,
        )
        result.to_csv(self.output_dir / "walkforward_oos.csv", index=False)
        if result.empty:
            return {
                "walkforward_n_folds": 0.0,
                "walkforward_mean_oos_sharpe": np.nan,
            }
        return {
            "walkforward_n_folds": float(len(result)),
            "walkforward_mean_oos_sharpe": float(result["oos_sharpe"].mean()),
            "walkforward_min_oos_sharpe": float(result["oos_sharpe"].min()),
        }

    def _run_cost_stress(
        self,
        weights_path: Path | str,
        prices: pd.DataFrame,
    ) -> None:
        stress_values = self.config.get("evaluation", {}).get("cost_stress_bps")
        if not stress_values:
            return
        rows = []
        for value in stress_values:
            bps = float(value)
            if bps < 0:
                raise ValueError("evaluation.cost_stress_bps cannot contain negatives")
            result = run_trading_pipeline(
                weights_path=weights_path,
                prices_override=prices,
                cost_model=self._cost_model(slippage_bps=bps),
                initial_capital=float(
                    self.config.get("backtest", {}).get("initial_capital", 1_000_000.0)
                ),
                output_dir=self.output_dir / "cost_stress" / f"{bps:g}bps",
                benchmark_market=self._benchmark_market(),
                benchmark_path=self.config.get("backtest", {}).get("benchmark_path"),
                risk_free_rate=float(
                    self.config.get("backtest", {}).get("risk_free_rate", 0.0)
                ),
            )
            rows.append({"slippage_bps": bps, **result["performance"]})
        pd.DataFrame(rows).to_csv(
            self.output_dir / "cost_stress_summary.csv",
            index=False,
        )

    def _run_regime_split(
        self,
        backtest_result: Dict[str, Any],
        factors: pd.DataFrame,
        regime_col: Optional[str],
    ) -> None:
        if not self.config.get("evaluation", {}).get("regime_split", False):
            return
        if not regime_col or regime_col not in factors.columns:
            raise ValueError("evaluation.regime_split requires a valid regime column")
        daily = backtest_result.get("daily_pnl", pd.DataFrame()).copy()
        if daily.empty:
            return
        if "date" not in daily.columns:
            daily = daily.reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        return_col = "net_ret" if "net_ret" in daily.columns else "gross_ret"
        regime = factors.groupby("date", observed=True)[regime_col].mean()
        daily["_regime_score"] = daily["date"].map(regime)
        threshold = float(
            self.config.get("multi_factor", {}).get("regime_threshold", 0.60)
        )
        daily["_regime"] = np.where(
            daily["_regime_score"] > threshold,
            "turbulent",
            "normal",
        )
        rows = []
        for name, group in daily.dropna(subset=[return_col]).groupby("_regime"):
            values = group[return_col]
            std = values.std(ddof=1)
            rows.append({
                "regime": name,
                "n_days": len(values),
                "mean_daily_return": float(values.mean()),
                "annualized_sharpe": (
                    float(values.mean() / std * np.sqrt(252))
                    if len(values) > 1 and std > 0
                    else np.nan
                ),
            })
        pd.DataFrame(rows).to_csv(
            self.output_dir / "regime_split.csv",
            index=False,
        )

    def _run_ablation_scenarios(
        self,
        factors: pd.DataFrame,
        prices: pd.DataFrame,
        universe: Optional[pd.DataFrame],
        portfolio_cfg: PortfolioConfig,
        regime_col: Optional[str],
    ) -> None:
        scenarios = self.config.get("evaluation", {}).get("ablation_scenarios")
        if not scenarios:
            return
        rows = []
        for scenario in scenarios:
            unknown = set(scenario) - {"name", "factors", "regime_overlay"}
            if unknown:
                raise ValueError(
                    f"Unsupported ablation fields in {scenario.get('name')}: "
                    f"{sorted(unknown)}"
                )
            name = str(scenario.get("name", "unnamed"))
            scenario_factors = [
                factor for factor in scenario.get("factors", [])
                if factor in factors.columns
            ]
            if not scenario_factors:
                raise ValueError(f"Ablation {name} has no available factors")
            use_regime = bool(scenario.get("regime_overlay", False))
            scenario_output = self.output_dir / "ablations" / name
            scenario_output.mkdir(parents=True, exist_ok=True)
            portfolio = run_portfolio_pipeline(
                method=self._portfolio_method(),
                factor_cols=scenario_factors,
                config=portfolio_cfg,
                output_path=scenario_output / "weights.parquet",
                factors_override=factors,
                prices_override=prices,
                universe_override=universe,
                multi_factor_method=self._combiner_method(),
                baseline_factors=self.config.get("baseline", {}).get("factors", []),
                regime_col=regime_col if use_regime else None,
                multi_factor_lookback=int(
                    self.config.get("multi_factor", {}).get("lookback", 126)
                ),
            )
            backtest = run_trading_pipeline(
                weights_path=portfolio["output_path"],
                prices_override=prices,
                cost_model=self._cost_model(),
                initial_capital=float(
                    self.config.get("backtest", {}).get("initial_capital", 1_000_000.0)
                ),
                output_dir=scenario_output / "backtest",
                benchmark_market=self._benchmark_market(),
                benchmark_path=self.config.get("backtest", {}).get("benchmark_path"),
                risk_free_rate=float(
                    self.config.get("backtest", {}).get("risk_free_rate", 0.0)
                ),
            )
            rows.append({
                "scenario": name,
                "factors": ",".join(scenario_factors),
                "regime_overlay": use_regime,
                **backtest["performance"],
            })
        pd.DataFrame(rows).to_csv(
            self.output_dir / "ablation_summary.csv",
            index=False,
        )

    def _load_factors(self) -> pd.DataFrame:
        data_cfg = self.config.get("data", {})
        self._require_explicit_cn_path(data_cfg, "factors_path")
        path = Path(data_cfg.get("factors_path", resolve_data_path("factors", "factors.parquet")))
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_prices(self) -> pd.DataFrame:
        data_cfg = self.config.get("data", {})
        self._require_explicit_cn_path(data_cfg, "prices_path")
        path = Path(data_cfg.get("prices_path", resolve_data_path("prices", "prices.parquet")))
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_universe(self) -> Optional[pd.DataFrame]:
        data_cfg = self.config.get("data", {})
        self._require_explicit_cn_path(data_cfg, "universe_path")
        configured_path = data_cfg.get("universe_path")
        universe_name = str(data_cfg.get("universe", "")).lower()
        exchange = str(data_cfg.get("exchange", "")).upper()
        if not configured_path and universe_name:
            if universe_name == "sp500" and exchange in {"", "XNYS", "XNAS"}:
                configured_path = resolve_data_path("universe", "universe.parquet")
            else:
                raise ValueError(
                    f"No dataset mapping exists for data.universe={universe_name!r}; "
                    "provide data.universe_path explicitly"
                )
        path = Path(configured_path or resolve_data_path("universe", "universe.parquet"))
        if not path.exists():
            return None
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _require_explicit_cn_path(self, data_cfg: Dict[str, Any], key: str) -> None:
        exchange = str(data_cfg.get("exchange", "")).upper()
        if exchange.startswith(("XSH", "XSHE")) and not data_cfg.get(key):
            raise ValueError(
                f"CN experiment requires explicit data.{key}; refusing to use "
                "the default US research dataset"
            )

    def _validate_point_in_time_inputs(
        self,
        universe: Optional[pd.DataFrame],
        factor_cols: list[str],
    ) -> None:
        contract = self.config.get("data_contract", {})
        mode = str(contract.get("mode", "production")).lower()
        allow_non_pit = bool(contract.get("allow_non_pit_research", False))
        if mode not in {"production", "research"}:
            raise ValueError("data_contract.mode must be production or research")
        if mode == "research" and allow_non_pit:
            logger.warning(
                "Running {} in explicitly permitted non-PIT research mode",
                self.name,
            )
            return

        if universe is None:
            raise ValueError("Production research requires a point-in-time universe")
        from quant_platform.core.data.universe import require_point_in_time_universe

        require_point_in_time_universe(universe, context=f"experiment {self.name}")

        fundamental_factors = {
            "EARNINGS_YIELD", "BOOK_TO_MARKET", "GROSS_PROFITABILITY",
            "ASSET_GROWTH", "ACCRUALS", "PIOTROSKI_F8",
        }
        if fundamental_factors.intersection(factor_cols):
            fundamentals_path = self.config.get("data", {}).get("fundamentals_path")
            if not fundamentals_path:
                raise ValueError(
                    "Production ML/factor research using fundamentals requires "
                    "data.fundamentals_path for PIT provenance validation"
                )
            fundamentals = load_parquet(Path(fundamentals_path))
            from quant_platform.core.data.fundamentals import (
                require_point_in_time_fundamentals,
            )

            require_point_in_time_fundamentals(
                fundamentals,
                context=f"experiment {self.name}",
            )

    def _filter_dates(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or "date" not in df.columns:
            return df
        data_cfg = self.config.get("data", {})
        out = df.copy()
        if data_cfg.get("start"):
            out = out[out["date"] >= pd.Timestamp(data_cfg["start"])]
        if data_cfg.get("end"):
            out = out[out["date"] <= pd.Timestamp(data_cfg["end"])]
        return out

    def _ensure_returns(self, factors: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        ret_col = self.config.get("multi_factor", {}).get("return_col", "fwd_ret_1d")
        if ret_col in factors.columns or "adj_close" not in prices.columns:
            return factors
        periods = [1]
        if ret_col.startswith("fwd_ret_") and ret_col.endswith("d"):
            periods = [int(ret_col.removeprefix("fwd_ret_").removesuffix("d"))]
        fwd = add_forward_returns(prices, periods=periods)
        ret_cols = [f"fwd_ret_{p}d" for p in periods if f"fwd_ret_{p}d" in fwd.columns]
        return factors.merge(fwd[["date", "ticker"] + ret_cols], on=["date", "ticker"], how="left")

    def _select_factors(self, factors: pd.DataFrame, factor_cols: list[str]) -> list[str]:
        red_cfg = self.config.get("redundancy", {})
        if red_cfg.get("enabled", True) is False or len(factor_cols) <= int(red_cfg.get("max_factors", 5)):
            return factor_cols

        directions, _ = infer_factor_metadata(factor_cols)
        redundancy_config = RedundancyConfig(
            max_signal_corr=float(red_cfg.get("max_signal_corr", 0.70)),
            max_return_corr=float(red_cfg.get("max_return_corr", 0.70)),
            max_exposure_similarity=float(red_cfg.get("max_exposure_similarity", 0.80)),
            min_factors=int(red_cfg.get("min_factors", 3)),
            max_factors=int(red_cfg.get("max_factors", 5)),
            min_incremental_sharpe=float(red_cfg.get("min_incremental_sharpe", 0.0)),
            min_incremental_alpha_tstat=float(red_cfg.get("min_incremental_alpha_tstat", 0.0)),
        )
        report = build_redundancy_report(
            factors,
            factor_cols,
            direction_map=directions,
            return_col=self.config.get("multi_factor", {}).get("return_col", "fwd_ret_1d"),
            config=redundancy_config,
        )
        self._save_redundancy_report(report)

        score_table = self._score_table(factor_cols)
        selected = select_complementary_factors(
            score_table,
            report,
            config=redundancy_config,
        )
        selected.to_csv(self.output_dir / "selected_factors.csv", index=False)
        selected_factors = selected[selected["selected"]].sort_values("selected_rank")["factor"].tolist()
        return selected_factors

    def _score_table(self, factor_cols: list[str]) -> pd.DataFrame:
        path = resolve_data_path("factors", "factor_catalog.csv")
        if path.exists():
            table = pd.read_csv(path, index_col=0)
            table = table.loc[[f for f in factor_cols if f in table.index]].copy()
            if "factor" not in table.columns:
                table["factor"] = table.index
            return table
        return pd.DataFrame({
            "factor": factor_cols,
            "selection_score": list(range(len(factor_cols), 0, -1)),
        }).set_index("factor", drop=False)

    def _save_redundancy_report(self, report: Dict[str, pd.DataFrame]) -> None:
        for name, df in report.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(self.output_dir / f"redundancy_{name}.csv")

    def _portfolio_config(self) -> PortfolioConfig:
        p = self._portfolio_section()
        sector_cap = p.get("max_sector_weight", 0.30)
        turnover_cap = p.get("max_turnover")
        return PortfolioConfig(
            mode=PortfolioMode(p.get("mode", "long_only")),
            weight_scheme=WeightScheme(p.get("weight_scheme", p.get("weight", "equal"))),
            max_stock_weight=float(p.get("max_stock_weight", 0.05)),
            max_sector_weight=(
                None if sector_cap is None else float(sector_cap)
            ),
            max_turnover=(
                None if turnover_cap is None else float(turnover_cap)
            ),
            n_quantiles=int(p.get("n_quantiles", 5)),
            long_quantile=int(p.get("long_quantile", 5)),
            short_quantile=int(p.get("short_quantile", 1)),
            top_n=p.get("top_n"),
            rebalance_freq=p.get("rebalance_freq", "M"),
            initial_capital=float(self.config.get("backtest", {}).get("initial_capital", 1_000_000.0)),
        )

    def _portfolio_method(self) -> str:
        return self._portfolio_section().get("method", "quantile")

    def _portfolio_section(self) -> Dict[str, Any]:
        if "portfolio" in self.config:
            return self.config["portfolio"]
        return self.config.get("baseline", {}).get("portfolio", {})

    def _combiner_method(self) -> str:
        mf = self.config.get("multi_factor", {})
        if mf.get("method"):
            return mf["method"]
        alpha = self.config.get("alpha_model", {})
        return alpha.get("combiner", "rolling_icir")

    def _cost_model(self, *, slippage_bps: Optional[float] = None) -> CostModel:
        c = self.config.get("costs", {})
        model_name = str(c.get("model", "us_equity")).lower()
        if model_name not in {"us", "us_equity"}:
            if model_name in {"cn", "cn_a_share", "a_share"}:
                raise ValueError(
                    "CN A-share costs are not compatible with the current trading "
                    "pipeline interface; refusing to substitute US costs"
                )
            raise ValueError(f"Unsupported costs.model: {model_name}")
        participation = c.get("max_participation_rate", 0.10)
        taf_cap = c.get("finra_taf_max", 8.30)
        return CostModel(
            slippage_bps=float(
                c.get("slippage_bps", 5.0)
                if slippage_bps is None
                else slippage_bps
            ),
            impact_coeff=float(c.get("impact_coeff", 0.1)),
            impact_exponent=float(c.get("impact_exponent", 0.5)),
            commission_per_share=float(c.get("commission_per_share", 0.005)),
            commission_pct=float(c.get("commission_pct", 0.0)),
            commission_min=float(
                c.get("commission_min", c.get("min_commission", 1.0))
            ),
            sec_fee_rate=float(c.get("sec_fee_rate", 8.0e-6)),
            finra_taf_per_share=float(
                c.get("finra_taf_per_share", 0.000119)
            ),
            finra_taf_max=(
                None if taf_cap is None else float(taf_cap)
            ),
            stamp_duty_pct=float(c.get("stamp_duty_pct", 0.0)),
            borrow_rate_annual=float(c.get("borrow_rate_annual", c.get("borrow_rate", 0.005))),
            adv_lookback=int(c.get("adv_lookback", 20)),
            min_adv_shares=float(c.get("min_adv_shares", 1_000.0)),
            max_participation_rate=(
                None if participation is None else float(participation)
            ),
        )

    def _benchmark_market(self) -> Optional[str]:
        if "backtest" in self.config and "benchmark_market" in self.config["backtest"]:
            return self.config["backtest"]["benchmark_market"]
        exchange = self.config.get("data", {}).get("exchange", "XNYS")
        return "cn" if str(exchange).upper().startswith("XSH") else "us"

    def _save_summary(self, result: ExperimentResult, portfolio_result: Dict[str, Any]) -> None:
        payload = {
            "name": result.name,
            "config_path": str(self.config_path),
            "selected_factors": result.selected_factors,
            "portfolio_path": str(result.portfolio_path) if result.portfolio_path else None,
            "backtest_dir": str(result.backtest_dir) if result.backtest_dir else None,
            "performance": result.performance,
        }
        (self.output_dir / "experiment_summary.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        if not portfolio_result.get("factor_weights", pd.DataFrame()).empty:
            portfolio_result["factor_weights"].to_csv(self.output_dir / "factor_weights.csv", index=False)
        if not portfolio_result.get("regime_controls", pd.DataFrame()).empty:
            portfolio_result["regime_controls"].to_csv(self.output_dir / "regime_controls.csv", index=False)


def collect_factor_names(config: Dict[str, Any]) -> list[str]:
    """Collect configured factor names from baseline/factors/advanced sections."""
    names: list[str] = []
    if config.get("factors", {}).get("names"):
        names.extend(config["factors"]["names"])
    if config.get("baseline", {}).get("factors"):
        names.extend(config["baseline"]["factors"])
    adv = config.get("advanced_features", {})
    for key, value in adv.items():
        if key == "regime_overlay":
            continue
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, Iterable):
            names.extend([str(v) for v in value])
    return list(dict.fromkeys(names))


def collect_regime_col(config: Dict[str, Any]) -> Optional[str]:
    """Return the configured regime column, if any."""
    mf = config.get("multi_factor", {})
    if mf.get("regime_col"):
        return mf["regime_col"]
    adv = config.get("advanced_features", {})
    if adv.get("regime_overlay"):
        return adv["regime_overlay"]
    overlay = config.get("alpha_model", {}).get("regime_overlay", {})
    if isinstance(overlay, dict):
        return overlay.get("column")
    return None
