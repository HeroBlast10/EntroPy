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
        self.output_dir = Path(output_dir) if output_dir else resolve_data_path("experiments", self.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        if run_backtest:
            backtest_dir = self.output_dir / "backtest"
            backtest_result = run_trading_pipeline(
                weights_path=portfolio_result["output_path"],
                cost_model=self._cost_model(),
                initial_capital=float(self.config.get("backtest", {}).get("initial_capital", 1_000_000.0)),
                output_dir=backtest_dir,
                benchmark_market=self._benchmark_market(),
                benchmark_path=self.config.get("backtest", {}).get("benchmark_path"),
                risk_free_rate=float(self.config.get("backtest", {}).get("risk_free_rate", 0.0)),
                capacity_capitals=self.config.get("backtest", {}).get("capacity_capitals"),
            )
            performance = backtest_result["performance"]

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

    def _load_factors(self) -> pd.DataFrame:
        path = Path(self.config.get("data", {}).get("factors_path", resolve_data_path("factors", "factors.parquet")))
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_prices(self) -> pd.DataFrame:
        path = Path(self.config.get("data", {}).get("prices_path", resolve_data_path("prices", "prices.parquet")))
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_universe(self) -> Optional[pd.DataFrame]:
        path = Path(self.config.get("data", {}).get("universe_path", resolve_data_path("universe", "universe.parquet")))
        if not path.exists():
            return None
        df = load_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

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
        report = build_redundancy_report(
            factors,
            factor_cols,
            direction_map=directions,
            return_col=self.config.get("multi_factor", {}).get("return_col", "fwd_ret_1d"),
            config=RedundancyConfig(
                max_signal_corr=float(red_cfg.get("max_signal_corr", 0.70)),
                max_return_corr=float(red_cfg.get("max_return_corr", 0.70)),
                max_exposure_similarity=float(red_cfg.get("max_exposure_similarity", 0.80)),
                min_factors=int(red_cfg.get("min_factors", 3)),
                max_factors=int(red_cfg.get("max_factors", 5)),
                min_incremental_sharpe=float(red_cfg.get("min_incremental_sharpe", 0.0)),
            ),
        )
        self._save_redundancy_report(report)

        score_table = self._score_table(factor_cols)
        selected = select_complementary_factors(score_table, report)
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
        return PortfolioConfig(
            mode=PortfolioMode(p.get("mode", "long_only")),
            weight_scheme=WeightScheme(p.get("weight_scheme", p.get("weight", "equal"))),
            max_stock_weight=float(p.get("max_stock_weight", 0.05)),
            max_sector_weight=float(p.get("max_sector_weight", 0.30)),
            max_turnover=p.get("max_turnover"),
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

    def _cost_model(self) -> CostModel:
        c = self.config.get("costs", {})
        return CostModel(
            slippage_bps=float(c.get("slippage_bps", 5.0)),
            impact_coeff=float(c.get("impact_coeff", 0.1)),
            impact_exponent=float(c.get("impact_exponent", 0.5)),
            commission_per_share=float(c.get("commission_per_share", 0.005)),
            commission_pct=float(c.get("commission_pct", 0.0)),
            borrow_rate_annual=float(c.get("borrow_rate_annual", c.get("borrow_rate", 0.005))),
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

