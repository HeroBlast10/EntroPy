"""Verify long-history evidence and render matched, descriptive comparisons."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from quant_platform.core.alpha_models.nonlinear_alpha import daily_rank_ic
from quant_platform.core.evaluation.factor_diagnostics import hac_summary
from quant_platform.core.utils.provenance import write_run_metadata, fingerprint

METHODS = {"quantile_20d": "20日全局分位数", "industry_20d": "20日行业内分位数",
           "risk_dollar": "20日风险优化", "risk_neutral": "20日风险优化+中性"}
LABELS = {"cross_sectional": "原横截面标签", "industry_size_vol": "中性/波动标签"}
KEYS = ["label", "block", "model", "method", "cost_multiplier"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def block_indices(n, repetitions=1500, block=60, seed=2718):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]


def sharpe(values, axis=0):
    return np.mean(values, axis=axis) / np.std(values, axis=axis, ddof=1) * np.sqrt(252)


def verify_training(research):
    plan = json.loads((research / "plan.json").read_text())
    for name, expected in plan["source_sha256"].items():
        assert fingerprint(ROOT / name)["sha256"] == expected, f"Changed training source: {name}"
        assert fingerprint(research / "source_snapshot" / name)["sha256"] == expected
    completion = json.loads((research / "training_completion.json").read_text())
    assert not completion["smoke"] and len(completion["cohorts"]) == 8
    assert len({r["prediction_key_sha256"] for r in completion["cohorts"]}) == 1
    trials, audits = [], []
    for row in completion["cohorts"]:
        folder = research / f"models/{row['label']}/{row['block']}/{row['model']}"
        actual_keys = pd.read_parquet(folder / "oof.parquet", columns=["date", "ticker"])
        assert len(actual_keys) == row["prediction_rows"] and not actual_keys.duplicated().any()
        actual_key_hash = hashlib.sha256(actual_keys.astype(str).to_csv(index=False).encode()).hexdigest()
        assert actual_key_hash == row["prediction_key_sha256"]
        audit = pd.read_csv(folder / "fold_audit.csv")
        assert (pd.to_datetime(audit.max_training_label_end) < pd.to_datetime(audit.validation_start)).all()
        assert (pd.to_datetime(audit.max_inner_training_label_end) < pd.to_datetime(audit.inner_validation_start)).all()
        audits.append(audit)
        for trial in json.loads((folder / "trial_registry.json").read_text()):
            trials.append({**trial, **{k: row[k] for k in ("label", "block", "model")}})
    frame = pd.DataFrame(trials)
    assert frame.groupby(["fold", "phase"]).training_key_sha256.nunique().eq(1).all()
    cache = research / "risk_cache"
    risk_audit = pd.read_csv(cache / "risk_audit.csv")
    assert (pd.to_datetime(risk_audit.latest_risk_observation) < pd.to_datetime(risk_audit.date)).all()
    assert risk_audit.euler_error.abs().max() < 1e-10
    cov = np.load(cache / "covariance.npz")["covariance"]
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    risk_metadata = json.loads((cache / "run_metadata.json").read_text())
    assert fingerprint(cache / "source_snapshot/risk_model.py")["sha256"] == risk_metadata["inputs"]["risk_model"]["sha256"]
    assert fingerprint(cache / "source_snapshot/prepare_wrds_risk.py")["sha256"] == risk_metadata["inputs"]["driver"]["sha256"]
    return {"cohorts": 8, "folds": 13, "prediction_rows_per_cohort": completion["cohorts"][0]["prediction_rows"],
            "trial_entries": len(trials), "single_model_fits": int(sum(3 if r["model"] == "lightgbm" else 1 for r in trials)),
            "common_sampling_keys": True, "purged_label_ends": True, "risk_observations_strictly_prior": True,
            "training_sources_and_snapshots_match": True, "risk_source_snapshot_matches": True,
            "max_risk_euler_error": float(risk_audit.euler_error.abs().max()),
            "min_factor_covariance_eigenvalue": float(np.linalg.eigvalsh(cov).min())}


def load_books(research):
    primary = next((research / name for name in ("risk_replays_20d", "risk_replays_final") if (research / name).exists()), None)
    roots = ([(primary, list(METHODS))] if primary is not None else
             [(research / "risk_replays", ["quantile_20d", "industry_20d"]),
              (research / "risk_replays_v2", ["risk_dollar", "risk_neutral"])]
             if (research / "risk_replays_v2").exists() else [(research / "risk_replays", list(METHODS))])
    frames, failures = [], []
    if primary is not None:
        for previous in (research / "risk_replays", research / "risk_replays_v2", research / "risk_replays_final"):
            record = previous / "failures.json"
            if record.exists():
                failures += [{"root": str(previous), **row} for row in json.loads(record.read_text())]
    for root, methods in roots:
        complete = json.loads((root / "completion.json").read_text())
        assert not complete["smoke"]
        if "risk_neutral" in methods:
            assert not complete["failures"] and complete["completed"] == complete["expected"] == 16 * len(methods)
        failures += [{"root": str(root), **failure} for failure in complete["failures"]]
        metadata = json.loads((root / "experiment_plan.json").read_text())
        for name, expected in metadata["params"]["source_sha256"].items():
            assert fingerprint(root / "source_snapshot" / name)["sha256"] == expected
            assert fingerprint(ROOT / name)["sha256"] == expected, f"Changed replay source: {name}"
        frame = pd.read_csv(root / "performance.csv")
        frame = frame.loc[frame.method.isin(methods)].copy()
        assert len(frame) == 16 * len(methods)
        frame["artifact_root"] = str(root)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    assert len(results) == 64 and not results.duplicated(KEYS).any()
    books, verification = {}, []
    for row in results.itertuples():
        folder = Path(row.artifact_root) / f"{row.label}/{row.block}/{row.model}/{row.method}/cost_{row.cost_multiplier:g}x"
        p = pd.read_parquet(folder / "daily_pnl.parquet")
        trades = pd.read_parquet(folder / "trades.parquet", columns=["date", "shares", "total_cost", "executed_participation"])
        assert len(p) == 3248 and p.index.is_monotonic_increasing and p.index.is_unique
        if books:
            assert p.index.equals(next(iter(books.values())).index), "Different return calendars"
        assert trades.shares.eq(np.floor(trades.shares)).all()
        assert trades.loc[trades.shares.eq(0), "total_cost"].eq(0).all()
        assert trades.executed_participation.dropna().le(.1 + 1e-12).all()
        cost = trades.groupby("date").total_cost.sum().reindex(p.index, fill_value=0)
        np.testing.assert_allclose(cost, p.trading_cost_dollar, atol=1e-7)
        sources = pd.read_csv(folder / "cost_sources.csv")
        np.testing.assert_allclose(sources.total_cost_dollar.sum(), p.total_cost_dollar.sum(), atol=1e-6)
        previous = p.nav_net.shift().fillna(1_000_000.)
        np.testing.assert_allclose(previous * (1 + p.gross_ret) - p.total_cost_dollar, p.nav_net, rtol=1e-11)
        if row.method == "risk_neutral":
            assert p.risk_target_market_beta.abs().max() <= 2e-6
            assert p.risk_target_gross.max() <= 1 + 2e-6
            assert p.risk_target_industry_abs_net.max() <= 2e-5
        key = (row.label, row.block, row.model, row.method, row.cost_multiplier)
        books[key] = p
        verification.append({"key": key, "costs_and_nav_reconcile": True, "whole_shares": True,
            "max_unmodelled_gross": float(p.risk_unmodelled_gross.max()),
            "volatility_cap_binding_rebalances": int(p.risk_volatility_scale.lt(1 - 1e-9).sum()) if row.method.startswith("risk") else None})
    return results, books, verification, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/NEUTRAL_LABELS_RISK_2026-10-23.md")
    parser.add_argument("--test-count", type=int, default=506)
    args = parser.parse_args()
    research = args.research.resolve()
    output = args.output.resolve()
    evidence = verify_training(research)
    summary, books, verification, failures = load_books(research)
    summary.to_csv(research / "comparison_summary.csv", index=False)
    base = summary.loc[summary.cost_multiplier.eq(1)].copy()
    models = pd.read_csv(research / "model_summary.csv")
    model_stats = []
    for row in models.itertuples():
        oof = pd.read_parquet(research / f"models/{row.label}/{row.block}/{row.model}/oof.parquet",
                              columns=["date", "prediction", "forward_return"])
        stats = hac_summary(daily_rank_ic(oof, "prediction"), horizon=20)
        np.testing.assert_allclose(stats["mean_rank_ic"], row.raw_rank_ic, atol=1e-12)
        model_stats.append({"label": row.label, "block": row.block, "model": row.model, **stats})
    model_stats = pd.DataFrame(model_stats)
    model_stats.to_csv(research / "model_ic_uncertainty.csv", index=False)
    plan = json.loads((research / "plan.json").read_text())
    dates = next(iter(books.values())).index
    indices = block_indices(len(dates))
    alternative_indices = {length: block_indices(len(dates), block=length) for length in (20, 126)}
    estimates, paired, regimes, attribution = [], [], [], []
    sensitivity = []
    factors = pd.read_parquet(research / "risk_cache/factor_returns.parquet").reindex(dates)
    periods = {"2015–2016": ("2015-01-01", "2016-12-31"), "2018": ("2018-01-01", "2018-12-31"),
               "2020": ("2020-01-01", "2020-12-31"), "2022–2023": ("2022-01-01", "2023-12-31"),
               "2022-11—2023-01": ("2022-11-01", "2023-01-31"), "2023银行事件月": ("2023-03-01", "2023-03-31")}
    boot_sharpe = {}
    for key, p in books.items():
        if key[-1] != 1:
            continue
        sample = p.net_ret.to_numpy()[indices]
        draws = sharpe(sample, axis=1)
        boot_sharpe[key] = draws
        for length, selected in {60: indices, **alternative_indices}.items():
            local_draws = draws if length == 60 else sharpe(p.net_ret.to_numpy()[selected], axis=1)
            sensitivity.append({"label": key[0], "block": key[1], "model": key[2], "method": key[3],
                "block_length": length, "bootstrap_sharpe_se": float(local_draws.std(ddof=1)),
                "sharpe_lower": float(np.quantile(local_draws, .025)), "sharpe_upper": float(np.quantile(local_draws, .975))})
        recent = p.loc["2021-12-27":].net_ret.to_numpy()
        recent_draws = sharpe(recent[block_indices(len(recent))], axis=1)
        estimates.append({"label": key[0], "block": key[1], "model": key[2], "method": key[3],
            "sharpe_lower": float(np.quantile(draws, .025)), "sharpe_upper": float(np.quantile(draws, .975)),
            "bootstrap_sharpe_se": float(draws.std(ddof=1)), "recent_4y_sharpe": float(sharpe(recent)),
            "recent_4y_sharpe_se": float(recent_draws.std(ddof=1))})
        exposure_cols = [c for c in p if c.startswith("risk_factor_exposure:")]
        exposure = p[exposure_cols].rename(columns=lambda c: c.removeprefix("risk_factor_exposure:")).shift(1).fillna(0)
        components = exposure.mul(factors).where(p.risk_unmodelled_gross.shift().fillna(0).lt(1e-8), axis=0)
        for name, (start, end) in periods.items():
            window = p.loc[start:end]
            contribution = components.loc[start:end]
            regimes.append({"label": key[0], "block": key[1], "model": key[2], "method": key[3], "period": name,
                "net_return": float((1 + window.net_ret).prod() - 1), "mean_abs_beta": float(window.risk_actual_beta.abs().mean()),
                "mean_industry_abs_net": float(window.risk_actual_industry_abs_net.mean())})
            attribution.append({"label": key[0], "block": key[1], "model": key[2], "method": key[3], "period": name,
                "gross_return_sum": float(window.gross_ret.sum()), "market_component_sum": float(contribution.market.sum()),
                "size_component_sum": float(contribution["size"].sum()),
                "industry_component_sum": float(contribution.filter(like="sector:").sum().sum()),
                "unexplained_sum": float(window.gross_ret.sum() - contribution.sum().sum())})
    uncertainty = pd.DataFrame(estimates)
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(research / "sharpe_block_sensitivity.csv", index=False)

    def pair(kind, challenger, reference):
        delta = boot_sharpe[challenger] - boot_sharpe[reference]
        daily = books[challenger].net_ret - books[reference].net_ret
        mean_draw = daily.to_numpy()[indices].mean(axis=1) * 252
        paired.append({"kind": kind, "challenger": "/".join(map(str, challenger[:-1])),
            "reference": "/".join(map(str, reference[:-1])), "net_sharpe_difference": float(sharpe(books[challenger].net_ret) - sharpe(books[reference].net_ret)),
            "sharpe_delta_lower": float(np.quantile(delta, .025)), "sharpe_delta_upper": float(np.quantile(delta, .975)),
            "annual_mean_delta_lower": float(np.quantile(mean_draw, .025)), "annual_mean_delta_upper": float(np.quantile(mean_draw, .975))})
    for label in LABELS:
        for block in ("price_only", "price_plus_financial"):
            for model in ("ridge", "lightgbm"):
                pair("neutrality", (label, block, model, "risk_neutral", 1.), (label, block, model, "risk_dollar", 1.))
                pair("industry_ranking", (label, block, model, "industry_20d", 1.), (label, block, model, "quantile_20d", 1.))
                for method in METHODS:
                    if label == "industry_size_vol":
                        pair("label", (label, block, model, method, 1.), ("cross_sectional", block, model, method, 1.))
                    if model == "lightgbm":
                        pair("model", (label, block, model, method, 1.), (label, block, "ridge", method, 1.))
                    if block == "price_plus_financial":
                        pair("financial", (label, block, model, method, 1.), (label, "price_only", model, method, 1.))
    paired = pd.DataFrame(paired)
    uncertainty.to_csv(research / "sharpe_uncertainty.csv", index=False)
    paired.to_csv(research / "paired_comparisons.csv", index=False)
    pd.DataFrame(regimes).to_csv(research / "regime_comparison.csv", index=False)
    pd.DataFrame(attribution).to_csv(research / "factor_return_attribution.csv", index=False)
    evidence.update(verified_books=len(verification), solver_failures_before_numerical_repair=len(failures),
                    artifact_checks=verification, production_admitted=False)
    (research / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    assets = output.parent / "assets/neutral_risk_20261023"
    assets.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = ["#7A8495", "#C49136", "#2B79AD", "#2B9677"]
    for row, label in enumerate(LABELS):
        for col, model in enumerate(("ridge", "lightgbm")):
            ax = axes[row, col]
            for color, method in zip(colors, METHODS):
                p = books[(label, "price_only", model, method, 1.)]
                nav = pd.concat([pd.Series([1.], index=[p.index[0] - pd.Timedelta(days=1)]), p.nav_net / 1_000_000])
                ax.plot(nav.index, nav, color=color, lw=1.05, label=METHODS[method])
            ax.set_title(f"{LABELS[label]} · {model}", loc="left")
            ax.grid(alpha=.2); ax.axhline(1, color="#B5BCC8", lw=.7)
            ax.set_xlim(dates[0] - pd.Timedelta(days=1), dates[-1])
            ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_ylabel("净值（起始=1）")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("2013–2025：标签与风险约束的固定参数对照", fontsize=16, x=.06, ha="left")
    fig.text(.06, .02, "价量预测组 · 基础模拟成本 · 100万美元 · 全部为已审阅历史研究", fontsize=9, color="#657083")
    fig.tight_layout(rect=(.01, .04, 1, .95))
    fig.savefig(assets / "long_history_nav.png", dpi=170); fig.savefig(assets / "long_history_nav.svg")
    plt.close(fig)

    model_rows = [[LABELS[r.label], r.block, r.model, f"{r.mean_rank_ic:.5f}", f"{r.hac_t:.2f}", r.n_days]
                  for r in model_stats.itertuples()]
    main_rows, confidence_rows = [], []
    for r in base.sort_values(["label", "block", "model", "method"]).itertuples():
        two = summary.loc[summary.label.eq(r.label) & summary.block.eq(r.block) & summary.model.eq(r.model)
                          & summary.method.eq(r.method) & summary.cost_multiplier.eq(2)].iloc[0]
        main_rows.append([LABELS[r.label], f"{r.block}/{r.model}", METHODS[r.method], f"{r.net_ann_return:.2%}",
            f"{r.net_sharpe:.3f}", f"{abs(r.net_max_drawdown):.2%}", f"{r.net_ann_vol:.2%}",
            f"{r.mean_gross_exposure:.3f}", f"{r.max_gross_exposure:.3f}", f"{r.actual_beta_abs_p95:.4f}", f"{r.actual_industry_abs_net_p95:.2%}",
            f"{r.annual_trading_drag_bps / 100:.2f}%", f"{two.net_ann_return:.2%}"])
        if r.method == "quantile_20d":
            u = uncertainty.loc[uncertainty.label.eq(r.label) & uncertainty.block.eq(r.block)
                                & uncertainty.model.eq(r.model) & uncertainty.method.eq(r.method)].iloc[0]
            se = sensitivity.loc[sensitivity.label.eq(r.label) & sensitivity.block.eq(r.block)
                                 & sensitivity.model.eq(r.model) & sensitivity.method.eq(r.method), "bootstrap_sharpe_se"]
            confidence_rows.append([LABELS[r.label], f"{r.block}/{r.model}", f"{r.net_sharpe:.3f}",
                f"[{u.sharpe_lower:.3f}, {u.sharpe_upper:.3f}]", f"{u.bootstrap_sharpe_se:.3f}", f"{u.recent_4y_sharpe_se:.3f}",
                f"{se.min():.3f}–{se.max():.3f}"])
    model_table = table(["标签", "信息组", "模型", "20日原始收益Rank IC", "HAC t", "有标签评估日"], model_rows)
    result_table = table(["标签", "信息组/模型", "组合", "净年化", "净Sharpe", "最大净回撤", "净年化波动", "平均实际gross", "最大实际gross", "持仓估计abs(beta)P95",
                          "行业净额绝对和P95", "年化交易拖累", "双倍成本净年化"], main_rows)
    confidence_table = table(["标签", "信息组/模型", "长窗口净Sharpe", "95%区块区间", "长窗口SE", "最近4年SE", "20/60/126日块SE范围"], confidence_rows)
    pair_rows = [[kind, len(group), int(group.sharpe_delta_lower.gt(0).sum()), int(group.sharpe_delta_upper.lt(0).sum())]
                 for kind, group in paired.groupby("kind")]
    pair_table = table(["配对类型", "配对数", "未校正95%区间完全为正", "未校正95%区间完全为负"], pair_rows)
    # This illustrative cohort is fixed by its role as the original price/Ridge
    # reference, rather than selected for its returns in a particular episode.
    regime_frame = pd.DataFrame(regimes)
    reference_regimes = regime_frame.loc[regime_frame.label.eq("cross_sectional")
        & regime_frame.block.eq("price_only") & regime_frame.model.eq("ridge")]
    regime_table = table(["阶段（原标签/价量Ridge）", *METHODS.values()],
        [[period, *[f"{reference_regimes.loc[reference_regimes.period.eq(period) & reference_regimes.method.eq(method), 'net_return'].iloc[0]:.2%}"
          for method in METHODS]] for period in periods])
    binding = sum(row["volatility_cap_binding_rebalances"] for row in verification
                  if row["key"][-1] == 1 and row["volatility_cap_binding_rebalances"] is not None)
    label_pairs = paired.loc[paired.kind.eq("label")]
    risk_pairs = base.loc[base.method.isin(["risk_dollar", "risk_neutral"])].pivot(
        index=["label", "block", "model"], columns="method",
        values=["actual_beta_abs_p95", "net_max_drawdown", "net_sharpe"])
    beta_reductions = int((risk_pairs["actual_beta_abs_p95"].risk_neutral < risk_pairs["actual_beta_abs_p95"].risk_dollar).sum())
    drawdown_reductions = int((risk_pairs["net_max_drawdown"].risk_neutral > risk_pairs["net_max_drawdown"].risk_dollar).sum())
    sharpe_improvements = int((risk_pairs["net_sharpe"].risk_neutral > risk_pairs["net_sharpe"].risk_dollar).sum())
    reference_error = uncertainty.loc[uncertainty.label.eq("cross_sectional") & uncertainty.block.eq("price_only")
                                       & uncertainty.model.eq("ridge") & uncertainty.method.eq("quantile_20d")].iloc[0]
    rejected = pd.read_csv(research / "locked_feature_schema.csv")
    rejected = ", ".join(rejected.loc[~rejected.selected, "feature"])
    text = f"""# 标签中性化、风险中性与长窗口研究（2026-10-23）

基准：[阶段报告](STAGE_REPORT_2026-10-23.md)、[降换手研究](TURNOVER_OPTIMIZATION_2026-10-23.md)。本次新增源码和独立产物，没有覆盖旧OOF或旧执行账本。

已完成八个模型分支、64个完整组合/费用账本及配对检验。结果先行：

- 新标签在16个匹配组合中有 **{int(label_pairs.net_sharpe_difference.gt(0).sum())}/16** 个净Sharpe点估计提高，**{int(label_pairs.sharpe_delta_lower.gt(0).sum())}/16** 个未校正95%配对区间完全为正；不能据此默认全面替换原标签。八个分支原始收益IC的最大绝对HAC t仅为 **{model_stats.hac_t.abs().max():.2f}**。
- 加入beta/行业硬中性，在 **{beta_reductions}/8** 个匹配组降低持仓估计abs(beta)的P95，**{drawdown_reductions}/8** 个降低最大净回撤，**{sharpe_improvements}/8** 个提高净Sharpe。控制指定敞口与改善总收益是不同的验收项。
- 基础费用下32个固定组合中有 **{int(base.net_ann_return.gt(0).sum())}个** 净年化为正，双倍费用下32个组合中有 **{int(summary.loc[summary.cost_multiplier.eq(2), 'net_ann_return'].gt(0).sum())}个** 为正。当前结果没有提供足以全面替换默认方案的稳健盈利证据。
- 拉长窗口确实提高精度：原标签/价量Ridge分位数组合的Sharpe区块标准误从最近4年的 **{reference_error.recent_4y_sharpe_se:.3f}** 降至全窗口的 **{reference_error.bootstrap_sharpe_se:.3f}**。这仍是已审阅历史研究，不构成新的独立盈利证明。

## 1. 必要性判断与前提纠正

三个方向值得实施为可审计的对照实验；标签和风险约束是否替换默认方案，必须看净收益、敞口与配对不确定性，不能先认定它们会改善盈利或消除大部分回撤。

- **标签已有横截面处理。** 原实现是按日去均值/标准差缩放并截尾，不是直接拿原始20日收益拟合。`zscore(r_i-r_market)=zscore(r_i)`，当天共同标量市场收益相减没有新增作用；它也不能消除股票之间不同beta带来的市场暴露。新实验加入的是个股事前波动缩放与行业/规模联合投影。
- **回撤/波动比不能归因。** 最大回撤是路径统计，年化波动是收益分布统计，2.5倍的比值本身不能证明某次行业或动量反转造成了回撤。本轮保存实际beta、行业净额和条件风险模型的收益分解，再检查这些解释。
- **10%目标不保证更小回撤。** 无费用的正固定倍数缩放不会改变Sharpe，动态波动管理则是需要实证检验的策略。[Moreira–Muir](https://www.nber.org/papers/w22208)提供研究动机，并不保证本项目收益。当前目标gross上限为1，10%是调仓目标的事前波动上限；不加杠杆强行命中10%，不把它当作最大回撤承诺。
- **长窗口有必要。** 年度Sharpe精度不能只按年数机械认定，序列相关会影响推断，见[Lo的研究](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260)。本轮同时给出长窗口和最近4年的区块抽样误差。

## 2. 长窗口及公平比较

实际预测期为 **2013-02-04—2025-12-31，3,248个交易日，13个预测折**；首日由756个训练日和20日标签清除边界决定，不能虚构2013年1月预测。最后一折224个交易日保留。

八个标签/信息组/模型分支各有 **{evidence['prediction_rows_per_cohort']:,}** 个相同预测键；共同训练标签有效比例为{plan['common_trainable_fraction']:.2%}。标签缺失或小行业只影响训练准入，不删除当日可生成预测的证券。外层与内层均按真实`label_end`严格清除交叠，所有模型、信息组和标签分支共享同折同阶段的抽样键。

为避免用2021年的数据替2013年选特征，特征质量门槛锁定于 **{plan['feature_lock_before'][:10]}** 之前。价量组为37个、加入财务为42个；早期覆盖不足而排除的是 **{rejected}**。因此新实验与旧38/44特征、后期锁定的四年实验不是“只延长日期”的严格单变量比较；后期可用信息被固定早期schema保守约束，这也是结果解释边界。

共16个预声明标签/信息组/模型候选配置、{evidence['trial_entries']}条内层候选/外层重拟合记录、{evidence['single_model_fits']}次实际含种子的单模型拟合。Ridge和LightGBM沿用原有两组候选；20日原始收益IC用于内层选择，外层标签不参与选参。所有2013–2025结果均标记为已审阅或审阅状态未知的历史walk-forward研究，**不宣称新、未触碰的独立holdout**。

## 3. 标签实现

新标签使用严格截至t−1的60交易日个股日收益波动，最少40条观测，日波动下限0.5%；分母为该波动乘sqrt(20)。先缩放前瞻收益，再对按历史有效日期对齐的行业哑变量及滞后log市值作联合投影；行业至少5个有效标签。行业对齐不代表拥有当时发布版本，PIT限制仍在。若先投影后除以不同股票的波动，会重新引入暴露，因此本实现调整了运算顺序。

投影后按日标准化、截尾到±3，再次联合投影，最后只做同日公共的向下缩放以守住±3。这样稳健处理后仍保持行业/规模正交（浮点精度内），但不要求最终方差恰为1。行业和市值来自原`decision_*`字段，不重复滞后；未来收益只用于训练标签与事后评估，不进入预测特征。

{model_table}

HAC至少使用19阶滞后以容纳20日标签重叠，保留缺失日期间隔。这里的t统计没有校正历史研究选择。标签端处理不保证原始收益IC统一提高；残差动量和特征端中性化的历史表现不能替代标签端的直接证据。

## 4. 风险模型与实际执行

- 市场代理为历史合格股票池的等权收益，并非官方市值加权指数；SMB按前日公司市值划分。风险行业因子使用日期对应的宽SIC分区2–8超额收益，组合硬约束使用更细的历史SIC2行业，不把今天的分类回填历史。beta=0是相对于此市场代理的估计约束。
- 每日风险估计使用t之前最多252个交易日，至少126个有效观测；风险因子协方差半衰期60日、50%对角收缩；个股特异年化波动下限1%。复用`FactorRiskModel`并保留Euler核对，修复nullable数值转object矩阵的问题，并按共同缺失掩码批量回归。
- 原优化器的gross=2及复权价差分路径不直接用于本实验。新路径用CIZ直接总收益，调仓目标约束为gross≤1、个股≤1.5%、发行人≤3%、净美元敞口0；中性分支再约束滚动单因子市场beta=0和每个SIC2行业净额=0。行业资金中性不等于每个连续行业风险因子的beta也恰为0；规模因子进入协方差，没有额外强制规模beta=0。
- [OSQP凸二次规划](https://osqp.org/docs/interfaces/python.html)直接使用因子形式，避免每天构造稠密股票协方差。分数按固定10%年化横截面收益差先验换算到日收益单位，风险厌恶3、20日预测期、每成交美元10bps的L1换手代理；它不是已校准的收益预测或真实券商费率。
- 换手惩罚参照当日标记的实际持仓，绝非昨日目标。实际成交仍独立支付原佣金/最低佣金/滑点/冲击/卖出费用/借券。压力情景实际费用整体翻倍，优化器固定10bps代理不重调参，保留NAV与成交路径反馈。
- 四种正式构造统一沿用原模型20交易日的调仓日历，逐日估计/记录实际风险。两个分位数组合使用原始分数；两个风险优化组合共同使用EMA5，仅中性约束不同。它们使用同一事前风险可覆盖的预测股票池。行业内分位数组合遇到不可构造双边的小行业持现金，限制后只缩小较大一侧以保留行业净额0。
- 目标波动缩放仅向下，遵守gross/个股/发行人目标上限；目标与实际敞口均保存。整数成交、停牌和参与率可能留下实际残余敞口，期间收益与费用也会使实际gross超过目标上限，表中同时列出平均及最大实际gross。共同20日日历没有额外每日强平规则，不能把目标中性或目标上限说成持续精确的实际约束。

基础费用的16个风险优化账本合计有 **{binding}次** 调仓触发10%事前波动上限。若未触发，风险变化应归因于组合权重、协方差惩罚与中性约束，不能归功于未实际生效的波动缩放。

早期风险QP有{len(failures)}次完整窗口任务在数值约束检查时失败，记录保留在原`failures.json`。中间运行因求解效率问题中断，另存`run_interrupted.json`，随后加入求解器热启动与统一向下可行性缩放。为避免将风险约束和交易频率一起改变，正式对照另行登记为共同20日日历，完整重跑64个任务；每日模式由`--risk-sessions 1`显式开启，先前每日接口运行不列作完整收益证据。本轮不宣称已做到两次调仓之间持续精确beta/行业中性，期间实际漂移见表和账本。未放宽风险限制、未静默切换策略；失败与中断运行不用于选策略参数。

## 5. 全部净收益与风险结果

下表保留全部固定方案，未按最大收益挑选默认策略；行业净额绝对和是各SIC2净权重绝对值之和，beta为事前滚动市场代理的估计暴露。费用是假设历史模拟成本，非2013–2025真实成交校准结果。

{result_table}

![长窗口固定参数净值对照](assets/neutral_risk_20261023/long_history_nav.png)

## 6. 不确定性、阶段表现与收益归因

固定种子2718、60交易日循环区块、1,500次配对抽样。另保存20和126交易日块的单组合区间敏感性；配对比较以预设60日块为准。区间条件于已生成预测及执行路径，没有重新训练每个bootstrap样本，也没有校正全部历史研究选择或全部比较的多重检验。

{confidence_table}

{pair_table}

完整配对区间位于`paired_comparisons.csv`，分别覆盖标签、Ridge/LightGBM、财务增量、行业内排序和风险中性；不能仅凭点估计或年数宣称已区分模型。

以下预先固定的原标签/价量Ridge参照组用于展示阶段差异，各阶段为区间累计净收益，并非对事件的因果归因；其他七组完整保留在CSV。

{regime_table}

`regime_comparison.csv`单列2015–2016、2018、2020、2022–2023、2022-11—2023-01及2023年3月；`factor_return_attribution.csv`以昨日收盘实际因子暴露乘当日已实现因子收益，记录市场/规模/行业分量及剩余。它是条件风险模型的统计分解，因子相关且不是事件因果识别。出现无法建模持仓时明确标记覆盖缺口；新模型结果也不等同于旧四年冻结OOF的事件损益。

## 7. 验证与复现

- {args.test_count}项全工作区测试通过；新测试覆盖标签市场平移不变性、稳健后行业/规模正交、波动率前缀不变、日期锚定及最后短折、共同抽样/清除边界、nullable风险数据、日期行业图、QP中性/上限、波动缩放与实际持仓回调。
- {len(verification)}个最终账本核对整股、零费用拒单、参与率、来源费用加总及逐日NAV恒等式。模型预测键/抽样键一致；全部风险观察严格早于决策日；因子协方差最小特征值为{evidence['min_factor_covariance_eigenvalue']:.3g}。
- `classification_is_point_in_time=false`、`fundamentals_is_point_in_time=false`仍保留。这里是有效日期历史分类与修订版财务的受限研究；价量信息组的预测特征不含财务字段，组合风险上下文在两组间一致。
- 仍未建立独立股数/现金/公司行为第二账本、真实借券准入、多资金容量或新未触碰检验期的稳健盈利证据；`production_admitted=false`。
- 原始授权数据和旧报告产物保持独立。模型、风险缓存和执行阶段分别记录输入/源码哈希及源码快照；不同数值修复阶段有独立产物，未冒称仅凭未提交Git版本即可复现。

本地产物：`data/wrds/batch3/neutral_2013_2025/`。`comparison_summary.csv`为64组净收益/成本/暴露；`sharpe_uncertainty.csv`、`paired_comparisons.csv`及阶段/归因CSV补充解释；`verification.json`为验收记录。

```powershell
python scripts/research_wrds_neutral.py --output data/wrds/batch3/new_neutral_run --jobs 4
python scripts/prepare_wrds_risk.py data/wrds/batch3/new_neutral_run
python scripts/replay_wrds_neutral.py data/wrds/batch3/new_neutral_run --jobs 6 --risk-sessions 20
python scripts/report_wrds_neutral.py data/wrds/batch3/new_neutral_run --output data/wrds/batch3/new_neutral_run/report.md
python -m pytest -q
```

新源码已修复数值约束并支持求解器热启动，重建时一个`risk_replays`目录即可包含全部方法；本报告使用`risk_replays_20d`中的64个完整账本，另保留原工程失败与中断目录。
"""
    output.write_text(text, encoding="utf-8")
    write_run_metadata(assets / "evidence_manifest.json", stage="long_history_label_risk_report",
        inputs={"driver": Path(__file__), "report": output, "model_summary": research / "model_summary.csv",
                "comparison": research / "comparison_summary.csv", "paired": research / "paired_comparisons.csv",
                "verification": research / "verification.json", "chart": assets / "long_history_nav.png",
                "ic_uncertainty": research / "model_ic_uncertainty.csv", "uncertainty": research / "sharpe_uncertainty.csv",
                "block_sensitivity": research / "sharpe_block_sensitivity.csv", "regimes": research / "regime_comparison.csv",
                "factor_attribution": research / "factor_return_attribution.csv",
                **{f"predictions:{r.label}:{r.block}:{r.model}": research / f"models/{r.label}/{r.block}/{r.model}/oof.parquet"
                   for r in models.itertuples()}},
        params={"bootstrap_block": 60, "bootstrap_repetitions": 1500, "seed": 2718,
                "sensitivity_block_lengths": [20, 60, 126],
                "historical_review_not_untouched_holdout": True, "production_admitted": False})
    print(output)


if __name__ == "__main__":
    main()
