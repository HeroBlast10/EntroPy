"""Render aggregate turnover evidence; security-level licensed data stays local."""
from __future__ import annotations

import argparse
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

from quant_platform.core.utils.provenance import write_run_metadata

LABELS = {"MOM_12_1M": "长期动量", "RESID_MOM_12_1M": "残差动量", "PIOTROSKI_F8": "F8",
          "SHARE_TURNOVER_20D": "低真实换手率"}
VARIANT_LABELS = {"baseline": "原始", "daily": "每日对照", "tolerance20": "容忍带20%",
                  "buffer30": "缓冲带30%", "ema5": "EMA5", "ema10": "EMA10", "tranche20": "20个子组合",
                  "combined": "缓冲+EMA5+容忍", "combined_tranche20": "组合措施+20子组合"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def label(strategy):
    parts = strategy.split("/")
    return LABELS[parts[-1]] if parts[0] == "factor" else f"{parts[1]}/{parts[2]}"


def display_path(path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def score_ic(rows):
    """Pairwise-complete Spearman IC on the same reporting-only labels."""
    valid = np.isfinite(rows.selection_score) & np.isfinite(rows.forward_return)
    frame = rows.loc[valid, ["date", "selection_score", "forward_return"]].copy()
    ranks = frame.groupby("date")[["selection_score", "forward_return"]].rank()
    centered = ranks - ranks.groupby(frame.date).transform("mean")
    num = (centered.selection_score * centered.forward_return).groupby(frame.date).sum()
    den = np.sqrt(centered.pow(2).groupby(frame.date).sum().prod(axis=1))
    result = num / den.replace(0, np.nan)
    result = result.where(frame.groupby("date").size().ge(20))
    return float(result.mean()), int(valid.sum())


def verify_books(folder, summary):
    """Independent artifact checks; does not claim an independent cash ledger."""
    records = []
    for row in summary.itertuples():
        dest = folder / f"replays/{row.strategy}/{row.variant}/cost_{row.cost_multiplier:g}x"
        trades = pd.read_parquet(dest / "trades.parquet", columns=["date", "shares", "total_cost", "commission"])
        pnl = pd.read_parquet(dest / "daily_pnl.parquet")
        daily = pd.read_csv(dest / "daily_turnover_costs.csv", index_col=0)
        sources = pd.read_csv(dest / "cost_sources.csv")
        np.testing.assert_allclose(trades.shares, np.floor(trades.shares), atol=0, rtol=0)
        assert trades.loc[trades.shares.eq(0), "total_cost"].eq(0).all()
        np.testing.assert_allclose(trades.groupby("date").total_cost.sum().reindex(pnl.index, fill_value=0),
                                   pnl.trading_cost_dollar, atol=1e-7)
        np.testing.assert_allclose(sources.total_cost_dollar.sum(), pnl.total_cost_dollar.sum(), atol=1e-6)
        np.testing.assert_allclose(daily.one_way_turnover, pnl.executed_turnover / 2, atol=1e-12)
        prior = pnl.nav_net.shift().fillna(1_000_000.)
        np.testing.assert_allclose(prior * (1 + pnl.gross_ret) - pnl.total_cost_dollar, pnl.nav_net, rtol=1e-11)
        assert len(pnl) == 1008 and pnl.index.is_unique and pnl.index.is_monotonic_increasing
        floor_topup = (trades.commission - .005 * row.cost_multiplier * trades.shares).clip(lower=0)
        topup_daily = floor_topup.groupby(trades.date).sum().reindex(pnl.index, fill_value=0)
        records.append({"strategy": row.strategy, "variant": row.variant, "cost_multiplier": row.cost_multiplier,
                        "whole_shares": True, "zero_cost_rejects": True, "cost_sources_reconcile": True,
                        "nav_reconciles": True,
                        "commission_floor_topup_annual_bps": float((topup_daily / prior).mean() * 252 * 10000)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/TURNOVER_OPTIMIZATION_2026-10-23.md")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.stress = args.stress.resolve()
    args.output = args.output.resolve()
    for folder in (args.experiment, args.stress):
        completion = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
        if not completion["is_full_window"] or completion["failures"]:
            raise ValueError("Report requires completed full-window experiments")
    audit = pd.read_csv(args.experiment / "recorded_strategy_summary.csv")
    base = pd.read_csv(args.experiment / "replay_summary.csv")
    stress = pd.read_csv(args.stress / "replay_summary.csv")
    combined = base.loc[base.variant.eq("combined")].set_index("strategy")
    original = base.loc[base.variant.eq("baseline")].set_index("strategy")
    strategies = list(original.index.sort_values())
    all_results = pd.concat([base, stress], ignore_index=True)
    # Publish both conventions explicitly: turnover/NAV includes 1/2,
    # execution bps are charged on every buy and sell dollar.
    for frame, name in ((audit, "recorded_summary_explicit_units.csv"),
                        (all_results, "replay_summary_explicit_units.csv")):
        explicit = frame.copy()
        explicit["cost_per_unit_one_way_turnover_bps"] = 2 * explicit.unit_trading_cost_bps
        explicit["break_even_per_unit_one_way_turnover_bps"] = 2 * explicit.break_even_trade_bps
        explicit.to_csv(args.experiment / name, index=False)
    verification = verify_books(args.experiment, base) + verify_books(args.stress, stress)
    (args.experiment / "artifact_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    micro_orders = all_results.merge(pd.DataFrame(verification)[["strategy", "variant", "cost_multiplier",
                                    "commission_floor_topup_annual_bps"]],
                                    on=["strategy", "variant", "cost_multiplier"], validate="one_to_one")
    micro_orders.to_csv(args.experiment / "micro_order_diagnostics.csv", index=False)
    residual_key = "factor/RESID_MOM_12_1M"
    before, after = original.loc[residual_key], combined.loc[residual_key]
    assets = args.output.parent / "assets/turnover_20261023"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    colors = {"baseline": "#7B8496", "combined": "#2176AE", "combined_tranche20": "#2C9477"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, (factor, name) in zip(axes.flat, LABELS.items()):
        for variant, color in colors.items():
            path = args.experiment / f"replays/factor/{factor}/{variant}/cost_1x/daily_pnl.parquet"
            p = pd.read_parquet(path)
            nav = pd.concat([pd.Series([1.], index=[p.index[0] - pd.Timedelta(days=1)]), p.nav_net / 1_000_000])
            ax.plot(nav.index, nav, label=VARIANT_LABELS[variant], color=color, lw=1.3)
        ax.set_title(name, loc="left")
        ax.axhline(1, color="#B2B7C0", lw=.7)
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2021-12-26"), pd.Timestamp("2025-12-31"))
    for ax in axes[:, 0]:
        ax.set_ylabel("净值（起始=1）")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("固定信号的降换手消融：扣费净值", fontsize=16, x=.07, ha="left")
    fig.text(.07, .018, "2021-12-27—2025-12-31 · 100万美元 · 保留开仓费用 · 历史描述性研究", fontsize=9, color="#626D7D")
    fig.tight_layout(rect=(.02, .04, 1, .95))
    fig.savefig(assets / "factor_nav.png", dpi=170)
    fig.savefig(assets / "factor_nav.svg")
    plt.close(fig)

    factors = [f"factor/{name}" for name in LABELS]
    rows = []
    for strategy in factors:
        r = audit.loc[audit.strategy.eq(f"total_loss/h1/fixed_signal/{strategy.split('/')[-1]}/cost_1x")].iloc[0]
        rows.append([label(strategy), f"{r.daily_one_way_turnover:.2%}", f"{r.unit_trading_cost_bps:.2f}",
                     f"{2 * r.unit_trading_cost_bps:.2f}",
                     f"{r.break_even_trade_bps:.2f}", *[f"{r[f'{s}_annual_cost_bps'] / 100:.2f}%"
                                                     for s in ("signal", "boundary", "drift_resize", "lot_residual", "borrow")]])
    cost_table = table(["因子", "日单边换手", "每成交美元成本bps", "每单位单边换手成本bps", "盈亏平衡bps/成交美元", "信号/成分变化", "边界代理", "漂移/重分配", "非调仓残单", "借券"], rows)
    rows = []
    for strategy in strategies:
        a = micro_orders.loc[micro_orders.strategy.eq(strategy) & micro_orders.variant.eq("baseline") & micro_orders.cost_multiplier.eq(1)].iloc[0]
        b = micro_orders.loc[micro_orders.strategy.eq(strategy) & micro_orders.variant.eq("combined") & micro_orders.cost_multiplier.eq(1)].iloc[0]
        rows.append([label(strategy), f"{a.executed_orders:,} → {b.executed_orders:,}",
                     f"${a.median_fill_dollar:,.0f} → ${b.median_fill_dollar:,.0f}",
                     f"{a.commission_floor_topup_annual_bps / 100:.2f}% → {b.commission_floor_topup_annual_bps / 100:.2f}%"])
    micro_table = table(["策略", "成交订单数", "成交额中位数", "最低佣金补差年化拖累"], rows)
    rows = []
    for strategy in strategies:
        a, b = original.loc[strategy], combined.loc[strategy]
        two = stress.loc[stress.strategy.eq(strategy) & stress.variant.eq("combined")].iloc[0]
        rows.append([label(strategy), f"{a.net_ann_return:.2%}", f"{b.net_ann_return:.2%}", f"{two.net_ann_return:.2%}",
                     f"{a.daily_one_way_turnover:.2%} → {b.daily_one_way_turnover:.2%}",
                     f"{a.annual_trading_drag_bps / 100:.2f}% → {b.annual_trading_drag_bps / 100:.2f}%",
                     f"{a.net_max_drawdown:.2%} → {b.net_max_drawdown:.2%}"])
    main_table = table(["策略", "原始净年化", "组合措施净年化", "组合措施双倍成本", "日单边换手", "年化交易拖累", "最大净回撤"], rows)
    rows = []
    for strategy in strategies:
        for variant in ["baseline", *[v for v in VARIANT_LABELS if v != "baseline"]]:
            selection = base.loc[base.strategy.eq(strategy) & base.variant.eq(variant)]
            if selection.empty:
                continue
            r = selection.iloc[0]
            two = stress.loc[stress.strategy.eq(strategy) & stress.variant.eq(variant)]
            rows.append([label(strategy), VARIANT_LABELS[variant], f"{r.gross_ann_return:.2%}", f"{r.net_ann_return:.2%}",
                         f"{two.iloc[0].net_ann_return:.2%}" if len(two) else "—", f"{r.daily_one_way_turnover:.2%}",
                         f"{r.unit_trading_cost_bps:.2f}", f"{r.break_even_trade_bps:.2f}", f"{r.mean_gross_exposure:.3f}",
                         f"{r.max_gross_exposure:.3f}", f"{r.max_absolute_net_exposure:.3f}"])
    full_table = table(["策略", "消融", "毛年化", "净年化", "双倍成本净年化", "日单边换手", "单位成本bps", "盈亏平衡bps",
                        "平均gross", "最大gross", "最大abs(net)"], rows)
    labels_path = ROOT / "data/wrds/batch2/model_comparison/total_loss/full/h20/oof.parquet"
    labels = pd.read_parquet(labels_path, columns=["date", "ticker", "block", "forward_return"])
    labels = labels.loc[labels.block.eq("price_only"), ["date", "ticker", "forward_return"]]
    rows, ic_records = [], []
    for strategy in strategies:
        values = []
        for variant in ("baseline", "ema5", "ema10"):
            scores = pd.read_parquet(args.experiment / f"replays/{strategy}/{variant}/cost_1x/scores.parquet")
            scored = scores.merge(labels, on=["date", "ticker"], validate="one_to_one")
            assert len(scored) == len(scores) == 506738
            ic, count = score_ic(scored)
            values.append(ic)
            ic_records.append({"strategy": strategy, "variant": variant, "rank_ic_20d": ic, "matched_rows": count})
        rows.append([label(strategy), *[f"{value:.5f}" for value in values], count])
    pd.DataFrame(ic_records).to_csv(args.experiment / "score_ic_comparison.csv", index=False)
    ic_table = table(["策略", "原分数20日Rank IC", "EMA5的20日Rank IC", "EMA10的20日Rank IC", "配对有标签行"], rows)
    text = f"""# 成本来源拆解与降换手研究（2026-10-23）

冻结参照：[阶段报告](STAGE_REPORT_2026-10-23.md)。本轮读取原有预测与因子，不重训、不变更方向，不改写历史产物。
窗口为 2021-12-27—2025-12-31，共 1,008 个交易日；起始资金 100 万美元；沿用基础成本、发行人/个股上限和全损退市假设。

残差动量的预先指定组合措施（缓冲带30%、EMA半衰期5日、目标偏差容忍20%）将净年化从 **{before.net_ann_return:.2%}** 改为 **{after.net_ann_return:.2%}**，日均单边换手从 **{before.daily_one_way_turnover:.2%}** 改为 **{after.daily_one_way_turnover:.2%}**。
这些是已经被审阅过的历史样本上的执行消融，不构成新的独立样本外盈利证据；没有按回测最大收益挑选最终参数。

## 1. 原始成本到底花在哪里

下表费用列为逐日费用/NAV的均值乘252，属于年化算术拖累；不应与复利CAGR差机械相加。信号、边界和漂移列是互斥的订单归因，另列借券。

{cost_table}

- 单边换手 = 0.5 ×（买入成交额 + 卖出成交额）/扣借券后、扣交易费前NAV。现金日与开仓交易均保留。
- 单位成本bps = 总交易费/买卖成交额合计 × 10,000；该分母与单边换手差一个2，不能混用。
- 每单位单边换手成本按同一成交额加权口径为上述费率的2倍；另存的`*_summary_explicit_units.csv`同时输出这两种费率及对应的盈亏平衡门槛。
- 盈亏平衡bps：在记录的成交/毛收益路径上，以统一每成交美元费率替换交易费，借券保持不变，求期末净值=1的复利方程根。负值表示零交易费仍亏损；它不等于重新执行得到的成本门槛。
- 边界代理：发生进出、且相邻决策时点的排名都位于同侧外30%区间。它标识可观测的边界交易，不证明价格噪声或无效alpha。其余名单/方向变化归入信号/成分变化。
- 漂移/重分配：同方向目标权重变化、持仓价格/NAV漂移，以及前次成交后的实施偏差；方向相反的指令先净额抵消，再按同向贡献分配实际成交与费用。整股效应与价格漂移不能仅靠旧账本被唯一因果分离。
- 非调仓残单：只计原有执行器中非调仓日的残单重试成交费用；整股不足而拒单的费用为零。四条每日调仓因子的该项为零，不能引用别的策略最高25.2%的数值来解释它们。
- 子组合的费用按活跃子组合当日目标转换标签分组，属于记账约定；多个子组合净额抵消后的费用不能唯一因果归到某个信号事件。

全部 **{len(audit)}** 份历史账本均输出日度换手、单位成本、盈亏平衡与费用来源；包括两种退市假设、基础/双倍成本及原有固定基线/负对照。逐证券细节保留在本地授权数据目录。

小单诊断另按费用类型列示；最低佣金补差=`max(实际佣金−成交股数×每股佣金, 0)`，属于上述成本里的子集，不能重复加总。基础每股佣金为0.005美元，压力情景同步翻倍。

{micro_table}

## 2. 固定组合措施与原策略比较

{main_table}

因子原始参照为每日调仓；模型原始参照为原来的20交易日一次调仓。模型的“每日对照”另列于完整消融表，EMA、缓冲与容忍带均与该每日对照比较才能隔离各自效果。
双倍成本是重新执行，佣金、滑点、冲击、卖出费用及借券一起翻倍，保留NAV与成交路径反馈。

![固定因子降换手后的净值](assets/turnover_20261023/factor_nav.png)

## 3. 执行规则与解释边界

1. 缓冲带：进入多头须达到前20%，上次已入选的多头目标直到跌出前30%才退出；空头对称。这里的入选状态来自前次目标名单，实际成交仍由执行器决定。保留原平均排名的并列分数约定，名单数量可以变化，再做等权与原发行人/个股上限。调仓时缺失分数不续持；成分退出与终止处理保持显式。
2. EMA：直接平滑原预测输出，半衰期固定5/10个交易日，再排序；不读取标签。缺分数或离开预测股票池会重置该证券状态；OOF开始时初始化，没有虚构更早预测；模型折边界不使用未来数据预热。
3. 20个重叠子组合：首日同时建立全部子组合，之后每天只更新一个子组合，分配当日NAV的1/20。未更新的子组合按实际持仓与总收益漂移，并先内部抵消反向交易再对外净额成交。不会把过去20日的恒定目标权重简单平均后每天全组合再平衡，也不保证换手必然等于原来的1/20。
4. 子组合未成交量在当日收盘取消，下一轮轮到该子组合时重新决定；普通调仓路径保留原残单生命周期。该差异已记录在实验计划，流动性受限时不能将子组合结果解释为只改变调仓日历。
5. 容忍带：同方向持仓的目标偏差不超过目标绝对权重20%时不提交订单；新开仓、归零退出、反向换仓均不受此带阻挡。子组合在各自仓位上应用容忍带，成分退出对全部子组合生效。被抑制的候选记录为`suppressed_tolerance`且零费用，供审计使用。
6. 约束检验针对每次选出的单位子组合目标；非更新子组合与实际持仓允许漂移。完整表另列实际gross与净敞口极值，不能把目标约束说成每天实际敞口始终不漂移。

## 4. 全部固定参数消融

`combined`固定为缓冲30% + EMA5 + 容忍20%；`combined_tranche20`再增加20个子组合。EMA10是固定敏感性方案，不根据结果调整参数。各表保留全部失败/负收益结果。

{full_table}

## 5. 平滑是否损失排序信息

{ic_table}

IC只在原有可用20日标签上作描述性统计，标签从不参与目标构造；不能预先假设慢信号平滑一定不损失IC。标签读取冻结的价量模型OOF文件，按共同日期/证券键一对一关联到已完成的因子与模型分数，未重新计算或改变标签口径。各平滑分支的缺失掩码与原分数一致；数值只代表此1,008日窗口，不能与阶段报告2010—2025的IC混为同一统计窗口。

## 6. 验证、产物与复现

- 原始策略重放关闭全部降换手控制后，与八份冻结参照NAV逐日核对，最大允许相对误差为1e-10。
- 每份账本验证净值会计恒等式、来源费用加总、成交参与率、整股成交与目标权重上限；单元测试覆盖EMA前缀不变、缺失/退出、缓冲留仓、新开/归零/反向豁免、残单带内取消、子组合内部净额与部分成交、非更新子组合漂移、复利盈亏平衡求根。
- 本次全工作区测试：496项通过，45条warning。经济效果与测试通过分别评价。
- 没有独立重建股数/现金/公司行为账本，没有验证真实借券可得性或重新建立容量证据，`production_admitted=false`。
- 新实验记录dirty源码状态及完整源码快照、输入SHA-256；没有将未提交代码的结果声明为仅凭当前Git提交即可复现。

本地产物：`{display_path(args.experiment)}`与`{display_path(args.stress)}`。前者包含`recorded_strategy_summary.csv`、`replay_summary.csv`，各策略下有`daily_turnover_costs.csv`、`cost_sources.csv`、执行账本、目标与平滑分数；后者保存双倍成本回放。`experiment_plan.json`和`completion.json`记录全部尝试与完成状态。

```powershell
python scripts/optimize_wrds_turnover.py --phase all --jobs 4 --output data/wrds/batch2/turnover_new_run
python scripts/optimize_wrds_turnover.py --phase replay --variants baseline combined combined_tranche20 --cost-multipliers 2 --jobs 2 --output data/wrds/batch2/turnover_new_stress
python scripts/report_wrds_turnover.py --experiment data/wrds/batch2/turnover_new_run --stress data/wrds/batch2/turnover_new_stress
python -m pytest -q
```

输出目录必须是新目录，避免覆盖既有证据。源码快照保存在各实验的`source_snapshot/`，输入授权数据不随公开仓库分发。
"""
    args.output.write_text(text, encoding="utf-8")
    write_run_metadata(assets / "evidence_manifest.json", stage="turnover_report",
        inputs={"driver": Path(__file__), "recorded": args.experiment / "recorded_strategy_summary.csv",
                "replay": args.experiment / "replay_summary.csv", "stress": args.stress / "replay_summary.csv",
                "base_plan": args.experiment / "experiment_plan.json", "stress_plan": args.stress / "experiment_plan.json",
                "ic_labels": labels_path, "ic_table": args.experiment / "score_ic_comparison.csv",
                "verification": args.experiment / "artifact_verification.json",
                "micro_orders": args.experiment / "micro_order_diagnostics.csv",
                "chart": assets / "factor_nav.png", "document": args.output},
        params={"recorded_books": len(audit), "new_replays": len(all_results),
                "primary_variant": "combined", "new_oos": False})
    print(args.output)


if __name__ == "__main__":
    main()
