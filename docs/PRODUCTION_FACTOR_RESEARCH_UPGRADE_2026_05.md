# EntroPy 生产级因子研究升级说明

本文档记录 2026-05 这次升级的完整内容，目标是把 EntroPy 从“能跑多因子回测”推进到“能筛出互补、稳定、可上线候选因子”的研究框架。

## 1. 已修复的研究口径问题

### 1.1 统一 effective signal

新增 `quant_platform/core/signals/effective.py` 与 `orientation.py`，所有研究与组合入口统一使用：

```text
FactorMeta.direction -> winsorize -> neutralize -> z-score -> rank
```

影响范围包括：

- 单因子 IC、RankIC、分层收益、换手、容量评估。
- `run_portfolio_pipeline` 组合构建。
- ML alpha、cross-sectional ranker、time-series forecaster。
- walk-forward、ablation、report 评估链路。

这修复了低波、短反转、低资产增长等“低值更好”因子被错误当成“高值更好”的问题。

### 1.2 ASSET_GROWTH 同比修复

`ASSET_GROWTH` 不再用 `shift(252)` 处理基本面。基本面不是日频数据，旧逻辑会把披露事件频率误当交易日频率。

新逻辑按报告期对齐上一年同季，避免错配财报频率，减少因子定义层面的假信号。

### 1.3 共享特征缓存真正生效

`FactorRegistry.compute_all` 传入的 `_feature_cache` 现在能通过 `FactorBase.compute(..., **kwargs)` 进入 `_compute`。动量、波动、流动性等因子可复用中间特征，之前“写了缓存但没接上”的性能设计已经补通。

### 1.4 typed factors 接入主线

新增 `signals/catalog.py`，把 cross-sectional、time-series、regime、relative-value 的评估产物合并到统一 factor catalog。`run_factor_pipeline.py` 与报告入口不再只看 `factor_comparison.csv`，typed factors 可以参与自动筛选与批量报告。

### 1.5 walk-forward 从 sanity check 升级

`walkforward.py` 已使用 effective signal，并让 `factor_select_top_k` 真正基于训练窗 IC 进行因子选择。滚动 OOS 结果不再只是展示指标，而能模拟研究期训练、测试期验证。

### 1.6 执行层 NAV 演化修复

交易规模现在跟随组合 NAV 演化，借券成本也按动态 NAV 计算。旧逻辑长期回测会系统性低估或高估交易规模、成本和容量。

## 2. 单因子评估升级

`factor_tearsheet` 和 `compare_factors` 现在输出多门槛生产筛选指标：

- 多周期 IC/RankIC：1d、5d、10d、20d。
- IC decay 与 horizon sign consistency。
- 分层收益单调性。
- 因子换手。
- break-even cost。
- 10% ADV 容量估计。
- 上涨/下跌市场状态稳定性。
- 子区间符号稳定性。
- 滚动 OOS RankIC 稳定性。
- 成本后 long-short Sharpe。

新增 `factor_selection.py`：

- Benjamini-Hochberg FDR。
- Bonferroni 校正。
- White Reality Check 近似 bootstrap。
- Deflated Sharpe 近似惩罚。
- deployability hard filters。

筛选目标从“最好看”改成“可上线”：方向正确、OOS IC 为正、成本后 Sharpe 为正、换手不过高、容量不过低、子样本符号不反复翻转、对 horizon 不敏感。

## 3. 冗余剔除

新增 `quant_platform/core/signals/redundancy.py`。

它从三个维度识别重复因子：

- effective signal correlation。
- factor long-short return correlation。
- exposure-vector cosine similarity。

主要输出：

- `signal_correlation`
- `factor_return_correlation`
- `exposure_similarity`
- `factor_return_panel`
- `clusters`
- `selected_factors`

选择逻辑不是“保留 10 个都有效的因子”，而是按 deployability/selection score 排序后做逐步增量检验，优先保留 3 到 5 个互补且稳定的因子。若候选因子与已选因子相关过高，或对已选因子回归后的 residual alpha Sharpe 太弱，就会被剔除。

## 4. 多因子组合器

新增 `quant_platform/core/alpha_models/multi_factor.py`，支持三类以上可比较组合器：

- `rolling_icir`：滚动 RankIC 均值/波动得到 ICIR 权重，只使用历史窗口，避免未来函数。
- `mean_variance`：使用因子 long-short 收益均值和协方差估计因子权重。
- `risk_parity`：按因子收益波动做 inverse-vol/risk-parity 风格权重。
- `orthogonal_incremental`：对 baseline 因子正交后，只组合增量 alpha。

多因子输出统一写入 `alpha_multi`，并保存：

- factor weights。
- regime controls。
- composite alpha。

## 5. Regime 接入多因子层

regime 不再只是 report 里的 scorecard。`RegimePolicy` 可以影响：

- 因子权重。
- 高波动/高 turbulence 下禁用某些 category。
- category multiplier。
- 组合净暴露。
- 换仓阈值。

`run_portfolio_pipeline` 会读取 combiner 产出的 `regime_controls`，并在组合权重层按 `net_exposure` 缩放。long-only 组合允许保留 cash sleeve，因此 turbulent regime 下可以真实降仓。

## 6. 风险模型接入优化器

`OptimizedPortfolio` 不再只用股票样本协方差。现在优先使用已有 `FactorRiskModel`：

```text
stock covariance = B * factor_cov * B' + specific_var
```

风险因子包括：

- market return。
- alpha-spread return。

优化目标同时支持：

- alpha。
- factor risk covariance。
- turnover penalty。
- stock/sector/turnover constraints。

如果风险模型估计失败，会自动回退到原有 shrunk covariance，保证流水线不中断。

## 7. Benchmark 与容量接入主回测

`run_trading_pipeline` 现在会尝试加载 benchmark，并把 benchmark returns 传给 `performance_summary`，输出：

- active return。
- tracking error。
- information ratio。
- CAPM alpha/beta。
- alpha t-stat/p-value。
- beta contribution 与 residual contribution。

新增 `quant_platform/core/evaluation/capacity.py`，主回测会保存：

- `capacity_summary.csv`
- `capacity_curve.csv`

容量指标包括：

- 平均、P95、最大参与率。
- 平均、P95、最大 ADV notional 占比。
- 10% ADV 对应容量估计。
- 不同资金规模下重新估算交易成本。
- 不同资金规模下 estimated net Sharpe。

## 8. Experiment Runner

新增真正可执行的 YAML experiment runner：

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --config quant_platform/experiments/us_baseline.yaml
python scripts/run_experiment.py --config quant_platform/experiments/us_signal_lab.yaml
```

runner 会消费 YAML 中的：

- data 日期与路径。
- factor set。
- redundancy thresholds。
- multi-factor combiner。
- baseline factors。
- regime column。
- portfolio constraints。
- cost model。
- benchmark 与 capacity capital grid。

输出目录：

```text
data/experiments/<experiment_name>/
```

核心产物：

- `selected_factors.csv`
- `redundancy_*.csv`
- `weights_<name>.parquet`
- `factor_weights.csv`
- `regime_controls.csv`
- `backtest/performance_summary.csv`
- `backtest/capacity_summary.csv`
- `backtest/capacity_curve.csv`
- `experiment_summary.json`

## 9. CLI 入口变化

### 9.1 build_portfolio.py

新增多因子与风险模型参数：

```bash
python scripts/build_portfolio.py \
  --factors MOM_12_1M --factors STR_1M --factors VOL_20D \
  --combiner rolling_icir \
  --method optimize \
  --turnover-penalty 0.1
```

可选：

- `--combiner rolling_icir|mean_variance|risk_parity|orthogonal_incremental`
- `--baseline-factor <factor>`
- `--regime-col HMM_TURBULENCE_PROB`
- `--no-factor-risk`
- `--turnover-penalty`

### 9.2 run_backtest.py

新增 benchmark 参数，并默认输出容量文件：

```bash
python scripts/run_backtest.py --benchmark-market us --risk-free-rate 0.03
```

### 9.3 run_experiment.py

新增 YAML 实验入口：

```bash
python scripts/run_experiment.py --config quant_platform/experiments/us_baseline.yaml
```

## 10. 简历与面试表述

这次升级可以概括为：

> Built a production-oriented multi-factor research stack with unified signal orientation, multi-horizon IC validation, multiple-testing controls, redundancy pruning, regime-aware factor allocation, factor-risk-model-based portfolio optimization, benchmark-relative analytics, capacity stress curves, and YAML-driven experiment orchestration.

可展开讲的工程点：

- 修复 factor direction 未进入主流程导致的系统性方向错误。
- 把单因子评估从单一 RankIC 升级成多门槛 deployability filter。
- 用 FDR/White Reality Check/Deflated Sharpe 处理 data snooping。
- 从“等权 z-score”升级到 ICIR、mean-variance、risk parity、orthogonal incremental alpha。
- 用因子风险模型替代裸股票协方差，接近生产组合优化范式。
- 用 capacity curve 评估资金规模扩大后的成本弹性和 net Sharpe 衰减。
- 用 experiment YAML 固化研究配置，提升可复现性。

## 11. 主要文件清单

- `quant_platform/core/signals/effective.py`
- `quant_platform/core/signals/orientation.py`
- `quant_platform/core/signals/factor_selection.py`
- `quant_platform/core/signals/redundancy.py`
- `quant_platform/core/alpha_models/multi_factor.py`
- `quant_platform/core/portfolio/optimize.py`
- `quant_platform/core/portfolio/pipeline.py`
- `quant_platform/core/evaluation/capacity.py`
- `quant_platform/core/experiments/runner.py`
- `scripts/run_experiment.py`
- `scripts/build_portfolio.py`
- `scripts/run_backtest.py`
- `quant_platform/experiments/*.yaml`
- `tests/test_research_pipeline_upgrades.py`

