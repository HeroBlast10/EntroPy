<div align="center">

# EntroPy

**面向生产化研究的股票多因子量化平台**

EntroPy 用于检验状态空间、市场状态、熵、均值回复与传统横截面因子，是否能在点时间数据、滚动样本外验证、真实交易成本和容量约束下产生稳健可交易 alpha。

[English README](README.md) | [升级说明](docs/PRODUCTION_FACTOR_RESEARCH_UPGRADE_2026_05.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-172%20passed-brightgreen.svg)](#测试)

</div>

---

## 研究目标

EntroPy 围绕一个更接近真实投研的问题构建：

> 在交易成本、因子冗余、多重检验、基准风险和容量约束之后，信号处理类特征是否还能提供可验证、可交易的增量 alpha？

项目目标不是找到“回测最好看的因子”，而是搭建一条生产化研究流水线：

- 统一因子方向和 effective signal 口径。
- 多周期单因子有效性评估。
- 多重检验控制，降低 data snooping 风险。
- 冗余剔除，筛出 3 到 5 个互补稳定因子，而不是一堆高度重复因子。
- 多种可比较的多因子组合器，而不是固定等权 z-score 平均。
- regime-aware 因子配置和组合降仓。
- 基于因子风险模型的组合优化。
- 基准相对表现、交易成本仿真和容量曲线。
- YAML 驱动实验配置，保证研究可复现。

## 核心功能

| 模块 | 能力 |
| --- | --- |
| 数据层 | US/CN 市场日历、动态股票池、点时间价格和基本面对齐、benchmark 加载 |
| 因子库 | 横截面、时间序列、regime、relative-value 四类因子 |
| Effective signal | 全流程统一使用 `direction -> winsorize -> neutralize -> z-score -> rank` |
| 单因子研究 | 1/5/10/20d IC 和 RankIC、IC decay、分层单调性、换手、break-even cost、容量、regime/子区间/OOS 稳定性 |
| 多重检验 | Benjamini-Hochberg FDR、Bonferroni、White Reality Check 近似、Deflated Sharpe 近似 |
| 冗余剔除 | 因子值相关、因子收益相关、暴露向量相似性、聚类诊断、逐步增量 alpha 检验 |
| 多因子 alpha | 滚动 ICIR 加权、因子收益 mean-variance、因子收益 risk parity、baseline 正交后的增量 alpha |
| Regime 集成 | regime 驱动因子权重、category 启用/禁用、净暴露、换仓阈值和 alpha 缩放 |
| 组合构建 | 分层组合、优化组合、等权/市值/信号强度/逆波权重、行业/个股/换手约束 |
| 风险模型 | Barra 风格因子风险模型：暴露、因子协方差、特异风险、风险分解 |
| 执行与成本 | 动态 NAV 交易规模、滑点、冲击、佣金、借券成本、成本归因 |
| 回测分析 | gross/net NAV、回撤、VaR/CVaR、benchmark alpha/beta/IR、容量和资金规模曲线 |
| 实验系统 | YAML 配置因子集、组合器、成本、walk-forward、benchmark、capacity |

## 因子库

| 因子类型 | 例子 | 主要用途 |
| --- | --- | --- |
| 横截面因子 | `MOM_12_1M`, `STR_1M`, `VOL_20D`, `ILLIQ_AMIHUD`, `BOOK_TO_MARKET`, `ROE`, `ASSET_GROWTH` | 股票排序和组合构建 |
| 时间序列因子 | `KF_VELOCITY`, `KF_TREND_STRENGTH`, `SPECTRAL_ENTROPY_60D`, `HURST_60D`, rolling skew/kurtosis | 单资产状态、趋势和噪声特征 |
| Regime 因子 | `HMM_TURBULENCE_PROB` | 市场状态感知的因子配置和风险暴露控制 |
| Relative value | `OU_ZSCORE` | 均值回复质量和价差风格诊断 |

每个因子都有 `direction`、`category`、`signal_type`、lookback、lag 和 description 元数据。低波、低资产增长等负向因子会先被翻转成“高值更好”的统一信号，再进入评估、组合和 ML 主流程。

## 研究流水线

```mermaid
flowchart LR
    A["原始价格和基本面"] --> B["点时间数据集"]
    B --> C["因子注册表和批量计算"]
    C --> D["Effective signal 层"]
    D --> E["单因子评估"]
    E --> F["多重检验控制"]
    F --> G["冗余剔除"]
    G --> H["多因子组合器"]
    H --> I["Regime-aware alpha"]
    I --> J["组合构建"]
    J --> K["执行和成本仿真"]
    K --> L["基准、风险、容量分析"]
    L --> M["HTML 报告和实验产物"]
```

## 多因子组合器

EntroPy 支持多种可比较的生产化组合器：

| 组合器 | 方法 | 适用场景 |
| --- | --- | --- |
| `rolling_icir` | 用滚动 RankIC 均值除以波动得到权重 | 默认自适应因子配置 |
| `mean_variance` | 使用因子 long-short 收益均值和协方差 | 因子层收益/风险优化 |
| `risk_parity` | 使用因子收益流的 inverse volatility | 预期收益噪声较大时的稳健配置 |
| `orthogonal_incremental` | 对 baseline 因子正交后再组合残差 alpha | 检验高级信号是否真的有增量价值 |

多因子输出统一为 `alpha_multi`，并保存 factor weights、regime controls 和 composite alpha。

## Experiment Runner

`quant_platform/experiments/*.yaml` 是可执行实验配置，不再只是设计草稿。YAML 可以定义数据范围、因子集、冗余阈值、组合器、组合约束、成本、benchmark 和 capacity grid。

```bash
# 查看可用实验
python scripts/run_experiment.py --list

# 运行 baseline 多因子实验
python scripts/run_experiment.py --config quant_platform/experiments/us_baseline.yaml

# 运行高级信号实验
python scripts/run_experiment.py --config quant_platform/experiments/us_signal_lab.yaml
```

典型输出目录为 `data/experiments/<experiment_name>/`：

- `selected_factors.csv`
- `redundancy_*.csv`
- `weights_<experiment>.parquet`
- `factor_weights.csv`
- `regime_controls.csv`
- `backtest/performance_summary.csv`
- `backtest/capacity_summary.csv`
- `backtest/capacity_curve.csv`
- `experiment_summary.json`

## 快速开始

```bash
pip install -r requirements.txt

# 1. 构建点时间数据集
python scripts/build_dataset.py

# 2. 计算并评估所有因子
python scripts/build_factors.py --evaluate

# 3. 构建单因子组合
python scripts/build_portfolio.py --signal MOM_12_1M

# 4. 运行带执行成本的回测
python scripts/run_backtest.py

# 5. 生成研究报告
python scripts/generate_report.py --signal MOM_12_1M
```

多因子组合示例：

```bash
python scripts/build_portfolio.py \
  --factors MOM_12_1M \
  --factors STR_1M \
  --factors VOL_20D \
  --factors ILLIQ_AMIHUD \
  --combiner rolling_icir \
  --method optimize \
  --turnover-penalty 0.1
```

一键因子流水线：

```bash
python scripts/run_factor_pipeline.py --factors MOM_12_1M
python scripts/run_factor_pipeline.py --auto-best
python scripts/run_factor_pipeline.py --all-factors --quick
```

因子参数调优：

```bash
python scripts/tune_factors.py --objective ric_icir --top 5
```

## 组合与回测

### 组合构建

- 支持 long-only 和 long-short 分层组合。
- 支持优化组合，优先使用因子风险协方差，失败时回退到 shrink 后的股票协方差。
- 支持等权、市值权重、信号强度权重和 inverse-vol 权重。
- 支持个股权重、行业权重和换手约束。
- 支持 regime 降仓，即通过 net exposure 缩放保留现金仓位。

### 执行与成本

- 根据每日权重变化生成交易。
- 交易规模跟随组合 NAV 演化。
- 成本包括佣金、滑点、平方根市场冲击、监管费用、印花税占位和借券成本。
- 借券成本基于动态 NAV 和空头暴露计算。

### Benchmark 与容量

回测可输出 benchmark-relative 指标：

- 主动收益和 tracking error。
- Information ratio。
- CAPM alpha、beta、alpha t-stat 和 residual volatility。

容量分析包括：

- 参与率。
- ADV notional 占比。
- 10% ADV 容量估计。
- 不同资金规模下的成本弹性。
- 不同资金规模下的 estimated net Sharpe。

## 项目结构

```text
quant_platform/
├── core/
│   ├── data/                  # PIT 数据、日历、股票池、benchmark、行业映射
│   ├── signals/               # 因子基类、注册表、effective signal、筛选、冗余剔除
│   │   ├── cross_sectional/    # 动量、波动、流动性、价值/质量因子
│   │   ├── time_series/        # Kalman、entropy、Hurst、高阶矩
│   │   ├── regime/             # HMM turbulence probability
│   │   ├── relative_value/     # OU 均值回复特征
│   │   └── evaluation/         # 不同类型因子的评估 scorecard
│   ├── alpha_models/           # Ranker、ML alpha、regime overlay、多因子组合器
│   ├── portfolio/              # 分层组合、优化器、约束、风险模型、pipeline
│   ├── execution/              # 成本模型、向量化回测、PnL
│   ├── evaluation/             # walk-forward、ablation、benchmark、capacity、report
│   └── experiments/            # YAML experiment runner
├── experiments/                # 实验 YAML 配置
├── scripts/                    # CLI 入口
├── tests/                      # 单元测试和回归测试
└── docs/                       # 设计与升级文档
```

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `python scripts/build_dataset.py` | 构建价格、股票池、基本面数据 |
| `python scripts/build_factors.py --evaluate` | 计算因子并生成 factor catalog |
| `python scripts/build_portfolio.py` | 构建组合权重 |
| `python scripts/run_backtest.py` | 运行执行成本回测 |
| `python scripts/generate_report.py` | 生成 HTML 研究报告 |
| `python scripts/run_factor_pipeline.py` | 运行 factor-to-report 流水线 |
| `python scripts/run_experiment.py` | 运行 YAML 实验 |
| `python scripts/tune_factors.py` | 因子参数调优 |

## 测试

```bash
python -m compileall quant_platform scripts
pytest -q
```

最近验证结果：

```text
172 passed
```

## 已知限制

- 默认 US 股票池用动态市值和流动性过滤近似 large-cap index，不是官方历史 S&P 500 成分股。
- 日频 OHLCV 无法捕捉日内执行和微观结构细节。
- 因子风险模型相对商业 Barra 模型更紧凑。
- White Reality Check 和 Deflated Sharpe 是轻量近似，主要用于研究卫生检查，不是完整学术复现包。
- 基本面数据质量依赖 SimFin 覆盖和报告滞后假设。
- CN A 股支持包含配置和成本模型接口，但生产使用前仍需验证本地数据可得性和市场特有执行规则。

## 文档

- [生产级因子研究升级说明](docs/PRODUCTION_FACTOR_RESEARCH_UPGRADE_2026_05.md)
- [数据字典](docs/data_dictionary.md)
- [因子字典](docs/factor_dictionary.md)
- [组合字典](docs/portfolio_dictionary.md)
- [交易字典](docs/trading_dictionary.md)

## 技术栈

`pandas` · `numpy` · `scipy` · `numba` · `scikit-learn` · `pyarrow` · `matplotlib` · `plotly` · `yfinance` · `simfin` · `exchange_calendars` · `loguru`

## License

[MIT](LICENSE)

