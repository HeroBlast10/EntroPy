# EntroPy 正确性审计与修复记录（2026-07）

本轮修复的目标不是提高某一个历史指标，而是统一数据、因子、模型、
组合和执行的时点口径，使结果可以复现、可以解释，并在缺少关键
point-in-time（PIT）数据时明确失败或降级为 research-only。

## 1. 数据与数据契约

- 价格调整现在真正遵守 `adjustment.method` 和
  `adjustment.point_in_time`；支持 split-only 与 total-return 的因果
  forward adjustment，并校准增量下载重叠区间。
- 新建价格文件会保存同名 metadata sidecar，记录数据源、调整方法、
  PIT 口径、日期和行数。
- 当前成分股快照不再伪装成历史 S&P 500 成分。股票池逐行保存
  `membership_source` 与 `membership_is_point_in_time`。
- production 模式要求显式的历史成分区间文件
  (`ticker`, `membership_start`, `membership_end`)；缺少时 fail closed。
- `in_index=False` 不再可能通过 `pass_all_filters`；IPO 年龄改为交易
  观测日计数。
- 股票池的市值改为“最近已知 shares × 当日 raw close”；缺少 shares
  时，以事件日 market cap 和首个可交易日 close 推导 shares，不再在
  两个报告日之间冻结市值。
- yfinance 基本面快照逐行标记为 non-PIT；production 模式只接受有真实
  publication date 的 PIT 来源。
- US/CN benchmark 分别使用 XNYS/XSHG 交易日历。

## 2. 因子计算与评估

- `FactorBase` 取消 `raw.values` 位置赋值，统一按原始行索引或
  `(date, ticker)` 键对齐，解决 ticker-major/date-major 排序造成的
  静默错配。
- neutralization 会保留并使用请求的行业/连续暴露列；连续暴露使用
  截面 OLS residual。
- probability、regime、time-series 和 relative-value 信号不再被默认
  截面 z-score 改变原始含义。
- Kalman 修正 `A P Aᵀ` 协方差递推，并取消 full-sample min/max 缩放。
- HMM 改为历史窗口参数估计、定期 refit、单向 filtered probability；
  turbulent state 由最大拟合波动率确定。日度状态使用递推 forward
  step，避免每天从头重算完整窗口。
- 修正 `SPREAD_HL` 常数、downside-volatility 分母与最小样本、
  `TAIL_RISK`/`VOL_*` 方向。
- TTM 按“当时已知的最近四个不同报告期”累计；E/P、B/M 的会计分子
  PIT carry-forward，但分母按每日 raw close 重估。
- Residual Momentum 的 skip 只应用一次，并兼容旧 `skip=` 参数。
- Amihud cache/no-cache 使用一致的 raw dollar-volume 口径。
- forward return 统一使用 `adj_close`、按 ticker 计算，禁止跨 ticker
  shift。
- registry 对全空、常量、重复键硬拒绝；覆盖率、横截面非恒定比例、
  dominant value 和近零方差形成质量报告。严格构建模式会拒绝所有
  warning 因子。

## 3. 机器学习与统计评价

- LightGBM/MLP 在截面排名前先按 `(date, ticker, pass_all_filters)`
  过滤可交易股票。
- 多日标签按共享交易日历精确连接 `t+h`；停牌或缺失目标日价格不会
  被错误替换为“该股票下一条观测”。
- 所有 CV/OOF split 按完整交易日切分，显式 purge forward-return
  horizon；同一截面不会分到 train/test 两侧。
- target 只允许唯一 `(date, ticker)` 键连接或完全相同的唯一索引；
  模糊位置对齐和重复键直接报错。
- OOF 指标包括逐日 RankIC、Newey-West t-stat、真实分位数组合换手和
  明确的 overlapping-return 年化口径。
- LightGBM、MLP、MOM_12_1M 和 equal-weight alpha 使用相同 OOF 样本；
  模型间及模型对基线均输出配对 block bootstrap。
- multi-factor ICIR/风险平价/均值方差权重按收益 horizon 延迟，只使用
  决策时已经完全实现的历史收益。
- ExperimentRunner 校验所有 YAML 字段；CN 数据/成本接口未接通时
  fail closed；headline walk-forward 在每个训练窗内选因子并先过滤
  tradable universe。全样本结果只标记为 diagnostic in-sample。

## 4. 组合、执行与成本

- 统一为 close-target 账本：昨日收盘后实际持仓先获得今日
  close-to-close return，再收借券费，随后在今日收盘执行今日目标。
  今日新目标不会获得今日已经发生的收益。
- 实际持仓会漂移；调仓差额相对于 drifted holdings，而不是相对于
  上一期静态 target。
- 非调仓日不会免费恢复目标权重。周/月频只在计划调仓日交易；拒单或
  部分成交可以继续尝试。
- 缺价、无历史 ADV、整股取整为零均有明确 fill status；股数向下取整，
  participation cap 默认 10% ADV。
- ADV 使用截至昨日的历史窗口，并读取 `CostModel.adv_lookback`。
- constraints 改为同时可行投影；个股、行业、袖套和换手上限在最终
  权重上复查。启用行业约束却缺少映射时 fail closed。
- long-short fallback 保持真实 `+1/-1` 双袖套；short SIGNAL 权重方向
  与“越低越应做空”一致；market-cap 全空时退回等权。
- carry-forward 保留全现金目标，确保清仓指令和后续现金日期不会因
  删除零权重而消失；也会持有最后有效目标直到评估区间末尾。
- event backtest 使用 raw close 做股数/整手、adjusted close 做收益，
  并按方向执行涨跌停限制。
- US FINRA TAF cap、CN 最低佣金和 volatility slippage 均已修正。
- capacity curve 使用动态 NAV、requested order 和 borrow drag。
- Sharpe/Sortino 使用逐期超额收益；FactorRiskModel 的因子与特异风险
  单位统一为年化，并对缺 exposure/specific risk 的非零持仓 fail closed。

## 5. 可复现性与验证

- 新增 Python 3.12 GitHub Actions：compileall + 全量 pytest。
- `requirements-lock.txt` 记录完整研究环境的精确直接依赖；
  `requirements-ci-lock.txt` 排除 PyTorch/CUDA、Streamlit、IBKR 等
  非测试依赖，避免 CI 下载多 GB 可选运行时。
- 非线性研究产物记录输入 SHA-256、Git commit、Python/包版本、
  feature schema cutoff、fold 和模型参数。
- 成本后比较使用独立脚本
  `scripts/evaluate_alpha_after_costs.py`，统一组合、调仓、成本与执行时点。

## 仍然存在的数据边界

当前本地价格种子来自“今天的成分股”，基本面也包含 yfinance snapshot
fallback。因此本轮重评只能标记为：

> **explicit non-PIT research，存在成分股幸存者偏差，不是可用于上线
> 声称的 production backtest。**

production 结论仍需补齐：

1. 历史成分加入/退出区间及退市证券价格；
2. 有真实 publication timestamp 的 PIT 基本面；
3. 历史 PIT 行业分类（否则应像本轮横向比较一样明确关闭行业约束）；
4. 对现有 legacy price parquet 做有来源记录的重新下载/迁移。

## 2026-10-16：Tier A 正确性修复与验收

本轮修复已完成，测试从 297 项增加到 **344 项，全部通过**。新增的
47 项回归场景集中在 `tests/test_tier_a_correctness.py`；3 个旧执行测试
补足了 ADV 预热历史，保留原有成交和残单断言。

验收口径作了以下修正：一般协方差下，增大 λ 保证最优组合方差不升，
不保证最大单股权重下降，因此另用等方差对角协方差验证集中度下降；
`OLS 截距 / HAC 标准误` 是 alpha t 值，和年化截距/残差波动率分列；
Market、Size、Sector 是三类风险来源，实际维度为 Market + Size +
K−1 个行业超额收益因子，不能固定宣称为“四因子”。

| 修复点 | 代表回归场景 |
| --- | --- |
| 优化目标量纲 | 日收益先验跨度、信号缩放不变性、λ 与方差/对角情形最大权重 |
| 静默回退 | 缺协方差和求解失败抛专用异常；显式降级写元数据；回测继承且不误读内存权重的 lineage |
| 结构化风险 | 滞后 log-cap SMB、未来市值不影响过去、行业映射、候选信号不影响风险协方差、Euler 加总 |
| 增量 alpha | 截距非零、独立 HAC sandwich 数值核对、自相关修正、t 值筛选阈值、共线性 |
| 容量 | ADV 预热/门槛/不可交易量、p95 同口径、零股重试不稀释 p95、NAV 增长剔除、缩放后重新限额 |
| 两个回测引擎 | 缺价不填充、全缺价日期不被 pivot 删除、价格与信号可交易标记 |
| PRS | 缺 crowding 时剔除并重分配权重、DSR 不用命中率代替、试验离散度公式、目录与 HTML 使用同一评分 |

使用原有三个 quantile 候选配置完成全历史执行重跑，并按最终容量口径
从保留的成交/PnL 重算容量。输出位于
`data/experiments/candidate_backtests_20261016_tier_a/`：

| 候选 | p95 容量（美元） | 所有有效订单日期均满足 10% ADV 的容量 | 实际最大成交参与率 |
| --- | ---: | ---: | ---: |
| MOM_12_1M | 48,188,302 | 23,038,319 | 0.4335% |
| STR_1W | 29,828,702 | 14,286,330 | 0.7304% |
| TURNOVER_ACCEL | 27,307,402 | 10,498,556 | 0.9125% |

p95 对“有正整股请求的日期”的每日最大请求参与率取分位数；不足一股
的零请求重试日不进入样本。容量归一到初始资金，剔除 NAV 增长影响。
三个候选均满足预期的 10⁶–10⁸ 量级，资金网格成交参与率均 ≤ 0.1。
5,000 万美元档位触发限额改变成交路径，收益与 Sharpe 估计留空并标注
需要完整执行重放；成本曲线仍输出实际限额后的成交成本估计。

DSR 使用逐期 Sharpe、试验 Sharpe 的离散程度与完整目录试验数，明确
披露“独立试验”的假设；所选目录小于完整搜索范围时，应提供更完整
的 `n_trials`。当前静态行业图和上文 non-PIT 数据边界仍然存在，修复
正确性不等于获得 production 数据认证或保证因子成本后盈利。

复算命令：

```powershell
python scripts/backtest_factor_candidates.py --refresh-capacity --output-dir data/experiments/candidate_backtests_20261016_tier_a
```
