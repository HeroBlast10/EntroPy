# IBKR ETF 与个股多因子模拟研究流程

统一入口为 `scripts/ibkr_workflow.py`。ETF 使用 8 只固定当前标的，个股
使用 12 只固定当前标的；SPY 是独立基准。原有 IBKR 默认配置继续保持
只读，生成的历史数据、权重、账户账本和报告保存在被 Git 忽略的 `data/`。

## 当前运行边界

当前阶段按用户选择保留研究、订单预演、券商模拟提交撤单验证和只读对账。
延迟行情不会触发策略订单；未观察到已结算 USD、有效 USD 损益或账户汇率时，
预检明确阻止策略提交。HKD 现金可以单独对账，不会被当作 USD。

ETF 的券商接口测试是在常规时段以外挂一笔 1 股、低于参考价 2% 的 IEF
模拟限价单，`outsideRth=False`，然后撤单。它验证提交、永久订单编号、
撤单确认和重连恢复；它不是策略成交、滑点或收益验证。当天重复运行不会新发同一测试单。

## 每日使用

在仓库根目录和已安装依赖的 Python 环境中运行以下命令；Gateway 需登录
Paper Trading 并监听 4002。TWS 模拟账户可在每个命令后加 `--port 7497`。

```powershell
# 首次取数，或更新到最新已完成交易日：下载、研究、回测、订单预演和对账。
python scripts/ibkr_workflow.py daily --profile etf --refresh

# 扩展到个股：相同的研究、组合、预检与对账边界。
python scripts/ibkr_workflow.py daily --profile equity --refresh

# 离线复算缓存，不连接券商。
python scripts/ibkr_workflow.py research --profile etf
python scripts/ibkr_workflow.py research --profile equity

# 查看实时账户条件下的订单计划；不提交、不撤单。
python scripts/ibkr_workflow.py preview --profile etf

# 读取券商成交、现金、持仓并与持久化账本核对。
python scripts/ibkr_workflow.py reconcile --profile etf
```

`daily` 的默认行为是只读。建议在美股收盘且经过 15 分钟数据缓冲后运行，
或在下次开盘前更新前一个完整交易日。交易日由 XNYS 日历决定，处理夏令时、
节假日和提前收市。程序未安装任何后台定时任务，也不自行登录 Gateway。

退出码 0 表示所请求步骤通过；2 表示有明确阻碍、对账需复核或操作失败。
只读预演在延迟行情或闭市时返回 2 是预检结果，详细原因在报告中。
首次账户观察仅初始化基线；随后的观察才构成基线之后的独立核对。

## 策略与回测约定

`config/ibkr_etf_workflow.yaml` 与 `config/ibkr_equity_workflow.yaml` 分别控制策略。
两者以 USD 12,000 为研究资金、目标总敞口 20%、单标的目标权重上限 10%。
截断权重留下现金，不强行重新归一化。两个 profile 同时运行会叠加敞口，
账户级每日订单数量和名义金额预算共用同一个账本。

ETF 按 12-1 月动量选择正动量的前三只，再按 20 日波动率倒数配置。
个股用动量、低波动、短期反转三个固定方向的截面百分位秩等权合成，选择前五只，
再按波动率倒数配置。没有在全样本上拟合因子方向或选择最佳参数。
当前个股扩展不包含价值、质量、财报事件或历史市值筛选。

回测在 T 收盘计算信号、T+1 开盘成交，每周首个交易日再平衡，首个有效信号
用于初始化。其后持仓随价格漂移，现金独立记账。低于 USD 50 的差额保留原持仓，
与订单预演的门槛一致。成本为成交额 10 bps 加每个实际调仓标的每次 USD 1，
同时报告 20/40 bps 压力结果。报告比较相近敞口的研究池等权和 SPY 基准。
最近 252 个交易日单列为时间留出区间，不代表已经完成前瞻样本外观察。

研究使用供应商复权后的总回报单位，允许小数单位；它不是历史真实整股订单回放。
复权开盘价由 `open * adjusted_close / close` 构造，会继承供应商复权修订。
现金利息按零计，成本为模型值。固定当前存续标的也有选择和幸存者偏差。
券商端单独使用当前买卖报价和合约最小价格单位计算整股限价单。

IBKR 的 `ADJUSTED_LAST` 调整拆股和分红，`TRADES` 调整拆股；原始缓存分别保存，
研究只使用共同起始日以后的完整配对数据，不前向填补缺失交易日。
每份研究摘要记录数据 SHA-256、配置与信号日期。
参见 [IBKR 历史数据说明](https://www.interactivebrokers.com/docs/tws-api/doc/md/historical-bars)。

## 文件与运行证据

每个 profile 的输出位于：

```text
data/research/ibkr_workflow/etf/etf_momentum_v1/
data/research/ibkr_workflow/equity/equity_multifactor_v1/
```

| 文件 | 用途 |
| --- | --- |
| `research_summary.json` | 数据来源、配置、全样本/开发/留出结果、成本压力与边界 |
| `backtest_daily.csv` | 实际漂移后的每日研究净值、收益、现金、成本和换手 |
| `backtest_trades.csv` | 调仓日期、信号日期和带方向成交额 |
| `baseline_daily.csv`、`market_benchmark_daily.csv` | 相近敞口的对照序列 |
| `latest_signals.csv` | 最新特征、波动率、动量和流动性 |
| `target_weights.parquet` | 带日期、显式零权重退出的目标持仓 |
| `latest_broker_report.json` | 本次账户快照、订单计划、阻碍与对账结果 |
| `transport_receipt.json` | 券商接口测试的永久订单编号、终态及成交数量 |
| `workflow_report.html` | 汇总表、净值图和券商证据 |
| `<command>_<timestamp>.json` | 每次运行的独立历史记录 |

持久化账本为 `data/live/ibkr_workflow/ledger.sqlite3`。先提交本地订单意图，
再调用券商；不确定的发送结果阻止自动重发。执行 ID 去重，并接收晚到的佣金。
资金按实际币种核对，遗漏现金流、缺失币种、头寸不一致和执行历史覆盖缺口都会
留下待复核状态，不会把失败的检查点覆盖为新基线。

全账户现金对账并不自动解释分红、利息、外部换汇和转账。若这些发生，需要补入
有来源的现金流水；程序保留差额。API 历史覆盖不足时也需要人工记录或报表补齐，
不能以当前持仓相同替代整段历史审核。

订单恢复同时核对账户、持久化引用和 broker permId。IBKR 的已完成订单回执
可能省略 clientId/orderId，不会因此丢失终态。取消只针对本次创建的订单，
不会使用全局撤单；`PendingCancel` 不被当作已取消。

## 后续启用策略模拟订单

策略提交需要显式 DU 账户与本次命令的开关，例如：

```powershell
python scripts/ibkr_workflow.py submit --profile etf --account DU你的模拟账号 --submit-paper-orders
```

程序仍会检查实际为模拟账户、数据与权重一致、最近完整交易日、常规时段、
新鲜实时买卖报价、价差、已结算 USD、币种一致的损益、持仓所有权、挂单、
持久化日限额、未决订单和对账。账户只读选项需要在模拟 Gateway 中关闭。
不会自动购买行情、转换货币或借入 USD。

API 接口测试命令为 `transport-check --profile etf --account DU... --submit-paper-orders`。
它只用于上述闭市提交撤单演练；实际策略模拟成交应在账户条件满足后独立验证。
IBKR 的模拟撮合与真实撮合有差异，参见
[官方模拟交易限制](https://www.interactivebrokers.com/docs/tws-api/doc/notes-limitations/tws-api-limitations)。

## 验证

```powershell
python -m pytest -q tests/test_ibkr_workflow.py tests/test_ibkr_paper_safety.py tests/test_ibkr_daily_research.py
```

这些测试不连接券商，覆盖交易日、收盘缓冲、信号时序、成本现金记账、数据缺口、
整股与零权重退出、账户币种、晚到佣金、跨日覆盖、发送前持久化、部分成交、
撤单等待、重连恢复和重复运行。实际券商回执另存于运行证据中。
