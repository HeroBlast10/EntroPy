# 五个量化开源项目的成功机制，与 EntroPy 的对照反思

> 研究日期：2026-10-26 ｜ 对象：hftbacktest / vectorbt / KalishMakerMaker(Kalshi MM) / tensortrade / nautilus_trader ｜ 对照：EntroPy（本仓库）
>
> 说明：星标、fork、commit、tag 等数字取自截图时点与公开检索；licence、发布节奏、商业模式为当日核对结果。EntroPy 侧数字取自本仓库 `README.md`、`docs/STAGE_REPORT_2026-10-23.md`、`NONLINEAR_ALPHA_RESEARCH_2026_07.md` 与 `git log`。

---

## 0. 结论先行

1. **五个项目没有一个靠"功能多"成功**，全部靠一句可以在 10 秒内验证的承诺：成交真实性（hftbacktest）、参数吞吐（vectorbt）、回测实盘一致（nautilus）、RL 可组合（tensortrade）、改个 config 就能下单（Kalshi MM）。
2. **它们都在"数据免费、场景高热度、边际未被学术榨干"的场地上竞争**：加密 L2/L3、预测市场、crypto RL。EntroPy 选了相反的一侧——WRDS 授权数据 + 美股日频截面 alpha，triple penalty：**不可分发、不可复现、alpha 被挖了六十年**。
3. **EntroPy 真正的资产不是"平台"，是一套"防自欺的验收协议"**（PIT/PERMNO/总收益/退市标签/NW HAC/FDR/SPA/CSCV/PRS/双倍成本/trial registry）。这套东西的严谨度**高于上述五个项目中的任何一个**，但它们全部被埋在两万字的内部报告里，没有变成可被别人使用的产品。
4. **最大的定位错位**：EntroPy 本质上是一个"研究结论工厂"（结论是 `production_admitted=false`，双倍成本下全部为负），却按"平台"（44 因子、449 测试、7 维 PRS）去包装；而平台必须具备的分发面（可安装包、文档站、release、社区入口、30 秒 demo）一个都没有。
5. **品牌与实测背离**：项目叫 EntroPy，最弱的恰恰是熵类因子（`SPECTRAL_ENTROPY_60D` 20 日 IC −0.0029、q=0.70；`HURST_60D` IC −0.0009）。承诺的名词和交付的结果不是一回事，这在叙事上比任何 bug 都致命。

---

## 1. 五个项目：成功机制逐项解剖

### 1.1 nkaz001 / hftbacktest — 用"一个物理量"击穿整个品类

| 维度 | 观察 |
| --- | --- |
| 规模 | ★4.8k / fork 936 / 1,038 commits / 56 tags / **19 open issues** |
| 承诺 | "限价单到底会不会成交"——queue position + feed/order latency |
| 技术选择 | Python 侧策略写在 `@njit` 里（Numba JIT），Rust 侧负责实盘；**回测和实盘共享同一套算法内核** |
| 数据策略 | 内置 Binance / Bybit 的 L2(MBP) / L3(MBO) 数据下载与格式转换，用户当天就能跑 |
| 文档形态 | readthedocs + 一组"方法论文章"级教程：GLFT 做市、order book imbalance alpha、APT 定价框架 |
| 冷启动 | 单维护者，但 issue 中位响应 11.6 小时，星标增长靠教程被引用/转发 |

**为什么赢**：它没有做"更好的回测框架"，它做的是**别人假装不存在的那个变量**。所有 bar/trade 级框架都在回避"排在你前面还有多少单"，它把这个问题当成产品本身。一旦"成交真实性"成为卖点，用户对功能数量的容忍度就极高——它连股票都不支持。

**一句话教训**：把"行业集体回避的那一个量"做成产品，比补十个功能更快获得信任。

---

### 1.2 polakowo / vectorbt — 把一个数学洞察变成一个 API

| 维度 | 观察 |
| --- | --- |
| 规模 | ★9.4k / fork 1.2k / 1,109 commits / 8 tags / **113 open issues**（高活跃的代价） |
| 承诺 | "thousands of trading ideas before others finish one"——参数扫描吞吐 |
| 核心洞察 | 回测 = 数组代数。每一组参数是一列，broadcast 之后一次算完 |
| 技术 | Numba；2026-04 的 v1.0 又加了可选 Rust 引擎（自动派发） |
| 商业模式 | Apache-2.0 + **Commons Clause**（禁止转售）开源版做漏斗 → VectorBT PRO 私库订阅（约 $25/月，$500 买断），PRO 才给 Rust 原生模拟器、MCP server、purged CV |
| 反面教材 | `from_signals()` 默认**当根 K 线收盘成交**（`price=np.inf`），配合 `ma_crossed_above` 也是当根计算→教科书级未来函数，**且库永远不会警告你** |

**为什么赢**：它卖的不是正确性，是**"提问成本"**。当一个问题从 40 分钟变成 0.4 秒，用户会自愿接受"答案可能不准"。这是极其清醒的产品选择：先让你上瘾，再在 PRO 里卖严谨（purged CV、更细的订单模拟）。

**一句话教训**：便利性和正确性是两门生意。EntroPy 只做了后者，且没有做前者的入口。

---

### 1.3 roofit / KalishMakerMaker — 小项目的正确形态

| 维度 | 观察 |
| --- | --- |
| 规模 | ★438 / fork 79 / **31 commits / 1 branch / 1 tag** |
| 承诺 | "Deploy simple market making strategies on Kalshi"——动词是 **deploy** |
| 形态 | 不是库，是模板：`config.yaml` + `runner.py` + `mm.py`（Avellaneda–Stoikov），改完参数就能挂单 |
| 场景 | 2024–2026 预测市场爆发期；生态里已有 35+ 个同类仓库被目录站排名（botforkalshi 等），注意力按"能不能跑"分配 |
| 维护成本 | 近乎零。范围窄到不需要 roadmap |

**为什么"成功"**：它不是技术胜利，是**场景胜利**。Kalshi 的做市机器人门槛极低（二元合约、单一 API、RSA 签名），任何人 fork 之后一小时内能见到实盘订单。438 星在一个小生态里已经是头部。

**一句话教训**：小项目靠"场景 + 可执行"赢得注意力，不靠功能数量。EntroPy 有 44 个因子，却没有一个"改一下就能用"的东西。

---

### 1.4 tensortrade-org / tensortrade — 恒星也会变成化石

| 维度 | 观察 |
| --- | --- |
| 规模 | ★7.2k / fork 1.3k / 1,093 commits / 7 tags |
| 承诺 | "train, evaluate, deploy robust trading agents"——RL 的可组合积木 |
| 设计 | Exchange / ActionScheme / RewardScheme / Observer / Portfolio 五件套，任意替换；论文和课程直接引用 |
| 真实轨迹 | 2022 之后基本停摆，v1.0 拖了三年；**2026-02 靠 AI 辅助维护复活**（仓库里赫然有 "All issues fixed with CURSOR"、"MIGRATION_GUIDE — All issues fixed with CURSOR"），补上 CI、Python 3.12、教程课程体系 |
| 最值得学的一点 | 新 README 把自己最难看的结果放到了首屏：PPO on BTC/USD，0% 佣金 +$239（vs 买入持有 −$355），**0.1% 佣金 −$650**；并明确写"手续费超过预测收益，核心问题是交易频率" |

**为什么"半成功"**：抽象做得漂亮，所以被引用、被 star；但没有维护者和可验证结果，7.2k 星换不来一个生产用户。**星标是延迟的信任，不是资产。**

**一句话教训**：一个项目真正的护城河是"别人能跑出和你一样的数字"。做不到这一点，热度只会变成考古现场。

---

### 1.5 nautechsystems / nautilus_trader — 把"工程纪律"本身做成卖点

| 维度 | 观察 |
| --- | --- |
| 规模 | ★36.1k / fork 4k / **21,819 commits / 217 tags / 只有 7 个 open PR** |
| 承诺 | "same code, backtest → live, no code changes"（研究/生产 parity） |
| 架构 | Rust 内核（tokio、mimalloc、PyO3）+ Python 控制面；纳秒级确定性事件时间模型；Redis 状态持久化；v2 重写期间 Cython v1 仍在支持 |
| 治理 | 公司主体 Nautech Systems（© 2015–2025）、LGPL-3.0、CLA、Discord、master/nightly/develop 三轨、MSRV 跟随 Rust stable、Docker + JupyterLab 镜像、roadmap 明确写出"什么在开源范围内、什么不在" |
| 获客 | **适配器矩阵即渠道**：IB / Binance / Bybit / Databento / Betfair / dYdX / Polymarket… 每接一个 venue 就吃一批该 venue 的用户 |
| 商业 | 开源引擎 + 企业支持/咨询；LGPL 既允许被广泛使用，又阻止闭源白嫖 |

**为什么赢**：它把"回测和实盘不一致"这个所有量化团队都痛、但所有人都用"重写一遍"来忍受的问题，做成了架构承诺。然后用 217 个 tag、三轨分支、CLA 告诉你：**这不是玩具，是基础设施**。

**一句话教训**：当你的产品是"可信"，那么发布纪律、版本号、issue triage 就是产品的一部分，不是附加项。

---

## 2. 横向归纳：六条共同规律

| # | 规律 | hftbacktest | vectorbt | Kalshi MM | tensortrade | nautilus | **EntroPy** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 一句话可验证承诺 | 成交真实性 | 参数吞吐 | 能下单 | RL 积木 | 回测=实盘 | **无（首屏是能力清单）** |
| 2 | 主动放弃大量功能 | 无股票、无组合 | 无实盘、无撮合 | 只做 Kalshi | 只做 RL | 不做因子研究 | **什么都做** |
| 3 | 30 秒可运行 | pip + 内置数据 | pip + yfinance | 改 config | pip + notebook | pip + Docker | **5 条命令 + 授权数据** |
| 4 | 数据可获取 | 自带下载器 | 免费源 | 交易所有 API | CCXT | 适配器 | **WRDS 授权，外部无法复现** |
| 5 | 分发面 | PyPI + RTD | PyPI + 文档站 | 仓库即产品 | PyPI + RTD | PyPI + 文档站 + Discord | **无 pyproject / 无 release / 无站点** |
| 6 | 结果可被别人验证 | 教程可复跑 | 示例可复跑 | 能真下单 | EXPERIMENTS.md 表格 | 官方回测示例 | **clone 后跑不出任何数字** |

补充两条不在表里但很致命的观察：

- **工程痕迹就是信任凭证**。21,819 commits / 217 tags 与 44 commits / 0 tag，传递的是完全不同的信号。潜在用户不会读你的代码，只会看你的 tag 列表和 issue 关闭率。
- **诚实本身就是增长策略**。tensortrade 把"扣费亏钱"放在首屏，反而被引用；hftbacktest 反复强调"过度悲观和过度乐观都错"。EntroPy 的诚实度远高于它们，但**藏在 docs/ 里**，只有已经信任你的人才会读到第 6 节。

---

## 3. EntroPy 现状体检（客观描述，不加修饰）

**它实际是什么**：一个单人维护的、面向美股日频截面 alpha 的**研究验收流水线**，附带 44 个因子、一个执行级回测器、一套报告生成器，以及一份罕见的、诚实的负结果记录。

**硬数字**：

| 项 | 现状 |
| --- | --- |
| 代码规模 | `quant_platform` + `scripts` + `tests` ≈ 数万行 Python，449 项测试（CI 只跑 pytest + compileall） |
| 数据资产 | 857 个历史 PERMNO、2,903,815 行 CRSP CIZ 日线、48,958 × 720 财务事件、2,021,564 合格研究行 |
| 因子 | 44 个（34 横截面 / 8 时序 / 1 regime / 1 relative value） |
| 统计结果 | 224 个检验里**只有 1 日周期 10 个条目 q<0.05**；5/20/60 日**全部不显著** |
| 执行结果 | 20 日价格特征 Ridge 净年化 0.873%、净 Sharpe 0.218；**双倍成本全部为负** |
| 结论 | `production_admitted=false`；112 组回放，最大参与率仅 0.103% ADV |
| 工程痕迹 | 44 commits、0 tag、无 `pyproject.toml`/`setup.py`、无 release、无文档站、无 CONTRIBUTING |

**它做得比五个项目都好的地方**（必须说清楚，否则反思没有靶子）：

1. **时间点正确性**：PERMNO 主键、历史成分区间、CIZ 直接总收益、退市收益只计一次、231 条终止路径逐条对账。五个项目里没有一家做到这个层级（vectorbt 连幸存者偏差都不检测）。
2. **多重检验的自觉**：BH-FDR + Hansen SPA + 完整组合 CSCV + Deflated Sharpe + trial registry，并且明确写"12 个配置不等于 12 次独立试验"。
3. **成本与容量的双档压力**：基础/双倍成本 + 全损/零收益两种退市情景 + 72 组正式回放 + NAV 会计恒等式核对。
4. **拒绝自我欺骗的措辞纪律**："描述性补充研究，不能作为新的独立样本外验证"、"已审阅窗口，需要新的未开启期间"。这是**学术界水平**的表述习惯。

也就是说：**EntroPy 的方法论是资产，工程包装是负债。**

---

## 4. 七个需要提升的地方（按严重程度排序）

### P0-1 定位错位：它是"结论"，却打扮成"平台"

平台的必要条件是分发面：能装、能跑、能提问、能被贡献。EntroPy 一样都没有；而它已有的最强内容——"我们用最严格的口径，没有找到能扣费存活的 alpha"——恰恰是**一篇能被引用的结论**，却被埋在 `docs/STAGE_REPORT` 第 6 节。

竞品的首屏是承诺，EntroPy 的首屏是资产清单（44 因子 / 449 测试 / 7 维 PRS）。读者读完之后不知道"我为什么需要它"。

### P0-2 场景选择把自己锁死：数据不可分发 = 结论不可验证

这是最结构性的一条。WRDS 授权数据意味着：

- 外部用户 clone 之后**跑不出任何头条数字**，只能选择"信"或"不信"；
- 你无法把结果发布成可复现的 artifact（论文可以，仓库不行）；
- 日频美股截面 alpha 是学术挖了六十年的赛道，残余边际（IC 0.01 量级）在 5bps 滑点面前基本归零——报告自己已经证明：残差动量毛 23.53% → 净 5.95%，长期动量 17.62% → 0.37%。

而五个竞品全部站在"数据免费、场景高热、边际未被榨干"的一侧。这不是能力问题，是**场地选择问题**，它比因子数量重要一个数量级。

### P0-3 品牌与实测背离

叫 EntroPy，主打的却是熵/Hurst/状态空间；而实测这两个因子是全表最弱的（IC −0.0029 / −0.0009，q≈0.7/0.94）。反而最强的残差动量、F8、真实换手率，是"传统"因子。名字承诺了一个故事，数据否掉了这个故事。

后果不是学术性的：任何认真读报告的人，都会在"熵"这个词上先减一分信任。

### P1-4 成果无法被别人复跑（分发缺失的具体清单）

| 缺什么 | 后果 | 竞品对照 |
| --- | --- | --- |
| `pyproject.toml` / `setup.py` | 无法 `pip install`，无法被依赖 | 五家全部上 PyPI |
| 合成/脱敏 demo 数据集 | `notebooks/*.ipynb` 走了合成面板，但需要手动 cd + 手算 | hftbacktest 内置下载器 |
| 语义化版本 + tag + release | 无"稳定点"，无升级路径 | nautilus 217 tags |
| 文档站（mkdocs/RTD） | 13 个 md 散落，无搜索、无导航 | 四家有站点 |
| 单一 CLI 入口 | `scripts/` 下 40+ 个 `wrds_*`（replay_/repair_/report_/verify_） | 竞品都是 `entropy <verb>` |

### P1-5 研究债已经工程化成了化石层

`scripts/` 里同时存在 `acquire_wrds_alternative.py`、`repair_wrds_broad_terminal.py`、`repair_wrds_ibes_basis.py`、`replay_wrds_alternative.py`、`replay_wrds_extensions.py`、`replay_wrds_neutral.py`、`report_wrds_extensions.py`、`report_wrds_neutral.py`、`report_wrds_turnover.py`……这是**每一轮研究迭代留下的沉积岩**，没有收敛成"阶段 + 动作"的参数化 CLI。根目录还残留 20+ 个 `.pytest_*` 目录和 `.log`。

这些对你自己无害，对"路过的人"是**专业性的直接反证**——它会让人怀疑：如果连临时目录都不清，结论的口径能有多干净？（这恰恰是你最在乎的东西。）

### P1-6 依赖与安装：全量 requirements 把所有用户劝退

`requirements.txt` 一次性拉 `torch + lightgbm + streamlit + ib_insync + baostock + simfin + numba + plotly`。一个只想跑因子诊断的人要装 2GB+ 的 torch。nautilus 用 extras 分流（`[ib]`, `[docker]`, `[dydx]`），EntroPy 应该分 `[ml] [ibkr] [cn] [dashboard] [wrds]`。

### P2-7 继续"加因子"的边际收益已经归零

20 日周期 224 个检验全部不显著，说明**瓶颈不在信号，而在"信号→可交易性"的映射**。报告自己给出了证据：日频调仓把 23.5% 的毛收益打到 5.95%，双倍成本下全负。真正的研究问题应该是持仓期结构、缓冲带、成交算法、事件驱动 alpha（IBES 修正、指数调整、期权流——你有权限但未落地），而不是第 45 个 IC=0.011 的因子。

---

## 5. 战略选择：五条路，建议走 A + E，留 C 做扩张

| 选项 | 做什么 | 优势 | 风险 | 判断 |
| --- | --- | --- | --- | --- |
| **A. 做"审计器"** | `entropy audit <returns.csv | weights.parquet | signals.parquet>` → PRS / DSR / CSCV / 成本压力 / 容量曲线 / 一句话裁决 | **数据无关**，任何人能跑；差异化极强（五家都不做统计严谨性）；你的协议立刻变成产品 | 需要把协议从 WRDS 上解耦，做成纯函数层 | **强烈推荐** |
| **B. 做"可复现负结果"案例** | 把 WRDS 那套整理成一篇"最严格口径下美股日频 alpha 不存在"的报告 + 聚合级 artifact | 学术引用价值高，最能体现你的方法论 | 数据授权限制分发；受众窄 | 值得做，但**不能作为唯一主线** |
| **C. 迁移到免费数据场景** | 把协议搬到 crypto / ETF / 预测市场，直接对上 hftbacktest、Kalshi 那批用户 | 数据可得、场景高热、能立刻被复跑 | 需要重做数据层与成交模型（资金费率、L2 撮合） | 中期扩张首选 |
| **D. 继续补全栈平台** | 加因子、加模型、加 dashboard | 沉没成本最低 | 边际收益已归零，分发面仍为零 | **不建议** |
| **E. 做教学资产** | `notebooks/EntroPy_quant_research_course_CN.ipynb` 已是雏形；扩成"如何不做假回测"课程 | 获客成本最低，与 A 天然互补（课程即文档即营销） | 不直接产生技术资产 | **推荐，作为 A 的引流** |

### 为什么 A 是最优解

你的对手都在卖"更快/更真/更方便"。**没有人卖"别骗自己"**。而这恰恰是你唯一做到世界级的东西：

- vectorbt 默认当根收盘成交，且不警告；
- tensortrade 三年后才承认手续费吃掉全部收益；
- hftbacktest 只解决成交真实性，不管多重检验；
- nautilus 强调工程正确性，不做统计裁决；
- **EntroPy 已经有 PRS + CSCV + DSR + 双倍成本 + trial registry，能对任意策略给出"可部署/有条件/拒绝"的裁决。**

把这件事解耦出来，就得到一个**不需要 WRDS、不需要美股、不需要你的因子库**的产品：任何人上传一条净值曲线或一张权重表，就能得到"你的夏普有多少是运气"。这是可以分发、可以验证、可以裂变的形态。

而 EntroPy 现在的全部负结果，会从"我们失败了"变成"**我们用它证明了别人的成功多半是假的**"——叙事瞬间反转。

---

## 6. 30 / 90 / 180 天：可验收的动作清单

原则：每条都写成 EntroPy 自己的语言——**可验证的承诺 + 验收标准**。

### 30 天：让它能装上、能跑、能被看懂

| 动作 | 验收标准 |
| --- | --- |
| 建 `pyproject.toml`，包名 `entropy-quant`（占位），extras 分层 `[ml] [ibkr] [cn] [dashboard] [wrds] [all]` | 干净环境 `pip install -e ".[ml]"` 成功；`pip install -e .` 不含 torch |
| 抽 `entropy` CLI 单一入口，`scripts/` 40+ 个 `wrds_*` 收敛为 `entropy wrds build/replay/report --stage {p0,batch2,ext,neutral}` | `entropy --help` 覆盖全部历史入口，旧脚本变成薄壳 |
| 生成 `data/demo/`：10 只票 × 3 年**合成或授权允许的**价格 + 权重，配套 `entropy demo` | 新机器 5 分钟内产出完整 tearsheet + PRS，无外网、无授权数据 |
| 清根目录：`.pytest_*`、`*.log` 入 `.gitignore` 并移出 | `git status` 干净；根目录 ≤ 15 个条目 |
| README 首屏重写：一句话结论 + 一张图 + 一条命令 | 首屏不再出现"44 因子 / 449 测试"；出现的是"我们发现 X 不成立，这是证据" |
| 打第一个 tag `v0.1.0` + CHANGELOG 语义化 | GitHub release 可下载 |

### 90 天：把"审计器"做成产品

| 动作 | 验收标准 |
| --- | --- |
| 抽出 `entropy/audit/`：输入 `returns` / `weights` / `signals` 三种形态，输出 PRS、DSR、CSCV、成本压力、容量、裁决 | 不依赖 `core/data`（无 WRDS、无 yfinance）即可运行；纯 numpy/pandas |
| `entropy audit` 支持任意第三方回测输出（quantstats / vectorbt / backtrader 的净值 CSV） | 对 vectorbt 官方示例跑一遍，输出"默认当根成交导致 DSR 虚高 X"的对照 |
| 建立 `docs.entropyquant.io`（mkdocs-material）：概念页 / 协议页 / API / 负面结果档案 | 站内搜索可用；每页有"可复跑命令" |
| 把 STAGE_REPORT 的负结果做成 3 篇可独立阅读的"审计报告"（动量、ML、换手） | 每篇有固定结构：假设→口径→数字→裁决→复跑命令 |
| 增加 `CONTRIBUTING.md` + issue 模板 + Discussions | 第一个外部 issue 在 30 天内出现 |

### 180 天：迁移到免费数据场景，开始获客

| 动作 | 验收标准 |
| --- | --- |
| 加 `entropy.data.crypto`（Binance/Bybit 日频+资金费率）与 `entropy.data.etf` | 复用同一 `audit`，无需改协议代码 |
| 复现一个公开策略（如 hftbacktest 的网格做市，或 Kalshi MM 的 AS 模型）并出审计报告 | 报告能被该社区引用；在对应 issue/Discussions 里发一次 |
| 课程 notebook 上线（"如何不做假回测" 6 讲） | 每讲末尾都有 `entropy audit` 实操 |
| 定治理：LICENSE（MIT 保留）+ 版本节奏（双周 patch / 季度 minor） | 连续 3 个 release 不跳票 |

---

## 7. 明确不建议做的事

1. **不要加第 45–60 个因子。** 20 日全不显著已经是答案。加因子只会扩大多重检验的分母，让 PRS 更难看。
2. **不要做 GUI/dashboard 的扩展。** `streamlit` dashboard 已存在但 README 都没提，说明它不产生价值。竞品里唯一做得好的是 nautilus 的 JupyterLab 镜像（因为它是**运行环境**，不是界面）。
3. **不要为了星标去做"全资产类别支持"。** nautilus 支持七类资产是因为它有公司和 21k commits；单人项目做这个是自杀。
4. **不要隐藏负结果。** 你最大的差异化就是敢说"没有 alpha"。把它放首屏，不是放第 6 节。
5. **不要在没有 audit 产品之前重写数据层。** WRDS 那套已经很干净，先复用，不要重构。

---

## 8. 一句话总结

五个项目的共同点是：**它们都只在回答一个问题，并且允许别人当场验证答案**。

EntroPy 回答了最难的那个问题（"这些信号扣费后还活着吗"），答案是否定的，而且证明过程比它们所有人都严谨——但它把答案锁在了只有自己能打开的数据和脚本里。

**要提升的不是因子、不是模型、不是功能，而是"让第二个人能跑出同一个数字"这件事。** 一旦做到，EntroPy 从"一个失败的项目"变成"一把别人没有的尺子"。

---

## 附录：五个项目关键事实速查

| 项目 | ★ | fork | commits | tags | 许可 | 变现 | 数据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hftbacktest | 4.8k | 936 | 1,038 | 56 | MIT | 赞助/数据托管 | 自带 L2/L3 下载 |
| vectorbt | 9.4k | 1.2k | 1,109 | 8 | Apache-2.0 + Commons Clause | PRO 订阅 $25/mo | 免费源 + PRO 连接器 |
| KalishMakerMaker | 438 | 79 | 31 | 1 | MIT | 无 | Kalshi API |
| tensortrade | 7.2k | 1.3k | 1,093 | 7 | Apache-2.0 | 无（曾停摆） | CCXT |
| nautilus_trader | 36.1k | 4k | 21,819 | 217 | LGPL-3.0 | 企业支持/咨询 | 适配器矩阵 |
| **EntroPy** | — | — | **44** | **0** | MIT | 无 | **WRDS 授权** |
