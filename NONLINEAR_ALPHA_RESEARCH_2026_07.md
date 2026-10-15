# EntroPy 非线性 Alpha 研究（2026-07）

基于本地真实因子面板的 LightGBM/MLP 样本外研究记录。

## 结论

本次研究为 EntroPy 增加了两个可复用的非线性截面 alpha：

- `LIGHTGBM_ALPHA`：带 L1/L2、列采样和行采样的 LightGBM 回归模型；
- `MLP_ALPHA`：两层 GELU + Dropout 的 PyTorch 表格网络。

二者在 2022-07-13 至 2026-07-20 的 1,008 个样本外交易日上均显著优于现有因子的简单等权组合，但没有在平均 RankIC 上显著超过样本内事后识别出的最佳单因子 `MOM_12_1M`。机器学习 alpha 的 IC 波动更低，多空组合的毛 Sharpe 更高。

| Alpha | 平均 RankIC | 年化 ICIR | IC t-stat | IC 命中率 | 多空年化毛收益 | 多空毛 Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MOM_12_1M` | 0.0254 | 1.71 | 3.42 | 57.5% | 15.0% | 0.72 |
| `MLP_ALPHA` | 0.0205 | **1.94** | **3.87** | 57.2% | 17.7% | 1.16 |
| `LIGHTGBM_ALPHA` | 0.0197 | 1.90 | 3.81 | 55.0% | **18.6%** | **1.24** |
| `EQUAL_WEIGHT_ALPHA` | 0.0011 | 0.11 | 0.23 | 52.4% | -7.7% | -0.60 |

表中的多空收益为每日最高/最低五分位等权收益差，不含手续费、冲击成本和组合约束。因此它是 alpha 研究指标，不是可直接复现的净回测收益。

## 与 EG take-home 的对应关系

实现复用了 `D:\EG take-home` 中最重要的研究约束，而不是简单复制模型调用：

1. 每日截面标准化目标并裁剪至 `[-3, 3]`；
2. 只按唯一交易日生成 expanding walk-forward folds；
3. 在训练期和验证期之间 purge 完整的预测期限；
4. 仅使用 OOF 预测计算逐日 IC；
5. 对同一天的模型预测再做截面标准化；
6. 保存分折元数据、逐日 IC、特征重要性和行级 OOF 预测；
7. 用 paired circular block bootstrap 检查 IC 改善，而不是只比较点估计。

EntroPy 的输入不是匿名原始字段，而是已有因子面板。因此模型输入先在每个交易日内做 percentile rank，并映射到 `[-1, 1]`。缺失值填为当日截面中点 0；覆盖率低于 80% 或全局常量的字段不进入模型。本次共使用 33 个现有因子。

特征集合在首个验证区间之前锁定；不会依据后续验证期的覆盖率决定输入字段。

## 无泄漏的验证设计

- 预测目标：下一交易日复权收益；
- 训练数据起点：2010-01-04；
- 验证：4 个连续的 252 交易日区间；
- purge：1 个交易日，与 forward-return horizon 一致；
- LightGBM：每折使用全部可用历史行；
- MLP：每折从完整 expanding history 中按固定种子抽取最多 500,000 行；
- 验证集：不抽样，共 507,073 行、1,008 个交易日。

| Fold | 训练截止日 | 验证区间 | LightGBM 平均 RankIC | MLP 平均 RankIC |
| --- | --- | --- | ---: | ---: |
| 1 | 2022-07-11 | 2022-07-13 → 2023-07-13 | 0.0176 | 0.0192 |
| 2 | 2023-07-12 | 2023-07-14 → 2024-07-15 | 0.0135 | 0.0107 |
| 3 | 2024-07-12 | 2024-07-16 → 2025-07-17 | 0.0264 | 0.0255 |
| 4 | 2025-07-16 | 2025-07-18 → 2026-07-20 | 0.0211 | 0.0265 |

两个模型在四折中均保持正 IC。它们的逐日 IC 相关系数为 0.862，说明深度模型与树模型高度相关，暂不宜把二者当作两个完全独立的 alpha 来源。

## 配对显著性

20 日 circular moving-block bootstrap（2,000 次）结果：

| Challenger - Baseline | 平均 ΔIC | 95% CI | `P(ΔIC ≤ 0)` |
| --- | ---: | ---: | ---: |
| LightGBM - 等权组合 | 0.0186 | [0.0084, 0.0291] | 0.0000 |
| MLP - 等权组合 | 0.0194 | [0.0090, 0.0308] | 0.0000 |
| MLP - LightGBM | 0.0008 | [-0.0039, 0.0056] | 0.3565 |
| LightGBM - `MOM_12_1M` | -0.0058 | [-0.0154, 0.0041] | 0.8835 |
| MLP - `MOM_12_1M` | -0.0049 | [-0.0129, 0.0042] | 0.8740 |

由此可以支持“非线性模型优于简单等权组合”，但不能支持“MLP 优于 LightGBM”或“模型优于最佳单因子”。后两项的置信区间均跨过零。

## 特征解释

LightGBM 的平均 gain importance 前五名为：

1. `DOWNVOL_60D`
2. `MOM_12_1M`
3. `VOL_60D`
4. `STR_1W`
5. `MOM_6M`

MLP 文件中的 importance 是第一层连接权重绝对值均值，只适合做粗略诊断，不能解释为因果贡献。若要做模型治理，应另行增加严格样本外 permutation importance 或 SHAP 分析。

## 运行方式与产物

```bash
python scripts/evaluate_nonlinear_alpha.py
```

主要产物位于 `data/experiments/nonlinear_alpha_20260726/`：

- `oof_predictions.parquet`：行级 LightGBM/MLP OOF alpha；
- `alpha_comparison.csv`：模型、等权组合及所有输入因子的同口径对比；
- `daily_rank_ic.csv`：逐日 RankIC；
- `paired_ic_delta.csv`：配对 block-bootstrap；
- `feature_importance_*.csv`：分折及平均特征重要性；
- `fold_metadata.json`：训练/验证日期和样本数；
- `run_metadata.json`：特征、方向、超参数和评估口径。

## 使用边界与下一步

- 当前比较不含交易费用，也未经过 EntroPy 的组合约束与执行回测；
- 当前股票池可能存在历史成分近似和幸存者偏差；
- 超参数没有在 validation folds 上大规模搜索，避免在同一 OOF 区间过度调参；
- `MOM_12_1M` 是看过全部 OOF 结果后识别出的最佳单因子，不能视为预注册基线；
- 上线前应将 OOF alpha 接入现有 portfolio/backtest pipeline，加入成本、换手和容量约束；
- 若继续选模，应保留新的、未开启的 confirmation period。
