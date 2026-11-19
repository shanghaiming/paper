# 心理书籍 → 算子/策略台账 (2026-11-07)

> 本地已有 `behavioral_layer_20261020.md`（TFAS/Kahneman 237 行机制映射 + 9 篇原文 PDF）。
> 本文件只做一件事：把书里的原则压成**可回测的算子改动**，并记录已测结果。
> 书是假设，不是证据；每条都必须预注册 + 反向对照。

## 1. 书目与核心原则

| 书 | 核心原则 | 对应算子 |
|---|---|---|
| Kahneman《思考，快与慢》 | System 1/2；前景理论（参考点、损失厌恶 λ≈2.25、概率加权）；锚定；可得性；过度自信；均值回归 | 止损/止盈参考点、注意力入场、换手频率 |
| Thaler《Misbehaving》 | 处置效应（卖盈持亏）；心理账户；禀赋效应；自我控制 | 部分止盈、止损承诺装置、再平衡 |
| Freeman-Shor《The Art of Execution》 | 1866 笔/30,874 交易；49% 盈利；Rabbits/Raiders/Assassins/Hunters/Connoisseurs；止损与让赢家跑 | 止损倍数、partial/trail、top-k 集中度 |
| Lo《Adaptive Markets》 | 效率随制度/环境变化；适应性市场 | 状态条件权重、regime 门 |
| Douglas《Trading in the Zone》 | 概率思维、接受风险、不预测 | 固定风险预算、过程评估 |
| Duke《Thinking in Bets》 | 结果 ≠ 决策质量；概率化 | train/validation、过程指标而非单年收益 |

## 2. 已测：书→算子结果

| 算子 | 书的原则 | 改动 | 结果 | 判 |
|---|---|---|---|---|
| 止损 | Assassin / 损失厌恶 | −8% 硬止损（kl008） | 赢家 MAE 中位 −2%，输家 −9.8%；止损后不反弹 | **保留** |
| 部分止盈 | Raider / 心理账户 | 拿掉 partial（kl015） | 65.41→58.67%（−6.74pp），胜率 63.4→42.8% | **FAIL：partial 是承重件** |
| 部分止盈 | Connoisseur / 让赢家跑 | partial 推迟到 +1.5 ATR（kl015） | 65.41→61.08%（−4.33pp） | **FAIL** |
| 入场注意力 | 可得性/注意力 | 排除高 INTRA 群（kl006） | 65.41→45.27%（−20.14pp） | **FAIL：高 INTRA 是赢家群** |
| 入场注意力 | 可得性/注意力 | 只留高 INTRA / comp 正倾斜（kl007） | −12.34 / −3.53 / −2.87pp | **FAIL** |
| 再平衡 | 过度自信/过度交易 | 3 日 / 10 日（kl014） | 3 日 −22.7pp；10 日 −32.6pp | **5 日是甜点** |
| 袖集成 | 1/N 先验 | 逆波动袖权（kl014） | 65.42 vs 65.41（袖波动几乎相同） | **无效** |
| 止损宽度 | Assassin / Rabbit | STOP06 / STOP10（kl011） | 57.31 / 63.82；STOP10 验证 +3.80pp 但训练半 −355pp | **基准 0.8 稳健；制度依赖** |
| partial 0.1@1.0 | Raider | PART10（kl011） | 64.68（−1.52pp） | **FAIL** |
| trail 0.75 / 1.25 | Connoisseur | TRAIL075 / TRAIL125（kl011） | 63.86 / 66.70；验证 67.35 / 67.90；两半不同号 | **FAIL** |
| 入口权重 | Adaptive Markets | V1 状态条件 w_mom（kl010） | 68.11/17.79；验证 +0.51pp（不足 2pp 门）；mdd −4.87pp | **FAIL（风险改善、收益验证不足）** |
| 入口形式 | 凹性/交互 | V2 凹化 / V3 交互（kl010） | 63.21 / 64.44；验证 −2.90/+0.47pp | **FAIL** |
| 选择集中度 | 过度自信/注意力 | TOP1 / TOP3（kl012） | 90.18/36.52（mdd +14.45pp）；49.42/21.51（验证 −18.95pp） | **FAIL：top1 是集中度假象** |
| 宇宙冷热 | 可得性/流动性 | COLD / STRICT（kl012） | 66.54 / 64.20；验证 −0.51/−1.14pp | **FAIL；Ap 宇宙 80% 在最热端** |
| 仓位/1/N | 过度自信 / 1/N 先验 | SIZE_SW / POS60 / CORR85（kl013） | 65.46 / 47.74 / 65.42；验证 +0.10/−20.31/+0.05pp | **FAIL；负分 clamp 退化，维持等权** |
| 52 周高点锚定 | Kahneman 锚定 / George-Hwang | NEAR / FAR / TILT（kl016） | 44.81 / 42.95 / 59.33；验证 −35.29/−13.55/−10.58pp；rankIC +0.103→−0.112 | **FAIL；样本外翻号** |
| 峰终/CLV 出场 | Kahneman 峰终 / System 1 | 弱收盘即砍（kl016） | 验证净均值 +0.52% vs 固定 +1.29%；MAE 改善但 MFE 腰斩 | **FAIL；杀右尾** |
| 分数倾斜仓位 | 过度自信 / 1/N 先验 | 正区平移 +2/+3、rank²（kl018） | 验证 +0.17/+0.11/−0.00pp；w1 仅 0.504-0.507 | **FAIL；平移把分差压到 ~1%** |
| regime 条件止损 | Lo 适应性市场 | COND HIGH0.8/LOW1.0（kl017） | val +1.081 vs max FIX +1.128（Δ−0.048pp）；胜率 −1.53pp | **FAIL；是时期效应不是 regime** |
| 保留分差倾斜 | 过度自信 / 风险预算 | SHIFT0.3/0.5、SOFTMAX5（kl019） | 验证 +3.41/+1.27/+2.20pp；训练全负；mdd +5.03/+2.01/+3.21pp | **FAIL；尾部放大，非稳定增益** |

## 3. 书→算子的四条规则（从结果反推）

1. **原则不是算子**：书说“让赢家跑”，但 Ap 的 trail 只在 partial 后生效；拿掉 partial 等于同时拿掉 trail。映射必须落到引擎的真实机制。
2. **先测反向对照**：每个书原则都测“反向/去掉/推迟”版本；只有正向与反向显著不同，才算原则在起作用（kl015 的 No-Part 就是反向对照）。
3. **书预测方向，数据决定幅度**：Rabbits/Assassins 的方向被 kl008 验证；Raiders/Connoisseurs 的方向被 kl015 否证。书不能替代回测。
4. **行为原则常常是“弱机的补丁”**：INTRA_20 在 compB 上 +3.23pp，在 Ap 上 −20.14pp；同样的心理原则在不同算子上符号可以相反。

## 4. 待测：下一条书→算子（已部署 agent）

| 待测算子 | 书 | 预注册改动 | 判据 |
|---|---|---|---|
| 52 周高点锚定门 | Kahneman 锚定 + George-Hwang | dist52=C/max(C,252)−1；只买近高点 / comp 加 0.1·dist52 | 验证窗 +2pp，mdd +2，两半同正 |
| 峰终 CLV 出场 | Kahneman 峰终 + System 1 | 持仓日 CLV=(C−L)/(H−L)<0.2 且当日下跌 → 次日退出 | 事件级均笔 ≥ 固定 20 日 +0.5pp |
| 可得性/注意力门 | Barber-Odean 注意力 | 只在成交量放大但未极端时入场 | 验证窗 +2pp |
| 适应市场 regime | Lo | 按市场波动/广度切换 comp 权重 | 两半稳健 |

## 5. 结论（当前）

- **已经被数据支持的书原则**：损失厌恶/Assassin → −8% 止损；过度自信 → 不要 3 日高换手；1/N → 三袖等权。
- **被数据否证的书原则**：Raider 的“早点落袋”不是问题（partial 是承重件）；Connoisseur 的“让赢家跑”不能字面套用；可得性“排除高注意力”在 Ap 上反向。
- **尚未测**：锚定（52 周高点）、峰终（CLV 出场）、适应市场（regime 权重）。agent kl016 正在跑前两条。

## 6. 来源

- 本地：`papers/behavioral_layer_20261020.md`；`papers/pdfs_theory/Kahneman_Tversky_1979_Prospect_Theory.pdf`、`Kahneman_Tversky_1974_Judgment.pdf`、`Barber_Odean_2000_Trading_Hazardous.pdf`、`Barberis_Thaler_2003_Survey.pdf`、`Barberis_Xiong_2009_Disposition.pdf` 等。
- 网络书评/摘要：Wall Street Mojo behavioral finance books；Snappchart trading psychology books；Tradoshi guide；Validea《The Art of Execution》study；Investing.com book review；Harriman House；AAII "Being wrong and still making money"。
