# 论文深读笔记 B 组（12 篇）— 2026-11-13

## 元信息与方法声明
- 覆盖: 索引行 132, 153, 164, 168, 186, 192, 198, 201, 206, 212, 225, 230（索引 §1 总表按相关度指派的 12 篇）。
- 方法声明: **全文 pdftotext -layout 提取后逐节深读**（正文+表格数字逐项抄录+附录/脚注核对），非摘要级阅读。每篇读完当篇立即落盘工作稿 `.scratch_deepread_B.md`（边读边落盘纪律），本文件为其最终合成稿。
- 页码口径: 所有页码 = pdftotext -layout 物理页（form-feed 计数 + 1），已对印刷页脚抽查核对。
- 错配文件注: 行 153/164/168/225 四篇 PDF 物理位于 `pdfs/_mismatched/`（文件名与内容错配已隔离，索引按实际内容评级为高）。本组从该目录**只读提取**至 `.scratch_txt_B/`（未改动隔离区任何文件），笔记按实际内容撰写并逐篇注明"错配文件, 按实际内容读"。
- 引用纪律: 学生级/预印本/样本内/超短样本的数字均在篇内标注折扣条件；所有"可检验命题候选"均为 [假设生成,未判]，不构成已判例。
- 本地判例术语（出场机/移植半径/制度符号翻转律/机器吸收率定律/CGO 弧线/相位彩票等）按 `quant_trade-main/docs/alpha_registry.md` 登记簿口径使用。

---

## [132] RVRAE: A Dynamic Factor Model Based on Variational Recurrent Autoencoder for Stock Returns Prediction（Yilun Wang, Shengjie Guo, NCSU, 2024, arXiv:2403.02500, q-fin.PM）
- 核心主张: 把动态因子模型的因子当潜随机变量，用 VRAE（RNN+VAE，每步一个 VAE、先验由 RNN 隐状态生成）同时做收益预测与风险（方差）估计；beta 由 LSTM 从公司特征回归得到，r̂t = β̂t·f̂t。卖点是"先验-后验学习"——训练期用当期收益 r_t 编码出最优后验因子，预测期只用历史数据训练的 predictor+decoder，防泄漏。声称在低信噪比、时序依赖市场数据上优于 IPCA/CA/FactorVAE/ALSTM/Transformer 基线。
- 方法与数据域: 美股（CRSP），**月频个股收益**，5682 只，46 个公司特征（Freyberger et al. 口径），2000-01~2020-12，15 年训练/3 年验证/3 年测试，因子数=5。预测视界=下月（月度截面）。未披露成本假设以外的执行细节（月频换手，30bps 情形）。
- 关键定量读数（原文+页码）:
  - p5 Table I: Total R2 RVRAE 20.06 / ALSTM 20.33（最高为 ALSTM）/ Trans 19.23 / FactorVAE 18.71 / IPCA 18.32 / CA 17.52；Predictive R2 RVRAE 1.57（最高）/ Trans 1.46 / FactorVAE 0.92 / ALSTM 0.91 / CA 0.74 / IPCA 0.36。
  - p5 Table II: OOS Sharpe 无成本 RVRAE 2.26 / ALSTM 2.15 / FactorVAE 1.92 / Trans "12.03"（原文如此，显系 2.03 排版错）/ CA 1.12 / IPCA 0.86；30bps 成本后 RVRAE 2.03 / ALSTM 2.01 / Trans 1.92 / FactorVAE 1.43 / CA 0.96 / IPCA 0.64。
  - p6 Table III（随机剔除 m 只股票再预测其收益，10 seeds）: m=50: RVRAE Rank IC 0.055(0.008) ICIR 0.295(0.039)；m=100: 0.053(0.004)/0.361(0.040)；m=150: 0.053(0.007)/0.363(0.052) — 全档居首。
- 可检验命题候选 [假设生成,未判]:
  1. 月频截面上，时序依赖模型（RNN/VRAE 系）相对静态特征模型（IPCA）的 Predictive R2 增益（0.36→1.57）主要来自波动聚集期的协方差结构而非条件均值改善 → 若在 A股日线域复刻，预期增益集中在高波动 regime。
  2. RVRAE 潜空间分布的 σ 可作风险代理：高 σ 股票随后实现波动更大 → 条件"σ_z 高于截面中位"→ 预期未来 20 日实现波动高于中位。
  3. 剔除训练股票后 IC 几乎不降（0.055→0.053）→ 新股/冷门票外推稳健，与本地"池构成 vs 位置密度"判例（kl109: 因子缺失池=富矿）可对话。
- 与本地判例的接口: ①月频/月度视界与本地 ENS 入场机（周级窗口）不同频，移植半径有限；但其"潜变量方差=风险估计"与本地出场机 R5 棘轮/波动带的交互值得探针——出场机的 A/PP 参数是否应随潜空间 σ 条件化，本地尚无此判例。②其 Rank IC ~0.05（月频）与本地 factor_cache 的日频 IC t+3.4 量级（iter157）不同口径，仅作量级参照。③prior-posterior 防泄漏构造与本地相位测试/PIT 纪律同构——但该文训练集内用当期收益编码，若复刻须严守"预测期只用 predictor"防线，否则就是前视。④无 A股/港股数据，无微观结构内容。

---

## [153] BDLOB: Bayesian Deep Convolutional Neural Networks for Limit Order Books（Zihao Zhang, Stefan Zohren, Stephen Roberts, Oxford-Man Institute, 2018, arXiv:1811.10041, NeurIPS 2018 BDL workshop）【错配文件, 按实际内容读; 物理位于 pdfs/_mismatched/, 只读提取】
- 核心主张: 对 DeepLOB（CNN+Inception+LSTM）加 MC dropout 变分推断（100 次随机前向），用后验预测分布的不确定性（predictive entropy H）做仓位 sizing——低不确定加仓 1.5×、高不确定减仓 0.5× 并回避劣质交易，在类似风险水平下显著提升利润；dropout 同时作为随机正则提升分类精度。
- 方法与数据域: 伦敦证券交易所 5 只股票（LLOY/BARC/TSCO/BT/VOD），**tick 级 LOB（40 档挂单，>1.34 亿条报价）**，2017 全年（6 月训练/3 月验证/3 月测试）；预测视界 = 下一时刻 mid-price 三分类（up/neutral/down，DeepLOB 标准 100-tick 窗口）；无交易成本、mid-price 成交假设、日内平仓。**预测视界为 tick/秒级，与日线域无直接移植通道**。
- 关键定量读数（原文+页码）:
  - p3 Table 1: Precision/Recall/F1/AUC — CNN 0.52/0.52/0.48/0.672；DeepLOB5 0.60/0.60/0.58/0.803；BDLOB 0.60/0.61/0.60/0.811。
  - p3: 仓位规则 "we upsize our positions to 1.5 × µ if H < β1 or keep original size µ if β1 < H < β2 but downsize our positions to 0.5 × µ if H > β2. We exit the position if H < β2." µ = 首档 ask/bid 量的 30%（p2）。
  - p4: 超参 "We set α = 0.7 for all strategies that use softmax probability vectors and set β1 = 0.1, β2 = 0.9 for Bayesian trading strategy." α=0.8/0.9 时交易太少，单次错判冲击放大；β2 控制出场: "A strict exit rule allows us to take even fewer trades amplifying profits, but the impact of wrong predictions is also more obvious and we can hold a bad trade for too long"（p4）。
  - p4: 风险度量用 DDR（Downward Deviation Ratio）替代 Sharpe，只罚负收益。
  - 图 2/3（p4）: 归一化日利润 boxplot，BDLOB_bayesian 在 5 只票上利润最高且风险相当（定性，无单一数字）。
- 可检验命题候选 [假设生成,未判]:
  1. **不确定性分层 = 出场/仓位信息**: 模型预测熵 H 高的持仓应优先减仓/离场 → 日线域版本: 预测分歧度（ensemble 方差/相位谱宽）高的状态，出场机应收紧（A↓/PP↑）。与本地"出场机参数面"判例（kl085/kl087: trail/出场参数是收益宿主）构成同构假设: 出场参数按不确定性条件化。
  2. 置信门 α 的非单调性: 提高入场置信门→利润升→过门后交易太少→单错冲击放大（α=0.7 峰值）→ 本地 ENS top_n/袖宽轴（kl081）同型: 集中度有内生最优非越尖越好。
  3. β2（出场不确定门）的利润-风险交换: 放宽出场门→利润升但坏仓持有过久→方差大 → 与本地 kl115/kl116（出场路径 PnL 解剖: rotate/棘轮/止损分账）对话。
- 与本地判例的接口: ①无 A股/港美股日线接口（tick 级 LOB，LSE），移植半径≈0（数据粒度不可得）；但其**仓位×不确定性×出场三合一框架**是本地出场机轴（A/PP/T/PC/R）的一个未测维度——本地出场机参数是全局常数，该文主张按 per-trade 不确定性动态化。②DDR（只罚下行）作为风险度量与本地 Calmar/零负年口径的亲和性: 零负年判据本质也是下行单侧；若做折层评估可用 DDR 替代 Sharpe 做 sanity。③"hold a bad trade for too long"机制 = 本地 kl115 LEFT_TRUNCATED（左尾截断）的镜像表述。

---

## [164] Machine learning approach to stock price crash risk（Abdullah Karasan, Ozge Sezgin Alp, Gerhard-Wilhelm Weber, Annals of Operations Research, 2025 accepted, DOI 10.1007/s10479-025-06596-7）【错配文件, 按实际内容读; 物理位于 pdfs/_mismatched/, 只读提取】
- 核心主张: 用 Minimum Covariance Determinant（MCD，稳健马氏距离+χ²(0.975) 截断）替代规则式 3.09σ 定义崩溃风险（NEGOUTLIER=1），以公司特有投资者情绪指数（7 变量 PCA）为核心解释变量做截面 logistic 面板回归: 情绪高→崩盘概率高，该关系在小市值组最强、大市值组反号；detoning（去第一主成分）后中-大市值组仍稳健。
- 方法与数据域: 美股 NYSE+NASDAQ（CRSP/Compustat），**周频收益→年度崩溃变量，年度回归**；1980-2020，6171 家公司 / 51,903 firm-years（p4）；扩展市场模型取残差（周频，±2 周滞后市场收益）。预测视界 = t+1 年。**频率与本地日线/周级出场机不同频，属会计-慢因子域**。
- 关键定量读数（原文+页码）:
  - p7 Eq(9): SENT = −0.136·P/E + 0.208·TURN + 0.052·EQS − 0.216·CEFD + 0.006·TOBIN + 0.17·LEV + 0.048·BSI（公司级 PCA 情绪指数；detoning 去第一主成分）。
  - p8 Table 1: NEGOUTLIER 均值 0.04（4% firm-year 至少一次 ML 崩盘），CRASH 均值 0.18，NCSKEW −0.143，DUVOL −0.179。
  - p9 Table 2: corr(NCSKEW,DUVOL)=0.889；corr(NEGOUTLIER,NCSKEW)=0.273，corr(NEGOUTLIER,CRASH)=0.089。
  - p11 Table 3（logistic，t+1 因变量）: SENTt 系数 NEGOUTLIER 0.106**(0.0442) / CRASH 0.209***(0.0613) / NCSKEW 0.141*** / DUVOL 0.134***；RETt 0.916***(13.96) 在 NEGOUTLIER 模型最大；SIGMAt 12.00***(3.122)；SIZEt 0.131***(0.0145)。
  - p15 Table 5（Arellano-Bond GMM 分规模）: 小公司 SENTt +0.2650**/+0.3883**（正显著），大公司 −0.2313*（负显著）— **情绪→崩盘关系符号随规模翻转**。
  - p17 Table 6（detoning 后）: 大公司组 SENTt 0.594***(0.199)，小公司组第二列 0.0604 不显著、0.344* 反号 — 稳健性部分崩（作者如实呈现）。
  - p18 Table 7: RMSE Decision tree 0.62 / Random forest 0.64 / XGBoost 0.68；特征重要性: turnover（dturn）最强，log_size 最弱（Fig.1, p18）。
- 可检验命题候选 [假设生成,未判]:
  1. 换手变化（DTURN）是崩盘风险第一驱动（XGBoost 重要性榜首）→ A股日线域: 高换手/量能异常票的左尾风险集中 → 与本地 kl113（US D10=彩票型 skew）/ kl110（极端态收割）对话: 量能轴两端两机制。
  2. 情绪→崩盘符号随市值翻转（小正大负）→ 港股微盘 B20 腿（冷门票）若构造 SENT 类指标，预期小市值侧情绪高→崩盘概率升，大市值侧反号 → 这是一条**制度符号翻转律候选**（跨规模而非跨市场）。
  3. MCD 稳健马氏距离（vs 固定 3.09σ）作为崩盘标签的自适应优势 → 可测: A股 2015/2024 微盘崩期间，MCD 标签 vs NCSKEW/CRASH 标签的事件覆盖差异。
- 与本地判例的接口: ①"高情绪=坏消息囤积=崩盘前兆"与本地 CGO（资本利得过量）心理线是同族（锚定/囤积心理→左尾），本地已判 CGO 全腿封存（机器吸收率定律），该文的 SENT 指数（P/E+TURN+CEFD+BSI 混合）是 CGO 的镜像变量，移植前须预期被机器吸收；②SIZE 交互与本地 #0c31 SIZE 条件单（中小盘侧门非微盘门）的门-腿结构同构；③周频残差化（扩展市场模型）与本地 RMT/残差动量判例（kl097: 共因子即信号，清除反伤）直接冲突——该文把残差当因变量，本地已判 A腿残差动量 RMOM_HARMFUL；④慢因子（年度）无出场机接口，移植半径=组合层防御门（崩盘风险门可作折层择时条件候选而非选股因子）。

---

## [168] Understanding stock market instability via graph auto-encoders（Dragos Gorduza, Stephan Zohren, Xiaowen Dong, Oxford, NeurIPS 2022, arXiv:2212.04974）【错配文件, 按实际内容读; 物理位于 pdfs/_mismatched/, 只读提取】
- 核心主张: 用 GAE（2 层 GCN 编码器+内积解码器）在日 t 的股票相关图上训练、在日 t+1 图上测边重构精度 AUROC(t+1)；AUROC 低 = 图异质化 = 市场不稳定代理。实证 S&P500 分钟数据: AUROC 与波动率反向共动，且**在波动率上冲前先行下弯**；加入 AUROC 后 HAR 类 log-RV 预测 R2 显著提升。
- 方法与数据域: 美股 S&P500 成分（**分钟级**价格，EOD API），2015-2021/2022，401 只（全期有数据者，p3）；相关图 = 20 日滚动窗 Pearson 相关，阈值 0.7 二值邻接（p3）；波动率 RV = 分钟收益平方和，log-RV；预测视界 = t+2（次日/1 小时频率）。日线域可移植性: 相关图构造只需日线 20 日窗，**机制本身频率无关**。
- 关键定量读数（原文+页码）:
  - p3: "Auv = 1 if the (u, v) return pair has a correlation higher than 0.7 and otherwise we set Auv to 0"；S=20 日窗。
  - p6 Table 1（1 小时频率 log-RV 预测）: Linear R2 0.360（含 AUROC）vs 0.107（不含），p=0.00；Tree 0.357 vs 0.310，p=0.043；MLP 0.380 vs 0.317，p=0.003（bootstrap 单边）。
  - p5-6: "AUROCt+1 is high in periods of relative market stability where volatility is lower and sharply fall during downturns"；"the AUROCt+1 seem to dip slightly before a volatility upswing"（p6）。Spearman 相关系数只在图 5 中给出，正文未落数字。
  - 六个灰带标注不稳定期（2015 闪崩、2020 疫情崩盘等，p5）。
  - p7（讨论）: 限制 = 需 ≥20 日价格数据成网，新上市公司无法入网；当前为静态嵌入，未来 DynGAE。
- 可检验命题候选 [假设生成,未判]:
  1. A股日线域: 20 日窗相关图（阈 0.7）的 GAE 次日边重构 AUROC 下跌 → 预期 t+2 指数/个股波动率上升（分钟→日线的移植）。
  2. 更廉价的同构代理: 相关图密度（corr>0.7 边占比）或截面平均相关系数的日变化，是否已捕捉 AUROC 的大部分信息（该文未做此消融）→ 若是，无需 GAE，一行公式即得不稳定指标。
  3. "先行下弯"假说: AUROC 峰谷领先波动率峰谷 1-2 日 → 可作折层/腿层的**波动 regime 门**（与本地 CRIT/MB20_lo 条件门同族但基于图结构）。
- 与本地判例的接口: ①本地 corr 门判例（kl154/kl155: 引擎原生 corr 门 CORR_WIN 三门全过）是该文的镜像——本地在组合内用 corr 门选权重，该文用图同质性做波动预测，两者可在"图异质性→出场机收紧"上汇合: 图同质性跌破阈值时收紧出场（A↓/PP↑），属出场机条件化的图论版（与 #153 BDLOB 的不确定条件化同型）。②20 日窗 = 本地 C10/CRIT 门同窗量级，移植半径大（日线可算）；③2024 微盘崩/2015 A股崩可作图异质性事件的天然检验场（本地已有 kl093 HEDGE_BREAK_2026 对冲不对称判例可对照）；④与 #230（THGNN 相关性预测）构成"图结构×波动"双联: 168 用图做波动预测，230 用图预测相关性本身。

---

## [186] Learning Universal Multi-level Market Irrationality Factors to Improve Stock Return Forecasting（Chen Yang, Jingyuan Wang, Xiaohan Jiang, Junjie Wu, 北航, KDD'25, DOI 10.1145/3690624.3709328）
- 核心主张: 非理性因子分两级: 股票级 = 实际价与"协整理性价"之差 u_t（用可学注意力组合全市场候选股票价格 βij·p_j 构造软协整序列，DF 式平稳正则 |ρ|<1 约束）；市场级 = 异常同步波动（δ 涨跌家数占比超 H_m）通过双子市场对比学习+市场同步预测两个自监督任务嵌入市场表示 m_t。因子喂给 Transformer+图注意力混合模型（RankIC 正则）预测下期收益。中美两市场全指标居首，因子对基线模型普遍增益。
- 方法与数据域: **A股（沪深 5148 只，2006-01~2023-03，测试 2018-03~2023-03）+ 美股（AMEX/NYSE/NASDAQ 8993 只，2000-03~2020-10，测试 2015-10~2020-10）**（p6）；**日频**（期=交易日），预测视界 = 次日收益；Qlib Alpha360 六个量价特征；每 30 交易日滚动重训（p7）；多空对冲策略 N=10% 市场股票数，单边成本 0.1%（p7）。
- 关键定量读数（原文+页码）:
  - p8 Table 1（US）: IC UMI 0.057 vs DA 0.054；RankIC 0.061；ICIR 0.663；RankICIR 0.666。投资: AR 0.289 / AV 0.144 / SR 2.007 / MDD 0.129 / CR 2.236（vs 最强基线 Co-CPC SR 1.366，CR 1.165）。
  - p8 Table 1（CN）: IC 0.078 / ICIR 0.831 / RankIC 0.073 / RankICIR 0.777；AR 0.358 / SR 2.680 / MDD 0.072 / **CR 4.958**（vs 最强基线 Co-CPC SR 1.921）。
  - p8 UMI+CM 跨市场因子迁移: US SR 2.126 / CR 2.434，CN SR 3.011 / CR 5.515 — 用对方市场学的市场级因子再迁移，双市场再升一档（**跨市场迁移正增益**）。
  - p8 消融序: NS < NM < ND < NR < UMI（股票级因子贡献 > 市场级）。
  - p8 Table 2 普适性: 因子加到 DA/D-Va/Co-CPC 上，US SR +19~38%，CN SR +27~41%。
  - p9 案例: Wisesoft（SZ002253）2016-11-09 涨停日（公司名与"特朗普大胜"谐音），u_t 出现异常尖峰 — 股票级因子捕获纯谐音型非理性事件。
- 可检验命题候选 [假设生成,未判]:
  1. **协整残差 u_t 是股票级非理性标签**: u_t 极端（低于/高于阈值）→ 预期随后均值回复 → A股日线域可用"价格 vs 行业/全市场注意力组合价"之差的 z-score 廉价复刻（无需神经网络，用行业等权或 beta 加权代理）。
  2. 市场同步度（δ 涨跌家数净值占比）超阈值日 → 次日截面相关性上升/反转机会集中 → 与本地 THS/港股广度恐慌择时叠层（iter191/212）同构: 广度恐慌日已判 A股 beta 收割 +44.4bp/触发，美股更弱（+19.4bp）— 该文市场级因子即广度信号的表示学习版。
  3. 跨市场因子迁移正增益（CN↔US）→ 港股 B20 腿可用 A股/美股学的市场级因子增强，但本地已判 corr(B20,A/US)≈0.15-0.2 低相关的来源是微盘流动性而非市场状态，预期迁移增益集中在波动 regime 识别。
- 与本地判例的接口: 最直接的一篇。①u_t = 协整偏离 = 本地 CGO（锚定心理）的价格侧镜像 — 本地 CGO 六例全向封存（机器吸收率定律），预期 u_t 类因子也会被 ENS 机器吸收，但其"事件标签"用法（谐音涨停例）与本地 kl110 极端态收割（blend 空间微笑）可对话；②市场级同步因子 = 广度恐慌的连续版，本地已有判例: A股 H5 +44.4bp/触发、美股 H5 -22.7bp（择时超额弱）— 该文未报告纯广度基线消融（NM 消融只去掉表示，未对比原始 δ 信号），这是引用其结论时的最大保留；③多空对冲+RankIC 损失的评估口径与本地 ENS 入场机不同（对冲 vs 纯多+出场机），移植半径: 入场信号侧而非出场侧。

---

## [192] Dynamic Asset Pricing: Integrating FinBERT-Based Sentiment Quantification with the Fama–French Five-Factor Model（Chi Zhang, 天津城建大学, 2025, arXiv:2505.01432）
- 核心主张: FinBERT（新闻+Twitter/Reddit）构造日频情绪指数 S_t 及其 21 日滚动波动 HV_t，嵌入 FF5 回归。**诚实读法: 全样本 OLS 情绪项不显著**（St 0.0062, SE 0.0068；交互项 0.2353, SE 0.3077，均无星），论文的实质发现全在滚动回归与事件研究里: 情绪系数 γ 时变，2021 risk-on 期正（峰值 >0.10），2022 紧缩熊市期反转为负（谷 ~−0.11）；FOMC 2022-06-15 加息 75bp 事件窗内情绪增强模型只把 |CAR| 降低 ~4bp 且仅在 t=0~+2 显著，+5 日后不显著。摘要宣称的 "consistently positive impact" 与正文 Table 2 不符，引用时以正文数字为准。
- 方法与数据域: 美股（S&P500/Nasdaq/DJ 指数 + 102 只大市值个股，Yahoo Finance），日频，2020-2022（724 交易日，p14/p15）；情绪源 = Reuters/Bloomberg/WSJ + Twitter cashtag + r/WallStreetBets；因变量 = 日超额收益；Newey-West(5)。样本期仅 3 年且覆盖 COVID+加息双 regime — 事件研究外部效度有限。
- 关键定量读数（原文+页码）:
  - p14 Table 1: S_t 均值 0.0458 / std 0.0678 / [−0.3164, 0.1935]；HV_t 均值 0.0343；VIX 均值 25.17 [13.68, 82.69]；n=724。
  - p15 Table 2: 基线/情绪/交互三列，Market−RF 0.9442*** 恒显著；SMB/HML/RMW/CMA/ΔDGS10 均不显著；St 0.0062(0.0068) 无星；St×HV_t 0.2353(0.3077) 无星。
  - p16: 60 日滚动 γ̂w — 2021 重开牛市爬升至 2021-08 峰 >0.10，"nearly 85% of windows in this subsample reject H0: γ = 0 at the 10% level"；2022-03 首次前置加息后转负，秋季回撤谷 "trough near −0.11"。
  - p17: 事件研究 [−10,+10] — "reduces the absolute CAR magnitude by roughly 4 basis points at the trough"；BMP 检验 102 只大市值股，均值 |CAR| 增强模型在 5% 水平显著更低**仅两日窗**，"differences become insignificant beyond five trading days"。
  - p18 稳健性: 90/120 日窗符号反转时点漂移 ≤6 交易日；分源 — 危机期新闻情绪主导，低波动期散户情绪有增量；安慰剂事件不显著；IV（滞后情绪冲击+Google Trends 金融焦虑指数）与 OLS 一致。
- 可检验命题候选 [假设生成,未判]:
  1. **情绪符号的制度依赖**: risk-on 期情绪正载荷/流动性紧缩期反转负载荷 → A股日线域: 用融资余额/北向/成交额构造的 proxy 情绪，预期在 2015 牛市段正、2018/2022 熊市段负 — 这是本地"制度符号翻转律"（CGO 跨腿）的时序版候选: 同一因子跨 regime 符号翻转而非跨市场。
  2. 情绪增量只在 2 日事件窗: → 日线出场机的 t+1/t+2 检查窗（R5 棘轮 5 日窗）是情绪信息的自然宿主，更长持有窗情绪无增量。
  3. 全样本不显著+滚动显著 = 相位彩票结构: 若用固定系数全样本判生死会漏掉条件效应 → 与本地相位测试（k0 谱顶 vs 幸运相位）的方法论同构，支持本地"条件单"传统而非全局因子。
- 与本地判例的接口: ①情绪时变符号直接呼应本地 CGO 弧线（kl172-179: 心理因子跨腿/跨池符号翻转，机器吸收率定律）— 该文是 regime 维度的同型证据，可作制度符号翻转律的外部文献锚；②"4bp 事件窗增益"量级与本地出场机锁利 1.7pp/笔（iter206）不同口径但同向: 短窗行为信息在出场侧；③102 只大市值股+FF5 残差化与本地 C3/ENS 截面机器不同（其对冲式 FF5 vs 本地出场机主导），移植半径 = 情绪作折层条件门（如 S_t×HV_t 高时收紧出场），不建议作入场主因子（本地已判量类/情绪类入池多被吸收）；④样本期 2020-2022 太短，单作者单校，引用权重要打折。

---

## [198] A Comparative Analysis of Statistical and Machine Learning Models for Outlier Detection in Bitcoin Limit Order Books（Ivan Letteri, 单作者, 2025, arXiv:2507.14960, L'Aquila 大学）
- 核心主张: 在统一回测环境 AITA-OBS 内对比 13 个无监督异常检测模型（4 统计 + 9 机器学习），把异常分数经 P95 动态阈值转成二元信号后按"逆近期动量"（mean-reversion）方向交易，发现最简单的经验协方差（Empirical Covariance，Mahalanobis 距离 + 97.5 分位阈值）以 6.70% 收益居首，胜过全部 9 个 ML 模型与 Buy-and-Hold。作者主张简单稳健统计方法在加密 LOB 异常驱动策略中优于复杂 ML。
- 方法与数据域: 单一资产 Bitcoin、单交易所（只称"a major exchange"）、1 分钟 K 线 + LOB 层特征、18 天样本（2024-12-26 至 2025-01-13）、26,204 条记录、初始资金 $1500、每笔 33.33% 固定比例仓位（p6）。预测视界=至下一信号或 bar 收盘（分钟级）。无 train/test 划分表述 — 全样本内回测。移植判据: 分钟级加密 LOB，与 A股/港美股日线域距离极远。
- 关键定量读数（原文+页码）:
  - p1 摘要: "a dataset of 26,204 records from a major exchange, demonstrates that the top-performing model, Empirical Covariance (EC), achieves a 6.70% gain, significantly outperforming a standard Buy-and-Hold benchmark"。
  - p6: "dataset spanning 18 days (from 26 December 2024 to 13 January 2025), comprising 26,204 records at a 1-minute timeframe"；门槛 "τM = P95({sM(Xt) | ∀t})"（p6 式(4)）。
  - Table 2（p7）: EC 341 多/315 空、累积利润 $100.47、+6.70%；HBOS $70.27/+4.68%（562/723 笔）；EE −0.55%、MCD −0.59%。
  - Table 3（p8）: CBLOF $75.48/+5.03% 但 649/662 笔、KNN +4.79%、LOF +4.02%、OC-SVM $43.59/+2.91% 仅 140 笔、K-Means 8852 笔仅 +0.07%、OPTICS/IsoF +0.86%；"the B&H benchmark, which incurred a loss of -$37.06"（p8，即 −2.47%）；费率 "a market maker fee of 0.08% per trade"（p8）。
- 可检验命题候选 [假设生成,未判]:
  1. 若日线域以小特征集（Amihud 流动性、量偏离、波幅）构造 Mahalanobis 距离并按分位数门槛打极端旗标，则旗标后短期（1-5 日）条件收益应呈反转（负动量后续跌）→ 可在 A股/美股日线复刻"简单协方差打旗 + 逆向"骨架。
  2. 模型交易频率越高，计入成本后每笔利润越低（原文 OC-SVM 140 笔 +2.91% vs K-Means 8852 笔 +0.07%，p8）→ 频率-成本权衡在日线域对应持有期拉长应改善净收益。
  3. 简单统计异常检测 ≥ 复杂 ML 异常检测（原文 13 模型横截面，但无显著性检验）。
- 与本地判例的接口: (a) 本质是"极端态逆向收割"的分钟级版本 — 与 kl110 极端态收割判例同构，其证据支持"异常旗标后做反转"的方向，但域不可比（分钟 vs 日线、加密 vs 股票）；(b) "简单统计胜过全部 9 个 ML 模型"的横截面与本地对复杂模型的怀疑一致，可作外部锚引用（注意其无显著性检验）；(c) 高频交易笔数侵蚀利润的结论与本地 5bp 成本线（主线 57.9@5bp）同向，可作成本敏感性讨论的外引；(d) 方法论缺陷清单可直接复用为开卡时的反面教材: 异常分数 Min-Max 归一化 "over the entire dataset"（p6 式(3)）与 P95 门槛 "∀t"（p6 式(4)）均为全样本统计 — 前视偏差；18 天单资产无显著性检验，数字本身不具引用价值。

---

## [201] Deep Learning for Short-Term Equity Trend Forecasting: A Behavior-Driven Multi-Factor Approach（Yuqi Luan, 单作者, 2025, arXiv:2508.14656; 作者身份 "Master of Science UZH ETH in Quantitative Finance" p1 — 硕士论文级，未经同行评审）
- 核心主张: 把 40 个公式化技术/行为 alpha 因子（量价背离、底部反转、动量追涨三类结构）喂给双任务 MLP（回归头预测 5 日前瞻收益 + 分类头预测涨跌方向，共享隐层），在 IC 与组合回测上均胜过 CNN 与线性 SVR 基线，主张行为驱动的分结构深度建模能从短期市场数据中提取 alpha。
- 方法与数据域: 股票市场未明说（由 "January 2025…drop in the Nasdaq" p12 推断为美股）；**股票池、股票数、样本起止期均未披露 — 重大缺口，全文只在 p7 给出 "data before 2023 is used for training, and data from 2023 onward is reserved for validation"，p12 顺带提及 "in 5 years"**；日频（附录公式全为日线 OHLCV 构造，尽管正文称 "derived from high-frequency price and volume data" p5 名实不符）；预测视界 5 日；组合构造为每日 top-5/bottom-5 多空（p7）。移植判据: 域与日线接近，但样本面不透明，数字不可复刻。
- 关键定量读数（原文+页码）:
  - Table 1（p11）: MLP（dual-task）IC 0.0340 / IC Std 0.2097 / IR 0.1621；CNN IC 0.0137 / IR 0.0686；SVM IC 0.0108 / IR 0.0518 — MLP 的 IC 是 CNN 的 2.5 倍。
  - Table 2（p13）: 年化 Sharpe MLP 1.6075（年化收益 0.5349/波动 0.3321）、CNN 1.1487、SVM 0.7709。
  - p12: "The Top 5 cumulative return in 5 years for MLP model reaches above 800%"（股票池未定义故无法评估集中度）。
  - p14 自认零成本: "the current experimental design abstracts away from trading frictions, as transaction costs, liquidity constraints, and turnover effects are not fully embedded"。
  - SHAP 归因（p14）: "macd_rsi_product, momentum5_rank, and volume_above_adv20 exhibit the highest positive attribution"；"Momentum & Herding factors are dominant in the positively attributed region…Volume-Price Divergence factors appear more frequently in negatively attributed areas, consistent with their reversal detection nature"。
  - 损失（p7）: "Ltotal = Lreg + 0.5 × Lcls"。
  - 附录 Table 3（p16）给出全部 40 因子公式（如 alpha_momentum_5d_min_rank = −1·rank(Close−Close.shift(5)), alpha_volume_spike_ratio = Volume/ma(Volume,5), alpha_rsi_bounce_strength = I(rsi14↑ & <30)·(rsi14−50), alpha_volspike_times_body = 量能尖峰×K线实体强度）。
- 可检验命题候选 [假设生成,未判]:
  1. 若附录底部反转族（rsi_bounce_strength、boll_rebound、macd_cross_strength）在 A股/港美股日线上作入场条件化，条件 5 日前瞻收益应为正。
  2. 若美股日线上动量×超买交叉特征（macd_rsi_product）SHAP 为正，同一族在 A股应被本土因子集部分吸收（机器吸收率定律方向检验）。
  3. 双任务损失（回归 + 0.5×方向 BCE）相对纯回归提升 IR（原文无消融，仅横截面对比）。
  4. 量价背离族做入场信号方向为负（原文 SHAP 负区），即"价升量缩"更宜做空/回避而非做多。
- 与本地判例的接口: (a) **附录 40 因子公式表（p16）是本篇唯一具直接移植价值的资产** — 全部为日线 OHLCV 可计算公式，与本地 A股/港美股日线域零摩擦，可直接开卡做条件化/出场机条件变量候选（量价背离族尤其契合 kl110 极端态收割的"量能异常"判据）；(b) 该文是纯入场侧研究（无出场机、无成本），与本地"全平台 alpha 在出场机不在入场"判例正面互补: 其 MLP IC 0.034 的入场端数字恰好可用于对照本地 C3 入场×V134 出场的组装范式 — 引用时应标注其未计 5bp 成本线，top-5 年化 53.49% 与本地主线 57.9@5bp 不可直接比较；(c) 其"分行为 regime 定制模型"思路（动量/背离/反转三分）与本地制度分域/条件门（#0c31 SIZE 条件门、CGO 弧线分域）同构，但该文用特征预分类实现而非显式门控；(d) 引用权重要打折: 硕士论文级、单作者、股票池与样本期缺失、IR 与 ICIR 定义重复（p9 式(7)(8) 相同）、无显著性检验。

---

## [206] Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles（Aryan Ranjan, 单作者, 2025, arXiv:2510.22348, Oxford St Cross; 学生级工作）
- 核心主张: 用 MLP+XGBoost+CatBoost 软投票集成预测 SPY 未来 5 日累计回撤超 1% 的二元事件，特征为 12 资产跨资产域（股/债/汇/商品/波动率）的 178 个工程特征（滚动矩、Hurst 指数、跨资产 beta/相关、KL 散度），按预测概率 0.5 门槛做多/做空 SPY。声称 Sharpe 2.51、年化 CAPM alpha +0.28、beta 0.51。**诚实读法: 全部核心数字是全样本内训练的结果** — p7 明文 "the final ensemble is trained on the entire dataset. Metrics such as ROC-AUC, precision, and recall therefore reflect in-sample performance"；Table 4 标题即 "In-sample trading strategy performance metrics (2005-2025)"（p15）；p16 注 "These metrics do not currently account for transaction costs nor have they been forward tested"。索引行的 Sharpe 2.51 引用必须带此三重折扣。
- 方法与数据域: 美股单一标的 SPY（工具池 12 个: SPY/QQQ/IWM/TLT/VIX/GLD/CL=F/DX-Y.NYB/EURUSD=X/JPYUSD=X/TNX/IRX, Table 1 p3）；日频；预测视界 5 日（"where h = 5 trading days, and δ = 1%" p3 式(2)）；样本期 2005-2025, 5423 观测（p4 "178 engineered variables, spanning 5423 total observations over the 2005-2025 period"）；特征筛选 178→134（去零方差/相关≥0.95）→80（互信息 top80, p4-5）；有效样本 5000-6000 日观测（p6 自述）。移植判据: 美股指数级、日频、5 日视界 — 与本地美股腿折层/择时域同构，可移植性中等。
- 关键定量读数（原文+页码）:
  - Table 2（p5）: MI 前 13 特征全是各资产短程 Hurst（τ=16）: CL=F 0.1092 / JPYUSD 0.1076 / IRX 0.1074 / GLD 0.1069 / … / SPY_hurst_short 0.0920，唯一非 Hurst 为 IRX_vol_63d 0.0950。
  - p5: "signals from oil prices, currencies, and interest rates show greater statistical association with short-term SPY drawdowns than SPY itself. This implies that the equity market is often reactive rather than proactive"。
  - MI-SHAP 背离（p9）: "Once short-term, cross-asset signals are included, the Hurst metrics' predictive power is largely absorbed by more immediate drivers"。
  - 崩溃周 SHAP 主驱动（p9-10）: SPY 收益、IRX beta、油偏度/峰度、IWM 收益与 beta；"systemic vulnerability often manifests first in smaller, more liquidity-sensitive segments"（p10）。
  - Table 3（p8）: 类 1 precision 0.95 / recall 0.82 / accuracy 0.95（自认 in-sample）。
  - 风险五分位（p12）: "the average 5-day SPY return during HIGH-risk periods comes out to −1.89%, while the average 5-day SPY return during LOW-risk periods comes out to +0.75%"。
  - Table 4（p15）: Sharpe 2.51 / IR vs SPY 1.73 / MDD −18.12% / 年化收益 40.84% / 年化波动 13.23% / CAPM alpha(日) 0.00111 / beta 0.51 / t 值 14.03。
  - 策略规则（p13）: 概率<0.5 做多、≥0.5 做空、按概率调仓、逐日交易。
- 可检验命题候选 [假设生成,未判]:
  1. 若对 A股/港股域计算跨资产短程 Hurst（原油/汇率/短利率代理），其对指数 5 日回撤的边际 MI 应为正但被短程跨资产收益条件吸收（吸收律外部复制）。
  2. 若以崩溃概率五分位门控本地美股腿（SCENG-v2 折层），Q5 周条件 5 日收益应为负且收紧出场应改善 Calmar。
  3. 小盘/微盘对大盘的应力领先性: A股微盘/中证1000 相对沪深300 在崩前周应先出现收益与 beta 抬升（对应其 IWM 发现）。
  4. MI（边际）与 SHAP（条件）背离可作为本土因子集"吸收完成度"的检测器。
- 与本地判例的接口: (a) **MI-SHAP 背离段（p9）是"机器吸收率定律"的独立外部表述** — 原文语言 "predictive power is largely absorbed by more immediate drivers" 与本地吸收律判例（如 CGO 弧线、kl066 相位审计降级）几乎逐字同构，可作开卡引用的外部锚；(b) 跨资产先行性（商品/汇/利率领先股指，"equity reactive not proactive" p5）为本地广度恐慌择时（iter191/204: A股 +44.4bp/触发 vs 美股 +19.4bp beta 收割）提供了**跨资产维度的互补版**: 本地已判广度维度，跨资产 Hurst/尾部矩维度未测 — 这是最直接的开卡缺口；(c) Q5 −1.89% vs Q1-Q4 +0.75% 的 2.64pp/5 日条件价差（p12）是折层时序条件门的量级参照（注意其 in-sample 性质，真实 OOS 价差应显著缩水）；(d) 数字引用纪律: Sharpe 2.51 / alpha 0.28 / beta 0.51 全部 in-sample 无成本无前向测试（p7/p15/p16 三处自认），与本地主线 57.9@5bp（含 5bp 成本 OOS）不可同日而语，引用时必须并排标注。

---

## [212] Interpretable Deep Learning for Stock Returns: A Consensus-Bottleneck Asset Pricing Model（Changeun Kim, Younwoo Jeong, Bong-Gyu Jang, 2025/2026, arXiv:2512.16251v5, POSTECH; 工作稿但成熟度高: 多轮会议报告、v5 修订、附 Internet Appendix）
- 核心主张: CB-APM 把分析师共识嵌入为结构瓶颈（concept-bottleneck，源自 Koh et al. 2020）: 第一阶段神经网络 f 用 114 个公司特征+宏观潜变量**逼近 9 个分析师共识变量** Ĉ，第二阶段**线性**定价模块 g(Ĉ) 输出预期收益，联合损失 L = L_R + λ·L_C 一步优化。共识瓶颈作为内生正则器，同时提升 OOS 预测精度并保证解释性。核心发现: 年度收益 OOS R2 从 λ=0（朴素 NN）的 7.63% 升至 λ=0.3 的 10.46%（+37%）；十分位组合 H-L 价差从 1.64% 升至 ~2.3%/月；GRS 检验显示随 λ 收紧，传统因子模型（CAPM~FF6）越来越无法定价其组合 — 学到的共识编码了经典因子未张成的定价变异。
- 方法与数据域: 美股 NYSE/Amex/Nasdaq 全市场，**月频观测 + 年度（12 月）收益预测视界**（p17 明言 "we focus on annual return prediction"，理由是分析师发多季度至年度预测）；样本期 1994-01 至 2023-12, 605,722 firm-month 观测、4,683 家公司（p8, p18, p56-57）；数据源: Chen-Zimmermann Open Asset Pricing 114 特征 + 9 共识变量（AnalystRevision/ChangeInRecommendation/ChForecastAccrual/EarningsForecastDisparity/FEPS/ForecastDispersion/REV6/AnalystValue/AOP, p15-16）+ CRSP + FRED-MD 115 宏观 + Goyal 8 宏观（自编码器压缩为宏观潜嵌入）；扩展窗 OOS（"strictly out-of-sample and free from look-ahead bias" p19）；组合月度再平衡、市值加权、多 top10% 空 bottom10%。移植判据: 美股月频/年视界 — 与本地日线域频率错位一档，移植半径受限但架构思想可移。
- 关键定量读数（原文+页码）:
  - Table 1（p41）: 股票收益 R2 按 λ: 0→7.63, 0.3→10.46（峰值）, 1.0→9.37；共识平均 R2 单调升 7.33（λ=0.1）→24.21（λ=1.0）；Analyst EPS 逼近最强 R2=71.43（λ=1.0），ChangeInRecommendation 恒为负（−0.16~−0.25）；共识-only 规格（λ→∞）共识 R2=30.30%（p14 脚注 5）。
  - Table 2（p42）: 单排序 H-L: λ=0 1.64 → λ=0.3 2.30 → λ=1.0 2.34%/月，低十分位恒负收益（−0.36~−0.96）、高十分位 ~1.3%/月。
  - Table 3（p43）双排序（FEPS×预测收益 5×5）: 预测收益 H-L 在低 FEPS 组 2.12~2.57%/月、高 FEPS 组仅 0.53~0.83%/月 — **信号在分析师悲观态最强、乐观态衰减**（"state-dependent attenuation" p9/p26）；FEPS 的 H-L 在低预测收益段为正（+0.93）、高预测收益段翻负（−0.66~−0.91）— 符号翻转结构。
  - Table 4（p44）: 长短仓月均对数收益 λ=0: 1.53%/Sharpe 1.10 → λ=0.3: 2.20%/Sharpe 1.44；Max DD 全部 <21.3% vs S&P500 24.77%；换手率 ~60%/月（p30，成本稳健性 relegated 至 Internet Appendix D.5.3 "robust to proportional trading costs" p28 脚注 12）。
  - Table 5（p45）: GRS F 从 λ=0 的 ~1.84-1.99（p=0.05-0.07）升至 λ≥0.2 的 3.6-4.7（p≤0.001）— 瓶颈越紧越逃逸线性因子张成。
  - Table 6（p46-47）: 单共识维度排序组合大多不被拒绝（Analyst EPS GRS F=0.93, p=0.51）— 共识维度可被低维特征组合近似，但组合成的预测信号不可。
- 可检验命题候选 [假设生成,未判]:
  1. 若本地机器（A股/港美股日线）把预测路由过可解释中介瓶颈（如 CGO 或情绪代理），OOS 预测力应在中度瓶颈权重达峰、瓶颈主导时回落（λ 曲线复制）。
  2. 横截面信号价差应在"分析师/情绪悲观态"放大、乐观态衰减（其 FEPS 态依赖结构）。
  3. 瓶颈收紧后的组合应更强烈拒绝经典因子张成（GRS 逃逸复制）。
  4. 线性定价头+非线性共识头两段式优于单段黑箱（其架构消融）。
- 与本地判例的接口: (a) **瓶颈 = 吸收律的工程化利用** — 本地判例把"信号被机器吸收"当风险（CGO 镜像、kl066 降级），该文反其道把"强制通过中介变量"做成正则器: 与其对抗吸收，不如把中介设为瓶颈后让吸收为你工作。这是对本地吸收律判例最有启发性的架构级回应；(b) 态依赖衰减（悲观态信号强/乐观态弱）+ FEPS 符号翻转（低预测收益段正/高段负）与本地**制度符号翻转律**同族 — 又一个"符号/强度随状态翻转"的外部例证（CGO 跨腿跨池翻转、情绪随 regime 翻转[192]、size 翻转[164] 之后的第四例，本次深读已集齐四个独立外部例证）；(c) GRS 逃逸框架可直接移植为本地"新信号是否被既有因子集张成"的检验器（对应本地残差动量 RMOM 判例的传统角色）；(d) 频率错位警示: 月频年视界 vs 本地日线 R5 视界，任何数字移植须先过频率换算，不可直接对照 57.9@5bp 主线。

---

## [225] Understanding the Impacts of Dark Pools on Price Discovery（Linlin Ye, 单作者, 2016, arXiv:1612.08486, CUHK-Shenzhen; 纯理论论文）【错配文件, 按实际内容读; 索引行原标 "LLM_Guided" 名实不符, 物理位于 pdfs/_mismatched/, 只读提取】
- 核心主张: 在噪声信息结构下（知情者信号有异质精度 σe，区别于 Zhu 2014 的完美信息假设）建立交易所+暗池两地博弈: 交易所做市商保证成交但有价差，暗池按交易所中间价撮合但概率成交。两个核心结论: (1) **分选效应** — 强信号知情者进场内、中等信号进暗池、弱信号不交易（流动性交易者同构分选）；(2) **放大效应**（Proposition 4, p33）— 存在阈值 σ̄（σ=σv/σe "信息优势"），信息精度高时暗池分流的主要是噪音交易者，交易所知情/不知情比上升 → 价格发现增强；精度低时中等信号知情者涌向暗池"缓冲区"（buffer zone, p3）避险 → 价格发现受损。该机制调和了 Ye(2011) 与 Zhu(2014) 的对立预测及实证分歧。
- 方法与数据域: **纯理论模型，无实证估计**（数值例子用参数 μz=60, μ=30, Gamma 分布, p28 脚注 31）；机制背景为美股碎片化市场（美国约 300 个交易场所、11 家交易所、40 个活跃暗池, p6）；暗池份额事实: 美国 7.51%（2008）→16.57%（2015）（p2/p6），澳大利亚暗流动性占总成交额 26.2%（p6, ASIC 2013）。移植判据: 无频率/视界概念（一阶段模型），移植半径≈0（数字层面），机制层面可借。
- 关键定量读数（命题与预测，全为模型推导）:
  - Proposition 3（p31）: 加暗池使两类交易者的场内参与率均降、总参与率升（流动性创造），信息精度高时场内价差变宽、精度低时方向不定（Remark 2 p31: 可能变窄）。
  - Proposition 4（p33）: "there exists a threshold, σ̄ > 0, such that, a) when σ < σ̄, adding a dark pool impairs price discovery, and b) when σ is large, adding a dark pool enhances price discovery"。
  - Proposition 5（p36）: 暗池伤害概率 σ̄v 随 σe（不精度）升、随 μ/μz（知情者相对规模）升；价格发现度量 = 信号噪音比 I(γe−γe,αe) 与 scaled RMSE（p33）。
  - 可检验预测汇总（6.2 节, p38-41）: (1) 暗池不成交概率随信息精度升，且场内价差随暗池不成交概率升（p38-39）；(2) 暗池市场份额与信息精度、与场内价差均呈**倒 U**（p39）；(3) **暗池交易的信息含量与信息精度、与场内价差均呈倒 U — 中等流动性资产暗池交易信息含量最高**（p41; 结论 p44 "assets with modest exchange liquidity have a high information content in their dark pool trades"）；(4) 流动性外部性: 暗池使场内量降、总量升（p41）。
  - 信息精度代理（6.1, p37-38）: 公司年龄/规模/分析师覆盖数/R&D 比/分析师预测分歧与误差；宏观层面用法律制度与披露系统质量。
- 可检验命题候选 [假设生成,未判]:
  1. A股大宗交易（最接近的"暗"渠道）信息含量（折价率对未来收益的预测力）应随标的流动性呈倒 U — 中等流动性标的大宗交易信号最强。
  2. 场内成交量异常因子的信号强度应随"分流渠道活跃度"（大宗/盘后成交占比）衰减 — 隐性流动性迁移稀释场内量信号。
  3. 港股（存在暗池）可用暗池份额数据直接复刻倒 U 预测。
- 与本地判例的接口: (a) **放大效应是"制度符号翻转律"的微观结构版第五外部例证** — 同一结构变动（加暗池）的效应方向由状态变量（信息精度）决定，与 164（情绪→崩盘风险随规模翻符号）、192（情绪符号随 regime 翻转）、212（FEPS 价差随乐观态衰减+符号翻转）同族；开"制度分域/条件门"卡时可作为理论支撑引用；(b) 倒 U 信息含量预测为本地量价背离/量能异常因子（201 附录因子族、kl110 极端态收割）提供了**流动性分层条件化**的假设来源: 量信号的有效性可能不是单调随流动性，而是倒 U；(c) "中等信号者进暗池避险"机制提示: 场内可见订单流是**经过分选的强信号子集**，对量价因子是幸存者偏差警告；(d) 信息精度代理（分析师覆盖/预测分歧）与 212 共识变量族、与本地情绪条件化共享同一测量面；(e) 移植纪律: 全文无实证数字（除市场份额事实），任何引用只能引机制不可引数字。

---

## [230] Temporal-Heterogeneous Graph Neural Networks for Predicting Stock-Stock Correlations（Fanshawe, Masih & Cameron, UQ, arXiv:2601.04602v1）
- 核心主张: 滚动窗相关矩阵是"对过去的固定总结"，制度切换时反应迟缓；用 THGNN（每股票 Transformer 时序编码器 + 边感知 GAT）直接预测 10 日后股票-股票相关，以 Fisher-z 残差形式 Δẑ 相对 30 日滚动基线学习，预测矩阵喂 SPONGEsym 聚类构建市场中性篮子交易，相对 20 日持续基线把边级 MAE 从 0.3071 降到 0.2302（Pearson 0.310→0.778）。
- 方法与数据域: 美股 S&P500 日线，CRSP/Compustat WRDS + FRED 宏观；30 日×37 特征窗口，预测视界 10 日；训练 2006-2018，OOS 2019-04~2024-10（60 日防泄漏缓冲）；图结构=每节点 top-50/bottom-50 相关 + 75 随机中间伙伴。频率域=美股日线/10 日视界 — 与本地美股腿域重合，但属 corr/篮子域而非单票 alpha 域。
- 关键定量读数（原文+页码）:
  - Table 1（p14）: 269,121,442 条边，MAE 0.3071→0.2302，RMSE 0.3852→0.2940，Bias −0.0049→−0.0012，Pearson 0.310→0.778（R2 9.6%→60.5%），Spearman 0.314→0.795（对 20 日持续基线）。
  - 交易（Table 3 p16）: ARC 19.20% vs S&P500 14.43%，Sharpe 1.837 vs 0.647，Sortino 2.473，MDD −9.43% vs −33.93%，MLD 82 天 vs 512 天，CR 2.035；Ledoit-Wolf 单侧 bootstrap p=0.0200，95% percentile 区间 [0.054, 2.147]（p16-17）。
  - p15: "under comparable assumptions, we cannot reproduce the large performance gains reported for the ML-filtered strategy"（Korniejczuk & Slepaczuk 2024 框架复现失败，分类器指标对齐但组合层断裂，改以 S&P 买持有为基准）。
  - 解释性（p21）: 37 特征中 "only a handful dominate" — VIX/原油 DCOILWTICO/10 年期 DGS10/美元指数 DTWEXBGS/板块码 gsubind 五者主导；稳定期宏观主导，危机恢复期 gsubind 变关键。
  - Energy 注意力（p18）: COVID 期板块内注意力 0.05→0.35 尖峰；p19 Real Estate 注意力极端波动（Health Care >0.7）"promise and limitation of attention-based interpretability"。
  - 局限（p23）: 无滑点成本、无 LSTM/CNN/GNN 对照、单一超参、单一归因族。
- 可检验命题候选 [假设生成,未判]:
  1. 相关预测残差改善在危机期收敛而非扩大（原文 Fig 2: 压力期模型仍优 ~0.05）→ 本地域: 港美股 corr 门（kl154/155）若换 10 日前瞻相关，门信号在 VIX 高位月的稳定性应提升。
  2. VIX/油/利率/美元四宏变量主导相关预测 → 日线篮子对冲腿的特征集可以从 37 砍到 <10 而损失有限。
  3. 聚类数 k 按 90% 解释方差自适应（p12）于制度切换月会跳变 → 作为 corr 门状态机触发器。
- 与本地判例的接口: 直接对接 corr 门判例（kl154/155）: 本地用历史滚动相关做门，本文提供"前瞻相关替代滚动窗"的完整外部证据链（预测残差减半+交易 Sharpe 1.84），但成本滑点未计（p23 自认）故引用时须打折。与 [168] GAE instability 成对: 168 证明 corr 结构不稳定是可预测信号源（edge-AUROC），230 证明 corr 本身可被预测并改进组合 — 两者合起来支持"相关矩阵当预测变量而非固定摘要"的立项方向。Cartea et al.（2023）SPONGEsym 基线 12.2%/Sharpe 1.1（p2）是 3 日重平衡；230 的 10 日重平衡+ML 过滤后 19.2%/1.84，增量归因未拆分。市场中性篮子与本地出场机体系正交（组合构建层 vs 腿层），无出场机直接接口；ML 过滤器（Brier 加权 soft-voting，top 10% 信号≈45 股，p13）结构上是"信号层置信度门"，与 kl115 出场路径置信度思想同构。复现失败坦白（p15/p23）本身是方法论判例: "backtest fragility" 外部证词，与本地 iter260 cache withdrawn 同精神。

---

## 跨篇合成（供开卡选题）

### A. 制度符号翻转律 — 本次深读集齐五个独立外部例证
同一"信号→结果"关系的符号/强度随状态变量翻转，出现在五个互不相关的论文与四个不同状态变量上:
1. **[164] 市值维度**: 情绪→崩盘风险，小公司 +0.2650**/+0.3883** vs 大公司 −0.2313*（p15 Table 5）。
2. **[192] 时序 regime 维度**: 情绪系数 2021 峰 >0.10 → 2022 谷 −0.11（p16 滚动回归）。
3. **[212] 乐观态维度**: 预测收益价差在低 FEPS（悲观）组 2.12~2.57 vs 高 FEPS 组 0.53~0.83%/月；FEPS 自身价差 +0.93 → −0.66~−0.91 翻转（p43 Table 3）。
4. **[225] 信息精度维度（微观结构版）**: 暗池对价格发现的效应方向由阈值 σ̄ 决定（Prop 4 p33）。
5. （弱第六例 [201]）: SHAP 归因显示动量族正区/量价背离族负区的分区结构（p14）。
→ 开"制度分域/条件门"卡时，外部文献锚已从零散变为成族；本地 CGO 弧线（kl172-179 跨腿跨池翻转）是该族的本地判例。

### B. 机器吸收率定律 — 两个外部锚 + 一个架构级回应
- **[206] p9**: MI-SHAP 背离，原文 "predictive power is largely absorbed by more immediate drivers" — 吸收律的独立逐字表述。
- **[212]**: 瓶颈架构 = 吸收律的工程化利用（与其对抗吸收，不如让中介瓶颈为你工作；λ 曲线 7.63→10.46→9.37 的峰值结构证明中度瓶颈最优）。
- **[186] 反面保留**: 该文未做"原始 δ 广度信号 vs 表示学习因子"消融 — 表示学习论文普遍不做廉价公式基线，引用其增量时须自查（与 [168] 未测图密度廉价代理同型缺口）。

### C. 相关矩阵当预测变量（[168]+[230] 成对）— corr 门升级方向
168: 图同质性 AUROC 是波动先行指标（Linear R2 0.107→0.360, p6）；230: 相关本身可预测（MAE 0.3071→0.2302）且进组合 Sharpe 1.84（无滑点）。两者共同指向本地 corr 门（kl154/155）的两条升级路径: (a) 滚动窗→前瞻预测; (b) 图异质性→出场机收紧条件。共同保留: 两文均未测廉价代理（图密度/平均相关）是否已捕捉大部分信息。

### D. 简单结构稳健性
[198] 最简单 EC 胜过 9 个 ML 模型（但 18 天无显著性，数字不具引用价值）；[212] 线性定价头优于黑箱；[230] 37 特征砍到 5 宏观变量主导；[201] 附录 40 因子公式（纯日线 OHLCV）是唯一零摩擦移植资产。→ 与本地出场机判例（结构简单+条件化优于复杂黑箱）同向。

### E. 复现脆弱性与引用体检清单（四类反面教材）
- 复现失败: [230] p15 无法复现 K&S(2024) ML 过滤增益（"fragility of backtests" p23）。
- 摘要-正文不符: [192] 摘要 "consistently positive" vs 正文 Table 2 无星。
- 全样本内数字: [206] p7/p15/p16 三处自认 in-sample+无成本+无前向测试。
- 样本面不披露: [201] 股票池/样本期缺失；[198] 18 天+归一化前视（式 3/4）。
→ 引用外部数字前的体检: 复现性/摘要-正文一致性/OOS 声明/样本面披露。

### F. 引用权重速查
- **高权可引**: [164]（Annals OR 评审）、[168]（NeurIPS）、[186]（KDD）、[212]（v5 成熟工作稿）。
- **打折引用**: [192]（摘要超卖，3 年样本）、[230]（无滑点、无对照模型）、[206]（全样本内、无成本）、[132]（月频域+排版错）、[201]（硕士级、池未披露）。
- **只引机制不引数字**: [198]（18 天无显著性+前视偏差）、[225]（纯理论）、[153]（tick 域不可比）。
