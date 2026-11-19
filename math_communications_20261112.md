# 通信工程 × A股日线：从匹配滤波到噪声地板 (2026-11-12)

> ⚠️ 判例修正横幅 (2026-11-13, papers 笔记层重验 — papers/_notes_verification_map.md): M5 混叠实验判据 “周频 70% vs 日频 51%” 系 attack_plan L143 已作废频谱 (2026-10-25, 定版口径物理引擎 44-47); M5 立项前提需复核。


> 任务背景：用户指令"自由探索，通信/半导体/工程控制的数学公式都可以拿来研究"。
> 本篇把通信工程的一组成熟数学结构映射到我们的挖掘循环上。
> 内部约束：A 股全市场日线 OHLCV+vwap（无 tick），池 = csi800 PIT，不可做空，
> 往返成本 ~31bp，T+1，四条准正交因子轴 park/mom/nh250⊥/vp（两两 |corr|<0.26），
> 入场信息边际≈0、alpha 住在出场端（恐慌反弹窗 day2-5）。
> 已覆盖理论（本篇不重复，只引用）：Shannon 式因子信息量、Hawkes、Lo-MacKinlay 分解、
> 弛豫时间、遍历性、Kyle。

## 0. 诚实声明：类比的三层等级

| 等级 | 含义 | 本篇条目 |
|---|---|---|
| **Tier A（定理搬运）** | 数学命题在市场模型假设下严格成立 | Cover 侧信息增长率定理；SPRT/Wald 最优性；删除信道 2× 纠错；Welch 界；Kalman vs 固定窗的 13% 上界；无交易带 c^(1/3) 标度 |
| **Tier B（结构对应）** | 公式结构可精确计算，但最优性依赖的假设（已知模板/高斯噪声/平稳）在市场上只是近似 | 匹配滤波出场、混叠相位测试、Allan 方差选窗、MI 上界审计的 BR 组装 |
| **Tier C（修辞类比）** | 只提供研究品味，不提供可计算量 | Turbo/LDPC 三教益、频谱生态位叙事、kTB 比喻 |

每个映射单独标注。**总原则：Tier C 只用来提品味，Tier B 必须落到可算的数，Tier A 用来关研究门。**

## 1. 引用文献表（检索于 2026-11-12）

| # | 文献 | 用途 | URL |
|---|---|---|---|
| 1 | Kang (2025) *Optimal Signal Extraction from Order Flow: A Matched Filter Perspective on Normalization and Market Microstructure* | 匹配滤波在金融的直接应用；归一化=匹配滤波 | https://arxiv.org/html/2512.18648v1 |
| 2 | Witte (2015) *Volatility Harvesting: Extracting Return from Randomness* | Shannon's Demon / 再平衡收益与成本 | https://arxiv.org/abs/1508.05241 |
| 3 | Garivaltis (2018) *Exact Replication of the Best Rebalancing Rule in Hindsight* | 最优再平衡规则复制 | https://arxiv.org/html/1810.02485v2 |
| 4 | Hsieh & Wong (2023) *Frequency-Based Kelly Optimal Portfolio with Transaction Costs*, IEEE CSL 7:3489 | 再平衡频率×成本的最优性条件 | https://arxiv.org/abs/2301.02754 |
| 5 | Chen, Gao, He, Jiang, Xiong — *Daily Price Limits and the Magnet Effect*（VoxChina 专栏） | 涨跌停磁吸效应（删除信道的市场侧文献） | https://voxchina.org/show-3-49.html |
| 6 | *The cooling-off effect of price limits in the Chinese stock market* | 涨跌停冷却效应 | https://ui.adsabs.harvard.edu/abs/arXiv:1803.09422 |
| 7 | Baek et al. (2005) *Transfer Entropy Analysis of the Stock Market* | 转移熵监测轴间信息流 | https://arxiv.org/pdf/physics/0509014 |
| 8 | Maasoumi & Racine (2002) *Entropy and Predictability of Stock Returns* | 熵可预测性度量 | https://www.researchgate.net/publication/222710004_Entropy_and_Predictability_of_Stock_Returns |
| 9 | Liu et al. (2020) *The Flow of Information in Trading: An Entropy Approach to Market Regimes* | 熵方法识别 regime | https://pmc.ncbi.nlm.nih.gov/articles/PMC7597144 |
| 10 | Chaudhuri (2019) *Dynamic Alpha: A Spectral Decomposition of Investment* (SSRN 3184092/MIT) | 投资收益的谱分解 | https://dspace.mit.edu/bitstream/handle/1721.1/135075/SSRN-id3184092.pdf |
| 11 | Bezuidenhout (2021) *Spectral analysis and the death of value investing*, Cogent E&F | 价值/动量的谱分析 | https://www.tandfonline.com/doi/full/10.1080/23322039.2021.1988380 |
| 12 | Haeri (2022) *Optimal Moving Average Estimation of Noisy Random Walk* | Allan 方差用于金融随机游走 | https://par.nsf.gov/servlets/purl/10379088 |
| 13 | NIST — *Analysis of powers-of-two calculations of the Allan variance* | Allan 方差计算规范 | https://tf.nist.gov/general/pdf/3052.pdf |
| 14 | Wikipedia: Allan variance | Allan 方差综述 | https://en.wikipedia.org/wiki/Allan_variance |
| 15 | Wikipedia: Matched filter + UT Austin Bevans 讲义 | 匹配滤波推导 | https://en.wikipedia.org/wiki/Matched_filter ; https://users.ece.utexas.edu/~bevans/courses/realtime/lectures/14_Matched_Filtering/MFslidesJacobellisF21.pdf |
| 16 | Stanford EE359 Lecture 18: *Spread Spectrum, Multiuser Systems* | CDMA/多用户检测公式 | https://web.stanford.edu/class/ee359/pdfs/lecture18_handout.pdf |
| 17 | Andrews (UT Austin) *CDMA for Wireless Communications* 讲义 | 处理增益/干扰消除 | https://users.ece.utexas.edu/~bevans/courses/realtime/lectures/21_Spread_Spectrum/CDMA.pdf |
| 18 | Wikipedia: Shannon–Hartley theorem; MIT News *Explained: The Shannon limit* | 容量公式 | https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem ; https://news.mit.edu/2010/explained-shannon-0115 |
| 19 | *The Fundamental Law of Active Management: Time Series* (NYU, 纪念 Avellaneda) | IR=IC·√BR 的时间序列修正 | https://math.nyu.edu/inmemoriam/avellaneda/FundamentalLawFT.pdf |
| 20 | Clarke, de Silva & Thorley (2002) 相关材料（SSRN 1528484） | TC（转移系数）扩展 | https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID1528484_code551960.pdf?abstractid=1528484&type=2 |
| 21 | Kakushadze (2014) *Can Turnover Go to Zero?* | 换手率与成本结构 | https://arxiv.org/pdf/1406.0044 |
| 22 | Wikipedia: Adaptive equalizer; MathWorks LMS vs RLS | 自适应均衡器 | https://en.wikipedia.org/wiki/Adaptive_equalizer ; https://www.mathworks.com/help/dsp/ug/compare-rls-and-lms-adaptive-filter-algorithms.html |
| 23 | DSP.SE: *Relative performance of RLS and LMS filters* | LMS 更适合时变跟踪的结论 | https://dsp.stackexchange.com/questions/8184/relative-performance-of-rls-and-lms-filters |
| 24 | *Information-Theoretic Approach to Financial Market Modelling* (arXiv 2602.14575) | 信息论市场建模近期工作 | https://arxiv.org/pdf/2602.14575 |
| 25 | *Information-Theoretic Model for the Growth Optimal Portfolio of Stocks* (arXiv 2611.07457) | 增长最优组合的信息论模型 | https://arxiv.org/html/2611.07457v1 |
| 26 | Garivaltis (2019) *Super-replication of the best pairs trade in hindsight* | Shannon's Demon 推广 | https://www.tandfonline.com/doi/full/10.1080/23322039.2019.1568657 |
| 27 | 经典无 URL：Wald (1945) SPRT；Cover & Thomas《信息论基础》ch.6（赌博与侧信息）、ch.16（通用组合）；Kelly (1956)；Verdú (1998) 多用户检测综述；Whalley & Wilmott (1997) 对冲带；Davis & Norman (1990)；Gârleanu & Pedersen (2013) *Dynamic Trading with Predictable Returns and Transaction Costs*, JF | 定理出处 | — |

## 2. 映射总表

| # | 通信概念 | 精确公式 | 市场对象 | 等级 | 落地物 |
|---|---|---|---|---|---|
| M1 | 匹配滤波+SPRT | Λ_t=Σ[y_h m(h)/σ²−m(h)²/2σ²]，Λ_t≤lnB 出场 | 恐慌反弹 day2-5 模板的出场规则 | A(核心)+B | 时间斜坡出场算子 |
| M2 | CDMA/多用户检测/Welch 界 | (SᵀS)⁻¹、1/(1−ρ²)、ρ̄²≥(K−N)/(N(K−1)) | 四轴=准正交码；第 5/6 轴准入 | A | 轴准入检验+R⁻¹ 组合 |
| M3 | 侧信息容量（Cover/Kelly） | Δg=I(F;R) nats；I=−½ln(1−IC²)；Δg≈BR·IC²/2 | 四轴信息总量的增长率上界 | A | MI 上界审计 |
| M4 | 删除信道/RS 突发纠错 | RS(n,k) 纠 n−k 个删除、(n−k)/2 个错误；C=1−p_e | 一字板/停牌=删除符号 | A+B | erasure 标记+解冻日因子 |
| M5 | 采样定理/混叠 | f_a=\|f−f_s·round(f/f_s)\|；Nyquist=f_s/2 | 周频 70% vs 日频 51%；GRU 窗口 | A(数学)+B(假设) | 相位敏感性测试 |
| M6 | 信道估计/自适应均衡 | P*=(Q+√(Q²+4QR))/2；最优固定窗 MSE≈2√(RQ/3) | regime 自适应因子权重 | A | 否定定理+Q/R 监控 |
| M7 | 噪声地板/逼近香农限 | v_n=√(4kTRB)；带半宽∝c^(1/3) | 31bp 成本 floor 上的收割结构 | C(教益)+A(带) | 无交易带+软权重 |
| M8 | Allan 方差（时钟计量学） | σ²(τ)=½E[(ȳ_{k+1}−ȳ_k)²]，τ*=argmin | 因子估计窗的噪声-漂移权衡 | B | lookback 选择算子 |

## 3. M1 匹配滤波器 → 出场规则的形式化

### 3.1 公式

已知模板 s、加性噪声 n～N(0,σ²) 时，最大化检测 SNR 的线性滤波器是 h=s（时间反演对实信号相同），
输出 SNR²_max = E/σ²，E=Σ s(h)² 为**模板能量**。有色噪声下最优滤波器是 h=R_n⁻¹s（先白化再匹配）。
Kang (2025) 在订单流上证明：**归一化方式本身就是匹配滤波**——除数必须匹配信号生成过程的标度行为
（市值归一匹配容量约束机构、成交额归一匹配 VWAP 执行者），错配使可提取信号折半 [#1]。

**出场规则的正确形式化不是单次检测，是序贯检测（SPRT）**。事件后逐日超额收益
`y_h = m(h) + ε_h`，ε～N(0,σ_h²)（用 Parkinson/GK 日内波动估计做逐日白化，这正是 R_n⁻¹ 的对角近似：
σ_P²=(ln(H/L))²/(4ln2)，σ²_GK=½(ln(H/L))²−(2ln2−1)(ln(C/O))²——OHLCV 原生白化器）。
模板存活(H1) vs 模板已死(H0) 的对数似然比累计：

```
Λ_t = Σ_{h=1..t} [ y_h·m(h)/σ_h² − m(h)²/(2σ_h²) ]
出场（判死）当 Λ_t ≤ ln B = ln(β/(1−α))
```

常数模板 m 的特例（σ 恒定）：

```
出场当 Y_t < t·m/2 − c,   c = (σ²/m)·ln((1−α)/β),  Y_t=Σy_h
```

**结构性新结论：出场边界带时间斜坡项 t·m/2。** 价格 trail stop 只有价格棘轮，没有时间棘轮；
SPRT 推出的是"持有越久，累计收益必须跟上的模板半能量线越高"。这把审计发现的
"止损切输家+partial 激活 trail+让右尾跑"从经验结构变成定理结构：右尾交易
Y_t≫t·m/2 永不触线，落后于模板线一半的持仓自动离场。

与弛豫时间定律的关系：**匹配滤波不替代弛豫定律，它消费弛豫定律**。m(h)=m₀e^{−h/τ}
代入后滤波器权重 w_h∝e^{−h/τ}/σ_h²——模板形状参数 τ 正是弛豫时间；最优持有窗由
Wald 期望停时 E[T]≈(α lnA+β lnB)/E[ΔΛ] 自动给出，而不是外生网格。

### 3.2 付钱方故事

恐慌日的被迫卖方（强平、赎回、割肉）把反弹溢价付给流动性供给者。模板 m(h) 就是
"卖压能量释放曲线"；Λ_t 跌破阈值 = 卖方能量耗尽/接力买盘未现的信号物理判据。
出场=付钱方交接完成的检测，不是价格美学。

### 3.3 日线 OHLCV 可测变换

- 事件触发：跌 d% 以上 + 放量 z>1 + 收盘近最低（CLV 弱），沿用现有恐慌定义。
- 模板估计：只用 t−1 之前的事件窗（PIT），滚动 500 事件估计 m̂(h), h=1..10。
- 白化：σ̂_h 用 GK 日波动（含 60 日平滑）。
- 算子输出：Λ_t 逐日更新，替换/对比现有 trail。
- 模板能量监控：E=Σm̂(h)²。

### 3.4 预期方向与 mom 换皮风险

预期：SPRT 时间斜坡让"慢输家早出、右尾晚出"，期望持有效率（单位持有期的模板能量捕获）
高于固定 trail；对右尾的保护有机制（斜坡由 m/2 决定，非外生参数）。
换皮风险：恐慌事件集是 mom 底部分位的子集，条件化后与无条件 mom20 的截面相关**必须实测**；
按纪律报告与 mom 轴的 MI/corr，corr>0.5 即换皮。区分点：模板的可预测内容在**事件后路径形状**，
不在事件日截面排序——评估语义要改（见 §11）。

### 3.5 等级

SPRT/Wald 最优性 = Tier A（在"简单假设+已知模板+高斯白化噪声"下）。
市场适用 = Tier B：模板是估计的（plug-in），噪声厚尾（池化 kurt 4.7，事件窗内 CLT 部分救济），
且 SPRT 优化的是期望停时而非对数增长/右尾目标——这正是"任何更快结束痛苦的规则先杀右尾"
的数学解释：**SPRT 的目标函数就不是我们的目标函数**；时间斜坡项是对此的修正方向。

## 4. M2 CDMA/扩频/多用户检测 → 因子轴代数

### 4.1 公式

接收信号 `r = Σ_k √E_k b_k s_k + n`，签名序列 {s_k}，互相关 ρ_jk=⟨s_j,s_k⟩ [#16][#17]。

| 检测器 | 公式 | 金融对应 | 已知性质 |
|---|---|---|---|
| 单用户匹配滤波 | b̂_k=sign(s_kᵀr) | 单因子单变量 IC | 近远问题：强用户淹没弱用户 |
| 解相关器 | (SᵀS)⁻¹Sᵀr | 全因子联合回归/GLS | 噪声增强 (SᵀS)⁻¹_kk；近远抵抗 |
| SIC | 检最强→减去→循环 | 逐步正交化回归 | 误差传播：强者估计错污染全部后续 |
| MMSE 多用户 | (A⁻¹R+σ²I)⁻¹y | 有限样本最优线性组合 | 有限 SNR 下折中 |

**噪声增强定量**：等相关的 K=4 因子相关阵，(R⁻¹)_kk=(1+2ρ)/((1−ρ)(1+3ρ))。
ρ=0.26（我们的上界）：方差 +15.4%，t 值 −6.9%；ρ=0.15：+5.5%。
**结论：四轴已近似正交，分离代数（Gram-Schmidt/SIC vs 联合回归 vs MMSE）之间的差距 <15%——
当前瓶颈不在分离，在轴数与利用率。** Walsh-Hadamard 式"构造正交基"只做对角化，
不张出新空间（线性 span 相同），对本问题无增量。

**Welch 界（Tier A）给出"多少轴才够"的定量答案**：K 个单位向量在 N 维空间，
平均平方互相关满足

```
ρ̄² ≥ (K−N) / (N(K−1))
```

观测：K=4 轴、ρ̄≈0.15（ρ̄²≈0.0225）⇒ 反解 N_eff≥3.76（不束缚）。
预测：**第 6 轴若要平均相关保持 <0.2，要求环境有效维数 N_eff>5**
（K=6 时下界 (6−N)/(5N)<0.04 ⇒ N>5）。
所以"够不够"取决于截面 alpha 空间的有效秩 N_eff（用参与比 N_eff=(Σλ)²/Σλ²
从因子分数 Gram 矩阵估计）。N_eff 饱和后，新轴被 Welch 界强制与旧轴相关，
解相关器的 1/(1−ρ²) 噪声增强吃掉边际信息。

**处理增益=Grinold 定律**：扩频增益 SNR×N_chips ↔ 截面分散 IC·√N。
IR=IC·√BR [#19][#20] 就是 CDMA 解扩增益的资金版；"bets 必须独立"的告诫
= "码必须正交"。TC（转移系数，长_only 约束下 0.3-0.8）= 接收机实现损耗；
无做空 = 只能检测用户存在不能检测符号（功率检测 vs 相干检测），对称信号下
可用信息近似减半（见 M3）。

### 4.2 付钱方故事

每条轴的付钱人是不同的具名受限人群；CDMA 的意义是**分离付钱人**，不是创造信息。
近远问题的市场版：市场 β/规模是强用户（R²=0.26 的那部分），alpha 轴是弱用户——
所以所有 IC 都在残差截面里算（我们已这样做，理论在此补上理由）。

### 4.3 落地

① 不产新因子，产**轴准入检验**：新轴报告与 4 轴的 max|corr| 与 ΔN_eff。
② 组合权重实验：w=R⁻¹b vs 等权（理论差异 <15% t 值；若实验差异 >30% 说明有别的机制，值得查）。
③ 监控器：季度计算 N_eff 与 Welch gap（观测 ρ̄²−下界）；gap→0 = 轴空间饱和，停止挖轴。
④ 换皮探测器：装在 mom 上的候选在 R⁻¹ 下自动被压权重——正交化本身是换皮探测器。

## 5. M3 香农容量 / Cover 侧信息定理 → IC 的增长率上界审计

### 5.1 公式（Tier A）

Shannon-Hartley：C=½log₂(1+S/N) [#18]。对我们更重要的是**Cover 侧信息定理**
（赌博/对数最优框架）：侧信息 Y 每比特使对数最优财富翻倍，

```
E[log W*(侧信息)] − E[log W*] = I(Y;X)   （nats，赛马市场取等，一般为上界）
```

截面 IC 的信息量：二元正态 I=−½ln(1−ρ²)≈ρ²/2 nats（小 ρ）。
IC=0.035 → 0.000613 nats/name-day。按日独立截面组装（BR=有效独立 bet 数/年）：

```
年信息上界（每轴） ≈ BR·IC²/2 nats
四轴正交近似     ≈ 4·BR·IC²/2
```

数字：IC 0.035、BR=250 → 每轴 0.153 nats/yr ≈ **16.5%/年**；四轴正交合计 0.613 nats/yr
≈ **85%/年**（算术口径；log 口径 0.613）。
对照：Ap 纸面 65%/年 = 0.50 nats → 对四轴毛上界的利用率 ≈ 82%——同量级偏高，
但须按下面修正（长_only 折半→33-42%、成本 floor 再扣、BR 日间自相关再打折）后才能下结论：
修正后上界大概率落在 25-45%，此时 Ap 纸面 65% 与 C3 11% 分居上界两侧，
利用率排序即成为"挖轴 vs 提利用率"的判决数。
可用率修正：长_only 对称信号截走约一半（≈33%）；成本 floor 再扣（见 M7）；
BR 的日间自相关衰减（因子收益自相关）再打折扣。近期信息论建模工作见 [#24][#25]。

### 5.2 用途：一次审计回答"挖轴还是提利用率"

- 实测四轴精确 MI（秩相关→copula 变换或 KSG 估计器，与我们已排队的 MI 排重共用管线）。
- 上界 = Σ I_axis − 冗余项（用轴间 MI 扣除）。
- **利用率 = 实现的 log 增长（Ap/C3/Q×2 物理腿） ÷ 上界**。
- 利用率 <50% → 别挖新轴，改利用率（出场端、执行、无交易带）；
  利用率 >80% 且上界低 → Welch/M3 都说明该换问题（如换市场/换带）。

这是把 Clarke-de Silva-Thorley 的 TC 概念信息论化：TC 的一个下界来源就是
长_only 截断 + 成本 floor 的信息租金。

### 5.3 付钱方

信息租金总量=所有具名付钱人愿付的总租。上界审计=给"付钱人群总预算"画天花板。
与 Kyle 的关系：MI 上界量化 Kyle 信息份额的可收割部分——不重复 Kyle，给 Kyle 的
alpha 加了以 nats 计的总量。

### 5.4 换皮风险与等级

MI 对单调变换敏感度低于线性相关，天然抗"单调换皮"（mom 换皮构造 corr 0.67+
但若秩结构相同其边际 MI→0）。Tier：Cover 定理 A；BR 组装与长_only 折半 = B/C。

## 6. M4 编码理论/删除信道 → 一字板与停牌

### 6.1 公式（Tier A）

RS(n,k) 码：纠 **n−k 个删除**（位置已知的坏符号）但只纠 **(n−k)/2 个错误**（位置未知）。
**已知坏位置的处理能力是未知坏位置的两倍。** 删除信道容量 C=1−p_e（对比 BSC 的 1−H(p)）。

一字板（O≈H≈L≈C 触及涨跌停、成交量塌缩）和停牌在日线 OHLCV 上**可观测**——
它们是删除（erasure），不是错误（error）。正确处理：

1. **因子估计侧**：当日截面排名剔除冻结名字（冻结价算出的 mom/vp 是冻结伪影，
   当噪声混入等于放弃 2× 纠错红利）。
2. **回测语义侧（重大）**：删除信道容量 C=1−p_e 说，**回测不得假设被删符号上传送了信息**。
   kl025 已证：Q×2 纸面 232%→物理 46% 的 **97% 缺口来自停牌权重每日再分配给活票**
   ——纸面引擎正是在被删除的符号上计入了收益。这是"把删除信道当无噪信道"的定量代价
   （−80.87 log-pp）。通信视角把 kl025 的实证发现升级为信道语义定理。
3. **突发终止符号**：解冻日承载突发期间累积的信息（RS interleaving 逻辑：burst 后第一个
   可用符号含 burst 信息）→ 解冻日 gap 行为本身是候选因子。

### 6.2 付钱方与文献

一字板排队者付出队列时间+逆向选择（磁吸效应 [#5]、冷却效应 [#6]）：我们不排队就不付这笔。

### 6.3 日线变换/方向/换皮

- erasure 标记：E=1{H==L} ∪ {|ret|≥0.095 且 volume<q10} ∪ 停牌。
- 解冻日因子：unfreeze_gap = open_t/close_{t−1}−1；首日衰减比 = (close_t−open_t)/gap。
- 预期方向：erasure 剔除后 mom/vp 的 IC 变化 = 冻结伪影量（预期 mom 的反转 IC 变浅，
  因为涨停连板贡献了假动量）。
- **换皮风险高**：涨停连板延续因子=mom 换皮（预期 corr>0.6）；跌停/恐慌侧解冻因子
  与 mom 相关较低 → 全部按涨跌方向分侧报告。

## 7. M5 采样定理/混叠 → 周频 70% vs 日频 51%，以及频谱生态位

### 7.1 公式（Tier A 数学，Tier B 假设）

混叠：`f_a = |f − f_s·round(f/f_s)|`，Nyquist=f_s/2。周频采样 f_s=0.2/day →
只能看周期 >10 天的分量。恐慌反弹 day2-5 结构周期 2-5 天（f=0.2-0.5/day），
**整体在周频 Nyquist 之上**，混叠去向：f=0.25→f_a=0.05（伪装成 20 日慢波）；
f=0.2→f_a=0（伪装成 DC 漂移，符号依赖采样相位）；f=1/3→f_a≈0.067（15 日波）。

**判别实验（零成本，最重要）**：真低频信号对采样相位不变；混叠伪影对相位敏感。
把周频再归一的采样网格平移 1/2/3/4 天各重算一次（5 个相位）。若 5 个相位的年化
散度与 70%−51%=19pp 缺口同量级 → 周频优势部分是采样伪影；若相位不变 → 真低频
信息（成本故事成立）。这一实验决定跨市场腿的再平衡频率设计，且直接可用现有面板。

### 7.2 频谱生态位（Tier C 叙事，但有数字锚）

参与者采样率：HFT（tick，f_s≫1/day）；A 股日线玩家（f_s=1/day，T+1 把最短周期推到 ~2 天）；
机构月度 mandate（f_s=1/22，只看周期 >44 天）。**我们的可耕耘带=周期 2~44 天**。
day2-5 窗在这个带的上缘：HFT 嫌慢不做，mandate 够不着 → 解释其持续可收割性；
day0-1 入场带被更高采样率的玩家占用 → 解释"入场信息边际≈0"对我们成立（不是无信息，
是在我们的采样能力之上/已被占用）。GRU 佐证：60 日窗分辨不了 f<1/60（周期>60 天）的
分量，唯一住在那里的轴是 nh250 → **GRU 增量只来自 nh250 是频谱分辨率结论**，
与内部实证一致（非线性核对，非证明）。

### 7.3 合法用法与数据挖掘警戒

- IC(h)（IC 对持有期 h 的曲线）≈ 因子-收益互谱的 Fourier 版本（Chaudhuri 谱分解 [#10]、
  Bezuidenhout [#11]）：IC(h) 在 h=2-5 的峰=谱线，**描述性用法合法**；
  带通版"新因子"（动量的 5-20 日带通分量）默认按数据挖掘处理，须预注册+样本外。
- 监控器：IC(h) 峰位漂移=信号频移=竞争者进入生态位（类比载波频移告警）。

## 8. M6 信道估计/自适应均衡器 → regime 自适应权重的否定定理

### 8.1 公式（Tier A）

标量模型：真实权重随机游走 x_t=x_{t−1}+w_t（漂移方差 Q），观测 y_t=x_t+v_t（噪声方差 R）。
最优自适应（Kalman 稳态）：

```
P* = (Q + √(Q²+4QR))/2  ≈ √(QR)   (Q≪R)
```

最优**固定**尾随窗 N_w 估计的 MSE ≈ R/N_w + Q·N_w/3，最优 N_w*=√(3R/Q)，
最优固定窗 MSE ≈ 2√(RQ/3) = **1.155·√(QR)**。

**定理级结论：完美自适应均衡最多比最优调谐的固定窗省 13% MSE（≈7% t 值）；
且 Q→0 时增益→0。** LMS 失调 M=μ·tr(R_x)/2、RLS 等效窗 1/(1−λ) 都只能更差
[#22][#23]（LMS 在时变跟踪中反而常胜 RLS，因为 RLS 的快速收敛在非平稳下变成快速过拟合）。

### 8.2 用法：把 8-agent 审计升级为公式闸门

代入我们的量纲：轴 IC 的日估计噪声 R≈(1/√N_eff)²（N_eff≈200-800 截面有效宽），
轴真实权重的年漂移 Q 从滚动 IC 稳定性估（审计显示轴 IC 跨年稳定 → Q 极小）。
预期输出：自适应增益上界 <2%。**"regime 自适应因子权重"在数学上接近不可行，
除非 Q 增大一个数量级**——这是停止该方向研究的定理依据，与 kl027 的
"均线择时被打脸"、Ap 审计 regime 门全部失败的实证互证。
真正合法的自适应对象是**检测器结构**（M1 的模板随 PIT 滚动更新），不是**权重**。

### 8.3 监控器

Q 本身做监控：轴 IC 漂移方差突然增大 = 信道突变（regime 真变了），
此时才临时启用自适应；Q 平稳时锁死固定权重。这是"何时值得自适应"的可计算判据。

## 9. M7 噪声地板/逼近香农限 → 成本 floor 上的收割结构

### 9.1 映射

Johnson-Nyquist：v_n=√(4kTRB)。噪声功率∝带宽 B ↔ 成本拖累∝换手 Λ（年 31bp×Λ）；
温度 T↔波动率；R↔冲击系数。"你不能打败 kTB"="你不能打败 c·Λ"。
工程答案：**低于瞬时 floor 的信号靠积分检测**——信号能量随 T 线性积累、噪声按 √T 平均掉，
SNR=2E/N₀。资金版：日边际低于成本 floor 的 edge，靠**持有更久**（窄带积分）收割，
不靠提高采样率。day2-5 持有窗=窄带积分器；"两市场收割都死在成本 floor"的死法
=宽带策略（多票×高频再平衡）支付 kT·B 全带噪声税。

### 9.2 逼近 floor 的组合构造（Turbo/LDPC 的三教益，Tier C 品味 + Tier A 定理）

逼近香农限靠：软判决、长块、迭代软信息传递。对应：

1. **软判决**：硬 top-decile → 连续 rank-tilt 权重（Q×2 rank tilt 已验证方向）。
2. **长块**：持有窗=积分时间（M1 的模板窗而非固定 5 日）。
3. **迭代**：轴间一轮 backfitting（每轴用其他轴的软残差更新——turbo 迭代的统计版）。

严格的资金版是**无交易带**（Davis-Norman / Whalley-Wilmott / Gârleanu-Pedersen [#27]）：
带半宽 ∝ c^(1/3)（次线性：成本翻倍带宽只增 26%）；带内不动=对低于 floor 的
"比特"不解码。Hsieh & Wong [#4] 给出频率×成本的最优性条件（再平衡频率折衷的凹规划形式）。
Kakushadze [#21] 给换手下界。

### 9.3 落地

无交易带规则叠加现有腿零成本重放：|Δw|<x* 不交易，x*∝c^(1/3) 标定；
监控器=年化成本/信息比 c·Λ ÷ MI 上界（M3），→1 即逼近 floor，停止加频。

## 10. M8 Allan 方差（新增进口：时钟计量学/半导体表征）

### 10.1 公式（Tier B，Haeri 2022 已用于金融随机游走 [#12][#13][#14]）

```
σ²_A(τ) = ½ E[(ȳ_{k+1} − ȳ_k)²]，ȳ_k 为 τ 长度块平均
```

白噪声段斜率 −1（σ∝τ^{−1/2}）、闪烁噪声平台、随机游走段 +½。
**曲线最低点 τ* = 白噪声与漂移的交点 = 最优平均窗**。

### 10.2 与弛豫时间定律的分工（不重复）

弛豫时间=**支付端**衰减律（alpha 活多久）；Allan τ*=**估计端**噪声-漂移权衡
（信号估计该平均多久）。两者失配是诊断：
τ*_Allan < τ_relax → 估计受限（加长 lookback 有增益）；
τ*_Allan > τ_relax → 支付受限（lookback 再长也没用，出场端才是瓶颈——与我们
"alpha 住在出场端"的判断同向）。

### 10.3 落地

① 对四轴的日 IC 序列/因子分数序列算 σ_A(τ)，τ∈{1,2,4,...,128}（NIST 2 的幂规范 [#13]），
τ* 与现行 lookback 差 >2× 则改。② 监控器：Allan 曲线形态迁移（白噪声→随机游走）
= 信号消亡的物理签名，比 IC 单点衰减早报警。

## 11. "落到哪"清单（按落地纪律 ①②③④）

| 项 | ① 日线 OHLCV 候选因子/算子 | ② 评估语义修正 | ③ 零成本结构实验 | ④ 监控器 |
|---|---|---|---|---|
| **A. SPRT 时间斜坡出场**（M1） | Λ_t=Σ[y_h·m̂(h)/σ̂_h²−m̂(h)²/2σ̂_h²]，GK 白化，PIT 模板 | 事件窗路径 vs 模板能量相关的条件评估，替代无条件截面 IC | 既有事件面板重放：SPRT vs 固定 trail vs 固定窗；预注册"top10% 交易贡献保持率≥trail 的 90%"（右尾不被杀） | 模板能量 E 月度重估，E 跌 50% 告警 |
| **B. MI 上界审计**（M3） | 无新因子；I_axis=−½ln(1−IC²)·BR 组装 + 轴间 MI 扣冗余 | 利用率=实现 log 增长/上界，作为 TC 的信息论版 KPI | 四轴精确 MI（KSG/copula）一次计算；对比 Ap/C3/Q×2 物理腿 | 年度重算；I 增量<成本 floor 对应租金 → 停挖轴 |
| **C. 混叠相位测试**（M5） | 无新因子（诊断） | 周频读数必须附 5 相位散度 | 周频采样 5 个相位偏移重算年化；散度 vs 19pp 缺口 | IC(h) 峰位漂移监控 |
| **D. 删除信道 erasure 处理**（M4） | E 标记（一字板/停牌）；解冻日 gap+衰减比分侧因子 | 截面排名剔除 erasure 名字；纸面回测容量按 C=1−p_e 折减 | 现有回测 erasure 剔除前后对比（mom 伪影量）；解冻因子事件窗评估 | erasure 频率+连发长度月报 |
| **E. 轴准入 Welch 检验**（M2） | R⁻¹ 组合权重；新轴准入需 ΔN_eff≥1 | max|corr|+参与比 N_eff 双报告 | R⁻¹ vs 等权组合重放（差异>30% 即异常） | 季度 N_eff 与 Welch gap |
| **F. 自适应否定定理**（M6） | 无（关门的公式） | — | 估 Q,R 代入 P*/固定窗公式，输出自适应增益上界% | Q（轴漂移方差）突变告警 |
| **G. 无交易带**（M7） | 带宽 x*∝c^(1/3) 的 |Δw|<x* 不动规则 | 叠加现有腿重放 | c·Λ/上界 →1 告警 |
| **H. Allan 选窗**（M8） | 每因子 σ_A(τ) 曲线，τ* 定 lookback | τ*_Allan vs τ_relax 失配诊断 | 四轴算一遍，差 >2× 则改 lookback | Allan 形态迁移告警 |

## 12. 立项排序（按 期望值/成本）

1. **C（混叠相位测试）**：一天零成本实验，判定周频 70% 是真低频信息还是采样伪影——
   直接决定跨市场腿再平衡频率设计与 19pp 缺口的可信度。
2. **A（SPRT 时间斜坡出场）**：攻"alpha 住在出场端"的主战场，把审计发现的出场结构
   从经验升级为最优序贯检验，时间棘轮是现有 trail 没有的项，且有右尾保护机制。
3. **B（MI 上界审计）**：一次计算回答"继续挖轴还是提利用率"，且与已排队的 MI 排重
   共用管线；顺带把 long-only 的 TC 信息论化。
4. **D（erasure 处理）**：kl025 的 97% 缺口就是删除语义被违反的代价；删除 2× 纠错红利
   落在因子卫生与回测语义两处，解冻日因子是附带的新候选（分侧防 mom 换皮）。
5. **E+F（Welch 轴准入+自适应否定定理）**：两道"停止亏钱研究"的公式闸门，
   防止轴动物园与 regime 自适应两个方向的沉没成本继续累积。

## 13. 与已覆盖理论的关系（不重复声明）

- Shannon 式因子信息量（已覆盖）→ 本篇 M3 是其**总量审计与增长率翻译**（Cover 定理），
  不是重复度量。
- 弛豫时间（已覆盖）→ M1 消费它作为模板形状；M8 是估计端对偶。
- Kyle（已覆盖）→ M3 给 Kyle 信息份额加 nats 计的总量；M2 近远问题给 Kyle 的
  "为什么在残差里找 alpha"补通信理由。
- Hawkes（已覆盖）→ 生成事件强度；M1 在给定事件上做出场——互补层级。
- Lo-MacKinlay（已覆盖）→ 是谱分析的 ω→0 极限；M5 是全带版本+混叠工程。

## 14. 产物

- 本文：`papers/math_communications_20261112.md`
- 建议后续脚本（待立项后）：`kl031_alias_phase.py`（C）、`kl032_sprt_exit.py`（A）、
  `kl033_mi_ceiling.py`（B）、`kl034_erasure_semantics.py`（D）、
  `kl035_welch_admission.py`（E）。
