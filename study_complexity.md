# Arora & Barak《Computational Complexity: A Modern Approach》全书笔记 (读序 18 master)

- 作者: Sanjeev Arora, Boaz Barak; Cambridge University Press 2009 正式版的前身 — 本线读 2007-01-08 21:59 web draft
- 版权态: 作者官方免费 draft (complexitybook.com 502 → theory.cs.princeton.edu/complexity/book.pdf 作者正主镜像, 489p/4,572,986B); 书本身不复传云只本地读
- 读毕: 2026-11-21; 23,827 行逐块读毕 (43 块: 42×560 行 + 末块 307 行), 卷首+ch1-22 全覆盖 + 附录 A + §20.A 连分数 + Bibliography A-Z 全扫 (draft 无索引)
- 纪律: 本文件为读序 18 唯一 master (每域单一 master, 无 note/deep 拆分); study 笔记 ≠ kl 卡

---

## §〇 推导层 (本线自证的核心链)

读法: 全书跟定义-定理-证明走, 以下 17 条是本线合上书后能独立重推的主干 (其余为跟读理解)。

1. **可计算性-对角线链**: TM 单带充分性 (缓冲区模拟 O(T log T)); 普适 TM U(⟨M⟩,x)=M(x); 停机问题对角线 (跑则停/停则跑); 递归可枚举≠递归 (接受≠判定); 映射归约的传递偏序; "机器即字符串"=一切对角化的根。
2. **NP-完备链**: 验证器定义 (poly 可验证据 ⇔ NP); Cook-Levin 表算 (快照×时间步, **2×3 窗口局部性**, 机械可核); Karp 21 问题归约网; **Ladner** (P≠NP ⟹ NP 中间态, SATH 对角构造); padding; 自身归约 (搜索⇔判定); parsimonious 保数目归约。
3. **层次-相对化链**: 时间/空间层次定理 (对角线+计时器/空间重用); **BGS oracle** (∃A: P^A=NP^A ∧ ∃B: P^B≠NP^B) = 对角线技巧的边界申报; PH=交替量词塔 Σk/Πk; 塌缩定理 (相邻层等 ⟹ 全塔塌); TISP 联合权衡下界 (唯一已知的时空同时界)。
4. **空间-多项式链**: **Savitch** NSPACE(S)⊆DSPACE(S²) (构形图可达性, 递归中点法); PSPACE=NPSPACE=AP; **TQBF** PSPACE-完备 (交替量词=轮流走棋的博弈本质); **NL=coNL** (一步证书归纳, "非成员资格也有短证"); logspace 归约; P-完全=并行难。
5. **随机化链**: BPP/RP/ZPP/coRP 谱系; Chernoff 误差指数压缩+amplification; **Adleman** BPP⊆P/poly (随机性固化进 advice); Sipser-Gács 移位覆盖 BPP⊆Σ₂; 语义类无完全问题 (语义障碍); 随机自可归约 (离散对数均匀难=最坏即平均)。
6. **交互证明链**: IP 定义 (私币); GNI 三轮协议 (随机重排+挑战); 公私币等价 (Goldwasser-Sipser 集合大小换口径); **算术化+sumcheck=IP=PSPACE** 主干 (布尔式→低次多项式, 逐轮低次性+Schwartz-Zippel 抓谎, 度数控制改写); 协议编译器; 永久式 #P 协议; 程序检查/自检验/**自修正** (BLR); **MIP=NEXP** (隔离审讯); PCP 预告。
7. **计数链**: #P 函数类; **Valiant permanent #P-完备** (gadget 归约); 近似计数↔均匀抽样互推 (UPATH 自归约+乘法加权); **Toda** PH⊆P^#P (两步: PH⊆BP·⊕P ∧ ⊕P 指数封闭); **Valiant-Vazirani 隔离** (随机哈希逐层剥离, 4τ³+3τ⁴ 概率恰一解 ⟹ NP 随机归约 USAT)。
8. **密码-伪随机链**: OWF (平均难求逆); **GL 硬核位** (下一比特不可测⟺伪随机; 求逆器→预测器); Blum-Micali 迭代 PRG ↔ **Yao 混合论证** (逐位拆谎⟺整体不可区分); **GGM PRFG 树**; 零知识 (模拟器存在=零泄漏); GMW 任意心理博弈; PRG⟹学习下界 (KV94)。
9. **电路下界链**: Shannon 计数 (几乎所有函数需指数电路; 存在性易/显式构造难); **Håstad 切换引理** (随机限制星赋值, d 深度塌到常数 ⟹ AC0 下界); 向日葵引理 (Erdős-Rado 1960) + **Razborov 逼近归纳** (单调 CLIQUE: клика/非клика两测试分布); ACC0=GF(p) 多项式逼近度 (Razborov-Smolensky); depth-3 阈值 n^{Ω(log n)}; **Tardos** 单调/非单调指数分离; KW 对偶 (电路深度↔通信); KRW 直接和猜想。
10. **决策树-通信链**: 决策树 D(f)/证书复杂度 C₀,C₁ (双口径); **Yao 极小极大** (随机复杂度=分布复杂度取 max, 对手策略=概率分布); 敏感度 s→块敏感度 bs (bs² 封顶多项式度); 通信复杂度: monochromatic 矩形三工具 (**fooling set / tiling / 秩**); 多方额头协议 (GIP, BNS 下界 n/2^k); 单向通信。
11. **代数模型链**: 代数电路=直线程序; **部分导数空间维数** (Nisan-Wigderson permanent 指数下界); 多线性公式 (Raz 超多项式); det/perm 代数完备性对; 代数树 (Ben-Or: 半代数集+连通分支); BSS 实数精度陷阱; Mandelbrot 维数屏障。
12. **平均情形链**: distNP=(L,D) 分布对; 可处理/可采样分布; **Levin 峰消去** (重新加权把时间峰移进低概率尾); 平均完备; 启发式 vs 错误率两口径; **Impagliazzo 五世界** (Algorithmica/Heuristica/Pessiland/Land…draft 只列 4, 勘误)。
13. **去随机化链**: 硬度vs随机性纲领; **NW 生成器** (硬函数 truth table+组合设计部分相关族; 平均原理: 输出位与种子相关被硬度吸收); **去随机化⟹电路下界** (KI03: PIT 去随机化 ⟹ NEXP∉P/poly 或不可证); IW01 均匀假设; expander 双定义等价 (谱隙↔边展开)+混合引理+游走定理; **之字积** (RVW00) 常度显式扩张; **Reingold UPATH∈L** (保连通换扩张); 弱源提取器 ((n,k)-source 最小熵; 种子必要性; SV 源); **Trevisan 提取器** (编码视角+涂抹黑盒警告); **回收引理+Nisan 空间 PRG** (f(X)∘W≈f(X)∘Ext; 半程递归; "logspace 机器只记得 O(log n) 已见随机位"); Saks-Zhou log^{1.5}; BPL=L 直觉。
14. **硬度放大链**: 最坏→温和→强三阶程序; **Yao XOR 引理** (f^⊕k 优势 (1−2δ)^k 指数衰减); **Impagliazzo 硬核集** (硬度集中于密度 δ/2 子集; Russell-Noam 博弈+多数判决矛盾=min-max/boosting 血缘; Farkas 分离超平面⟺LP 对偶); ECC 谱系: WH (距离 1/2+线性+随机子和) / RS (距离 1−n/m, n−1 次多项式≤n 根) / RM (多变元+Schwartz-Zippel) / **级联 (距离相乘)**; **Berlekamp-Welch** (c(x)=g(x)e(x), 误差定位=根计数); 局部解码 (WH 两查询自修正 / RM 随机线+RS); **列表解码** (Johnson 界 ℓ≤1/(2δ²) / Sudan 双变量+因子分解 / 局部列表解码 **GL=WH 解码器**); 定理 17.24 (worst→mild) + **17.31 (worst→strong**, RM 参数+虚构算法换元分析)。
15. **PCP 链**: (r,q)-验证器+可靠性放大; **⟺gap-CSP 字典** (验证器↔实例/证明↔赋值/随机位↔约束数/可靠性↔NO 值); qCSP_W 框架+宽度归约; **FGLSS 图** (7 顶点簇, α(G)=val·m); **gap 指数化** (INDSET: 扩张游走在独立集内, 常数 gap→m^δ); **Dinur 主引理** (1−ε→1−2ε 迭代: 放大 g₆ 大字母表+字母归约 h; 幂次 ψ^t 半径 t+√t 球赋值+路径约束+**多数派赋值**+中途边 Claim 18.32 二项统计距离+重数上界+并集下界; 字母归约 WH 块+逐约束 Π+0.99 解码); WH 码 NP⊆PCP(poly,1) (**QUADEQ 三步**: BLR 线性测试+自修正+一致性 f(r)f(r′)=g(r⊗r′)+随机子和); 邻近 PCP (级联测试); **Raz 平行重复** ((1−δ^a)^{bℓ}, 不平凡=证明不一致可抓); **Håstad 3-bit** (线性测试 πi1+πi2+πi3=b; MAX E3LIN 最优 1/2+ε ⟹ 7/8+ε 3SAT 不可近似; **长码** 2^W 双指数膨胀+噪声测试 (1−2ρ)^|α| 压高阶+**折叠**奇支撑+h²(α) 投影+**Dᵢ=f̂² 采样重构**+GL 学习算法树剪枝 ≤4/ε 每层); 摊销自由位趋零 ⟹ CLIQUE n^{1−ε}。
16. **量子链**: 概率计算重表 (寄存器=分布向量+线性随机算子+初等≤3位 ≡BPP); 量子态=复酉向量 (|v_x|² 测度; 酉性四等价); **Hadamard=GF(2) 傅里叶=WH 码**三合一; 万有性 H+Toffoli+计数/维数下界; **Bell 博弈**经典 3/4 封顶 vs EPR ±π/8 条件旋转 cos²(π/8)≈0.85→0.8; BPP⊆BQP (置换变体自逆+**前算-反算清**); **Grover** 两反射=2θ 旋转 (sinθ=2^{−n/2}, O(2^{n/2}); 量子≠指数并行); **Simon** (干涉过滤 a·y=0+线性方程+高斯消元; 黑盒生日界 2^{n/2}); **QFT** (FFT 偶奇分推 O(m²); **变换态≠可读出**); **Shor 阶查找** (算术级数叠加+二次 Fourier+**连分数**唯一逼近+弦长引理 20.21/20.22) + 因式分解归约 (引理 20.23 CRT 1/4+引理 20.24 **非平凡平方根泄漏因子**); **BQP⊆PSPACE** (Coeff 递归模拟) 可加强至 P^#P; RFS 不在 PH; 噪声=环境测量+阈值定理。
17. **元方法论链**: 三大障碍=三层地图 (**相对化 BGS/自然证明 RR97/代数化**); **自然证明三条件** (大性 1/2^n+有用性+构造性 2^{O(n)}) ⟹ Φ 即 PRFG 区分器 ⟹ 破坏分解/离散对数; **引理 22.2** (形式测度 μ(f)≥c ⟹ ≥1/4 函数 μ≥c/4 — 下界证明必然涉随机函数); Razborov 次模测度判死 (μ(f∨g)+μ(f∧g)≤µ(f)+µ(g) ⟹ µ=O(n)); **Lovász Kneser 拓扑证=PSPACE-自然** (非构造≠难); **三对象统一** (PRG/提取器/列表码互造); 证明复杂度 (**Cook-Reckhow: SAT 多项式证明⟺NP=coNP**); **Fagin 二阶存在量词=NP**; Goldbach 诚实条款 (验证 10^17≠证明); 附录工具链 (Markov→Chebyshev→**Chernoff 哑变量全证**/Jensen/Stirling 熵形/容斥截断两向/**随机子和末位后选**/Vandermonde 根数界/Schwartz-Zippel 归纳)。

---

## 章节弧线 (四幕)

- **第一幕 基石 (ch1-6)**: 对角线不可计算性→NP=可验证性+Cook-Levin 局部表算→归约网经济学→时间层次+Ladner 中间态→BGS 相对化第一障碍→Savitch/TQBF (博弈=PSPACE 本质)→NL=coNL→PH 塌缩→ATM/TISP→P/poly advice+Karp-Lipton→NC 并行谱系。
- **第二幕 随机-交互-计数-密码 (ch7-10)**: BPP+误差消减+扩展图混合游走 (UPATH 谱隙)→IP 定义+GNI 私币+GS 口径换算+**算术化 sumcheck IP=PSPACE**→程序检查/MIP=NEXP/#P Valiant permanent+Toda+Valiant-Vazirani 隔离→OWF/GL 硬核位/Blum-Micali-Yao 混合/GGM PRFG/零知识 GMW。
- **第三幕 下界武器库 (ch11-15)**: 决策树+Yao 极小极大→通信矩形三工具+多方额头→Håstad 切换引理+向日葵+单调 CLIQUE+ACC0+Tardos/KW/KRW→代数模型部分导数+det/perm 完备对→平均情形五世界+Levin 峰消去。
- **第四幕 去随机化-PCP-量子-元反思 (ch16-22+附录)**: NW 设计+去随机化⟹下界+之字积+Reingold UPATH∈L+弱源提取器+回收引理/Nisan→硬度放大 (XOR/硬核集/ECC 全套/列表解码/GL=WH)→**PCP 定理全证** (字典/FGLSS/gap 指数化/Dinur 迭代/QUADEQUAD/邻近) + Raz 平行重复+Håstad 3-bit+Fourier 全套→量子 (Bell/Grover/Simon/Shor/BQP 定位)→逻辑 (Fagin/证明复杂度, sketchy)→**自然证明障碍**+Arora 个人观 (Lovász 拓扑突围)→附录 A 工具箱+Bibliography 全扫。

---

## 作者方法论指纹

1. **定义多等价并列**: 每个核心类给机器/验证器/oracle/逻辑多重定义并证等价 — 定义本身即定理 (NP 四重身份; BPP 双口径)。
2. **字典式双视角**: PCP⟺gap-CSP 对照表全列; 协议编译器; 三对象统一表 — 换视角不改事实改可解性。
3. **Proof by picture**: Grover 两反射图/Dinur 幂次构图/复数弦长图 20.5/Bell 几何 — 图先于公式承载直觉。
4. **障碍当导航**: 每遇不可能性即绘地图 (BGS→RR97→代数化; Impagliazzo 五世界), 把负结果转为选点依据。
5. **习题承重**: 关键步骤留题+提示 (GL 学习=习题正文化; Claim 18.36/18.37/18.38 移章末), 习题网络=正文骨架。
6. **老定理新身份**: GL 硬核位=WH 局部列表解码器; Hadamard=Fourier=WH 码三合一; 向日葵 1960 服务电路下界; von Neumann 1961 容错先声。
7. **组合-代数-概率三栖混用**: 同一证明内随机限制+GF(p) 多项式+Chernoff 自由切换, 无单一方言。
8. **draft 诚实态贯穿**: "To be written"/空节/断引 [?]/"References still incomplete" 直书不掩饰 — 未完成本身入档。
9. **个人观署名**: §22.4 "My personal view" 直书立场分歧 (Arora 乐观 vs Razborov 悲观), 文献志含苏联 perebor 史正名 Levin。
10. **epigraph 开章文化**: Feynman 双引 (1964 "close the doors"+1982 负概率)/Papadimitriou-Yannakakis "computation is inherently unstable"/Gödel 致 von Neumann 信 — 每章以一句立场锚定。

---

## 全书覆盖表 (诚实条款)

| 部分 | 内容 | 状态 |
|---|---|---|
| 卷首+ch1-2 (块1-4) | 导论/TM 形式化/NP/Cook-Levin/归约网/coNP/padding | 逐行读毕 |
| ch3-4 (块4-6) | 层次定理/Ladner/BGS/空间/Savitch/TQBF/博弈 | 逐行读毕 |
| ch5-6 (块6-8) | PH/ATM/TISP/P/poly/Karp-Lipton/NC | 逐行读毕 |
| ch7 (块8-10) | BPP/误差消减/素性/多项式恒等/扩展图混合 | 逐行读毕 |
| ch8 (块10-13) | IP/GNI/Goldwasser-Sipser/IP=PSPACE/程序检查/MIP/永久式协议 | 逐行读毕 |
| ch9-10 (块13-16) | #P/Valiant/Toda/VV/OWF/GL/PRG/PRFG/零知识 | 逐行读毕 |
| ch11-12 (块16-18) | 决策树/Yao 极小极大/通信复杂度/多方 | 逐行读毕 |
| ch13 (块18-19) | 切换引理/ACC/单调 CLIQUE/深度/分支程序/KW | 逐行读毕 |
| ch14 (块19-20) | 代数计算模型/det-perm/部分导数 | 逐行读毕 |
| ch15 (块20-21) | 平均情形/五世界/Levin 峰消去 | 逐行读毕 |
| ch16 (块21-25) | 去随机化/NW/组合设计/KI/扩展图/Reingold/弱源/提取器/回收/Nisan | 逐行读毕 |
| ch17 (块25-28) | XOR 引理/硬核集/ECC 构造/解码/局部/列表/17.31 | 逐行读毕 |
| ch18 (块28-31) | PCP 定理/字典/FGLSS/gap 指数化/Dinur 全证/QUADEQ/邻近 | 逐行读毕 |
| ch19 (块31-37) | 平行重复/Håstad 3-bit/Fourier 全套/学习算法/综述 | 逐行读毕 |
| ch20+§20.A (块37-40) | 量子全套 (Bell/Grover/Simon/Shor/BQP)/连分数 | 逐行读毕 |
| ch21-22 (块40-41) | 逻辑 (sketchy 如实)/自然证明障碍/个人观 | 逐行读毕 (空节照录) |
| 附录 A+Bibliography (块41-43) | 数学背景/有限域/多项式/文献志 A-Z 全扫 | 逐行读毕 (标题级+关键条目细读) |

**勘误记档 (draft 态+书误申报)**: 页眉 "DRAFT Web draft 2007-01-08 21:59" 与页脚 "References and attributions are still incomplete" 贯穿全书 (draft 态如实); 断引 [?]/?? 多处 (Thm 13.1/13.16/16.21/16.23/Lem 16.26/Thm 16.34/18.4/18.23 BLR/19.5 Håstad/19.18 学习/Raz 反例/Feige-Kilian/Raz-Safra/最佳经典分解算法/Thm 20.26/DeMillo-Lipton 等); **书误两处**: ①det/perm 公式对调 (~行12921-12943); ②Note 20.8 "real matrix is unitary iff symmetric" 应为 orthogonal (旋转阵反例); §17 章注 "many attributions still missing"; ch16 "still a lot missing"; §16.6.5 "still quite rough"; Lemma 16.42 证明留习题; §15.3 与 §15.2 重复; 五世界只列 4/5; §12.4/§11.5/§12.5 占位; GIP 证明 "to be completed"; DFT 例 "(or formula??; CHECK)"+"status of lowerbounds??"+"rect??"; Berlekamp-Welch "(??)" 断 ref; §19.4 节眉 "Proof of Theorem ??"; Lemma 17.9 证明缺右括号; §19.6.3 Unique games 空标题; **ch21 "Very sketchy"** (定理 21.2 Fagin "To be written."; §21.1.2 MAX-SNP/§21.2.1-21.2.3/§21.3 皆空节); 附录 A.5 空标题; Figures 16.5/18.5/22.1/22.2 "Figure unavailable in pdf file"; [AS98] "FOCS 1982" 疑 1992 之误; [AA02] 作者串排序混乱。**本线记档**: 映射 103 曾误写 "kl599" (应为 kl399), 正文映射体更正。

---

## 全映射目录 (215 条)

1. 1990 后成果单
2. 三部架构
3. draft 诚实条款
4. 读者-课程三层
5. 目录主轴预读
6. 双题词宣言
7. 自复制→DNA 预言
8. 六大问题清单
9. 算法效率>技术
10. 不可能证明学
11. TM 形式定义
12. 稳健性三引理
13. 机器即字符串
14. 普适 TM
15. 不可计算开篇
16. 对角化+归约两板斧
17. P 定义+子程序闭包
18. 强 CT 四异议
19. Edmonds 五钉
20. 章史+混凝土机器
21. 缓冲区摊还 O(T log T)
22. NP=可验证
23. 平凡夹心与中间地带
24. Karp 归约机器
25. Cook-Levin 局部性
26. 快照表机械可核
27. Levin/parsimonious 强化
28. 归约网经济
29. NP 硬度应对学
30. coNP ∀-对偶
31. Padding 技术
32. P=NP 乌托邦两面
33. NP=coNP 短反证
34. 章史+习题金矿
35. 三谱系定理+lazy 对角
36. Ladner SATH 构造
37. 谕示相对化
38. 局部性非相对化
39. 空间可复用+唯一已知关系
40. 构形图+博弈难于谜题
41. TQBF 证明
42. Savitch 平方确定化
43. 博弈=PSPACE 本质
44. logspace 归约
45. NL=coNL
46. PH 分层与塌缩
47. 交替机 ATM
48. TISP 联合下界
49. 谕示分层+VC 维
50. P/poly 非均匀
51. 建议机表征
52. Karp-Lipton 塌缩
53. 5n 屏障+概率法
54. NC 并行类
55. CKT-SAT+Tseitin+PH 电路化
56. BPP 胜率定义
57. 随机点测三例
58. RP/ZPP 单零侧
59. Chernoff 误差压缩
60. Von Neumann 提纯+扩展图
61. 扩展图游走误差压缩
62. Adleman 固化
63. Sipser-Gács 移位覆盖
64. BPP 语义性无完全问题
65. UPATH 游走+谱隙
66. 谱隙普适性
67. 扩展图双定义等价
68. 随机存在易显式构造难
69. 扩展游走算子证明
70. dIP=NP
71. IP 盲测定义
72. GNI 私币协议
73. 公币几乎等价+集合大小换口径
74. 哈希族+集合下界
75. 算术化+sumcheck 开题
76. sumcheck 逐轮抓谎
77. 度数控制改写
78. 证明者强于语言
79. 程序检查/实例检查
80. MIP 隔离审讯+PCP 预告
81. 计数难于存在
82. 保数目归约
83. Valiant 永久积
84. 近似计数与抽样
85. Toda+VV 隔离
86. 唯一解不更容易+逐层剥离
87. 4τ³+3τ⁴ 精度抬升
88. 单向函数
89. 随机自可归约
90. 一次一密+伪随机开篇
91. 预测⟺区分混合论证
92. GL 硬核位
93. 迭代 PRG
94. PRF 树+MPC
95. 去随机化+学习下界
96. 决策树对手论证
97. 证书复杂度双口径
98. 随机决策树
99. Yao 极小极大
100. 敏感度阶梯
101. 通信复杂度模型
102. 欺骗集
103. 单色矩形铺砌
104. 秩/差异/特征值工具箱
105. 多方额头数+GIP
106. 通信下界应用+KW 对偶
107. 滑铁卢+4.5n 屏障
108. Håstad 换路引理
109. ACC 多项式逼近
110. 单调电路+两测试分布
111. 向日葵引理+逼近归纳
112. 前沿五道清单
113. Valiant 深度削减+超集中器
114. 分支程序+Barrington
115. KW 对偶+KRW 直接和
116. 代数电路+det/perm 完备性
117. Ben-Or 拓扑下界
118. BSS 实数精度陷阱
119. Mandelbrot 维数屏障
120. 平均情形+分布敏感律
121. Impagliazzo 五世界+分布对
122. 平均多项式 E[t^{1/c}]
123. 支配条件归约
124. 峰消除+Levin 完备
125. 启发式口径+可采样
126. 硬度换随机
127. PRG 口径差异
128. 平均硬度 1/2+1/S
129. NW 设计部分相关
130. 平均原理+逐位拆谎
131. 黑盒归约+贪心设计构造
132. 去随机化⟹下界
133. 向下自可归约验证
134. 三积扩展器
135. 替换积随机循环
136. Reingold 保连通换扩张
137. 最小熵+SV 源
138. 提取器+种子必要性
139. Trevisan 涂抹+采样
140. 提取器应用族
141. 回收引理+Nisan 半程递归
142. BPL=L 直觉
143. PRG=提取器=列表码
144. Yao XOR 引理
145. Impagliazzo 硬核集
146. 硬核集博弈证明
147. Farkas 分离超平面
148. ECC 存在性与天花板
149. 高维几何直觉
150. 码族谱系
151. Berlekamp-Welch 误差定位
152. 随机线局部解码
153. Johnson 界
154. Sundan 列表解码
155. GL=WH 列表解码+度 3 曲线
156. 虚构算法换元分析
157. 定理 17.31 最坏→强
158. PCP=局部可验证明
159. 非鲁棒直觉被推翻
160. gap 问题+CSP 框架
161. PCP⟺gap-CSP 字典
162. FGLSS 图归约
163. 扩张游走 gap 指数化
164. BLR 线性测试+自修正
165. QUADEQ 三步验证器
166. PCP 主引理 gap 翻倍
167. 二段构图
168. 幂操作球赋值
169. 多数派赋值
170. 中途边+重数上界
171. 字母还原装配
172. 原证史+Dinur 革命
173. 预处理三 Claim
174. 平行重复引理
175. Håstad 线性三查
176. 傅里叶基+Parseval
177. 线性测试 Σf̂³
178. 长码+噪声测试
179. Håstad 投影测试
180. 学傅里叶系数
181. 摊销自由位复杂度
182. Bell 两证明者博弈
183. 双缝干涉
184. 概率计算新视角
185. 量子叠加开篇
186. Hadamard=傅里叶
187. 万有性+计数下界
188. EPR 对策 0.8
189. Grover 几何
190. Simon 隐藏结构
191. Simon 干涉过滤
192. QFT/FFT
193. 阶查找
194. 因式分解归约
195. BQP 定位
196. 连分数逼近
197. Fagin 二阶 NP
198. 证明复杂度
199. 自然证明障碍
200. 非构造突围
201. 障碍深度申报
202. Goldbach 诚实条款
203. 随机子和原理
204. 平均论证族
205. Chernoff 全证
206. GF(q^k) 构造
207. 根数界+Lagrange
208. Schwartz-Zippel
209. 文献志 A-L
210. 空节诚实条款
211. 文献对锚定
212. Tardos 指数鸿沟
213. 苏联 perebor 史
214. von Neumann 1961
215. Yao 五连+Valiant

---

## 收官注 (未发射算子点单制)

读书线只登记候选， 不自动发射； 用户点名才立项 (kl 卡另行查重+prereg):

1. **自然证明审计协议**: 每个候选通用判据申报三性质 (大性: 覆盖面份额/有用性: 分离力/构造性: 可算性) — 三性齐则按 RR97 逻辑反问 "为何市场没把它抹掉"; 接口 kl524 NO-VALUE 护栏 + kl336 求啥得啥。
2. **弱判据集成-硬度放大线**: XOR 引理/硬核集的市场版——单票 51% 弱优势经独立多票放大; boosting=弱判据集成的血缘申报; 接口 kl332 等权 E[K] + ε-greedy 纪律。
3. **误差定位-坏样本审查器**: Berlekamp-Welch c=ge 的市场版——结果对部分损坏样本的鲁棒性=码距申报; 误差定位器找出坏样本再重估; 接口 kl225b 数据缺陷 + kl331 最坏情形申报。
4. **双视角字典制度化**: PCP⟺gap-CSP 字典表的组合域版——每个现象强制双口径 (验证器视角/约束视角 = iso/brd) 分读后合判; 接口 iso/brd 第五域符号翻转群。
5. **摊销检验预算经济学**: 自由位复杂度=每单位置信的查询成本——判据设计报成本-置信紧关系而非只报置信; 接口 brute 有界点单 (计算预算申报) + kl524 薄截面护栏。
6. **间隙迭代放大引擎**: Dinur 主引理的市场版——小 gap 常数轮放大 (每轮 2×, 开销常数倍) 而非一次到位; 每轮保守放大+字母表归约防膨胀; 接口 kl526 换腿折正 BOARD 五门 + 组装家族日闸门。
7. **周期提取-连分数线**: 阶查找=均匀采样×周期结构→连分数取分母; 市场版=自相关/峰龄的显式周期复原器; 接口 kl311 峰龄 argmax + kl323 自相关 lag-1。
8. **傅里叶载荷谱正式化**: Parseval/Σf̂³/噪声 (1−2ρ)^|α| 压高阶全套——组合载荷谱的正式化 (大载荷=近结构; 噪声按阶衰减; 折叠杀偶阶); 接口 kl322 两毒正交 + kl347 矩形窗定律。

---

读序 18 收官。→ 读序 19: Wilf《generatingfunctionology》2e (math.upenn.edu/~wilf 作者站官方免费, DownldGF.html; 域=生成函数/组合数学, shelf 空白域无撞线, 2026-11-21 grep 检查过: 概率/组合/生成函数全空; 接口: 母函数=矩量谱生成 (Chernoff mgf 哑变量证明刚读)/递推闭式解→E[K] 闭式 kl332 式推导/分拆计数→brute 枚举经济; 备选 Hastings-Dekking《Introduction to Probability》Dartmouth 官方)
