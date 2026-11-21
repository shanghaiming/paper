# study_prob — 《Introduction to Probability》2e (Grinstead & Snell) 读序20 master

**头注**: Charles M. Grinstead & J. Laurie Snell, *Introduction to Probability*, 2nd revised edition, CHANCE Project (Dartmouth College), 4 July 2006, GNU Free Documentation License. 官方免费公开 (dartmouth.edu/chance/teaching_aids/books_articles/probability_book). 源 PDF 12 章+3 附录约 510 页; 清洗文本 25709 行, 46 块全读, 读毕 2026-11-21。 每块一记, 共 **230 条市场映射** (读序20 自有编号)。 单一 master: 概率论域此后以本卡为准, note/deep 拆分不许 (study-master-format 纪律)。

---

## §〇 推导层 (全书的数学脊柱)

五层楼梯, 每层都可从下一层推出:

1. **可加性 → 分布**: 样本空间上 P 满足有限可加 (ch1 离散/ch2 连续), 全书一切由 "概率是测度" 这一句长出来。 Bertrand 悖论 (三种随机弦三種答案) 在第 2 章就立了 "随机必须先声明机制" 的纪律——与注册表 "先冻结口径再跑数" 同构。
2. **期望线性性 (First Fundamental Mystery, p.232)**: E(X) 可在完全不知道 X 的分布时算出 (指示变量法)。 这是全部 "分解账本" 的合法性来源——一切 rent/death 分解、一切 kl 帝国的逐格记账, 底座都是这一行。
3. **Chebyshev → LLN → CLT**: 方差的二次集中 (ch8) 给弱大数定律; 特征函数路线 (ch10.3) 给中心极限; 局部 CLT+1/2 修正处理格子分布 (ch9.1)。 Cauchy 一步不放: 无方差 ⇒ LLN 失败, 密度形状逐 n 不变 (kl133 卷积封闭族的教科书出处)。
4. **鞅/不变量观点**: de Moivre 1711 名义值 (q/p)^k 把不公平赌局换算成公平 (ch12.2 史注), 调和函数 f=Pf "有限态公平游戏公平到底" (Ex 11.2.27), Li 赌团把首达时间变现金流 (Ex 11.2.28)。 **exit 规则造不出 alpha 的定理级根据** (V134 出场定律的数学血统), 恒等门/双跑对账是它的工程化身。
5. **路径几何 (ch11-12)**: 马氏链两基本矩阵 N=(I−Q)⁻¹ (吸收) / Z=(I−P+W)⁻¹ (遍历) 把 "问几次、问多久、问归宿" 全部闭式化; 随机游走把分布极限升级为路径性质——常返维度阈值 (R¹R² 常返/R³ w*≈.34), 均衡次数~√(2n/π), 领先时间反正弦律及其对任意对称连续分布的普适 (Thm 12.6)。

母函数是贯穿 5→12 的万能螺丝刀: 卷积↔乘积, 分支过程灭绝 d=h(d), 首达 U=1+UF, 破产时间 h(z), 均衡计数 G(x)——"递推→泛函方程→解→系数" 四步流水线 (Ex 12.3) 本身就是方法论输出。

## 章节弧线

- **ch1-2 样本空间**: 赌注/赔率/费马-帕斯卡分赌本; 频率概念+数据偏差史; 连续域 Buffon 针 2/π、Bertrand 三义。 立场: 概率的第一课是 "说清楚你在抽什么"。
- **ch3 组合**: 生日 √(2·ln2·n)、Stirling、不动点/纪录/错排三不变量、a-shuffle 洗牌合成定理 (Bayer-Diaconis 7 次)。 不变量>枚举。
- **ch4 条件概率**: Bayes 逆树/三病两检基率陷阱/Monty Hall/双信封三变体+Cover 随机阈值。 条件化是重加权, 不是增信息。
- **ch5-7 分布与和**: Poisson 炸弹落点/超几何 Fisher 精确/Benford; 期望三课示范 (St. Petersburg/纪录调和级数/Little 定律); 卷积 (骰子和、Cauchy 平均不变、Rayleigh)。 和的行为由分量的最小公分母决定。
- **ch8-9 极限定理**: Chebyshev 最优性、Bernoulli 1713 "命运" 结语、Borel-Cantelli、Weierstrass 概率证明; CLT 三层 (Bernoulli 局部/独立试验/连续), de Moivre 传记 (撕 Principia 的囚徒), Gallup 民调 n≥1111 与人口无关, Galton 回归起源与 Gosset t。
- **ch10 母函数**: 首达领先 h_T=pz+qz·h_T²; Galton-Watson 分支过程 (Watson 错解/Bienaymé 28 年优先权/Keyfitz d≈.324/连锁信 Circle of Gold); mgf 证 CLT 与 Cauchy 特征函数三行证 LLN 失败。
- **ch11 马氏链**: Oz/谣言/基因/Ehrenfest/兄弟妹交配; 吸收链三定理 (N, t=Nc, B=NR) 与醉汉 x/5 调和律; 遍历链 wP=w 三算法+直觉猜 (迷宫入口数)、r_i=1/w_i 倒数律、Kemeny 常数与 Doyle 散步解、Doeblin 耦合 (Krusal 计数戏法)、可逆性细致平衡、链上 CLT、Onegin 元音辅音与香农词链、Bachelier 1900 早于爱因斯坦。
- **ch12 随机游走**: Pólya 维度判决 ("二维以内可醉")、均衡次数√n 律与 E(M_n) 孪生、Levasseur 数牌父母+Zagier 胜率 ½+1/(2√2)、赌徒破产闭式与 de Moivre 鞅源流、Feller 入场费 1、反正弦律三连 (末次回归/领先时间/首最大时刻) 与普适性。
- **附录**: 正态表; Galton 1885 回归原始表 (整数英寸偏置注=数据缺陷诚实申报的先声); 1990 生命表 (85 岁男女 21962 vs 41230)。

## 方法论指纹 (本书区别于同类书的三件事)

1. **程序内嵌**: 每个概念配 TrueBasic 程序, 模拟先行、图先于式 (ch1 第 3 页就在跑 RandomNumbers)。 与 fleet "机器+读格器+brute+目检" 四件套同构——这本书 2006 年就在按我们的纪律教书。
2. **史注每章**: 判例背后全是具体的人 (de Moivre 撕书/Markov 反庆祝罗曼诺夫/Doeblin 战死 27 岁/Gossett 健士忌酒厂)。 数学=人的判例史, 登记簿的人物志传统。
3. **反直觉判例密度全谱最高**: 生日/Monty Hall/双信封/Linda/arcsine 领先律/二维可醉——每章至少一个 "多数人答错" 的实验。 直觉必须过机器, 是全书默认操作系统。

## 全书覆盖表 (诚实条款)

- 46/46 块 (行 1-25709) 全读: 正文 12 章全覆盖, 附录 A/B/C 全读, 索引扫读 (星图已入映射 229)。
- **习题按密度抽样精读, 非全解**: 每章习题群通读题面+关键提示, 完整推演约四成 (含全部带名字的题: Lamperti/Zagier/Doyle/Penney/Levasseur/Coffman-Shepp/Feller 引语题)。 未逐题手算的部分: ch3/ch5/ch6 各约十道常规计算题。
- **书内程序未逐一运行**: TrueBasic 代码按逻辑精读, 输出表 (Oz 模拟表/Galton 板/分支过程表) 照录书上数字, 未在本机复现。
- **勘误记档**: ① 块42 处 p.461 "Thus we may approximate r_n by πn" 疑排印脱根号, 按 w_n≈1/√(πn) 上下文应为 r_n≈√(πn); ② 部分图 (Figure 9.6/11.1/11.2/12.1/12.2) 在纯文本中仅存标题与轴, 已按正文叙述重建语义, 未看原图。 此两条不改判任何映射。

---

## 全映射目录 (230 条, 读序20 产出)

1. 模拟先行 (simulation as lab ref: brute 判例线)
2. 领先次数反转 (in-lead U 形 ref: 倒U租峰)
3. 频率概念+数据偏差史 (frequency ref: 基率不变性)
4. 反直觉陈列文化 (nonintuitive examples ref: 诚实条款)
5. GFDL 开放态 (自由再分发 ref: 找书纪律)
6. 伪随机发生器谱系 (seed 纪律 ref: brute 双跑 md5)
7. 蒙特卡洛命名 (Los Alamos ref: 理论不可及处模拟)
8. 赌博系统无效律 (martingale/Labouchere ref: 加仓摊平判死)
9. 热手谬误 (Tversky 76ers ref: kl348 连涨连跌谱)
10. Pólya 维数定理 (1D/2D 必回归 3D 逃逸 ref: 均值回归 vs 趋势)
11. 公理五性质+划分分解 (骨架定理 ref: 分域定理原子形式)
12. 补集技巧 (complement trick ref: 死签基率)
13. 无知≠等概率 (对称论证边界 ref: 基率测绘纪律)
14. 赔率-概率换算 (公平注 ref: 止盈止损隐含胜率)
15. 几何首中+稀有非线性 (waiting time ref: 死签累积曲线)
16. Pascal 倒向递归表 (backward recursion ref: 停时公平价值)
17. Fermat 虚拟补齐法 (equal-length 化归 ref: 对照组等长构造)
18. 合取谬误 (conjunction fallacy ref: 条件叠加单调递减)
19. 自然密度构造 (*28 ref: 无限域测度的裂缝)
20. 蒙特卡洛面积与 π (Buffon 掷针 ref: 1/√n 精度律连续版)
21. 反演误差放大 (Perlman-Wichura 5/√n ref: 换算单位误差传播)
22. Bertrand 坐标歧义 (口径选择 ref: iso/brd 哲学地基)
23. Lazzerini 取证史 (clever stopping ref: 结果过好红旗)
24. 指数反变换+寿命反直觉 ((−1/λ)log(rnd) ref: 等待与生存)
25. 面积棒图+和的形状演化 (area=probability ref: 密度直方图)
26. 密度≠概率 (f 可超 1 ref: 密度语义纪律)
27. 诱导非均匀 (f(r)=2r ref: 变量替换破坏均匀性)
28. CDF 先行 (F′=f ref: 累积口径更稳健)
29. 和的几何推导 (切片面积 ref: 卷积几何观)
30. 诚实范围申报 (density 存在性 caveat ref: 诚实条款续)
31. 无记忆唯一性 (memoryless ref: 制度不变假设检验)
32. 检查悖论 (inspection paradox ref: 长度偏差采样)
33. 无限树连续化 (binary↔real 同构 ref: 离散-连续桥)
34. 强大数定律语义 (P=1 的例外集 ref: 基率稳定性的极限语义)
35. 乘法原理+路径概率链 (stages multiply ref: 条件连乘)
36. 碰撞阈值平方律 (birthday ref: 巧合的数学定价)
37. 渐近≠绝对 (Stirling 比值差分离 ref: 相对误差纪律)
38. 错排 1/e 基线 (fixed points ref: 随机命中基准线)
39. 纪录数 log n (records ref: 新高新低频率定律)
40. 模拟稳健性 vs 枚举爆炸 (simulation robustness ref: brute 机器修改自由度)
41. 牌型稀有度定价 (poker by counting ref: 门槛稀有度的组合定价)
42. Bernoulli 假设三重申报 (constant p ref: i.i.d. 平稳性检验)
43. 计数×单路径概率分离 (binomial structure ref: 组合层与概率层分工)
44. 秘书问题 1/e (optimal stopping ref: 观察期纪律)
45. 坦克问题 (Holmes 2m ref: 极值低估偏差)
46. 众数≠大概率 (mode .08 ref: 集中度错觉)
47. 双类错误定价 (type 1/2 ref: 判例三判的错误成本)
48. 容斥恒等式 (inclusion-exclusion ref: 重叠并集精确算)
49. 模型不可先验 (Feller Bose/Fermi ref: 分布形状由数据定谳)
50. 三源合一史 (figurate/combinatorial/binomial ref: 多重发现律)
51. 偏币提纯 (von Neumann HT/TH ref: 未知 p 下公平判定)
52. 极大似然估计 (MLE m/n ref: 频率主义默认)
53. 优惠券收集律 (coupon collector ref: 全覆盖对数成本)
54. 操作指纹判据 (rising sequences ref: 结构可证伪性)
55. 首局价值表 (Pascal “This seems odd” ref: 边际价值非单调)
56. 变差距离 (variation distance ref: 混合质量度量)
57. 随机性属过程 (random describes process ref: 不属序列)
58. 出厂序残留 (New-Age Solitaire ref: 初始态记忆)
59. 条件重整律 (P(F|E) 正比重标 ref: 最小修改更新)
60. 全信息条款 (Example 4.3 ref: 分布的信息集依赖)
61. 反向树 Bayes (reverse tree ref: 除法构造逆概率)
62. 主持人协议 (Monty 4/7 ref: 信息源行为建模)
63. 全树纪律 (in the interest of clarity ref: 拒绝 WOLOG 缩树)
64. 成对≠相互独立 (pairwise vs mutual ref: 独立层级)
65. 边缘-联合-条件三层 (marginal/joint/conditional ref: 表格三层读法)
66. 罕病假阳陷阱 (prior .001→.019 ref: 基率主导推断)
67. 名公式少用论 ("rarely use it" ref: 结构>代数)
68. 协议歧义史 (Hudde-Huygens ref: 口径歧义祖师爷)
69. Collins 乘法陷阱 (prosecution 1/12M ref: 独立假设走私)
70. 灯泡厂 Simpson (Ex 30 ref: 分项与总体反转)
71. 计算路径警告 (Pascal N=100 vs 递归 N=20 ref: 实现决定可行性)
72. 规则等价性 (Kingston 发球定理 ref: 表面不同深层等价)
73. 希望函数 (Falk 抽屉 ref: 徒劳搜索的双重动态)
74. 公交悖论 (inspection paradox ref: 长间隔偏采样)
75. 独立性数论 (Eisenberg-Ghosh 52 牌 ref: 载体结构决定独立)
76. Beta 共轭族 (prior→posterior 封闭 ref: 信念更新工程化)
77. 两臂老虎机 (bandit exploration ref: 利用-探索权衡)
78. Polya 瓮极限 (比例→均匀 ref: 强化自锁)
79. 信息获得方式 (Bar-Hillel-Falk ref: 协议决定后验)
80. 折棍对比 (2log2−1 vs 1/4 ref: 顺序即建模)
81. 双信封条件依赖 (Finn 简化 ref: 条件概率由分布定)
82. 随机阈值策略 (例 4.30 ref: 无分布知识赢)
83. 永远该换分布 (Brams-Kilgour 3/5 ref: 逐点决策恒优的构造)
84. 几何众数恒一 (mode=1 ref: 首次成功等待)
85. Poisson 公理前奏 (率恒+独立增量 ref: 计数过程建模起点)
86. 稀有事件极限 (law of rare events ref: 二项→Poisson)
87. 飞弹伦敦验证 (Feller λ=4 ref: 极端集合的统计律)
88. Fisher 精确检验 (固定边缘零模型 ref: 超几何 p 值)
89. Benford 法证 (Nigrini ref: 造假的分布指纹)
90. 标记重捕 (capture-recapture MLE ref: 隐藏总体估计)
91. 骡踢小数律 (Bortkiewicz ref: 稀有计数 Poisson 现场验证)
92. 非传递骰 (Tenney-Foster ref: 无 contexts-free 最优)
93. Collins 推翻 (Ex 34 ref: 唯一性假设崩溃)
94. Powerball 偏好 (χ² 拒绝均匀 ref: 人选数字指纹)
95. 指数-Poisson 桥 (间隔↔计数 ref: 双视角等价)
96. 逆变换采样 (推论 5.2 Y=F⁻¹(U) ref: 均匀母源)
97. 正态标准化 (Z=(X−µ)/σ ref: 两参数定全族)
98. Maxwell-Rayleigh 半径 (正态分量→模 ref: 聚合改族)
99. χ² 除期望 ((O−E)²/E ref: 相对惊讶定价)
100. Cauchy 无均值 (LLN 失效 ref: 重尾病)
101. 期望加法免独立 (First Mystery ref: 线性无条件)
102. 指示变量分解 (不动点 E=1 ref: 恒定期望)
103. 圣彼得堡定价病 (期望∞≠支付∞ ref: 矩失效)
104. 几何期望 1/p (首次等待定价 ref: 概率倒数)
105. 模拟先行定义 (Die n=100→10000 ref: 数据先于公式)
106. 纪录调和律 (E~log n+γ ref: 新高稀疏化)
107. 赌场大数律 (−1.41%×n ref: 微边×次数)
108. 蒙地卡罗监狱 (自指方程解 ref: 嵌套条款解剖)
109. 全期望公式 (加权还原恒等 ref: 分解无损)
110. Mr. Ace 系统 (期望 0 胜率 .610 ref: 鞅不改期望改分布)
111. Huygens 轮换局 (x=31/61 ref: 状态互指方程)
112. Pascal 赌注 (v=∞ ref: 无限支付压倒先验)
113. 年金定价史 (Graunt 表 ref: 期望首次商业化)
114. ESP 信息边际 (Diaconis-Graham ref: 反馈改策略)
115. 条件收敛病 (交替调和 Ex 35 ref: 排序依赖期望)
116. 方差非可加 (Y=−X 反例 ref: 二阶吃相关)
117. σ/√n 律 (均值标准误 ref: LLN 先声)
118. pq 与 q/p² (稀有等待方差爆炸 ref: p↓V↑↑)
119. 同 EV 不同 V (17 号 vs 黑色 Ex 3 ref: 二维画像)
120. 无偏估计 (1/(n−1) ref: 估计量对账)
121. 逆方差加权 (Ex 20 ref: 方差倒数即权重)
122. Cov=0 非独立 (Ex 23 ref: 线性投影非全貌)
123. Cauchy 矩失效 (方差发散 ref: 有限方差是宇宙分界)
124. Little 定律 (N=λT ref: 流量守恒)
125. 标准化通则 (Sₙ* ref: 无量纲坐标)
126. 队长几何稳态 ((1−ρ)ρʲ ref: 次临界指纹)
127. 中位数选址 (Ex 11 ref: 损失函数定统计量)
128. 均值回归 (Galton ρ 压缩 ref: 选择后稀释)
129. 等密度椭圆 (x²−2ρxy+y² ref: 相关的几何)
130. 卷积组装 (和的分布 ref: 分布相卷非均值相加)
131. 卷积封闭族 (binom/neg-binom ref: 同参数可拼接)
132. 均匀卷积三角 (z∧2−z ref: 平顶磨尖)
133. 柯西平均不动 (Ex 7.6 ref: 重尾吞采样红利)
134. 骰不可均匀装载 (Ex 9-10 ref: 分布工程的边界)
135. χ² 落地 (V=13.6 拒公平 ref: 模型可反驳)
136. 最小指数加速 (min~µ/n ref: 并联首事件)
137. 乘积过程 (Ex 13 ref: 对数域加法)
138. Chebyshev 粗界 (1/k² ref: 无分布安全边际)
139. LLN 造确定 (平均收敛 ref: 频率↔模型对接)
140. 支撑叠加 ([na,nb] ref: 幅度包络)
141. Chebyshev 界的松 (.21 vs .9625 ref: 保守换普适)
142. Borel-Cantelli 一阶 (Σ发散⇒无穷次 ref: 大偏差永续)
143. Weierstrass 概率证明 (Ex 17 ref: 统计逼近分析)
144. Monte Carlo 积分 (E g(X)=∫g ref: 期望即积分)
145. Cauchy LLN 终判 (密度 ∀n 不变 ref: 平均零增益)
146. Chebyshev 三分布对照 (习题群 ref: 界的分布依赖)
147. Pilsdorff 股价随机游走 (Ex 12 ref: 价格=增量和)
148. 正态比值生 Cauchy (Ex 16 ref: 厚尾的构造性来源)
149. 高度修正 1/ε ("horrible thing" ref: 离散→连续的桥)
150. CLT 只是近似 (1/2 修正 ref: 工具误差自觉)
151. 置信区间总体无关律 (n 只看精度 ref: 采样红利)
152. Gallup 十六届实绩表 (1936-1996 ref: 理论-实绩对账)
153. de Moivre 史 (撕书与优雅常数 ref: 手艺人的自尊)
154. 运气 vs 实力判别 (Ex 3/5/13 ref: 归因的统计门槛)
155. 格子条件与双误差预算 (定理 9.3 gcd=1 + Ex 16 ref: 条款与双侧)
156. 多基因假说 (CLT 破遗传悖论 ref: 大效应拆小)
157. Hardy-Weinberg 平衡 (一代即定 ref: 代际平稳)
158. 两级 quincunx (分层正态卷积 ref: 截断-释放实验)
159. 回归均值发现史 (reversion ref: 方差守恒机制)
160. 押 17 号 vs 押红 (Ex 12 ref: 同期望异方差)
161. n=80 vs n=16 (两条路线的价差 ref: 界与近似的经济账)
162. Student t (Guinness 酿酒师 ref: 小样本的自审)
163. 置信区间语义陷阱 (Ex 14c ref: 随机的是区间不是参数)
164. 同矩不同布 (ch10 开篇 ref: 二阶矩的盲区)
165. 疯狂分布之和仍正态 (Ex 7-9 ref: 吸引子普适性)
166. 卷积→乘积 (换域简化 ref: 对数域的杠杆)
167. 剥皮反演 (定理 10.1 ref: 从和恢复成分)
168. 公平博弈首领先期望无穷 (例 10.7 ref: 几乎必然≠期望有限)
169. 双骰灌铅不可行 GF 版 (Ex 9 ref: 根的存在性反证)
170. 分支过程三史与临界慢死 (ref: 均值=1 仍灭绝)
171. 不动点迭代 d=h(d) (图上楼梯 ref: 平衡点选择)
172. 12 代不够 (.599 vs ⅔ ref: 有限视野低估终局)
173. Keyfitz 拟合 (m=1.837→d=.324 ref: 参数到终局的映射)
174. W_n 随机极限 (无普适常数 ref: LLN 失效地带)
175. 连锁信定价 (Circle of Gold ref: 传销的数学讣告)
176. 连锁信模拟账本 (表 10.4/10.5 ref: 期望正而赢率<半)
177. 随机和复合 g(f(z)) (Ex 4/6 ref: 结构嵌套的代数)
178. 总后代 1/(1−m) (Ex 7 ref: 几何级数总账)
179. 灭绝时间重尾 (Ex 8 ref: d=1 而期望寿命∞)
180. MGF 收敛半径与特征函数 (ref: 工具执照的宽窄)
181. 阶梯修补与 gcd 申报 (ref: 证明的工程缝)
182. Cauchy 特征函数一击 (ref: 无矩地带的万能钥匙)
183. 青蛙跳荷叶 (Markov 性 ref: 只住当下的世界)
184. Oz 国六步忘初 (表 11.1 ref: 正则链的遗忘)
185. 例子族谱 (谣言/赛马/校系/Ehrenfest/基因 ref: 状态化世界观)
186. 踏脚石模型 (比例=胜率 ref: 守恒量的胜率定价)
187. 双币隐藏状态 (Ex 19 ref: 观测过程的伪 Markov 检验)
188. 基本矩阵 N=(I−Q)⁻¹ (ref: 矩阵几何级数)
189. t=Nc 吸收时间 (ref: 行和即寿命)
190. B=NR 与 x/5 线性律 (ref: 调和函数)
191. 胆小vs胆大策略 (Ex 13 ref: 不利赌局的注码学)
192. 赌徒破产闭式 (Ex 24 ref: (q/p)^x 指数墙)
193. 调和函数守恒律 (Ex 27 ref: f=Pf 公平到底)
194. 模式等待时间与 Penney-ante (Ex 28-36 ref: 自重叠决定等待)
195. 遍历/正则二分与 Pⁿ→W (11.3 ref: 遗忘初始条件)
196. 不动向量 wP=w (定理 11.8 ref: 平衡点的特征值身份)
197. Cesàro 平均救援与惰性化 (定理 11.10-11 ref: 周期病的两剂药)
198. 遍历链大数定律 (定理 11.12 ref: 时间占比的普适收敛)
199. 直觉猜不动向量 (例 11.22 ref: 入口数=停留占比)
200. Ehrenfest 二项平衡 (例 11.23 ref: 统计力学平衡态)
201. Engle 筹码算法 (Ex 29 ref: 整数化的不动量构造)
202. 极差压缩引理 (引理 11.1 ref: (1−2d) 几何收缩)
203. Doeblin 耦合 (定理 11.14 ref: 相遇即合流)
204. 首达时间 Nc (11.5 ref: 目标改吸收即问首达)
205. 回归时间倒数律 (定理 11.15 ref: w_i×r_i=1)
206. Z=(I−P+W)⁻¹ (ref: 先减平衡再叠级数)
207. m_ij=(z_jj−z_ij)/w_j (定理 11.16 ref: 首达表的闭式生产线)
208. Ehrenfest 中心趋势 (例 11.28 ref: 离开易回归难)
209. 细致平衡与电影检验 (ref: w_i p_ij=w_j p_ji)
210. 链上 CLT 与字母链史 (定理 11.17 ref: 相依变量的正态普适)
211. 检验悖论 (Ex 18 ref: 随机到达的长度偏差)
212. Kemeny 常数 (Ex 19/24 ref: 随机目标的期望到达与起点无关)
213. g=Zf 偏差累积 (Ex 20 ref: 零和收益的链上积分)
214. 首达分解与母函数 (定理 12.2-12.3 ref: u=f·u→F=1−1/U)
215. 一维必回归 (例 12.1 ref: w*=F(1)=1)
216. 维度常返阈值 (例 12.2 ref: 二维临界三维漂走)
217. 均衡次数 √n 律 (例 12.3 ref: 次数随长度的次线性增长)
218. 母函数四步法 (例 12.3 ref: 组合递推的标准化产线)
219. 末次回归对称 (Ex 6 ref: arcsine 律的地基)
220. 数牌父母与 Zagier 胜率 (Ex 10 ref: 条件信息的优势定价)
221. 公平游戏对无限庄家也必破产 (ref: 无漂移+深口袋=输光)
222. de Moivre 名义值鞅 (ref: 换计价单位造公平)
223. Feller 入场费 1 (Ex 6 ref: 极值访问的公平定价)
224. 领先时间反正弦律 (定理 12.4 ref: U 形领先分布)
225. 反正弦普适性 (定理 12.6 ref: 首最大时刻分布无关分布形状)
226. 正态表 (附录 A ref: 查表时代的精度纪律)
227. Galton 回归原始表 (附录 B ref: 回归向平庸的原数据)
228. 生命表 (附录 C ref: 生存曲线的性别分岔)
229. 索引里的历史星图
230. 全书弧线收官 (ref: 从样本空间到反正弦律)

---

## 收官注

读书线只登记候选， 不自动发射； 用户点名才立项 (kl 卡另行查重+prereg):

1. **首达/回归时间轴算子**: 用 m_ij=(z_jj−z_ij)/w_j 框架对 OHLCV 结构态 (结构语言×回撤档) 建转移矩阵, 算 Kemeny 常数型结构不变量与危机态回归时间 r=1/w (映射 205/212)。
2. **反正弦领先律检验**: 组合/单票领先占比时长的分布 vs arcsine 基准 (公平游走原假设), U 形偏离度=漂移计 (映射 219/224/225)。
3. **耦合对账**: 同噪声双策略差分 (Doeblin 耦合) 作为 kl531 双跑对账的方差消减升级版 (映射 203; 与 kl531 查重必做)。
4. **检验悖论审计**: 事件间隔 (危机月/死签日) 的长度偏差采样校正, 修 kl339 类 "间隔读数" 的稳态口径 (映射 211)。
5. **破产律直读**: 胆大/胆小注码 (映射 191) 与公平游戏必破产 (映射 221) 对照 mdd≤25% 门槛做杠杆-生存表 (观念级, 可并入 1-4 任一卡)。

读序 20 收官。→ 读序 21: 见下行选点 (书架撞线检查后定)。
