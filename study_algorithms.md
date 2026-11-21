# Jeff Erickson《Algorithms》全书笔记 (读序 21 master)

- 作者: Jeff Erickson (UIUC); *Algorithms* 1st edition 1st printing, 2019-06-13, **CC BY 4.0** (书页明示; algorithms.wtf 作者官方自发布); 本线读 jeffe.cs.illinois.edu 正名版 Algorithms-JeffE.pdf (首探 URL 拼全名 404, 书页 grep 得正名)
- 版权态: CC BY 4.0 官方免费 — 按本线纪律仍只本地读不上云
- 读毕: 2026-11-21; pdftotext -layout 22926 行逐块读毕 (41 块: 40×560+末块 526); 绪论+ch1-12 全覆盖+主题/人名/伪码三索引全扫; 472p/25055430B/pdfTeX 2019-06-14
- 纪律: 本文件为读序 21 唯一 master (算法域, 每域单一 master); study 笔记 ≠ kl 卡; Roughgarden 算法博弈付费弃案后本域首本正主

---

## §〇 推导层 (本线自证的核心链)

1. **递归-递推链**: Hanoi 2ⁿ−1 下界 (奇偶性); 递归树三板斧 (全层等比/指数衰减/指数增长) 解 mergesort T(n)=2T(n/2)+O(n)→O(n log n); "trust the Recursion Fairy"; vacuous base case (n=0 空输入); Karatsuba Θ(n^log₂3) (三乘降四); repeated squaring O(log n); Pingala 幂 (韵律学两千年先声)。
2. **回溯链**: 一切搜索=决策序列 (sequence of decisions; summary of past decisions); n-皇后按列消解; SubsetSum 递归先正确后加速 ("correct algorithm that is sometimes slow beats fast algorithm that doesn't always work"); 博弈树 Zermelo 极小极大; Racetrack 配置图 (implicit graph)。
3. **DP 链**: memoization (Michie 1967; Theseus 1952 迷宫机器人继电器记忆) → 子问题依赖图 → 后序遍历; 四步法 (子问题数组/递推/初值/求值顺序); edit distance O(mn) Needleman-Wunsch; 空间降维 (滚动数组=只留相邻层); optimal BST Knuth 区间单调; SubsetSum O(nT) 伪多项式; LIS 换元 (下标→尾部值)。
4. **贪心-交换论证链**: "greedy algorithms are usually wrong — try dynamic programming first"; 交换论证模板 (取最优解逐换不改劣); 稳定匹配 Gale-Shapley (Boston Pool 史/NRMP 1952/男性最优=女性最劣); 区间调度最晚结束; Huffman 前缀码 (Fano 半盲 vs Shannon 证明; 最深叶放最轻权归纳); Jarník-Prim-Dijkstra 正确性=非负性交换。
5. **图遍历链**: 图史三源 (Königsberg 1736/Tabula Peutingeriana/cursus publicus); whatever-first search 统一 (stack=DFS, queue=BFS, heap=best-first); BFS 波前 wavefront+token 证明 (dist 有限⟺marked); flood fill/连通分量/二部性全归约为遍历; 隐式图按需生成邻居。
6. **DFS 结构链**: pre/post 时间戳区间嵌套; 边四分类 (tree/back/forward/cross) 按区间关系; 环⟺back edge; 拓扑序=reverse postorder; **DP=后序遍历的显式化** (memoized recursion is depth-first search); SCC: Kosaraju-Sharir (反转图二次遍历) 与 Tarjan low(v) 线性; condensation 是 dag。
7. **MST 链**: safe edge 引理 (割唯一跨界轻边必入某 MST); Borůvka 1926 (捷克电网) / Kruskal (排序+并查集) / Jarník-Prim-Minty 三同庚 (Stigler 律全场); 并查集 path compression+union by rank → inverse Ackermann α(n); 瓶颈生成树; 收缩统一观; 权互异⟺唯一。
8. **最短路链**: 泛型 Ford "Repeatedly relax tense edges"; 无权=BFS (Theseus/Zuse Plankalkül 1945 史注); dag 拓扑序 push/pull; **Dijkstra 两引理** (非负性买贪心; 无张力边即最短); Leyzorek 1957 军报先于 Dijkstra 1959; Shier-Witzgall Θ(2^(V/2)) 强制例 (本身是 dag); 负环=无界摧毁良定义; **Bellman-Ford** dist≤i 引理+V−1 轮; Moore 队列改进 (脏传播 SPFA); **Bellman DP 推导** (加单调递减结构参数破自指); **DP→松弛四步编译** (去缩进/i−1→i/去维 "iteration index i is completely redundant")。
9. **APSP 链**: **Johnson 重加权** (统一加法失败→势函数 π(v) 进门礼/出门税伸缩和; 超源 Bellman-Ford 定价; "Negative edges don't slow us down after all!"); Shimbel DP; Fischer-Meyer 中点分裂 (2 的幂 ℓ); funny min-plus 矩阵乘 (无减法逆); Strassen→n^2.372864; (Kleene-Roy-)Floyd-Warshall(-Ingerman) 中间顶点集合参数化 (单遍三循环序); Seidel 邻接幂计数。
10. **流-割链**: Harris-Ross 1955 解密铁路网 (1999 解密); 守恒+饱和; 割不对称; 弱对偶 (不等式弹跳); **Maxflow=Mincut** (Ford-Fulkerson 1954/EFS 1956); 残差图=可逆性; 增广路=改进方向; 整流定理; FF 非终止反例 (Zwick 1993 φ 收敛错值); Edmonds-Karp 最短增广 (水平单调引理按 level 归纳; 每边至多 V/2 次消失); Dinitz 1970; **Orlin 2013 O(VE) 封顶**; 流分解定理 (正组合=路径+圈→无圈最大流存在); 流空间维数 E−V+2 (生成树基)。
11. **流应用链**: 边不相交路径; 二部匹配 (Berge 1957/Kuhn 1955; Kőnig 1916; **Jacobi ~1836 先声**; Hopcroft-Karp O(√V·E)); 元组选择 (adjacent-pair 多项式/非邻接 NP-hard); 考试排程 (每边=一句要求); 不相交路径覆盖 (dag 拆点二部化, k 路径⟺匹配 V−k; 一般图 NP-hard⟺Hamiltonian); **棒球淘汰** ("saturates every edge leaving s" ⟺ 出线); **项目选择** (min-cut; ∞ 依赖边; profit(S)=P−‖S,T‖; source=什么都不做基线); 矩阵取整 (⌊⌋/⌈⌉ 保行列和⟺行列和整数); 时间展开图 (Neitherlands 门时窗)。
12. **NP-难链**: Adversary 黑盒 (未测尽 2ⁿ 不可宣布, 唯正答=全测); CircuitSat; P⊆NP; P≠NP 自然律 (Clay $1M; barrier results "we can actually prove that we have no idea how to prove"); **归约刺青律** (从已知难到新); **证书变换观** (可逆的是证书非实例; if 任意实例/only-if 特殊实例不对称); Tseitin 1966 门 gadget 编译链 CircuitSat→Sat→3Sat (新变量填三文字); MaxIndSet (子句三角+否定边); 补图对偶 (clique/vertex cover=n−k); 3Color (T/F/X+majority gate); Hamiltonian 双证明 (覆盖 gadget/2ⁿ 圈=赋值); SubsetSum base-4 无进位; **伪多项式陷阱** (多项式于数值非比特; 弱/强 NP-难); Karp 1972 单篇 21 问题; 选源法则 ("as simple as possible"); gadget 方法论; 国际跳棋合规即难 (Hearn 2010); PSPACE (QBF=量词交替博弈)/EXP/ELEMENTARY 阶梯+Stockmeyer 1974 带否定正则等价非初等。

---

## 章节弧线 (四幕)

- **第一幕 递归-回溯-DP 三级跳 (ch1-3)**: Recursion Fairy→递归树→回溯=决策序列→memoization→DP 四步法; edit distance/LIS/optimal BST/SubsetSum 全家桶; 全书最大章 (约 8 块), 奠基 "先正确后加速"。
- **第二幕 贪心+图遍历 (ch4-6)**: 交换论证+稳定匹配+Huffman→图史+whatever-first search 统一→BFS/DFS→边分类/拓扑序/SCC 双线性算法; "DP=后序遍历" 把第一二幕合流。
- **第三幕 图上优化全家桶 (ch7-11)**: MST (safe edge+三算法+并查集)→单源最短路 (Ford 泛型/Dijkstra/Bellman-Ford+DP↔松弛编译)→APSP (Johnson/Floyd-Warshall/min-plus)→最大流 (残差/增广/割对偶/Edmonds-Karp/Orlin)→流应用 (匹配/淘汰/项目选择/路径覆盖) — 五章一气 "图上优化=一族松弛"。
- **第四幕 NP-难+索引 (ch12+三索引)**: Adversary 赌局→Cook-Levin→归约工具箱 (刺青律+gadget+选源法则)→伪多项式→PSPACE/EXP 阶梯→open problems 收官; 三索引各配双 epigraph, Bernard of Chartres 侏儒巨人肩+Lang "dead people do your work" 收束引用主题。

---

## 作者方法论指纹

1. **免费开放+版本自持**: CC BY 4.0 全书自发布 (algorithms.wtf); 讲义迭代 20 年成书, "we now conclude our broadcast day" 收播。
2. **epigraph 文化**: 每章双引流行文化/古典 (Moore 1959 "no foresight or ingenuity"/Lao-Tzu/Mentat First Law/Peres-Rumsfeld "not to be solved, but to be coped with over time"/Bernard of Chartres/Ada Lovelace 1843/Monty Python 长笛/Billings 四律) — 立场先行。
3. **人格化教学**: Recursion Fairy/elves/Approximate Median Fairy/Mom (median-of-medians)/All-Powerful Malicious Adversary 全部进人名索引 — 抽象角色化。
4. **历史正义癖**: Stigler 冠名律 8 处落锤 (Leyzorek 先于 Dijkstra/Levin 莫斯科 1971/Jacobi 1836/Borůvka 1926/Tseitin 1966 先于 Cook-Levin; Cook 被 Berkeley 拒 tenure); "papers that aren't actually published don't count"。
5. **戏仿宇宙**: Sham-Poobanana 大学/SPU 吉祥物/Blagojevich 腐败系双聘/"Jeff actually did this"×2/Neitherlands/Magicians — 习题即段子。
6. **格言复现**: "First make it work, then make it fast"×13 页/"Don't try to be clever"×4/"solving the right problem"×5/"useful deliberate ignorance" — 索引可见的纪律回响。
7. **反黑盒接口纪律**: "none of your business"/"Only a graph. Vertices and edges. Nothing else." — oracle 输入类型洁癖。
8. **gadget 装配观**: 归约=子图语义模块拼接 (truth/variable/clause/vault/corner/crossing) — 结构设计先于证明。
9. **常数幽默学**: 374/42/31337/4:20/Karp and his 21 Assistants/大富翁真规则全脚注/"Jeff always gets the boot"。
10. **反直觉修辞**: 说话狗/"Why 3Sat? Because it has a 3 in it. You probably think I'm joking, but I'm not."/"Hey, where's my million dollars?!"/Gwynne 反 epigraph 故意并置 — 笑点即考点。

---

## 全书覆盖表 (诚实条款)

| 部分 | 块 | 状态 |
|---|---|---|
| 绪论+ch1 递归 | 1-3 | 读毕 |
| ch2 回溯 | 3-5 | 读毕 |
| ch3 动态规划 | 5-12 | 读毕 (全书最大章) |
| ch4 贪心 | 12-15 | 读毕 |
| ch5 图基础 | 15-19 | 读毕 |
| ch6 DFS | 19-23 | 读毕 |
| ch7 MST | 24-26 | 读毕 |
| ch8 最短路 | 26-29 | 读毕 |
| ch9 APSP | 29-31 | 读毕 |
| ch10 最大流 | 31-33 | 读毕 |
| ch11 流应用+习题 | 33-35 | 读毕 |
| ch12 NP-难+习题 | 35-40 | 读毕 |
| 三索引+图源+Colophon | 40-41 | 读毕 |

诚实条款: 41/41 块零跳读; 章界按块近似 (±1 块为分块读取自然漂移); 索引按条目+epigraph 判读式扫过 (不逐条精读); pdftotext 伪影 (排版错位/希腊字母混排/下标丢失) 照录不纠, 正文语义无损。

---

## 全映射目录 (205 条)

1. ADT≠数据结构双清单 punchline (Preface: Prerequisites)
2. 习题五色标注=难度与依赖显式申报 (About the Exercises)
3. 无解答哲学 (solutions are not the point)
4. Steal This Book! CC 重分发宣言 (书页几百页额外讲义+完整考试档案开放)
5. 目录骨架=搜索谱系五章+图六章 (TOC ch1-12)
6. 算法定义四要件+Martin 反例 (§0.1/§0.4)
7. Audit trail>易用性 (Liber Abaci 笔算取代算筹的理由) (§0.1)
8. 农夫乘法=递归本质披迭代外衣 (§0.2)
9. Huntington-Hill 正确性不依赖实现 (§0.3)
10. Martin 归约教学： 难题→"更容易的"问题 (§0.4)
11. What/How/Why/How fast 四组件 (§0.5)
12. Recursion Fairy=归纳假设人格化+黑盒魔法 (§1.1-1.2)
13. "Obvious is all too often a synonym for wrong" (§0.6)
14. 空间-时间权衡 (lattice O(n²) 存表 vs 飞加 O(n)) (§0.6)
15. HHGuess (d) 正确除子可能不存在 (习题 4)
16. “不要展开递归” (汉诺塔 STOP 教学 §1.3)
17. 通用机胜专用机 (von Neumann EDVAC 排序论证 §1.4)
18. 均匀假设驳回+恶意对手模型 (§1.5 脚注)
19. 递归树三 case: 根支配/平等/叶支配 (§1.7)
20. Vacuous base case: n=0 合法地什么都不做 (§1.3)
21. MOM 嵌套代表制 (块中位的中位数 §1.8)
22. Karatsuba 恒等式省一乘 (4→3 递归， §1.9)
23. 魔数 5 的选择理由=指数衰减 (§1.8 Sanity)
24. Pingala 平方折半+半群抽象 (§1.10)
25. SplitMultiply 失败照录 "Too bad" (§1.9)
26. Ruler function ρ(n) 驱动汉诺塔 (习题 1d)
27. 九连环 bit 模型+唯一 reduced 序列 (习题 2)
28. Oblivious 排序： 决策序列与数据无关 (习题 12 Cruel)
29. 加法链最优 5 乘 (§1.10 尾)
30. StoogeSort 正确但 O(n^2.71) + median-of-three 对手构造 (习题 10/11)
31. "Don't be like Euclid" 特例≠完整证明 (习题 25 脚注)
32. 多数党识别 >1/2 握手 (习题 27)
33. Central vertex 中心顶点 (习题 38)
34. 局部最小 O(log n): 边界夹逼存在性 (习题 32)
35. kd-tree 交替轴中位分裂 (习题 40)
36. 旋转保 inorder, BST 互转 O(n) 旋转 (习题 42)
37. Zermello good/bad 递归=状态价值函数原点 (§2.2)
38. "A heuristic is an algorithm that doesn't work. (Except in practice. Sometimes. Maybe.)" (§2.2 脚注)
39. 最简变体优先： boolean 先于构造 (§2.3 变体讨论)
40. SubsetSum include/exclude 二叉递归+传引用实现 (§2.3)
41. 过去决策摘要三档 (完整历史/当前局面/剩余目标， §2.4)
42. 局面历史无关性+例外申报 (§2.4 脚注7)
43. "Try everything. You can make the algorithm faster later." (§2.4)
44. ∨/∧ 换 max/+ = 布尔版与计分版同骨架 (§2.5 变体)
45. LIS 双分解： 输入驱动 vs 输出驱动 (§2.6/§2.7)
46. 最优 BST: 频率加权深度， 偏斜可胜平衡 (§2.8)
47. O(3ⁿ) vs Θ(4ⁿ/√n): 左右独立搜索省过全枚举 (§2.8 分析)
48. 序列谓词家族 lcs/scs/bitonic/oscillating/convex (习题 4)
49. Virahāṅka→Fibonacci 命名漂移与 Stigler 定律 (§3.1)
50. Memoization: “computes the same Fibonacci numbers over and over”→查表 O(n) (§3.1)
51. DP=无重复递归， 递归式才是主角 (§3.4)
52. Specification 先于 Solution: 写清解什么而非怎么解 (§3.4 两段式)
53. 评估顺序=依赖偏序的线性扩张 (§3.4 2d)
54. "Greedy algorithms never work! (hardly ever)" + greeDY→DY 口诀 (§3.5)
55. "Whoa! Not so fast!": 输出大小是复杂度下界， 加法不免费 (§3.2)
56. 编辑距离三操作递推+箭头事后重构 (§3.7)
57. 五人独立发明互不引用+Needleman-Wunsch 错挂 (§3.7 脚注14)
58. O(nT) 伪多项式≠P=NP (§3.8 脚注16)
59. "Dynamic programming isn't always an improvement!" (T≫2ⁿ 迭代反慢， §3.8)
60. 树上 DP: memo 挂在树上+后序+父祖分摊计时 (§3.10)
61. 最大子数组积要双跟踪 min/max (题3b 网络解大量错)
62. 最大子数组四变体： 环形/长度界/和上限 (题4)
63. 九连子序列家族恰一个贪心 O(n) (题5)
64. Elmo 三档对手： 四岁随机/八岁贪心/十三岁完美 (题12)
65. 状态机 DP 家族： 冷却/持糖/Ring-Ding≤3/DDR 双脚 (题13-17)
66. Scrabble 手牌 7 张+查词 O(1) oracle 申报 (题18)
67. bitonic TSP (凸位置多项式) vs 精确 TSP O(2ⁿ poly) (题27-28)
68. Vankin's Mile→Kilometer→Fathom: 方向自由度=复杂度悬崖 (题23+脚注21)
69. 校车 soda 积分成本+大理石反直觉价格 (题21/37)
70. 括号化三连： ±/×含零/布尔 T∧∨⊕ (题33-35)
71. DeNile 机器人 BST: α 移动/β 读/γ 掉头三参物理成本 (题40)
72. 最优 B-tree: O(n^c) 与分支因子 B 无关 (题41)
73. RNA 结构： 非交叉配对=树+"utterly fictional"评分诚实条款 (题44)
74. 树 DP 应用群： 派对 MIS/礼物染色/恰 k 人/广播轮数 (题46-49)
75. 公共子树家族+有根 minor (题52-54)
76. 磁带 L/F 比率律： 成本/频率比升序 (§4.1)
77. 排课最早结束贪心+“只对 biggest 有效” (§4.2)
78. 交换论证三板斧模板 (§4.3)
79. Huffman 合并律+总长不变量 649 (§4.4)
80. Huffman 顿悟史+Sallows Pangram Machine 理论剪枝 (§4.4 脚注4/5)
81. 双向贪心+拒约权=稳定匹配引擎 (§4.5)
82. 医院最优≡医生最差同一匹配 (系4.8/4.9)
83. Dubins-Freedman: 医生可谎报获益/医院不能 (§4.5)
84. NRMP 1998 反转 <1%+病人影响 open problem (§4.5)
85. 九贪心变体三对+加权版全灭 (题1/题2)
86. 币制贪心的结构域 (题14: 美国币✓/幂次币✓可证/任意币✗)
87. 三连贪心死例： 计数全对加权全死 (题20/21/23)
88. 加倍-递增二元算子最短路径 (题16: 二进制展开 O(log n) 步)
89. 圆桌楔形与环上区间 (题24: 旋转不变性提示)
90. 图=对； 画≠图 (§5.1-5.3: Peutinger 表/planar non-planar drawings)
91. 万物优先搜索统一框架 (bag 抽象： 容器决定算法)
92. parent 生成树与容器无关 (引理5.1)
93. 配置图=记忆化博弈树
94. 依赖图=DP 可行性判据 ("reasonably small dependency graph")
95. 隐式图表示与数据结构三态 (Table 5.1)
96. 黑盒规约纪律 (§5.7: 只依赖 specification 的复用)
97. "Why not just" 信号 (脚注10: 漏掉重要东西的标志)
98. 隐式图+输出敏感两优化 (像素阵即邻接表/颜色即标记)
99. 权威也会错+Zuse 史话 (脚注12 "flatly incorrect")
100. 状态扩维术 (题13 乘积态/题14 余数态/题4 锦标赛环)
101. 约束入状态 (题15 三连同色禁/题17 mod5 费用+优惠券层)
102. 全员同步移动的奇偶墙 (题19: 42 币 O(V+E))
103. Alice-Bob 相遇与握手引理 (题22: 奇度顶点成对)
104. Racetrack 速度入状态 (题25: state=(位置，速度), 加速度±1)
105. 贪心跳跃卡死≠无解+奇偶三呼 (题28)
106. 活跃区间不交即嵌 (先/后序时钟)
107. 边四分类的语境依赖性 (tree/forward/back/cross)
108. 环检测与源汇保证 (每 dag ≥1 源 ≥1 汇)
109. 拓扑序=逆后序白送+隐式处理 (§6.3)
110. DP=依赖图后序遍历+dag 最长路径 (§6.4)
111. 强分量=双向可达等价类+凝聚图必 dag (§6.5)
112. 反向图定位汇+两遍式流水线 (Kosaraju-Sharir)
113. Tarjan low 链+辅助栈+懒惰等价 (标记=删除)
114. Kosaraju 传说+多重发现+叙事偏差 (脚注5/6)
115. 割点/桥/传递约化与 Tarry 史前史 (题3-7)
116. 检验型判据： 判 T 是否 DFS/BFS 树 (题10)
117. 并行赋值串行化+依赖图 (题11)
118. dag 路径标签字符串族 (题14/17/18)
119. 谜题图建模四重奏 (题9 砸箱/题21 首登分/题22 方向定价/题23 对向 token)
120. MST 唯一性+唯一算法论+三态边 (§7.1-7.2)
121. Borůvka 全加安全边+对数轮收敛+天然并行
122. 三 MST 算法=同一不变量三种调度+工程选型 (§7.3-7.5)
123. useless 单调性 (贬死单向永不复活)
124. Kruskal 全局排序+Union-Find 小并大 (α≈4)
125. 习题群： 反删除/增量修复/次小树/收缩统一 (题3-8)
126. Moore 算法定义宣言 (无需先见无需才智)
127. 瓶颈距离=最宽路 (题9)
128. 路径压缩=记忆化 (脚注7)
129. 负边与负环 (§8.2)
130. Ford 松弛统一场+按图选型 (§8.3-8.4)
131. BFS 波前单调性+dist 即标记 (§8.4 收官)
132. dag 最短路=DP 可计算性 (§8.5)
133. Stigler 定律与灰色文献 (§8.6 史话)
134. Dijkstra 正确性=非负性买来 (引理8.3-8.5+脚注8)
135. 负边指数爆炸+慢而对胜快而错 (§8.6 尾+脚注9)
136. Bellman-Ford 有限轮界+超时即负环 (§8.7)
137. Moore 队列改进=脏标记传播 (§8.7)
138. 加参数破自指 (DP 推导)
139. DP→松弛四步编译链 (§8.7 尾)
140. 习题群： 验证不重算+替换路+负环四活检 (题0-11)
141. 三值语义+平局裁决+最短路计数 (题11-13)
142. 状态增广族 (题15-19: 续航/mod 3/无三连色/三色旗/费用 mod 5)
143. 时变边+多目标分离 (题20-24)
144. 概率路两变体 (题25-26)
145. Johnson 势函数重加权 (§9.3-9.4)
146. 中点劈开分治 (Fischer-Meyer §9.6)
147. min-plus 半环与无逆之痛 (§9.7)
148. Floyd-Warshall 中介集参数化 (§9.8)
149. 华尔街套利题+分区距离 (题5-6)
150. 传递闭包↔矩阵乘法互约 (题9-10)
151. 缓存感知+Seidel 递归+Yuval 作弊 (题11-13)
152. 流=守恒+容量 (§10.1)
153. 割=瓶颈+弱对偶 (§10.2+Harris-Ross 史话)
154. 强对偶+残余图 (§10.3)
155. 整性+贪心顺序病+黄金比死循环 (§10.4)
156. Zwick 黄金比收敛错值+浮点教训 (§10.4 尾)
157. 流分解定理+消环 (§10.5)
158. 流空间维数与基 (§10.5+题10)
159. Edmonds-Karp 单调性证明 (§10.6)
160. Orlin 上下界合拢+小抄文化 (§10.7)
161. 灵敏度分析+最佳割 (题13-16)
162. GreedyFlow 惨败+just 词典 (题17+脚注5)
163. 交替路径+最优性证书+Jacobi 1836 (§11.3)
164. 二部匹配 vs 稳定匹配 (§11.3+脚注1)
165. 元组选择+约束结构定复杂度 (§11.4)
166. 约束即边语义+不可行申报 (§11.4 尾)
167. 拆点二部化+三层套娃 (§11.5)
168. 最小教师数+脚注讽刺群 (§11.5 尾)
169. 棒球淘汰=流可行性 (§11.6)
170. 项目选择=最小割 (§11.7)
171. 覆盖类题群 (题7-11)
172. 配额+归属约束题群 (题12-15)
173. 矩阵取整守恒律 (题16)
174. 时间展开图 Neitherlands (题19)
175. 非交叉匹配+不可解性哲学 (题20+ch12 epigraph)
176. 说话狗类比+Cook-Levin 命运
177. 刺青律： 归约方向
178. CircuitSat→Sat→3Sat 编译链
179. MaxIndSet 三角构造： 分散化=独立集
180. 证书变换观： 可逆的是证书不是实例
181. 补图对偶三连
182. gadget 方法论+图着色编码
183. Hamiltonian 双证明： 等价的多路归约
184. base-4 位 gadget: 进位隔离
185. Caveat Reductor! 伪多项式陷阱
186. 弱/强 NP-难： 表示敏感性
187. Karp little list+归约工具箱
188. 选源经验法则： 形状识别
189. 国际跳棋： 合规即 NP-难
190. 复杂度阶梯+Stockmeyer 非初等
191. 魔盒六连： 判定→构造 self-reduction
192. 2 vs 12 相变： “数 3” 两侧
193. 常数花式不改本质 (题16-17)
194. X-3Sat 一次分类 (题15)
195. 戏剧化建模群+Algorithm March (题34-39)
196. NFA 对偶不对称 (题40-41)
197. Charon 摆渡+Alcuin 古典源 (题42)
198. 索引双 epigraph: 递归索引+模糊排序
199. 索引考古： see-also 网络与格言页码
200. open problems 清单+全书收官
201. 索引尾考古： 复盘层
202. 巨人肩上+死人干活 (人名索引)
203. Lovelace 1843+伪码总账
204. 图示之辩 (图源三 epigraph)
205. Billings 四律+广播收播 (Colophon)

---

## 收官注 (未发射算子点单制)

读书线只登记候选, 不自动发射; 用户点名才立项 (kl 卡另行查重+prereg):

1. **势函数重加权审计**: Johnson π(v) 进门礼/出门税——交易成本重加权不改最优路径序; 成本阶梯/融资费/冲击成本的重加权判读不变性申报; 接口 5bp 引用阶梯 (kl166)。
2. **残差-增广调仓器**: 残差图=当前解的边际可改进集; 增广=单步改进方向; 最短增广=最少换腿优先; 接口 kl526 S4 换腿折。
3. **DP→松弛编译纪律**: 加结构参数破自指→去维→显式循环: 窗口算子从递归定义到分块循环的标准编译路径; 接口 common.py 分块+OOM 护栏。
4. **流分解组合会计**: 组合=路径+圈正组合; 无圈流存在=换仓可去闭环; 整流定理=整数份额不损最优; 接口 kl332 等权 E[K] 闭式。
5. **证书往返验证协议**: self-reduction 判定→构造 (逐维试探+oracle 复验); prereg 证书可逆制度化 (双跑 md5=证书往返); 接口 brute 双跑 4709324e。
6. **复杂度度量申报制**: 伪多项式陷阱——回测成本按比特申报 (棒数×票数×数值位宽), 大参数隐性指数申报; 接口 brute 有界点单。
7. **量词交替对抗测试**: QBF=对手轮手——最坏对手 (regime 切换) 下的策略审计而非平均情形; 接口博弈论 master+226 出场机序贯决策。
8. **交互阶相变预算**: 2Sat 多项式/3Sat NP-难——双因子交互可全扫/三因子枚举爆炸的预算申报; 接口 kl524 薄截面护栏+kl337 池级侦察。

---

读序 21 收官。→ 读序 22: 撞线检查后选 (候选 Lyons-Peres《Probability on Trees and Networks》图论/网络概率域, 作者站官方免费, 与 study_prob 基础概率分域; 定谳见 SHELF 更新)
