# study_gf — Wilf《generatingfunctionology》2e 全书精读笔记 (读序 19)

- 书: Herbert S. Wilf, **generatingfunctionology**, Second Edition, Academic Press 1994 (1990 初版)。
- 来源: 作者站官方 Internet Edition www2.math.upenn.edu/~wilf/Downloads/gfology2.pdf (Academic Press 授权教育使用; **只读不上云**——版权书本地处理纪律)。下载判例: upenn 301→www2 后默认 curl UA 403, 须 Firefox 浏览器 UA 直取 (已记入书架行)。
- 底座: pdftotext -layout → 12,038 行 clean txt; 231 页。读毕 2026-11-21。
- 读法: 22 块 × 560 行 (末块 278 行) **零跳读**; 每块 5 条市场映射, 全书 110 条 (见全映射目录)。
- 纪律: 每域单一 master; study 笔记≠kl 卡 (本笔记不进判例链, 候选见收官注)。

## §〇 推导层 (全书骨架的推导链, 不按页序按逻辑序)

1. **表征即答案 (晾衣绳)**: 一个序列 {aₙ} 与一个形式级数 A(x)=Σaₙxⁿ互相唯一决定。求 aₙ 不必"解出"它——把 A(x) 的代数关系拿到手, 系数随取随用。"A generating function is a clothesline on which we hang up a sequence of numbers for display." 全书第一律。
2. **三种口径 ↔ 三种卷积 (口径-结构配对律)**: ops (普通 gf) ↔ Cauchy 卷积 (无标号/有序拼接); egf (指数 gf) ↔ 二项卷积 (标号结构合并: 合并⟺相乘, Fundamental Lemma of Labeled Counting); Dirichlet 级数 ↔ 因数卷积 (积性/数论结构)。选口径=选问题族: 标号计数用 1/n! 吸收标签, 因数结构用 n^{−s} 吸收整除性。同一序列可在多口径下各有 gf, 各答各的问题。
3. **形式/解析双通道**: 形式幂级数环里合法的代数 (无收敛焦虑, "formal power series: no guilt"), 与解析函数论 (收敛半径、奇点、留数) 服务同一对象。多数操作先在形式层算完, 需要渐近/存在性时切解析层——"how easily one can shift gears" (ch5 开篇)。证明内部也常两段: 先形式计算后补合法域 (LIF 证明)。
4. **奇点支配系数尺寸 (渐近第一律)**: R=1/limsup|aₙ|^{1/n}; 收敛圆周上必有奇点; **最近奇点独裁**大 n 系数增长。极点→主部支配 (减掉同圆全主部才扩盘, 系数差指数小); 代数奇点 (z₀−z)^α → Darboux (奇点邻域展开逐项"升"为系数渐近, [zⁿ](1−z)^β∼n^{−β−1}/Γ(−β)); 整函数→Hayman (a(r)=rf′/f 均值, b(r)=方差, 系数正态近似 aₙ∼f(rₙ)/(rₙⁿ√2πb))。增长不是拟合出来的, 是奇点位置读出来的。
5. **指数公式 (标号计数的总闸)**: 牌/牌组/手牌框架——手 = SET(牌)。H(x,y)=e^{yD(x)}。一族应用全部同构: 置换=循环集 (D=log(1/(1−x)), H=(1−x)^{−y} 上升阶乘), 集分拆 D=eˣ−1 (微笑兔子 e^{y(eˣ−1)}), 对合 e^{x+x²/2}, 2-正则图 e^{−x/2−x²/4}/√(1−x), 连通图 vice versa (由全图开方), 有根树 D=xe^D (Pólya, "Press on and keep the faith")。加权 y^k 数牌即双变量 gf, 逐格谱一发。
6. **无标号对偶 (prefab)**: ∏(1−yxⁿ)^{−dₙ} 乘积公式; 限制重数 W 则 1/(1−yxⁿ)→1+yxⁿ+⋯+(yxⁿ)^W。钱币兑换/Frobenius 数/Schur 定理 hₙ∼n^{M−1}/((M−1)!a₁⋯a_M) (主极 x=1 阶 M, "without ever writing it down") + 拟多项式 f(n)=n/ab+(a+b)/2ab+per(n) (周期项均值零)。
7. **矩=导数读出**: 均值 µ=F′(1)/F(1), 方差 σ²={(logF)′+(logF)″}|₁; xDlog 一律。随机变量全矩谱住 gf 的一阶对象上; 期望免费 (筛法中平均性质数=N₁/N), 退化礼貌 (k=1 时应约简)。
8. **筛法=换元**: E(x)=N(x−1)。"replaces the unfiltered data by the sieved quantities"——恰 0 性质/至少/至多一律一式; e_k 恰 k 性质同族 (多项式 e^{−y}Σrⁿyʳ/r!)。指纹: 展开在原点附近而求值在 t=1 附近。色多项式 (N_r=Σλ^{κ(S)})、避子串词、错排、奖券收集全走此门。
9. **反演对**: Möbius (Dirichlet 卷积下 zeta↔µ)、筛法对 (N(⊇S)↔e_k)、二项反演、LIF 版精美对 (b_n=Σ(n−k k)a_k ⟺ na_n=Σ(2n−k−1 n−k)(−1)^{n−k}k b_k)。工具源于实例: n=Σφ(k) 先于 φ 的反演式。
10. **Snake Oil 五步**: (a) 求和式 F(n); (b) 乘 xⁿ 对 n 求和; (c) 交换求和序内层变几何; (d) 内层求和识别已知 gf; (e) 读系数。"don't evaluate the sum—find the gf of the whole family"——整族宣言。失效申报: 自由参数在被求和项中出现太多时卡住; 解套=泛化加参数 m 再 set m=n。
11. **WZ 证书机器证明**: F(n+1,k)−F(n,k)=G(n,k+1)−G(n,k), 有理证书 R(n,k)=G/F; 证书在手验证只需多项式除法 ("no need for blind trust in the computer")。Gosper 算法完备 ("will not ever return 'I don't know'")。连续类比 F_x=G_y (e^{−y²} 自 Fourier)。
12. **Lagrange 反演**: u=tφ(u) 型函数方程不解出直接取系数 [tⁿ]{f(u(t))}=(1/n)[u^{n−1}]{f(u)φ(u)ⁿ}; f 任选 (要未知函数还是要其 k 次幂都行)。Cayley 第三证; 树/函数方程族的标准形。
13. **谱形-零点**: 实负零点⟹严格对数凹 (Rolle 链)⟹单峰; 单位圆上零点则"extravagantly violated"。mod 2 同余读整族奇偶。循环指标 C(x,t)=exp(Σx_j t^j/j) 编码共轭类谱。
14. **不变量常数族**: 1/e (错排/不动点 Poisson(1)/树端点概率 n(1−1/n)^{n−2}/置换开头递减段 e−1)、√5 (Binet)、γ (p(x)∼e^{−γ}/(1−x))、log2 (有序 Bell)。常数是结构族的指纹; 尺寸无关概率 (无小循环 e^{−H_q}, 误差超指数) 是基率稳定性的形式原型。
15. **渐近运算的边界**: ∼ 等式两边不能随便 n 次方 ((1+1/n)∼1 但幂后 ∼e); 处理 (·)ⁿ→exp{n log(·)}; 截断恰好处=乘 n 后仍→0 的项。三体制相图 (ε~n^a / n^{−a} / n^{−a}log n)。
16. **人机分工**: 递推→闭式→gf 三连机器包办 (RSolve "must leave us in awe"); 渐近机器 (Maple asympt) 给项数, 截断需求人定。机器做 routine, 人留判断与验收。
17. **诚实工程学**: 自曝作弊段 ("we cheated"), 勘误署名 (Knopfmacher-Warlimont "corrects an error in the previous printing"), Garsia 私人谈话归功, Katzman 学生解署名, (!!check this!!) 自注照登, 开问题申报 (polyomino gf 未知), 方法失效申报 (Snake Oil 卡住条件), 难度分级诚实 ("unapologetically difficult")。学术文本的验收文化。

## 章节弧线

- **ch1 初步** (块 1-3): Fibonacci 晾衣绳开篇→THE METHOD 六步→[xⁿ] 算子→三次样条边值 (字体判例)→二项/Stirling 两类→Bell 数与 egf 引入 (d₁ d₂ 符号阶乘)。主线: 离散问题连续化, 系数即答案。
- **ch2 级数形式论** (块 4-7): 形式幂级数环 ("no guilt")→ops/egf 演算规则 1-5/100-200→Cauchy 乘积=拼接→移位算子谱系→块喷泉判例→双口径成本表→Catalan 二次泛函→错排 1/e→解析理论三定理 (奇点即天花板)→limsup 双臂→单位根滤波→§2.5 参考表文化→Dirichlet 第三口径→ζ→Euler 积"三个 2"→Möbius µ 与反演→本原串→分圆多项式 Φ₁₂ (多项式幻)。
- **ch3 牌/牌组/手牌** (块 8-11): 指数族框架→**定理 3.4.1 指数公式**→合并⟺相乘→巫师学徒三步→置换=循环集→微笑兔子→对合→2-正则图→连通图 vice versa+xDlog 递推→二色化开方→Pólya 加点与 Cayley→binomial type→prefab 无标号→钱币兑换 conductor→Schur 定理→Euler 奇=离→有根树 formidable 方程。
- **ch4 矩与筛** (块 12-16): 矩=导数→调和律→筛法=换元→boxed (A)-(D)→错排 Poisson 化→车攻→"无 n 问题"→Snake Oil 五步全家桶→反演对→WZ 对与证书→Pfaff-Saalschütz/Dixon→对数凹与零点→mod 2→循环指标→平方根偶性→polyomino 开问题→HC 递推→ECS 覆盖序列判据; 习题 28 题金矿 (奖券收集/树上首返/Bonferroni/Cayley 度规格化 n/e/WZ 连续类比/exp_S 一般化/Bernoulli 判据 [Fr])。
- **ch5 解析与渐近** (块 16-19): 换挡宣言→**LIF** (证明+两例)→极点/主部支配→有序 Bell log2→无小循环 e^{−H_q}→Darboux (引理链+2-正则图)→Szegő 多奇点→Hayman (e^z 粗界→Stirling 精确→容许性 (A)-(E)→对合 tₙ 全流程含幂陷阱); 习题 (S_p(n) 七部曲/循环≤3 置换 LIF+Hayman 接力/对合平均循环 n/2+½√n)。
- **附录+解答层** (块 19-22): Maple/Mathematica (RSolve 敬畏/机器渐近)→全书解答=第二遍复习, 且方法故意换轨 (解析证→双射证, 双证文化); Frobenius 29 手证; 拟多项式定整; Katzman 署名; (!!check this!!) 照登; References 分层地图。

## 方法论指纹 (Wilf 式)

- 比喻负载教学: 衣夹 (clothespin)、晾衣绳、微笑兔子、巫师学徒 (trickle→flow→flood)、"formal beast"、同型火车 ("same train of manipulations")。
- 诚实段: 自曝作弊与改题、"which we won't"、难度分级、"unapologetically difficult"。
- 归功精确: Garsia (private conversation)、Katzman (学生)、[KnW] 自引短证、前版勘误署名。
- 参考表文化: §2.5 常用级数表 (12 条) + 文献志每方向指一本。
- 习题即正文: 28 题金矿承载一等方法 (Cayley 度规格化/WZ 连续/exp_S/Bernoulli 判据), 解答层复现双证。
- 开问题申报: polyomino gf 未知 "despite great deal of effort"——边界画线不装全知。

## 全书覆盖表 (诚实条款)

| 块 | 行 | 内容 | 状态 |
|---|---|---|---|
| 1 | 1-560 | 卷首/Preface/目录/ch1 至 Fibonacci 闭式 | 读毕 |
| 2 | 561-1120 | 样条边值/二项/Stirling/Bell/egf 引入 | 读毕 |
| 3 | 1121-1680 | Stirling 显式/Bell/egf 规则/形式环 | 读毕 |
| 4 | 1681-2240 | ops/egf 规则体系/Catalan/错排/解析三定理 | 读毕 |
| 5 | 2241-2800 | 单位根滤波/§2.5 表/Dirichlet 开篇 | 读毕 |
| 6 | 2801-3360 | ζ/Euler 积/Möbius/分圆 | 读毕 |
| 7 | 3361-3920 | ch3 指数公式主定理+证明 | 读毕 |
| 8 | 3921-4480 | 指数公式应用群/prefab | 读毕 |
| 9 | 4481-5040 | Schur/Euler 奇=离/有根树方程 | 读毕 |
| 10 | 5041-5600 | ch4 矩/调和/筛法开篇 | 读毕 |
| 11 | 5601-6160 | 筛法五例/Snake Oil 主体 | 读毕 |
| 12 | 6161-6720 | 反演对/WZ 对+证书 | 读毕 |
| 13 | 6721-7280 | Pfaff-Saalschütz/Dixon/对数凹/mod 2 | 读毕 |
| 14 | 7281-7840 | 循环指标/平方根/polyomino/ECS 开篇 | 读毕 |
| 15 | 7841-8400 | ECS 判据/覆盖序列/ch4 收官 | 读毕 |
| 16 | 8401-8960 | ch4 习题 1-28/ch5 开篇/LIF 开题 | 读毕 |
| 17 | 8961-9520 | LIF 定理证明+两例/极点主部/有序 Bell/无小循环 | 读毕 |
| 18 | 9521-10080 | Darboux/Szegő/Hayman/对合全流程 | 读毕 |
| 19 | 10081-10640 | ch5 习题/附录 Maple-Mathematica/ch1 解答开篇 | 读毕 |
| 20 | 10641-11200 | ch1-2 解答 (Faulhaber/反演工具箱)/ch3 解答开篇 | 读毕 |
| 21 | 11201-11760 | ch3 解答 (Katzman/e^{−γ}/拟多项式)/ch4 解答开篇 | 读毕 |
| 22 | 11761-12038 | ch4-5 解答收官 (泛化解套/Cayley 五部曲)/References/EOF | 读毕 |

**诚实条款**: 全书正文+习题+解答+附录+文献志 22 块零跳读; 每块笔记含两条原文核心句逐字引用。索引/扉页/版权页未逐行读 (无技术内容); 数值表只抽验 (有序 Bell 541.002、polyomino f(12)、Stirling 五项) 未全算。
**勘误记档**: (i) pdftotext 伪影——连分数/级数排版的空格错位与 ϵ/ε 混排为转换伪影, 非原书错误, 阅读时按数学惯例重排; (ii) 原书自带勘误已由作者申报 (Knopfmacher-Warlimont 条目 "corrects an error in the previous printing"), 照录不重复。

## 全映射目录 (110 条, 判例桥接)

1. 晾衣绳表征 (系数即身份) · 2. 双通道宣言 (离散×连续) · 3. 渐近代偿 (无闭式→近似式) · 4. 部分分数速定 (cover-up 消去) · 5. THE METHOD 六步 (协议化) · 6. Binet 主导项定律 (近似→更简精确) · 7. 边值两端未知数 (根代入钉死) · 8. 字体判例 (spline 存储) · 9. 两堆分解 (含n/不含n) · 10. 选对求和变量 (麻烦因子变常数) · 11. 公式自知边界 (适用域内嵌) · 12. 指数型引入 (标签用 1/n!) · 13. xD log 四步 (对数化乘为加) · 14. Cauchy 乘积=拼接 · 15. 形式级数无罪 (先代数后收敛) · 16. 衣夹符号论 (x 无值) · 17. 卷积反射律 (见形即触发) · 18. 部分和算子 (除 (1−x)) · 19. 移位算子谱系 · 20. 喷泉判例 (结构约束→隔代同列) · 21. 双口径成本表 (ops/egf 分治) · 22. Catalan 自嵌套 (二次泛函) · 23. 错排极限 (→1/e) · 24. 奇点即天花板 · 25. limsup 双臂定义 · 26. 单位根滤波 (等差下标抽取) · 27. 参考表文化 · 28. Dirichlet 第三口径 · 29. ζ 新生物判例 · 30. 素数幂自由+乘性 · 31. Euler 积选择唯一 (归因强制) · 32. Möbius 反演 (聚合→隐藏分量) · 33. 本原串剥离 · 34. log 化积为和 · 35. 染色-方差审计 · 36. 指数公式 SET 构造 · 37. 合并⟺相乘 · 38. 巫师学徒三步 · 39. 循环=腿 · 40. 表示＞闭式 · 41. 微笑兔子 (零信息维自由度) · 42. vice versa 反推 (xDlog 递推) · 43. 二色化-开方 · 44. Pólya 加点 · 45. Press on and keep the faith · 46. 主极独裁 · 47. 顺手取 c · 48. conductor 冷启动 · 49. 周期项均值零 · 50. Euler 奇=离一行证 · 51. 矩=导数读出 · 52. 循环调和律 · 53. 筛法=换元 · 54. 期望免费+退化礼貌 · 55. 泊松化渐近 · 56. 整族 gf 宣言 · 57. 同一列车 (computer trainable) · 58. 筛法指纹 (双展开诊断) · 59. Fibonacci 复活 (old friend) · 60. 多项式幻 (e^{−y} 截断) · 61. WZ 证书 (可审计机器证明) · 62. 不求值证相等 · 63. 特例难过一般 · 64. 域外 unfazed · 65. 闭式问题学 (Gosper 前奏) · 66. 完备算法 (无"不知道"档) · 67. 零点定谱形 (Rolle 链) · 68. mod 2 读同余 · 69. 循环指标 · 70. 无 n 问题 (1/e 型不变量) · 71. Poisson 渐近独立 · 72. 平方根偶性判据 · 73. 勘误申报 · 74. 开问题申报 · 75. 覆盖序列判据 · 76. 换挡宣言 (晾衣绳→解析) · 77. Bonferroni 夹逼 · 78. 奖券收集 (d·H_d) · 79. 树上首返 (端点 2n 内点 n) · 80. Cayley 度规格化 · 81. Lagrange 反演 (隐式腿直读) · 82. 主部支配 · 83. 尺寸无关概率 (e^{−H_q}) · 84. 模仿奇性策略 · 85. Cayley 三证合流 · 86. Darboux 升降律 · 87. 渐近幂陷阱 ((1+1/n)ⁿ∼e) · 88. Hayman 整函数法 (正态峰) · 89. 对合三件套 · 90. 容许性闭包 ((A)-(E)) · 91. 人机分工 (RSolve 敬畏) · 92. 最小项数自觉 · 93. formal beast 双证 · 94. 平均循环修正项 (n/2+½√n) · 95. 幂陷阱相图 (三体制) · 96. 双证文化 (奇=离双射证) · 97. Faulhaber 显影 · 98. 鸽笼下界 · 99. 反演工具箱 (φ/λ/squarefree) · 100. 递推直读 · 101. 算子定制戏法 · 102. 归功文化 (Katzman) · 103. 通用常数 e^{−γ} · 104. 拟多项式定整 · 105. 方法自省 (筛法 vs 递推) · 106. 泛化解套 (set m=n) · 107. 平均度剪叶 (2−2/n<2) · 108. 筛法期望元律 (N₁/N) · 109. 文献志分层 · 110. 收官全景 (三主线收束)。

## 收官注

全书=组合数学的"表示论": 不解方程拿系数 (LIF), 不求和拿恒等 (Snake Oil/WZ), 不闭式拿渐近 (奇点支配), 不数拿计数 (指数公式/筛法)。对判例线的最大馈赠是**方法论层**: 表示＞求值 (gf 拿关系不硬解), 双口径分治 (iso/brd 的数学原型在 ops/egf/Dirichlet 三口径), 误差预算的方向先知 (Bonferroni/截断恰好处), 机器+人分工的验收文化 (WZ 证书=可审计机器证明)。

读书线只登记候选， 不自动发射； 用户点名才立项 (kl 卡另行查重+prereg):

1. **奇点支配渐近移植**——判例线的容量/衰减轴 (kl331/kl334 容量梯) 用"最近奇点定增长"重读: 死签衰减率与租增长率的奇点解释, 是否统一容量梯 15× 与死亡梯 47.9× 两族。
2. **指数公式组装语义**——kl327-330 组装家族 (日闸门引擎+股选择装饰+避雷保险) 的 SET(牌) 形式化: 腿=牌, 引擎=牌组, 策略=手; 合并⟺相乘给组合层数学身份。
3. **筛法双展开指纹**——谱卡方法论的形式化: E(x)=N(x−1) 换元后"原点展开 vs t=1 求值"的分离, 对应判例 by-year 与 by-格双展开的诊断签名。
4. **Bonferroni 方向界**——误差预算表升级: 交错和截断的偶高奇低方向律, 给逐格置信区间的先验方向 (替代对称 ±)。
5. **拟多项式定整**——周期摆动 <数据粒度 ⟹ 主项反解精确值的骨架; 判例线周期项 (日历效应禁区) 的正律镜像: 何时周期项可忽略、何时必须显式。
6. **WZ 证书化 brute**——双跑 md5 → 有理函数证书: 判例断言从"两次运行一致"升级到"证书可独立复核", 验收文化的数学化。

读序 19 收官。→ 读序 20: 选点待 shelf 域空白检查 (备份候选: Hastings-Dekking《Introduction to Probability》Dartmouth 官方版; 先 grep 书架避免撞线)。
