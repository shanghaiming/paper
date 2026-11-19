# papers/ 学习笔记声明可溯性验证地图

- 验证人: notes-verify agent, 2026-11-13。过程底稿: `.scratch_notesverify.md` (逐笔记全量记录, 含证据行号)。
- 验单: papers/ 下 24 份自产研究笔记 (all_deep_analysis.md 已定谳跳过; 文本源 txt 不在验单)。
- 抽样: 每份 5-13 条声明级断言 (精确数字/直引/归属/计数), 共 382 条。
- 分类: **A** 本地源可回溯 / **B** 声称有源本地无源 / **C** 无源断言 (all_deep_analysis 病) / **D** 方法论免检。
- 合计: **A 199 / B 101 / C 18 / D 64**。C 分布于 13 份; 10 份非空笔记 C:0。

---

## 一、总表

| # | 文件 | A | B | C | D | 一句话判词 |
|---|---|---|---|---|---|---|
| 1 | behavioral_layer_20261020.md | 8 | 6 | 1 | 3 | 骨架极硬 (KT1979/Odean/BO2000 PDF 逐字命中 + registry 逐位), 病灶仅 Odean1999 一处外围数字 |
| 2 | essence_synthesis_2026030.md | — | — | — | — | **空文件 (0 字节)**, 疑似 20260930 的笔误残档, 建议删除/归档 |
| 3 | essence_synthesis_20260930.md | 9 | 8 | 1 | 3 | 内部数字密度极高且几乎全可溯 (9/9), 病在 1478 + 10-24/25 "手测"系列声称落档但检索不到 + 引用作废频谱 |
| 4 | internal_evidence_essence.md | 12 | 3 | 1 | 1 | 内部锚定最干净的笔记之一 (12 条抽验 11 中), 唯一硬伤是 1478 |
| 5 | kahneman_chapter_map.md | 12 | 3 | 0 | 2 | 引语纪律全目录最佳 (10 条原文逐字命中), 病灶仅 miner 状态引用 (F0033/F0037 与登记簿不符) |
| 6 | market_essence.md (2MB, 06-02) | 1 | 14 | 2 | 4 | **最重病例**: 1478 源头 + §8/§22-32 引用 14+ 篇 2026 arXiv 无一在库 + Hansen&Lunde 双重转引; 前 21 节数学推导自洽 |
| 7 | market_essence_cross_market_20261112.md | 10 | 0 | 0 | 2 | **最干净**: 抽验 10 条全部 futures_platform 逐位命中, 零 B 零 C |
| 8 | market_essence_from_data_20261107.md | 12 | 1 | 0 | 2 | "从数据读出"声明兑现, kl027→kl030 全数字 fleet_journal 逐位命中 |
| 9 | market_essence_v2_mechanisms.md | 7 | 14 | 0 | 3 | 诚实标注纪律标杆 ([快照]/[未证实] 自我声明), B 群全部带出处, 是"知道自己没读过什么"的对照样本 |
| 10 | math_communications_20261112.md | 8 | 3 | 1 | 4 | 数学搬运质量高 (Tier 分级), 但 M5 立项第一的 "周频 70 vs 日频 51" 引用已作废频谱 — 实验前提失真 |
| 11 | math_control_theory_20261112.md | 6 | 2 | 1 | 4 | 同病更重: 作废频谱被升格为 "实测" 且充核心命题3的验证性证据 |
| 12 | math_semiconductor_physics_20261112.md | 5 | 3 | 0 | 4 | 诚实纪律极强 (429 限流项不编 URL), 唯倒U +86/−9 系被同日改写的旧版读数 (版本漂移) |
| 13 | open_source_strategy_census_20261014.md | 8 | 10 | 0 | 3 | 情报官报告标杆: 明示对照底账 registry 且兑现, §7 诚实申报覆盖全部不可验项 |
| 14 | philosophy_coda_ledger_identity_20261025.md | 12 | 4 | 1 | 2 | "哲学笔记却全部可溯"的罕见样本 (17 条抽验 12 中 iter332-336 链), 唯 "300+ 封棺" 聚合计数无锚 |
| 15 | philosophy_layer_20261021.md | 9 | 5 | 3 | 2 | 判例与老庄引文全可溯, 但 3 处 C: 1478 + 300+ + h0 0.018→h365 0.170 全域零命中 |
| 16 | philosophy_quanxue_notes.md | 10 | 2 | 2 | 2 | 引文纪律最佳之一 (7 条古典全部镜像逐字), 但 1478 第五次复发 + 作废频谱被引为 "不折腾的数学" |
| 17 | practitioner_playbooks_2026.md | 3 | 10 | 0 | 3 | 82 URL 逐条挂源 + 附录C 证据等级 + [传闻]/[无公开证据] 标注 — 纯外部调研的诚实范本, 病在本地不可复核非虚构 |
| 18 | probability_theory.md | 6 | 1 | 2 | 4 | 关键定性: 与 tv_learning 原件 **md5 逐字节一致** (血统最硬), 病仅两个无源范围数 (50-80%/50-200%) |
| 19 | psych_books_to_operators_20261107.md | 12 | 3 | 0 | 2 | **可溯性最佳**: 20 行台账逐格抽验 13 行全部 fleet_journal+工件逐位吻合 |
| 20 | quanxue_philosophy_map_20261112.md | 11 | 1 | 0 | 3 | 映射最严格范本: 引句全镜像命中 + Tier 分级 + 主动拒绝清单 + 文本反调记录 |
| 21 | research_synthesis_400_20261107.md | 8 | 2 | 0 | 2 | 被 fleet_journal L4563 正式注册的综合备忘, 抽验 10 条证据链全落锚 |
| 22 | study_program_20261112.md | 9 | 2 | 1 | 3 | 三轨纲领自查枢纽, 五产物交叉引用与本验证全吻合, 唯 "12 节" (实 26 节) 引用失准 |
| 23 | theory_essence_20261017.md | 10 | 3 | 1 | 3 | 理论深读诚实标杆 (公式逐字抄录核验 + 扫描版申报 + [自推导核对]/[转引未核]), 唯 2015-07 接刀 −18.9% 孤数 |
| 24 | universal_method_v2.md | 11 | 1 | 1 | 3 | registry 注册的 "用户核心交付", 锚密度全场最高 (11 项数字全中), 唯 1478 继承 + iterV1 列位错置 |

---

## 二、C 类明细 (18 处 / 13 份) — 最重要

### C1. "1478 篇" 虚构计数链 (6 节点, 全目录最重的系统性病)

同一无源计数 "papers 库 1478 篇 (纯预测范式全灭)" 在 6 份笔记中复现, 全仓库无任何锚:
实际 papers/pdfs 231 篇 + pdfs_theory 27 篇 = 258 篇; all_deep_analysis.md 自称 651 篇 — 三个数字互相矛盾, 1478 与现实数量级都不符。

| 节点 | 文件 | 原文 (摘) |
|---|---|---|
| 源头 | market_essence.md (头部+文内 5 处) | "从 1478 篇金融工程论文中提炼" |
| 2 | essence_synthesis_20260930.md | "papers 库 1478 篇" |
| 3 | internal_evidence_essence.md | "papers/pdfs 的 1478 篇" |
| 4 | philosophy_layer_20261021.md | "papers 库 1478 篇" |
| 5 | philosophy_quanxue_notes.md | "纯预测范式 (papers 库 1478 篇) 全灭" |
| 6 | universal_method_v2.md L195 | "ML 全线 −34~−22% (1478 篇预测范式)" |

(注: "ML 全线 −34~−22%" 本身有 registry 锚, 是 A; 病只在 1478 计数。)
**修正口径建议**: 改为 "本地 258 篇库 + 全部 ML 迭代判例 (registry iter171-196 等, 全线 −34~−22%)" 或引用 strategy_knowledge.md 的 216 策略验尸。

### C2. 作废频谱被当活锚 (周频 70% vs 日频 51%)

`quant_trade-main/docs/attack_plan_200pct.md` L143 于 **2026-10-25 (iter332/333 终审)** 将简化引擎频谱 (固定 21/季 35/日折算 51/周频含成本 70/周频毛 88/纸面 232) 整体标为 **"[作废] (100% 可买性假设的产物, 勿再引用)"**, 定版改为物理引擎 M≥5 = 44-47/−22/零负年。以下笔记 (日期均晚于作废 18 天) 仍将其作活数字引用:

| 文件 | 原文 (摘) | 危害 |
|---|---|---|
| math_communications_20261112.md | "周频 70% vs 日频 51%, 19pp 缺口" (M5 混叠相位实验, 立项排序第 1) | 实验判据前提失真 |
| math_control_theory_20261112.md | "周频(70%) > 日频(51%), 差 19pp, **实测**" (核心命题3 的验证性证据 + §3.4 目标函数实测基础) | 危害最高: 被升格为定理级支撑 |
| philosophy_quanxue_notes.md | "周频再归一 70% vs 日频折算 51% (不折腾的数学)" | 哲学结论垫在作废数字上 |
| essence_synthesis_20260930.md L177 | "70% 为 Q×2 简化引擎的最诚实全史实盘数字" | **与 attack_plan 同日定版直接矛盾** (该档自称"最高权威—覆盖以上全部历史菜单数字") |

同型病第三例 (版本漂移): math_semiconductor L9 "恐慌日跌 5-7% 反弹峰 +86bp, 7%+ 翻负 −9bp" — 系被同日 (2026-11-12) miner_2 改写前的旧版倒U读数; IDEAS.md L150-157 执行版为 "d_c≈9.2-9.5%=跌停板非 7%, 桶梯 +54/+174/+18/−42, 恐慌日 +53.6bp vs 平静日 −23.1bp" 并明记 "原 7% 判读是粗桶伪阈值"。

### C3. 其余孤立无源断言 (8 处)

| 文件 | 原文 (摘) | 为什么是 C |
|---|---|---|
| behavioral_layer | "Odean 1999: 换手 1.5×/年, 交易使净收益每年少 2.65pp" (自称 "PDF 已精读") | 本地 Odean_1999_Trade_Too_Much.pdf 全文无 2.65 亦无 1.5×; 实际读数为月换手 6.5%、买-卖组合月超额 −0.234% — 记忆漂移型虚构 (all_deep_analysis 病的轻症) |
| market_essence.md | "Hansen & Lunde (2006): 超过 40% 的高频金融数据包含微观结构噪声" | 经 2602.06394 转引, 而 2602.06394 本身非本地 — 双重不可溯 |
| philosophy_coda | "程序描述为否证运动 (300+ 封棺)" | registry 全文 "封棺" 仅出现 191 次, 无任何条目给出 300+ 计数; 按被杀配置累计方向可辩护但作为陈述数字无锚 (未申报口径的聚合计数) |
| philosophy_layer | "300+ 次封棺" | 同上, 第二处 |
| philosophy_layer | "港股冷池 h0 0.018 → h365 0.170 回复" | 全域 (registry/fleet_journal/JOURNAL/docs/papers/us_stock_platform/miner_2/futures_platform) 检索**零命中** — 全目录最孤立的无源精确双数字 |
| probability_theory | "RMT 去噪后 Markowitz 权重波动减少 50-80%" | 教材方向对, 范围数无锚 |
| probability_theory | "秩方法在 A 股小盘股优势 50-200%" | 4.5% 最坏损失是 Pitman ARE 定理推论, 但 50-200% 是裸范围数 |
| theory_essence | "(2015-07 接刀 −18.9% 靠出场机雕平)" | 全档案无 −18.9%; 同月最近档案读数是最差日 −11.1 / 最差 5 日 −29.5 — 语境相容数字不符的孤数 |
| study_program | "probability_theory.md (12 节…)" | 实际 26 节 — 对本地资产的失真描述 (引用失准, 轻症) |

### C 边缘已改判记录 (不计入 18)

- "往返成本 ~31bp" (三份 math 笔记): 原判 C, 后在 `miner_2/scripts/mi_upper_bound.py` L102 (`cost_floor="31bp/往返 成本地板"`) + IDEAS.md L199/L173-174 (M6 注册) 找到双重硬锚, **改判 A**。检索教训: 必须扫 scripts/*.py 硬编码, 不能只查 docs。
- universal_method_v2 "5,507 剔除事件": 与 iter594 的 8,042 不冲突 — 5,507 是 iter252 自报 A 股剔除事件数 (`_output/iter252_index_events.md` L3 逐位), 8,042 是 fleet 后重建的 code×effective 口径。**判 A**。
- philosophy_coda "iter263 幸存者 (美股 94% 幻影)": 94% 数字 A, 但 registry L349 定性为 "反向拆股漏检 (非幸存者)" — A 带定性漂移。
- universal_method_v2 L27 "美股" 格 "VRP 已判薄 (iterV1)": iterV1 是商品期权 (期货域, registry L189), 应归期货列 — 内容真实, 列位错置。

---

## 三、B 类明细 (101 处) — 缺什么源, 去哪补

B 类绝对量大的结构性原因: papers/ 是"学习笔记"层, 大量外部书/论文/研报不在本地库。其中大部分**带出处申报或诚实标注** (v2 的 [快照]、math_* 的两档 URL 标注、playbooks 的证据等级), 属可接受的 B; 真正需要补源的是以下缺口:

### B-1. 高频被引但本地缺库的源头 (优先补)

| 缺失源 | 被引处 | 补源建议 |
|---|---|---|
| **TK1992** (λ=2.25/α=β=0.88/γ=0.61/δ=0.69) | behavioral_layer / philosophy_layer / psych_books (3 处) | 目前只能经 BHS2001 PDF L419 二级引用 ("follow TK1992 in setting λ=2.25"); 下载 Tversky & Kahneman 1992 JRU 原文入 pdfs_theory 即闭环 |
| **BLO 2009** (台湾散户 −3.8pp/GDP 2.2%) | market_essence_v2 (自称 "PDF 已下载并核对") + essence_synthesis | v2 自称已下载但 pdfs/ 无此篇 — 下载件未入库存; 入库或改标 |
| **George-Hwang 2004** (52 周高点) | behavioral_layer **声称 "本地已有 PDF"** — pdfs/ 与 pdfs_theory/ 均无 | **虚假本地性声明, B 类中最重一条**; 补 PDF 或删声明 |
| Zuckerman《The Man Who Solved the Market》(Medallion 66% 毛/39% 净) | v2 / playbooks / essence_synthesis | 书目, 无 PDF; 保留 URL 转引即可, 但不应出现在 "已核验" 语境 |
| Barber-Odean 2008 "All That Glitters" | behavioral_layer | 本地无此 PDF (有 BO2000) |
| Freeman-Shor (1866 笔/30,874 交易/49% 盈利) | psych_books | 书不在库, 来源节自列网络书评 |
| Korzybski 1933 / Simmel《货币哲学》/ Popper / Taleb / 康德 | philosophy_coda / philosophy_layer | 哲学归名, 无直引要求 — 可保留但建议标 [归名未直引] |
| 中文文献群 (冯旭南 2016 / 武佳薇 2020 / 赵学军-王永宏 2001 / 刘莉亚 2020 / 肖欣荣 2024 / Chen et al 2007) | behavioral_layer | 均称 "检索核对" 无本地源; 补 PDF 或改标转引 |
| S&P 纳入溢价 ~0.8% (美股读数) | universal_method_v2 L24 | 本地只有 iter252 的 A 股 IN 日二手转述; 补 Harris-Gurel 1986 / Shleifer 1986 / Greenwood-Sammon 2021 |

### B-2. market_essence.md 的 arXiv 引用群 (14+ 篇, 单文件最大 B 簇)

§8 表 8 条 "真正发现" (Shi&Zhou / Park 2406.08448 / Carr 008 / Nikolopoulos 2604.15531 / Fouque 009 / Kim 2605.13407 PRISM-VQ RankIC=0.065 等) 与 §22-32 十余节 "论文来源:XXXX.XXXXX" (自由能-熵对偶 / Kelly 因子分解 / IV 量子本质 94 特征 13 维流形 / 质量感知分词 Sharpe+30% / 微观结构噪声 40% 等) — **编号体系与现行 index_rebuild 不匹配, 论文均不在本地, 从未在本地存在过** (文件 6 月成稿, 早于 11-12 索引重建, 但当时库存即现 231 篇)。处置: 整体挂横幅 (见四), 逐条核验成本过高且引用语境 (因子名/表格/精确数字) 表明多半是生成性内容。

### B-3. "10-24/25 主会话手测" 系列 (essence_synthesis, 声称落档但无档)

倒U律四桶 (+37/+51/+86 峰/−9 翻负)、167 恐慌日、三段式 (d1+36/d2-5 真空/d6+49/d10+35)、周五事件 +166bp、孤岛收割 +20bp、复牌跳空 n=5880 中位 −0.7%、部署率 37%、B20 上行 647/下行 380bp — 笔记称 "全部落档", 但 registry/fleet_journal/docs 全文检索不到对应档案。其中部分主题在别处有同型档案 (倒U已由 miner_2 重做并改数, iter336 有复牌跳空 +3.8%/−3.1%) — 处置: 挂横幅, 数字以档案版为准。

### B-4. 其余 (轻, 多有诚实标注)

- 引用计数群: 6300+/7500+/1050+/8000+/8400/16000+/2300+/600+ (behavioral_layer / v2 / market_essence) — 无源计数, 建议删或改 "引用量级".
- 媒体/研报快照: 光大/中金/浙商研报数字、雪球/集思录、SSE 年鉴 82.01%、雪球 DMA、2015 两融 2.27→1.44 万亿、南向 24.2%、指增 1.9-3.6% — v2 已用 [快照] 自我标注, 保留标注即可.
- miner 状态漂移 (kahneman_chapter_map): F0033 标 "已活 (IC 0.035)" 实录 0.0313/Tier1 FAIL→观察区; F0037 标 "重验中" 实录 round_011 已判死 — 需小修 (这是唯一 "内部档案在库但笔记写错" 的 B 型).
- 半锚细节: kl021 笔级 / kl020 有效权重 6.67% / VRP corr 0.71 / ATR 12-17 / 602 列因子库 / "租金薄 10 倍" / "w=1 记号" / iter277 "失去耐心的债主" 文学化转写 — 工件可复核或方向有档, 低风险.
- math_* 三份的 URL 表 (27+25+~50 条): 自标 "检索于 2026-11-12" / "未逐条打开全文复核" / "凭书目知识给出" — 标注诚实, 属可接受 B.
- playbooks 10 条: 全部带 URL 出处 + 证据等级, 其中 Lou-Polk-Skouras 实与本地 pdfs_theory 同篇 (可升 A).

---

## 四、处置建议

### 4.1 需要修正横幅的笔记 (按严重度排序)

**第一梯队 (数字虚构/作废锚, 必须横幅)**

1. **market_essence.md** — 1478 源头 + 14+ arXiv 无一在库 + Hansen&Lunde 双重转引。横幅建议: "§8 与 §22-32 的论文引用未经本地核验 (含编号与现行索引不匹配), 前 21 节数学推导可读; 引用前须逐条溯源"。
2. **philosophy_layer_20261021.md** — 3 C: 1478 + 300+ + h0 0.018→h365 0.170 (全域零命中)。横幅: "1478 为虚构计数; h0/h365 回复双数字无档案出处"。
3. **math_control_theory_20261112.md** — 作废频谱升格 "实测" 且充核心命题3证据。横幅: "周频 70/日频 51 系 attack_plan L143 已作废频谱 (2026-10-25), 定版口径为物理引擎 44-47; 命题3 实测基础需重述"。
4. **math_communications_20261112.md** — M5 混叠实验以作废 19pp 为判据。横幅同上口径, M5 立项需复核前提。
5. **essence_synthesis_20260930.md** — 1478 + L177 与 attack_plan 作废判定直接矛盾 + 10-24/25 手测系列无档。横幅: "1478 虚构; 70% 系作废频谱; 10-24/25 手测数字未见档案, 引用前以 registry/iter336/miner_2 IDEAS 版本为准"。

**第二梯队 (单一虚构计数/孤数, 轻横幅或行内修)**

6. philosophy_quanxue_notes.md — 1478 (node 5) + 作废频谱 "不折腾的数学" 一句。
7. internal_evidence_essence.md — 仅 1478 一处 (其余 12 条抽验 11 中, 内部锚极干净)。
8. universal_method_v2.md — 仅 1478 一处 (L195) + iterV1 列位错置 (VRP 应归期货列)。
9. behavioral_layer_20261020.md — 仅 Odean 2.65pp/1.5× 一处 (建议行内改档案读数: 月超额 −0.234%)。
10. math_semiconductor_physics_20261112.md — 倒U +86/−9 版本漂移横幅: "倒U数字以 miner_2 IDEAS L150-157 执行版为准 (d_c≈9.5%, 桶梯 +54/+174/+18/−42)"。
11. philosophy_coda_ledger_identity_20261025.md — "300+ 封棺" 改 "百余次封棺判例 (registry '封棺' 191 处)" 或申报口径; iter263 "幸存者" 改 "反向拆股漏检"。
12. theory_essence_20261017.md — −18.9% 孤数: 改档案读数 (最差日 −11.1/最差 5 日 −29.5) 或删。
13. probability_theory.md — 两个范围数标 "教材方向性表述, 未经本地核验"。
14. study_program_20261112.md — "12 节" 改 "26 节"。
15. kahneman_chapter_map.md — F0033/F0037 状态行按 miner_2 REGISTRY 实录修正 (IC 0.0313 观察区 / round_011 判死)。

**零碎处置**

16. essence_synthesis_2026030.md — 0 字节空文件, 直接删除或改名 .empty 归档。
17. market_essence_v2_mechanisms.md — 不需横幅; 建议仅把 "George-Hwang 本地已有 PDF" 类虚假本地性声明清掉, 并统一 §1/§15 的 8000+/8400 两值。

### 4.2 干净笔记 (无需横幅, 可作诚实范本引用)

- **market_essence_cross_market_20261112.md** — 零 B 零 C, 全部落 futures_platform。
- **market_essence_from_data_20261107.md** — kl 链逐位命中。
- **psych_books_to_operators_20261107.md** — 台账逐格吻合, "书是假设不是证据" 纪律。
- **quanxue_philosophy_map_20261112.md** — Tier 分级 + 主动拒绝清单。
- **theory_essence_20261017.md** — [自推导核对]/[转引未核] 标注 (除 −18.9% 一处)。
- **open_source_strategy_census_20261014.md** — §7 诚实申报。
- **research_synthesis_400_20261107.md** — 正式注册 (fleet_journal L4563)。
- **practitioner_playbooks_2026.md** — 证据等级附录 (外部源 B 群属其体裁本性)。
- **probability_theory.md** — 副本血统 (md5 与 tv_learning 原件一致), 除两个范围数。
- **universal_method_v2.md** — 除 1478 一处外锚密度全场最高。

### 4.3 系统性观察 (给主会话的三条结论)

1. **病是局部且可定位的, 不是弥漫的**: 382 条抽验里 C 仅 18 (4.7%), 且高度聚集 — "1478" 一条链就占 6 处, 作废频谱占 4 处。内部实验数字 (registry/JOURNAL/fleet_journal/_output 工件) 的回溯率非常高 (essence_synthesis 9/9, internal_evidence 11/12, psych_books 13/13, universal_method 11/11) — 与 all_deep_analysis "没读过的虚构" 不同, 这批笔记对**自己做过的事**基本诚实, 病集中在**对自己没做过的事的转述** (外部文献数字、库规模计数、被作废的旧读数)。
2. **最有效的三个补丁**: ① 全局搜替 "1478" → 实际口径; ② 作废频谱 70/51 → 44-47 物理引擎口径 (4 份笔记); ③ 倒U → IDEAS L150-157 执行版。三个补丁覆盖 11/18 处 C。
3. **诚实范本已在库内**: psych_books (逐位台账)、quanxue_philosophy_map (拒绝清单)、theory_essence (标注分级)、playbooks (证据等级)、v2 ([快照]/[未证实]) 展示了五种可复制的自我申报模式 — 修横幅时可直接引用这些约定作为标准。
