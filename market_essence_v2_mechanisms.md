# 市场的机制学本质 v2 — Alpha到底从哪来, 谁把钱输给谁

> 机制学/经济学路线, 补 `market_essence.md`(v1, 数学哲学路线)之缺
> 调研方法: 33次独立网络检索 + 5份论文原文直接核验(NBER/加州大学伯克利/arXiv/上交所年鉴转引)
> 2026-09-30 | 机制学调研员(舰队)
> 标注约定: 未能在本次调研中直接打开核对原始页面的引用标 [快照](经搜索引擎摘要转引) 或 [未证实]; 其余URL均在本次调研中返回过内容。

---

## 0. 执行摘要(先给结论)

1. **每一分alpha都对应一个可识别的付钱方**。学术文献里所有可持续的收益来源只有五类: ①行为性(彩票偏好/锚定/处置效应——散户为心理满足付费), ②结构性被迫交易(指数再平衡/基金申赎/两融强平/雪球敲入——被规则或合同强迫的人付费), ③流动性提供(在别人必须成交的时刻做对手方——急迫方付费), ④信息性(内幕/快手指令流——慢的一方付费), ⑤保险性(卖波动率/卖尾部——风险厌恶方付保费)。没有第六类。
2. **v1报告缺的正是这个**: 鞦/遍历性/粗糙波动率描述的是价格的**统计形态**, 不是收益的**经济来源**。统计形态可以无限精修, 但不回答"我的对手盘是谁、他明年还在不在"。这句话是v2全部内容的起点。
3. **我们自己的体系(#0c17: 出场机+流动性池内低流动性端的流动性溢价)在文献里有精确对应物**: Nagel(2012)证明短期反转收益=流动性提供者的预期收益; Amihud(2002)证明非流动性溢价真实存在; A股T+1制度造成的**隔夜系统性负收益**(Qiao & Dam 2020)意味着"出场机"的alpha很可能有相当一部分=**规避负隔夜carry**——这是一个可用我们自有数据直接检验的假设(见§3与§15)。
4. **公开即衰减是定律级的现象**: 学术发表后因子收益平均-58%(McLean & Pontiff), 指数效应从7.6%衰减到0.8%(Greenwood & Sammon), A股可转债双低2022年后RankIC转负, A股量化指增超额从2015年的20%+压缩到2026年前5月的1.9-3.6%。**alpha是易腐品, 出租要在腐烂前完成**。
5. **200%年化的判定(详见§14)**: 长期(>10年)净200%在全球可查记录中**不存在**(Medallion扣费前66%已是人类天花板); 但"50万小资金 + corr≈0的四腿 + 各腿Sharpe≥2 + 3-6倍杠杆"的配置在**数学上**允许单年50-200%且多年100%+, 这正是Medallion结构的低配版——代价是杠杆与肥尾的死亡区距离很近, 2015年A股与2018年XIV的死法都写在尸体上。

---

## 1. Alpha食物链总图(taxonomy)

```
付钱方(输家)                    机制                        收租方                      我们能否收
─────────────────────────────────────────────────────────────────────────────────────────────
彩票偏好散户(为偏度付费)    →   MAX/低波动/低价因子        →   理性持有"非彩票"端者       →  能(多头端)
锚定/处置效应散户(卖赢持亏) →   动量/PEAD/52周高点         →   耐心持有者                →  部分(需事件数据)
被规则强迫的交易者          →   指数纳入/申赎/强平/敲入事件 →   提供流动性者              →  能(危机时段)
急迫成交者(要流动性)        →   短期反转/做市价差           →   流动性提供者              →  能(=我们的出场机)
慢的信息接受者             →   内幕/快手指令流/PEAD        →   快者                      →  基本不能(无tick)
风险厌恶的保险买方          →   波动率风险溢价(卖权)        →   卖方(带尾部风险)          →  能(需严格风控)
制度摩擦的承担者            →   T+1隔夜折价/涨跌停溢价/壳价值 →  利用制度结构者             →  能(A股特有)
```

三类"为什么输钱方不消失"的终极答案(Shleifer & Vishny 1997之后学界共识):
- **偏好不是错误**: 彩票偏好是效用函数的一部分, 散户在买娱乐, 不认为自己在犯错 (Bali et al. 2011; An & Li 2020);
- **被迫交易是合同内生的**: 赎回条款、保证金线、敲入线写在合同里, 价格越崩被迫交易越多 (Coval & Stafford 2007; 2015年A股; 2024年雪球);
- **套利者自身有资本约束**: "真套利需要资本且会被抽资, 恰在最该套利的时刻最无力" (Shleifer & Vishny 1997, 引用8000+次, https://www.jstor.org/stable/2329555 ; https://shleifer.scholars.harvard.edu/publications/limits-arbitrage)。

---

## 2. 机制一: 行为性alpha — 彩票偏好(付钱方: 为"以小博大"的快感付费的散户)

【谁付钱】彩票型偏好(homogeneous lottery preference)的投资者系统性高估有偏度、高MAX(近月单日最大涨幅)、高异质波动、低价的股票, 使其后继收益系统性偏低。做空彩票股/做多"非彩票"股者收租。

【为什么持续】(1) 偏好不是bug而是效用: 买彩票股的人消费的是**博彩服务**, 该需求随收入和风险偏好存在; (2) A股**卖空约束**使套利者无法砸平高估: "缺乏卖空使彩票股异象收益持续, 且收益主要来自空头腿"(Which Proxy: Capturing Lottery Feature Through Aggregation, Financial Management, 转引); (3) An & Li(2020)证明参考点依赖(浮亏时彩票需求更强)进一步放大该效应。

【证据与数据】
- Bali, Cakici & Whitelaw (2011) "Maxing Out: Stocks as Lotteries..."(JFE): 美股MAX最低与最高十分位组**月收益差>1%/月**(约年化12%+的多空), 控制常规因子后稳健 (转引, 原文检索可得)。
- An, Li (2020) "Lottery-Related Anomalies: The Role of Reference Dependence", Journal of Accounting Research: https://www.jstor.org/stable/48811828 ; 清华五道口版PDF: https://www.pbcsf.tsinghua.edu.cn/__local/4/AB/01/69E72890A1B609E8CEDA6071D26_D0CD8DAF_F0ECB.pdf
- 中国证据: Liu, Stambaugh & Yuan (2019, JFE 134:48-69) 系统检验A股**14个异象9大类(规模/价值/盈利/波动/反转/换手率/投资/应计/流动性)**, 波动与换手率类在中国极强 (引文见清华五道口因子动量研究: http://cfrc.pbcsf.tsinghua.edu.cn/__local/4/AE/67/89980D797AD790C70C6AD15BEAB_F3C5BFB8_73992.pdf?e=.pdf )。
- 上交所统计年鉴(2018卷): 沪市散户贡献**82.01%成交额但盈利占比约一成**; 2016年散户整体亏7090亿(机构亏3171亿), 2017年散户整体赚3108亿(机构赚11156亿), 机构户均盈利是散户的数千倍 — https://m.21jingji.com/article/20180903/herald/d82f143c18ecd9631c7ef6d6c967f563.html ; https://xueqiu.com/2364942209/113907548 ; 年鉴官方页: https://www.sse.com.cn/aboutus/publication/yearly/

【年化/容量/衰减】美国多空~12%/年, 但A股**无卖空工具, 只能做"买入非彩票端"的多头端**, 多头端年化约3-8%; 容量: 因子分散持股, 容量中等(百亿级), 但A股小盘端冲击成本是约束; 衰减: McLean & Pontiff (2016, JF): 97个因子样本外收益**-26%, 发表后-58%** ( https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623 ; PDF: https://www.fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf )。行为类因子衰减慢于技术类, 因为偏好不消失。

【对我们资产的适用性】**高**。A股日线2012-2026全史可直接计算MAX/IVOL/低价/单日涨停哑变量等彩票载荷因子。关键结构约束: 我们只能持多头, 所以应该把"彩票股"当作**选股池过滤器(剔除高彩票股)而非收益来源本身**; 这与我们"流动性池内低流动性端"的持仓结构天然兼容——但注意: A股低流动性端本身就是彩票股密集区, 两者方向冲突, 需要实证分辨我们的池子里是"彩票溢价"还是"非流动性溢价"在付钱(可做双重排序检验)。

---

## 3. 机制二: 制度性alpha — A股隔夜折价与T+1(付钱方: 日内追涨者+制度受限者)

【谁付钱】T+1制度禁止当日回转卖出, 造成**开盘价系统性折价("即时性价格")**: 日内买入者必须承担隔夜不可平仓的风险, 因此平均以折价开盘补偿——付钱方是每天用市价追入的短线资金(多为散户/游资)。这与散户主导叠加, 使A股成为全球唯一"隔夜收益均值为负、日内为正"的主要市场(美股相反)。

【为什么持续】制度不变, 摩擦不消失; 付钱方是流量(每天新进的追涨盘)而非存量, 流量由行为偏好持续供给。

【证据与数据】
- Qiao & Dam (2020) "The overnight return puzzle and the 'T+1' trading rule in Chinese stock markets", Pacific-Basin Finance Journal 50: https://ideas.repec.org/a/eee/finmar/v50y2020ics1386418120300033.html ; SSRN版: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3418356 — 结论: A股隔夜收益均值为负且1%水平显著, 可解释为T+1下的开盘折价(即时性价格)。
- An et al. (2022) "The Asymmetric Overnight Return Anomaly in the Chinese Stock Market"(组合隔夜收益显著为负): https://www.mdpi.com/1911-8074/15/11/534
- 诺丁汉大学宁波校区博士论文直接表述: "中国日内收益模式为隔夜负+日间正, **与美国和多数市场相反**, 独特的T+1规则…"(转引)。
- FT科普报道: "Chinese stocks go down at night because no one trusts…": https://www.ft.com/content/a5db3a7d-a22a-458c-a04f-717fc93ae002
- 日内beta与夜间beta在中国符号相反: "Factor Beta, Overnight and Intraday Expected Returns in China", SSRN 4138715 ( https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4138715 )。

【年化/容量/衰减】隔夜-日内分解本身的幅度量级: 市场平均隔夜拖累按日频bp级累积, 年化可达两位数(精确bp值依样本期而定, [未证实具体幅度, 需用自有数据计算]); 衰减: 制度不废除则不衰减, 但2015年以来T+0变体(ETF/可转债/期指对冲)部分侵蚀; 容量: 极大(全市场因子)。

【对我们资产的适用性】**极高——这是对我们自身归因的直接可检验假说**。我们的alpha已归因为"出场机+流动性池低流动性端溢价"。机制学翻译: **出场机的收益可能=规避隔夜负carry+在日内(对方急迫时)离场**。检验方法: 用日线数据把策略每笔持仓收益拆为 close→open 与 open→close 两段, 若alpha集中于日内段且隔夜段为正贡献(或至少非负), 则证实"出场机=隔夜规避"; 若隔夜段为正且占大头, 则更像纯非流动性溢价(次日开盘溢价卖出)。这一检验只需日线+我们的成交记录, 零新数据成本, **建议列为下一实验**。

---

## 4. 机制三: 被迫交易(结构性)— 指数再平衡 / 基金申赎 / 杠杆强平 / 雪球敲入(付钱方: 被合同和规则锁死的人)

【谁付钱】三类: ①指数基金持有人(纳入/剔除时被动买卖, 为可预测性付费); ②赎回潮中的基金持有人与基金公司(被迫贱卖, Coval-Stafford意义上的fire sale); ③杠杆/结构化产品持有者(两融强平、配资爆仓、雪球敲入、DMA强平)——**在价格最差时被法律强制成交**。

【为什么持续】被迫交易是合同条款的内生结果, 不依赖任何人的愚蠢; 且它在危机中自我放大(卖→跌→更多强平), 套利者此时自身也面临资本收缩(Shleifer-Vishny 1997), 所以**恰恰在定价错误最大时套力最弱**——这是全部市场中最坚硬的一条结构性裂缝。

【证据与数据】(本节是全报告数据最密集处)
- 指数纳入效应: Shleifer (1986, JF) "Do Demand Curves for Stocks Slope Down?" — S&P500纳入带来不可套利的异常上涨 ( https://ideas.repec.org/a/bla/jfinan/v41y1986i3p579-590.html )。衰减: Greenwood & Sammon "The Disappearing Index Effect"(JF 2025 / NBER w30748): 纳入异常收益**1980s平均3.4% → 1990s平均7.6% → 2010s仅0.8%**; 剔除效应2010-2020平均仅-0.6% ( https://www.nber.org/papers/w30748 ; PDF: https://www.hbs.edu/ris/Publication%20Files/23-025_563e45c6-df92-4d9c-ae05-608d4d0acab1.pdf )。Pavlova & Sikorskaya (RFS 2023) "Benchmarking Intensity"证明基准化资金比例上升产生可预测买压 (转引)。
- 基金火售: Coval & Stafford (2007, JFE 86:479-512) "Asset Fire Sales (and Purchases) in Equity Markets"(引用2300+): 被赎回强迫卖出的股票价格受压且事后**反转回升**——买入被迫卖出者有正alpha ( https://www.sciencedirect.com/science/article/pii/S0304405X07001158 ; https://www.nber.org/papers/w11357 )。
- 2015年A股: 清理场外配资触发链式强平; 两融余额从2015-6-18峰值**2.27万亿**降到7月初约1.44万亿(-36.6%)、9月跌破1万亿; 多次千股跌停, 7月8日超半数股票跌停或停牌, 全市场流动性枯竭; 下半年市值蒸发约**22万亿元**, 散户人均亏约4.37万 (人民网/中新网/财联社, [快照]转引: http://finance.people.com.cn 及 https://www.cls.cn/detail/2133462 )。
- 2024年1-2月雪球+DMA+微盘三连环: 挂钩中证500/1000的雪球存量约**2000亿元**, 敲入线集中于70-80%; 1月17日、2月5日集中敲入, 券商对冲盘抛IC/IM使贴水大幅走阔; 信达证券测算中证500雪球占IC总持仓最高约**六成**、中证1000占IM约**四成** (新浪财经2024-02-02《雪球产品敲入风险或已释放大半》[快照]; 信达金工研报[快照]); 随后DMA(约4倍杠杆的多微盘空期指中性产品)两头亏损触发强平死亡螺旋, 2024-2-6监管暂停转融券+汇金扩大ETF增持后才反转 (财新博客/德邦芦哲复盘 [快照]: https://finance.sina.com.cn )。
- 2026年最新生态: A股超短线交易者稳定盈利比例仅**21%**, 一线游资上榜次数同比-42%, 炸板率**68%** (什么值得买社区统计, 数据质量一般, [未证实]) — https://post.smzdm.com ; 中国基金报: 打板/接力盈利空间被量化与生态变化急剧压缩 — https://www.chnfund.com/article/AR20260412120446136

【年化/容量/衰减】事件驱动型: 平常年份贡献小, 危机年份贡献全年收益的大部分("危机alpha"); 容量=危机中可接的卖单量, 对50-500万资金**容量无限**; 衰减: 指数纳入类已从7.6%衰到0.8%(被前置套利), 但**强平类无法被套利掉**——因为套利者自己在危机中也缺资本, 这一类是永续的。

【对我们资产的适用性】**最高**。我们的"出场机"若理解为"在对方被迫急卖时接、在对方被迫急买时出", 正是这一机制+机制五的复合。A股全史含2015与2024两次完整强平螺旋, 是天然的验证样本。**最大教训**: 2024年2月证明"微盘+低流动性端"策略在拥挤后会成为被迫交易方本身(DMA就是我们的镜子)——所以腿的容量上限、杠杆上限和拥挤度监控不是风控装饰, 而是机制层面的生存条件。

---

## 5. 机制四: 锚定与处置效应 → 反应不足(付钱方: 卖出赢家太快、持有输家太久的散户)

【谁付钱】处置效应(disposition effect)投资者: 赢了急着落袋(压制价格上涨), 输了死扛(阻止坏消息消化), 其系统性行为为耐心持有者付费。

【为什么持续】前景理论的价值函数(S形)是人类决策硬件, 不是市场可教育的错误; An, Lou & Sadka (2016, JF)进一步显示"卖出极端赢家/输家"的倾向可预测价格压力。

【证据与数据】
- Frazzini (2006) "The Disposition Effect and Under-reaction to News", JF 61:2017-2046: 处置效应→对新信息反应不足→PEAD(盈余公告后漂移)与动量 [未附URL, 检索可得]。
- George & Hwang (2004) "The 52-Week High and Momentum Investing", JF 59(5):2145-2176: **接近52周高点的锚定解释了动量利润的大部分** (PDF: https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf )。
- An, Lou & Sadka (2016) "Asset Pricing When Traders Sell Extreme Winners and Losers", RFS (转引, https://www.jstor.org/stable/2489954 [未核验])。
- 动量的代价(收租方的风险): Daniel, Moskowitz, Ooi & Pedersen "Momentum Crashes"(JFE 2016): 动量在恐慌态+反弹中会遭遇可预测的崩塌(2009年最惨), 空头腿输家股呈期权型收益 ( https://www.nber.org/system/files/working_papers/w20439/w20439.pdf ; https://www.sciencedirect.com/science/article/pii/S0304405X16301490 )。

【年化/容量/衰减】PEAD/动量类多空在美股历史年化约5-12%, 容量大(巴菲特级资金可用); A股情况特殊: 动量弱、**反转强**(见机制五), 王永宏/赵学军(2001,《经济研究》)以来反复确认A股动量不显著而月内反转显著 [未附URL]。

【对我们资产的适用性】中。处置效应类alpha需要"投资者成本分布"型数据(筹码/换手结构), 我们有全史日线可构造代理(换手率异动、涨停打开、52周高点距离)。52周高点因子可直接加入因子库检验。动量崩塌文献同时警告: 任何含"卖输家"结构的策略在恐慌反弹月会爆亏——我们的出场机不应内置裸空头逻辑。

---

## 6. 机制五: 流动性提供溢价(付钱方: 急迫成交者)— 我们体系的核心对应物

【谁付钱】任何时候**必须现在成交**的人: 止损单、强平单、赎回引致的基金卖单、散户的开盘市价单。做市商与反转交易者是收租方。

【为什么持续】急迫性由外生事件(赎回/保证金/情绪)源源产生; 提供流动性有真实成本(库存风险+逆向选择), 所以补偿必须为正否则无人提供; 危机时流动性供给方撤退, 租金陡增。

【证据与数据】
- Nagel (2012) "Evaporating Liquidity", RFS 25(5): 短期反转策略收益可解释为**流动性提供的预期收益**, 且在流动性供给收缩(如2008)时反转收益激增 [RFS, 未附URL]。
- Amihud (2002, JFM, 引用16000+) "Illiquidity and stock returns": 预期非流动性正向进入事前超额收益——非流动性溢价存在 ( https://doi.org/10.1016/S1386-4181(01)00024-6 ); 国际证据: Amihud, Hameed, Kang & Zhang, NUS ( https://web2-bschool.nus.edu.sg [快照] )。
- 做市的极限稳定性: Virtu Financial 2014年IPO文件披露**1238个交易日仅1天亏损**——每天数百万笔小额做市, 靠大数定律把流动性租变成日频近确定收益 ( https://www.businessinsider.com/virtu-hft-only-one-losing-day-2014-3 ; https://www.cnbc.com/2014/03/11/virtu-financial-expects-to-raise-up-to-100-million-in-ipo.html ; Reuters: https://www.reuters.com )。
- A股反转与换手: A股**月内反转>动量**是20年共识(王永宏/赵学军2001; Liu-Stambaugh-Yuan 2019把反转与换手率列入中国14异象); 倪一宁(2023)《基于短期反转效应和MAX效应的联合交易策略研究》,《东北财经大学学报》: 短期反转是A股重要异象但多空波动大 ( https://zzs.dufe.edu.cn/zzcmcontent/conid_197.html )。
- 换手率/情绪类因子在A股的强度: 华创证券《流动性因子研究》: 换手率因子在中证800样本池中表现最佳 ( https://www.fxbaogao.com [快照] ); 国泰海通金工持续发布因子拥挤度周报, 以换手率与拥挤度预警反转因子失效 ( https://view.inews.qq.com [快照] )。

【年化/容量/衰减】Virtu式高频做市年化极高但容量小、需tick级基建(我们无); **日线级反转/低流动性端**: 美股反转多空历史Sharpe约1+但2009式反弹会重创空头腿; A股多头端(买入超跌/低流动性)年化贡献约3-10%且与彩票因子纠缠; 容量: 与持仓端成交额成正比, 50-500万资金处于容量曲线的平坦区; 衰减: Nagel证明该收益是**结构性补偿**而非错误定价, 理论上不衰减, 只随竞争强度波动(拥挤时变薄)。

【对我们资产的适用性】**这是我们已有的主粮, 且文献支持其永续性**——流动性提供是五大类alpha中唯一"理论上不该消失"的(它是服务的价格, 不是错误)。行动项不是找新alpha, 而是: ①把"我们到底在提供哪一类流动性"(开盘市价单? 强平盘? 赎回盘?)用5min数据画像; ②监测同类供给者(量化微盘基金)的规模作为拥挤度指标; ③按2024年2月情景做压力测试——当时流动性提供者的账面浮亏极深, 生存=有足够资本扛到2月6日。

---

## 7. 机制六: 信息性alpha(付钱方: 慢的一方)— 对我们基本关闭

【谁付钱】指令流慢的参与者: 收到信息晚、解读慢、下单慢的人。

【为什么持续】信息生产有成本, 快者收租是信息的价格; 但每条信息alpha半衰期以秒-分钟计。

【证据与数据】
- 台湾全样本(完整交易记录): Barber, Lee, Liu, Odean (2009, RFS) "Just How Much Do Individual Investors Lose by Trading?": 个人投资者整体年绩效惩罚**3.8个百分点**, 相当于台湾GDP的**2.2%**; 机构整体年增益1.5个百分点——个人的损失=机构+中介的收益, 零和结构被完整称重 (PDF: https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/justhowmuchdoindividualinvestorslose_rfs_2009.pdf , 本次已核验原文)。
- 台湾日内交易者: Barber et al. "The Cross-Section of Speculator Skill": 约45万日内交易者中**<1%**能持续净盈利 ( https://www.sciencedirect.com/science/article/abs/pii/S1386418113000190 ); "Do Day Traders Rationally Learn About Their Ability?" ( https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trading%20and%20Learning%20110217.pdf )。
- 巴西股指期货散户: Chague, De-Losso & Giovannetti "Day Trading for a Living?": 坚持>300天的日内交易者**97%亏损, 仅约1.1%收入超过最低工资** ( https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101 , 数字[转引]未直接核验)。
- 加密市场: Kogan et al. (2023) "Are Cryptos Different? Evidence from Retail Trading"(NBER w31317, JFE): 散户在加密上追涨杀跌, 系统性亏损 ( https://www.nber.org/system/files/working_papers/w31317/w31317.pdf ; https://www.sciencedirect.com/science/article/pii/S0304405X2400120X )。

【年化/容量/衰减】HFT/做市级信息alpha年化可达三位数, 但半衰期短、容量小、军备竞赛(设备/席位/费用)吞掉大半; 日线级残余信息alpha=PEAD等慢信息(见机制三[应为机制二/五])。

【对我们资产的适用性】**低(无tick)**。唯一可用的日线级信息残留: 公告/财报日历驱动的慢漂移, 需要事件数据支持。结论: 不投入, 做对手盘画像即可。

---

## 8. 机制七: 散户主导市场的本质(参与者结构变迁与可提取租金)

【谁付钱】按市场分: A股=散户(交易占比从2017年82%降至约60%); 港股=离岸机构为主+南向资金; 美股=机构主导(散户成交量占比~20-25%); 期货/期权=产业套保盘+散户投机盘。

【为什么持续/如何变化】A股机构化不可逆但缓慢; 每一轮"散户进场潮"(2015, 2020-21, 2024-9)都会重新打开行为性租金窗口, 然后再收紧。

【证据与数据】
- A股: 上交所年鉴2018卷: 2017年自然人交易占比**82.01%**/专业机构14.76% ( https://www.sse.com.cn/aboutus/publication/yearly/ ; 转引 http://finance.sina.com.cn/stock/stockptd/2018-08-30/doc-ihiixzkm2347848.shtml ); 注册制时代个人交易占比已降至**约60%**(中信/中金/申万/广发/国信联合口径, [快照]); 2024年新增投资者1274万(中登统计年鉴2024, [快照])。
- 美股: 散户成交量占比2011年约10% → 2021年**22%+**(Bloomberg Intelligence口径), 2025年约20-25%, 高波动日逼近35% (转引: https://coinlaw.io ; 原始BI报告付费)。
- 港股: 机构成交占比长期60-70%(外地机构约36%居首), 散户约25-30%; **南向资金占港股成交2025年达24.2%**(开通以来新高, 2014年约5%), 持股市值约13%, 部分交易日成交占比近50% (港交所《现货市场交易研究调查》与香港证监会SFC《2025年环球及香港证券市场回顾》, [快照]: https://www.hkexgroup.com ; https://www.sfc.hk ; 新浪: https://www.sina.cn )。
- 可转债(A股内的散户T+0孤岛): 光大证券回测2018-1至2021-5双低组合**平均年化21.3%/最高25.3%**(同期中证转债指数10.4%) ( https://m.hibor.com.cn/wap_detail.aspx?id=7070074da45d4db3398506ab45ad1ed5 ; https://www.sohu.com/a/508289558_121123927 ); 2018-2022双低+择时累计+156.51%/Sharpe1.97 (新浪研报: https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/10/rptid/728329674042/index.phtml ); **2022年后经典双低RankIC转负**(时序标准化改造后多空Sharpe 1.91→2.60) ( https://agents-quant.com/blog/convertible-bond-factor-normalization/ ) — 一个完整的"散户结构红利→被发现→拥挤→失效"标本。
- A股量化行业租金压缩: 私募排排网口径2025年量化指增平均超额**16.75%**(大牛市年), 而2026年前5月中证500/1000/2000指增平均超额仅**1.9%/2.94%/3.62%** (21财经2026-6-2《三问量化产品超额收益》[快照] https://www.21jingji.com ; 中证报《量化信仰》[快照] https://wap.eastmoney.com )。

【年化/容量/衰减】散户行为租金的年度供给≈散户交易量×行为成本率; 台湾数据(3.8pp/GDP2.2%)给出量级感觉; 衰减方向确定(机构化), 速度取决于散户流入周期。

【对我们资产的适用性】高。A股腿吃的是60%散户+制度摩擦; 港股腿注意: 南向定价权上升(24.2%)意味着港股的"行为租金"正在被内地资金重新供给(AH溢价收窄), 且港股小盘流动性悬崖(成交集中头部)使"低流动性端溢价"在港股**更厚但也更难退出**——港股腿的出场流动性必须单独建模; 期货腿的付钱方是产业套保盘(见§9 carry)。

---

## 9. 机制八: 期货端 — 展期收益/套保压力溢价(付钱方: 被迫持有期货头寸的产业与投机盘)

【谁付钱】商品期货: 产业套保者为锁定价格支付风险溢价(反向压力定价, Keynes-Hicks传统); 金融期货: 雪球对冲盘/中性产品空头(2024年1月贴水走阔时的IC/IM卖压是付费方)。

【为什么持续】套保需求来自实体经营, 不因金融定价而消失; 期限结构(backwardation/contango)持续再生产。

【证据与数据】Koijen, Moskowitz, Pedersen & Vrugt "Carry"(JFE 2018, 引用600+): carry(假设价格不变的预期收益)在**全球股票/债券/商品/FX/指数期货**上横截面与时序均预测收益, 多空carry组合在几乎所有资产类别为正 ( https://www.sciencedirect.com/science/article/abs/pii/S0304405X17302908 ; NBER: https://www.nber.org/papers/w19325 ; PDF: https://pages.stern.nyu.edu/~lpederse/papers/Carry.pdf )。2024年1月雪球对冲盘抛IC/IM致贴水走阔(见§4)是套保压力的实时案例。

【年化/容量/衰减】carry因子多空Sharpe约0.8-1.2, 单商品期货腿年化个位数到十位数; 容量极大; 衰减慢(供给方是实体)但2024年A股期指贴水长期化已把部分carry资本化进价格。

【对我们资产的适用性】高: 我们有期货主力日线+期限结构, 可直接构造展期收益/基差动量因子, 与A股腿相关性低——这正是四腿结构里期货腿的机制学定位: **吃套保盘的保费**。

---

## 10. 机制九: 波动率风险溢价(VRP)— 卖保险收租与被雷劈的数学

【谁付钱】愿意为"确定性"付保费的期权买方: 对冲基金买尾部保护、散户买彩票call/put。卖方=保险公司, 收的是保费。

【为什么持续】真实保险需求: 机构必须对冲(风控/赎回条款), 散户要买梦; 卖方承担尾部, 补偿必须为正。

【证据与数据】
- 理论+全球: Bollerslev, Tauchen & Zhou (2009, RFS) "Expected Stock Returns and Variance Risk Premia": 方差风险溢价(隐含-已实现方差)显著预测股市收益, 季度 horizon 最强 [RFS 22(11):4463-4492, 未附URL]; Carr & Wu 2009 (RFS) 证明delta对冲期权买方系统性亏损=卖方系统性收租 [未附URL]。
- 中国: CNVIX研究(Yue & Ruan 2023, Quarterly Review of Economics and Finance 91:40-55): 用50ETF期权构造中国波动率指数, 证实中国市场存在正VRP ( https://ideas.repec.org/a/eee/quaeco/v91y2023icp40-55.html ); "Variance Risk Premium and Return Predictability: Evidence from the Chinese SSE 50 ETF Options"(delta对冲组合证实中国期权买方付保费) ( https://www.researchgate.net/publication/335018450 )。
- 中国卖权的衰减: Peng (2024, arXiv 2403.00474) "Volatility-based strategy on Chinese equity index ETF options"(本次已核验原文): 50ETF期权**2015上市至2018年裸卖波动率策略回报优秀, 2018年后超额收益消失**(市场结构变化), 需叠加GARCH择时才有修复 — https://arxiv.org/abs/2403.00474
- 美股期权买方=散户亏损机: MIT Sloan综述"Retail investors lose big in options markets" ( https://mitsloan.mit.edu/ideas-made-to-matter/retail-investors-lose-big-options-markets-research-shows ); 0DTE热潮中做市商为净卖方收租(Bandi-Müller-Weigert系列 [未附URL])。
- 爆仓标本1: OptionSellers.com (2018-11): 裸卖天然气看涨期权, 单日标的+18%引爆, 290个账户全灭, 损失约1.5亿美元, 客户倒欠清算商 (WSJ: https://www.wsj.com/articles/energy-losses-prompt-emotional-video-to-options-firms-clients-1542709800 ; FT: https://www.ft.com/content/b7c525f6-ec44-11e8-89c8-d36339d835c0 ; CNBC: https://www.cnbc.com/2018/11/21/a-risky-natural-gas-bet-gone-awry-leads-to-weepy-youtube-confessional.html )。
- 爆仓标本2: XIV (2018-2-5 "Volmageddon"): 做空波动率ETN单日**-96.3%**触发提前赎回, 资产19亿美元一夜蒸发 ( https://www.etf.com/sections/news/inverse-vix-etn-shuts-down ; https://www.cnbc.com/2018/02/06/the-obscure-volatility-security-thats-become-the-focus-of-this-sell-off-is-halted-after-an-80-percent-plunge.html )。

【年化/容量/衰减】delta中性卖权组合历史年化约8-15%(美国), 中国50/300ETF期权2015-2018更肥; **回撤特征: 平稳多年+一夜归零型左尾**; 容量: 中国ETF期权流动性对50万-500万资金充裕; 衰减: 中国2018年后明显收窄(arXiv证据), 美国持续被0DTE供需重新定价。

【对我们资产的适用性】中高, 但必须以"保险公司的姿势": 卖=有保证金冗余+有对冲腿+单日最大亏损可预算(价差组合而非裸卖); 我们有ETF期权链+美股期权, 可做备兑/价差/现金担保put类结构。**纪律条款**: 任何卖权腿的杠杆上限必须使XIV式单日(-96%)情景不触发强平——这直接写入下一节Kelly计算。

---

## 11. 机制十: 杠杆的数学 — Kelly, 多腿与"√N免费午餐"

【谁付钱】(杠杆本身不创造alpha, 它是租金的放大器; 付钱方=利息与肥尾) 融资利息+参数估计误差+路径风险。

【数学与证据】
- 连续时间Kelly: 单资产最优杠杆 f* = μ/σ² (超额收益/方差), 对数增长最大; 过量下注(f>f*)会**降低**长期增长并在足够长时间内趋于破产(Thorp 2006, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market", Handbook of ALM; Thorp在Princeton/Newport Partners以"一系列不相关小赌注+半Kelly"实操, 19.1%年化/近20年无亏损年, 转引: http://compoundingmyinterests.com/compounding-the-blog/2012/10/12/how-did-ed-thorp-win-in-blackjack-and-the-stock-market.html ; https://macro-ops.com/lessons-trading-great-ed-thorp/ )。
- N个corr≈0、等Sharpe腿的组合: 组合Sharpe ≈ 单腿Sharpe×√N; 4腿→×2。组合波动≈单腿波动/2(等权等vol), 因此**Kelly杠杆不降反升**(f*∝S²/μ结构), 但现实中用**1/4-1/2 Kelly**对冲参数误差(Sharpe被高估是常态)。
- 算例(我们的配置): 设A股腿现实Sharpe 2.2(=现实82-97%@50万的历史值), 其他三腿Sharpe各1.2, corr≈0.05, 等风险权重 → 组合Sharpe≈2.2/4+3×1.2/4=1.45单腿均值→×√4≈**2.9**; 若组合未杠杆波动10%、期望29% → 理论f*=0.29/0.01=29倍(荒谬, 因μ/σ²对低vol组合爆炸), 现实约束: 融资可得性(A股两融≤~2x)、期权/期货保证金、肥尾、**corr→1的危机日**(2024-2-5 A股四箭齐发/XIV日)。
- 危机关联: 2015年A股强平螺旋中所有A股策略相关性→1; XIV日美股所有做空波动率策略同日死。**杠杆上限的约束不是平均日而是最坏日**: 若组合日σ=0.63%(年10%/√252), 6倍杠杆下单日6σ=-9.4%, 连续三日即-25%进入两融平仓区(维保线130%)。

【对我们资产的适用性】高, 且是通往200%的唯一数学通道(见§12)。落地: ①每腿先单独估现实Sharpe(用纸面×现实折扣系数, 我们历史折扣≈0.7-0.85); ②组合层按等风险而非等金额配置; ③总杠杆=0.25×f*且不超过"最坏三日情景回撤<25%"的约束解; ④每季度重估corr矩阵, 危机态(corr>0.5)自动降杠杆。

---

## 12. 200%年化出现在哪里 — 真实案例的完整分布

【数据点】(全部为公开可查记录)
| 案例 | 年化 | 机制 | 频率/容量 | 持续性 |
|---|---|---|---|---|
| Renaissance Medallion | **66%毛/39%净**(1988-2018) | 数千个弱信号×高频×高换手, 全球品种 | 日内-周内; 容量上限~100亿美元, 1993年后只对内开放, 定期强制分红缩规模 | 30年, 无亏损年 (Zuckerman《The Man Who Solved the Market》; 转引 https://iwpintel.com ) |
| Buffett/Berkshire | ~20%(1965-2024), Sharpe仅**0.79**, 杠杆1.7:1 | "便宜+安全+高质量"×低成本浮存金杠杆(BAB+质量因子) | 低频; 容量千亿美元级 | 60年 (Frazzini, Kabiller & Pedersen "Buffett's Alpha", FAJ 2018: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197185 ; PDF: http://docs.lhpedersen.com/BuffettsAlpha.pdf ; https://www.aqr.com/Insights/Research/Journal-Article/Buffetts-Alpha ) |
| A股量化指增(行业) | 超额2015年20%+→2025年16.75%→2026前5月1.9-3.6% | 散户行为租金(反转/换手/流动性) | 日-周频; 全行业容量千亿级 | 衰减中, 危机年(2024)反常高 [快照, §8] |
| 可转债双低 | 2018-2021年化18-25%, 2022后失效 | 散户T+0孤岛的结构性错价 | 日频; 容量十亿级 | 4年后被套利平 (§8) |
| 打板/超短游资 | 个体年化可达三位数, 群体稳定盈利仅~21%(2026Q1统计[未证实]) | 涨跌停制度的情绪溢价 | 日内; 容量极小 | 生态恶化中(炸板率68%) (§4) |
| Virtu做市 | 日频近确定盈利(1237/1238天) | 流动性提供 | 微秒; 容量=全部成交份额的固定比例 | 极稳 (§6) |
| 散户日内(台湾/巴西/加密) | 群体均值大幅为负(台湾-3.8pp/年) | —付钱方本体— | — | — (§7) |

【结论: 200%的可达条件】
200%长期净收益不存在于任何公开记录(Medallion的66%毛是人类天花板且依赖我们不具备的日内基建+数千信号)。200%**单年或少数年份**出现在且仅出现在以下全部条件同时满足时:
1. 资金小(冲击成本≈0): ≤500万, 我们满足;
2. 每日多次决策×大数定律(高Sharpe来自高频次而非高单次): 我们日线频率**不满足**——日线全策略组合现实Sharpe上限约3-4;
3. T+0或高换手市场(A股T+1是硬摩擦, 但期货/期权/美股/港股腿满足);
4. 杠杆(期货/期权/融资): 数学上, 组合Sharpe 3 + 波动15% + 杠杆4-5x → 期望年化 180-225%, 同时最坏情景进入死亡区。
**判定**: 在"50-300万资金 + 四腿corr≈0 + 现实组合Sharpe≈3 + 3-5x杠杆 + 严格危机降杠杆"的路径上, 200%**数学上可达但处于Kelly过度下注的边缘**(该配置的Kelly最优杠杆约在3x附近, 5x已超出); 更稳的定位是**目标100-150%/年**, 让2024年式的危机年(微盘反弹月)自然产出个别200%+年份。反之, 用裸卖权或单一微盘腿上高杠杆去"够"200%, 是OptionSellers/XIV已经演示过的死法。

---

## 13. 各市场谁能赚谁的钱(参与者结构速查)

| 市场 | 付钱方主力 | 收租方 | 对我们的开放度 |
|---|---|---|---|
| A股 | 散户(60%成交/10%盈利)+强平盘+追涨流量 | 反转/低流动性端/事件接盘者 | 高(已验证114%纸面/82-97%现实) |
| 港股 | 南向新贵散户化资金+小盘流动性悬崖 | 头部流动性提供者/老牌机构 | 中(低流动性端溢价厚但退出难, 南向占成交24.2%且集中大票) |
| 美股 | 散户期权买方(0DTE)+拥挤因子平仓盘 | 做市商/多空因子基金 | 中(我们无tick, 期权腿可用) |
| 中国ETF期权 | 保险买方/散户买方 | 卖方(2015-2018肥, 现收窄) | 中高(有期权链数据) |
| 期货 | 产业套保盘+雪球/中性对冲盘 | carry/期限结构交易者 | 高(有期限结构数据) |

---

## 14. 对我们资产栈的机制学适配清单(行动项)

1. **出场机alpha分解实验**(§3): 持仓收益拆隔夜vs日内, 检验"出场机=规避隔夜负carry"。零数据成本, 直接用现有日线+交易记录。
2. **彩票/非流动性双轴检验**(§2): 对现有持仓池按MAX/IVOL/换手/Amihud双重排序, 确认我们收的是哪份租(两者方向冲突, 须分辨)。
3. **流动性画像**(§6): 用5min数据把我们的成交对手方分类(开盘市价流/强平流/赎回流), 确定租金来源的实时指标。
4. **拥挤度仪表盘**(§4/§8): 量化微盘基金总规模、转债换手、打板炸板率、指增超额均值, 四个都是"我们的租金池水位计"。
5. **期货腿carry因子**(§9): 期限结构数据→展期收益因子, corr≈0的独立腿。
6. **期权腿保险化**(§10): 只做价差/备兑/现金担保, 杠杆上限按"单日-96%情景不清算"倒推。
7. **Kelly组合层**(§11): 等风险四腿+0.25f*杠杆+危机corr>0.5自动降杠杆; 目标年化定在100-150%而不是200%(数学见§12)。

---

## 15. 本质总结 — 市场的十条定律(附证据强度)

1. **零和加和定律**: 超额收益的每一分都有具名的付钱方; 找不到付钱方的alpha默认是回测错误。(强: 台湾-3.8pp/GDP2.2%; SSE年鉴82%成交10%盈利)
2. **偏好永续定律**: 行为性租金来自效用函数而非认知错误, 因此不可被教育掉, 只能被套利稀释到卖空约束边界。(强: Bali MAX; An-Li参考点依赖)
3. **套力守恒定律**: 定价错误最大时套利资本最少; 强平螺旋因此永续。(强: Shleifer-Vishny 8400引用; 2015/2024两次A股实证)
4. **流动性租金定律**: 反转收益=流动性提供服务的价格, 是唯一理论上不衰减的alpha; 危机时租金最高, 且危机时提供者账面最痛。(强: Nagel2012; Virtu 1237/1238)
5. **制度摩擦定律**: A股隔夜负收益=T+1的开盘折价; 涨跌停/强赎条款/T+0转债各制造一块可提取的错价。(强: Qiao-Dam; 可转债双低标本)
6. **公开即腐烂定律**: 因子发表后-58%; 拥挤后失效; 任何策略的有效寿命与其知道的人数成反比。(强: McLean-Pontiff; 指数效应7.6%→0.8%; 双低2022; 指增1.9%)
7. **保费与雷劈定律**: 卖波动率是卖保险: 长期正租金+归零型左尾; 裸卖方终将被单日-96%清算, 生存者都是买了再保险的。(强: BTZ2009/Carr-Wu; XIV; OptionSellers)
8. **√N定律**: corr≈0的N腿使Sharpe×√N, 4腿=×2; 这是散户唯一可得的"免费午餐", 也是Kelly杠杆能安全上调的唯一正当理由。(强: 数学; 实证中: Thorp/PNP)
9. **容量-频率-收益三角**: 年化与频率成正比、与容量成反比; 66%需要日内+千人团队, 20%可容纳千亿美元; 50万资金的日线体系理论上限在Sharpe 3-4。(强: Medallion vs Buffett两极)
10. **200%路径判定律**: 长期净200%无公开先例; 单年200%只在"小资金+多腿低相关+高Sharpe+3-5x杠杆+危机降杠杆纪律"的Medallion低配结构上数学可达, 且已处Kelly边界; 用单腿或裸卖权够200% = 复刻XIV。(结论性, 由1-9联合推出)

**对我们200%目标的直接回答**: 目标应重述为"组合现实Sharpe≥3 + 杠杆≤0.25f*下的年化100-150%, 高波动年份自然突破200%"。在这条路径上它是数学可达的; 在任何"单腿日线无杠杆"或"裸卖权高杠杆"路径上它数学不可达——前者上限是现实82-97%, 后者的期望寿命以XIV的14个月计。

---

## 附: 本次调研直接核验过的原始文献(其余见各节内联)

- Greenwood & Sammon, "The Disappearing Index Effect"(NBER w30748): https://www.nber.org/papers/w30748 — 纳入效应3.4%/7.6%/0.8%三级衰减, 已读原文摘要。
- Barber, Lee, Liu & Odean (RFS 2009)PDF已下载并核对: 个体-3.8pp/年, 机构+1.5pp/年, 损失=台湾GDP2.2%: https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/justhowmuchdoindividualinvestorslose_rfs_2009.pdf
- Peng Yifeng (arXiv 2403.00474)PDF已下载并核对: 中国ETF期权卖波动率2018年前后收益反转: https://arxiv.org/abs/2403.00474
- NBER w31317(Kogan et al.加密散户): https://www.nber.org/system/files/working_papers/w31317/w31317.pdf
- NBER w20439(Momentum Crashes): https://www.nber.org/system/files/working_papers/w20439/w20439.pdf
- George & Hwang 2004 (52周高点)PDF: https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf
- Thorp/PNP与Kelly实操(转引): http://compoundingmyinterests.com/compounding-the-blog/2012/10/12/how-did-ed-thorp-win-in-blackjack-and-the-stock-market.html

**主要[快照]/[未证实]项**(需后续二次核验): ①A股散户交易占比"约60%"的券商联合口径原文; ②2024年雪球存量2000亿/IC六成持仓的信达研报原表; ③2026Q1超短21%盈利统计(社区数据); ④巴西日内97%亏损的具体数字; ⑤上交所年鉴2018卷PDF原件(官方页可下: https://www.sse.com.cn/aboutus/publication/yearly/ )。
