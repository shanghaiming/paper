# 期货开源/卖方策略普查 — 2026-11-13 (iter296)

股票侧 open_source_strategy_census_20261014.md 的方法论移植: 同 census 打法,
对象换期货 CTA 生态 (用户令: 类似研究, 非照搬)。零回测, 纯文献 × registry 交叉。

## 主表 (12 族 × 判词)

时序动量/趋势=书内活 (vnpy 3/6 件, 华泰系列7) · 截面动量反转=死 (iter294 OOS净
Sh0.53) · carry展期收益率=死三封 (209/194/293) · **基差动量=零判词** (Boons-Prado
2019 JF: Δ斜率, 不同于209封的斜率水平与194封的价差水平) · 库存仓单=死 (291/m426)
· 波动率=死 conditioner (265) · 成交量=死 (m426-S4/M377VOL) · OI三轴=死 (275-280)
· 季节性=分裂 (日频窗死/M626活) · 日内突破=分钟墙外 (RBreaker/DualThrust) ·
跨期价差=死 K0.23 (194) · **产业链利润MR=零判词** · 品种筛选=meta conditioner ·
RMT 74品种=明文留格 (265)。

学术三分之二已判: Koijen 2018 (Carry死/MKT-E6留格/TSMOM活)。

## 空白格 K 预筛

产业链利润MR: 900bp 成本 vs 200-500bp 毛 → K0.3-0.8 门毙。
基差动量: 480bp vs 100-1500bp → K[0.2,3.1] 横跨门线 — 唯一预注册翻案候选,
先实测 K 再谈立项。

## 结论

抄研报无出路 (1:1 覆盖); 增量在期货原生结构面: 执行(展期时点, 免K) >
RMT截面+β结构 > 基差动量K测量。"鸡蛋链43%/yr"类卖方数字 = 定律三 IS 樱桃判例。

## 源链接

- vnpy_ctastrategy: github.com/vnpy/vnpy_ctastrategy (内置6件)
- 华泰期货 CTA 因子系列3 (2017, 期限结构): htfc.com/upload/20171025/201710251508893187512.pdf
- 华泰 2021 Curve因子与基差动量: htfc.com/wz_upload/png_upload/20210714/1626254361452e581a9.pdf
- 海通 FICC 系列研究之十五 (2021); "海量"135 品种动态筛选 (2019)
- 银河期货 商品期货CTA截面策略初探 (2025-06): yhqh.com.cn
- Boons & Prado 2019, Basis-Momentum, JF; Koijen et al. 2018 Carry; De Groot et al. 2014
- 信易科技 钢厂利润套利 (rb−1.6i−0.5j); 东方证券 衍生品五 (套利实证); 光大 跨品种套利专题
- awesome-systematic-trading: github.com/paperswithbacktest/awesome-systematic-trading
