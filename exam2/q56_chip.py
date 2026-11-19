# -*- coding: utf-8 -*-
"""考卷二 题五/题六. 预注册: 悟道附录@488. 申报: chip_dist 是成本区间摘要非全分布
(陈旧至 2026-07), 题五用 套牢质量代理 = 1-获利比例 + 90成本带上沿悬空距离, 方向不变.
题五 空穴密度线: 套牢质量高 -> 未来60棒上破+10% 失败率高 (只注册梯度).
题六 载流子判别: 放量大阳(量>=90分位∧涨>=7%)后 量半衰期长(仓位型) 前向60棒收益 >
  半衰期短(脉冲型), 分层 mom20 后仍立才 PASS (全被动量吸收=MISS)."""
import glob
import numpy as np
import pandas as pd
from common import Pool, save_json

P = Pool()

# ---------- 题五 chip ----------
rows = []
for f in sorted(glob.glob('/home/muller/git/quant_trade-main/data/chip_dist/*.csv')):
    try:
        df = pd.read_csv(f, usecols=['ts_code', 'trade_date', '获利比例', '90成本-高'])
    except Exception:
        continue
    rows.append(df)
cd = pd.concat(rows, ignore_index=True)
del rows
cd['trade_date'] = pd.to_numeric(cd['trade_date'].astype(str).str.replace('-', ''), errors='coerce')
cd = cd.dropna(subset=['trade_date'])
cd['ym'] = (cd['trade_date'].astype('int64') // 100)
cd = cd.sort_values(['ts_code', 'ym', 'trade_date'])
snap = cd.groupby(['ts_code', 'ym'], as_index=False).last()  # 月末
print(f"chip_dist: {snap.ts_code.nunique()} 股 {int(snap.ym.min())}-{int(snap.ym.max())} {len(snap)} 股月")

# 月末套牢质量 + 未来60棒上破判定
recs = []
c_of = {code: i for i, code in enumerate(P.codes)}
ym_of_row = P.yearmonth[P.didx]
_pr = snap['获利比例'].to_numpy()
_bhi = snap['90成本-高'].to_numpy()
_cd = snap['ts_code'].to_numpy()
_ym = snap['ym'].to_numpy()
for code, ym, pr, bhi in zip(_cd, _ym, _pr, _bhi):
    c = c_of.get(code)
    if c is None or pr != pr or bhi != bhi:
        continue
    s, e = P.start[c], P.end[c]
    sel = np.nonzero(ym_of_row[s:e] == ym)[0]
    if len(sel) == 0:
        continue
    i0 = s + sel[-1]
    if i0 + 60 >= e:
        continue
    close0 = P.close[i0]
    hi60 = P.high[i0 + 1:i0 + 61].max()
    recs.append((1.0 - pr, max(0.0, bhi / close0 - 1.0),
                 float(hi60 / close0 - 1.0 >= 0.10)))
df5 = pd.DataFrame(recs, columns=['trapped', 'overhead', 'fail'])
print(f'题五 样本 {len(df5)} 股月, 上破率 {1-df5.fail.mean():.3%}')
df5['t_dec'] = pd.qcut(df5['trapped'].rank(method='first'), 10, labels=False) + 1
df5['o_dec'] = pd.qcut(df5['overhead'].rank(method='first'), 10, labels=False) + 1
t_prof = df5.groupby('t_dec')['fail'].mean()
o_prof = df5.groupby('o_dec')['fail'].mean()
print('题五 套牢质量十分位失败率:', [f'{v:.3f}' for v in t_prof.values])
print('题五 上沿悬空十分位失败率:', [f'{v:.3f}' for v in o_prof.values])
print(f"题五 梯度: 套牢 D10-D1 = {(t_prof[10]-t_prof[1])*100:+.2f}pp, 悬空 D10-D1 = {(o_prof[10]-o_prof[1])*100:+.2f}pp")

# ---------- 题六 载流子 ----------
# 事件: 量 >= 滚动252日90分位(min120) 且 pct_chg>=7%; 半衰期 = 量回落到 <1.5x前60日中位 的棒数(cap 40)
evs = []
for c in range(len(P.codes)):
    s, e = P.start[c], P.end[c]
    if e - s < 180:
        continue
    v = P.vol[s:e]
    r = P.ret[s:e]
    if len(v) < 180:
        continue
    q90 = pd.Series(v).rolling(252, min_periods=120).quantile(0.9).values
    med = pd.Series(v).rolling(60, min_periods=30).median().values
    hits = np.nonzero((v >= q90) & (r >= 0.07))[0]
    for i in hits:
        if i < 120 or q90[i] != q90[i] or med[i] != med[i] or med[i] <= 0:
            continue
        thr = 1.5 * med[i]
        j = i + 1
        hl = 40
        while j < min(i + 41, e - s):
            if v[j] < thr:
                hl = j - i
                break
            j += 1
        if i + 60 >= e - s:
            continue
        fwd = P.close[i + 60] / P.close[i] - 1.0
        mom20 = P.close[i] / P.close[i - 20] - 1.0 if i - 20 >= 0 else np.nan
        evs.append((hl, float(fwd), float(mom20)))
df6 = pd.DataFrame(evs, columns=['hl', 'fwd60', 'mom20']).dropna()
med_hl = df6.hl.median()
pos, pul = df6[df6.hl >= med_hl], df6[df6.hl < med_hl]
raw_gap = pos.fwd60.mean() - pul.fwd60.mean()
print(f'题六 事件 {len(df6)}, 半衰期中位 {med_hl:.0f}棒: 仓位型 fwd60 {pos.fwd60.mean():+.4f} vs '
      f'脉冲型 {pul.fwd60.mean():+.4f} 原始差 {raw_gap*100:+.2f}pp')
# 分层 mom20 十分位
df6['m_dec'] = pd.qcut(df6.mom20.rank(method='first'), 10, labels=False) + 1
gaps, ws = [], []
for d, g in df6.groupby('m_dec'):
    p = g[g.hl >= med_hl].fwd60.mean()
    q = g[g.hl < med_hl].fwd60.mean()
    if len(g) >= 100:
        gaps.append(p - q)
        ws.append(len(g))
strat_gap = float(np.average(gaps, weights=ws))
print(f'题六 mom20 分层后差: {strat_gap*100:+.2f}pp ({len(gaps)} 层) vs 原始 {raw_gap*100:+.2f}pp')

save_json('q56', dict(
    q5=dict(n=len(df5), breakthrough_rate=float(1 - df5.fail.mean()),
            trapped_decile=[float(x) for x in t_prof.values],
            overhead_decile=[float(x) for x in o_prof.values],
            gap_trapped_pp=float((t_prof[10] - t_prof[1]) * 100),
            gap_overhead_pp=float((o_prof[10] - o_prof[1]) * 100)),
    q6=dict(n=len(df6), med_hl=float(med_hl),
            pos_fwd=float(pos.fwd60.mean()), pul_fwd=float(pul.fwd60.mean()),
            raw_gap_pp=float(raw_gap * 100), strat_gap_pp=strat_gap,
            n_strata=len(gaps))))
print('Q56 DONE')
