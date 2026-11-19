# -*- coding: utf-8 -*-
"""考卷二 题七/题八 (两融杠杆). 预注册: 悟道附录@488.
申报(数据适配): 盘上无全史流通市值 (circ_mv_history/daily_basic/margin_factors 皆短窗),
题七八改用 出清天数杠杆代理 lev = rzye / 20日均成交额 (余额vs对手盘消化能力), 方向不变;
并以 daily_basic(2016-01~2017-10 真实circ_mv) 验证代理与真 rzye_pct 的秩相关.
题七: 市场合计 lev 月 >90分位后 1/3/6月危机频率 > 基率.
题八: 个股 lev 月内十分位 × 危机月 -> 死亡梯度只在危机月正, 分层去"贵"后仍正才 PASS."""
import glob
import numpy as np
import pandas as pd
from common import Pool, save_json

P = Pool()
mon = P.month_frame()
pool_mon, crisis = P.crisis_months(mon)
crisis_set = set(crisis)
all_ym = sorted(pool_mon.index.tolist())
base_rate = len(crisis) / len(all_ym)

# ---- 每股 amount20 (月末前20棒均额) ----
am20 = np.full(len(mon), np.nan)
for r in mon.itertuples():
    c, irow = int(r.c), int(r.irow)
    s = P.start[c]
    if irow - 20 >= s:
        am20[r.Index] = P.amount[irow - 20:irow].mean()
mon['am20'] = am20
mon['code'] = P.codes[mon['c'].values]

# ---- margin_daily 月末 rzye ----
rows = []
for f in sorted(glob.glob('/home/muller/git/quant_trade-main/data/margin_daily/*.csv')):
    try:
        df = pd.read_csv(f, usecols=['ts_code', 'trade_date', 'rzye'])
    except Exception:
        continue
    if df.empty:
        continue
    df['ym'] = pd.to_numeric(df['trade_date'], errors='coerce') // 100
    df = df.dropna(subset=['ym'])
    df['ym'] = df['ym'].astype(int)
    df = df.sort_values('trade_date').groupby(['ts_code', 'ym'], as_index=False).last()
    rows.append(df[['ts_code', 'ym', 'rzye']])
rz = pd.concat(rows, ignore_index=True)
print(f'margin_daily: {rz.ts_code.nunique()} 股, {rz.ym.min()}-{rz.ym.max()}, {len(rz)} 股月')

key = pd.MultiIndex.from_arrays([mon['code'].values, mon['ym'].values])
mon['rzye'] = rz.set_index(['ts_code', 'ym'])['rzye'].reindex(key).values

# ---------- 题七 市场合计出清杠杆 ----------
m7 = mon.dropna(subset=['rzye', 'am20'])
m7 = m7[m7['am20'] > 0]
agg = m7.groupby('ym').apply(lambda g: g.rzye.sum() / (g.am20.sum() * 1000.0), include_groups=False)
agg = agg.sort_index()
thr = agg.quantile(0.9)
hot = agg >= thr
ym_arr = np.array(agg.index.tolist())
res7 = {}
for w in (1, 3, 6):
    hit = tot = 0
    for i in range(len(ym_arr)):
        if not hot.iloc[i] or i + 1 >= len(ym_arr):
            continue
        nxt = ym_arr[i + 1: i + 1 + w]
        tot += 1
        if any(int(y) in crisis_set for y in nxt):
            hit += 1
    res7[f'w{w}'] = dict(n_hot=tot, freq=hit / tot if tot else None,
                         ratio=(hit / tot) / base_rate if tot else None)
    print(f"题七 杠杆热后{w}月危机频率 {hit}/{tot} = {(hit/tot) if tot else float('nan'):.3f} "
          f"(基率 {base_rate:.3f})")
print('题七 热月:', [int(y) for y in agg.index[hot]], '| 杠杆比首末:', round(agg.iloc[0], 2), round(agg.iloc[-1], 2))

# ---------- 题八 杠杆×危机月 死亡交互 ----------
fwd60 = np.full(len(mon), np.nan)
prior60 = np.full(len(mon), np.nan)
sz = np.full(len(mon), np.nan)
for r in mon.itertuples():
    c, irow = int(r.c), int(r.irow)
    s, e = P.start[c], P.end[c]
    if irow + 60 < e:
        fwd60[r.Index] = P.close[irow + 60] / P.close[irow] - 1.0
    if irow - 60 >= s:
        prior60[r.Index] = P.close[irow] / P.close[irow - 60] - 1.0
    if r.Index == r.Index and irow - 20 >= s:
        sz[r.Index] = np.log(r.am20 + 1.0)
mon['fwd60'], mon['prior60'], mon['sz'] = fwd60, prior60, sz

sub = mon[mon['fwd60'].notna()].dropna(subset=['rzye', 'am20'])
sub = sub[sub['am20'] > 0].copy()
sub['lev'] = sub['rzye'] / (sub['am20'] * 1000.0)
sub['death'] = sub['fwd60'] <= -0.30
sub['lev_dec'] = sub.groupby('ym')['lev'].transform(
    lambda s: pd.qcut(s.rank(method='first'), 10, labels=False) + 1)
sub['is_crisis'] = sub['ym'].isin(crisis_set)
print(f'题八 样本: {len(sub)} 股月, 死亡 {sub.death.mean():.4%}, 危机股月 {sub.is_crisis.sum()}')


def gap(d):
    hi = d[d.lev_dec >= 8].death.mean()
    lo = d[d.lev_dec <= 3].death.mean()
    return float(lo), float(hi), float(hi - lo)


g_non, g_cri = gap(sub[~sub.is_crisis]), gap(sub[sub.is_crisis])
print(f'题八 常态月: D1-3 {g_non[0]:.3%} vs D8-10 {g_non[1]:.3%} 梯度 {g_non[2]*100:+.2f}pp (n={int((~sub.is_crisis).sum())})')
print(f'题八 危机月: D1-3 {g_cri[0]:.3%} vs D8-10 {g_cri[1]:.3%} 梯度 {g_cri[2]*100:+.2f}pp (n={int(sub.is_crisis.sum())})')

sub['p_ter'] = sub.groupby('ym')['prior60'].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False))
sub['s_ter'] = sub.groupby('ym')['sz'].transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=False))
strat = {}
for isc, nm in ((False, 'non'), (True, 'crisis')):
    d = sub[sub.is_crisis == isc]
    gs, ws = [], []
    for (pt, st), g in d.groupby(['p_ter', 's_ter']):
        nh, nl = int((g.lev_dec >= 8).sum()), int((g.lev_dec <= 3).sum())
        if nh < 50 or nl < 50:
            continue
        gs.append(g[g.lev_dec >= 8].death.mean() - g[g.lev_dec <= 3].death.mean())
        ws.append(min(nh, nl))
    strat[nm] = dict(n_cells=len(gs), stratified_gap_pp=float(np.average(gs, weights=ws)) * 100 if gs else None)
    print(f"题八 {'危机' if isc else '常态'} 分层去贵梯度: {strat[nm]['stratified_gap_pp']:+.2f}pp ({len(gs)} 格)")

# ---------- 代理校验: daily_basic 2016-17 真 rzye_pct 秩相关 ----------
import os
vb_files = sorted(glob.glob('/home/muller/git/quant_trade-main/data/daily_basic/*.csv'))
real = []
for f in vb_files:
    try:
        d = pd.read_csv(f, usecols=['ts_code', 'circ_mv'])
    except Exception:
        continue
    d['ym'] = int(os.path.basename(f)[:6])
    real.append(d)
rb = pd.concat(real, ignore_index=True).dropna()
rb = rb.groupby(['ts_code', 'ym'], as_index=False).last()
mm = sub.reset_index().rename(columns={'code': 'ts_code'}).merge(rb, on=['ts_code', 'ym'], how='inner')
mm = mm[mm['circ_mv'] > 0]
if len(mm) > 1000:
    mm['real_pct'] = mm['rzye'] / mm['circ_mv']
    mm['dec_proxy'] = pd.qcut(mm['lev'].rank(method='first'), 10, labels=False)
    mm['dec_real'] = pd.qcut(mm['real_pct'].rank(method='first'), 10, labels=False)
    sp = mm[['dec_proxy', 'dec_real']].corr(method='spearman').iloc[0, 1]
    print(f'代理校验: 2016-17 窗 n={len(mm)} 真比vs代理 十分位 Spearman {sp:.3f}')
else:
    sp = None
    print('代理校验: 样本不足', len(mm))

save_json('q78', dict(
    base_rate=base_rate,
    proxy='lev=rzye/(20d mean amount*1000), 全史circ_mv缺盘申报',
    q7=dict(thr=float(thr), hot_months=[int(y) for y in agg.index[hot]],
            lev_span=[float(agg.iloc[0]), float(agg.iloc[-1])], windows=res7),
    q8=dict(n=int(len(sub)), death_rate=float(sub.death.mean()),
            gap_non=g_non, gap_crisis=g_cri, stratified=strat),
    proxy_validation=dict(n=int(len(mm)) if len(mm) > 1000 else int(len(mm)), spearman=None if sp is None else float(sp))))
print('Q78 DONE')
