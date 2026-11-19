# -*- coding: utf-8 -*-
"""考卷二 题一/二/三 (纯日线池). 预注册方向冻结于悟道附录@488.
题一 安静×对齐先行量: 危机前1-2月 先行量 高于非危机月; 加分项 lead>=vol报警.
题二 触板腿: 触板常态月(月触板率top分位)后 1/3/6月危机频率 > 基率, 三窗并报.
题三 自由度坍缩: K_eff=(Σλ)²/Σλ² 危机月 < 常态月 (800只随机子样×3)."""
import numpy as np
import pandas as pd
from common import Pool, mw_pvalue, save_json

rng = np.random.default_rng(20261119)
P = Pool()
mon = P.month_frame()
pool_mon, crisis = P.crisis_months(mon)
crisis_set = set(crisis)
all_ym = sorted(pool_mon.index.tolist())
base_rate = len(crisis) / len(all_ym)
print(f'月数 {len(all_ym)} 危机月 {len(crisis)} 基率 {base_rate:.3f}')
print('危机月列表:', crisis)
KNOWN = [201507, 201601, 201810, 202402, 202111, 202108, 202106, 202003, 202204, 202401]
print('已知窗命中:', [m for m in KNOWN if m in crisis_set], '未入:', [m for m in KNOWN if m not in crisis_set])

# ---------- 题一 安静×对齐先行量 ----------
# 安静: 每股 amount/滚动252d中位 -> 池中位(日) -> 20日均值 -> 全期百分位; quietness=1-该分位
norm_am = np.full(P.n, np.nan)
# 分股向量化
cs, ce = P.start, P.end
amt = P.amount
for i in range(len(P.codes)):
    s = amt[cs[i]:ce[i]]
    if len(s) < 120:
        continue
    ser = pd.Series(s)
    med = ser.rolling(252, min_periods=120).median().values
    v = s / med
    norm_am[cs[i]:ce[i]] = v
dfr = pd.DataFrame({'d': P.didx, 'ym': P.yearmonth[P.didx], 'na': norm_am})
day_q = dfr.groupby('d')['na'].median()
day_q20 = day_q.rolling(20, min_periods=10).mean()
quiet_daily = 1.0 - day_q20.rank(pct=True)

# 对齐: 每月 |Σsign(月收益)|/N (ndays>=10)
ali = mon[mon['ndays'] >= 10].groupby('ym')['mret'].apply(lambda s: np.sign(s).sum() / len(s)).abs()
# 先行量 at 月末
last_day_of_ym = pd.Series(P.didx, index=np.arange(P.n)).groupby(P.yearmonth[P.didx]).max()
precursor = {}
for ym in all_ym:
    d = last_day_of_ym.get(ym)
    q = quiet_daily.get(d, np.nan)
    a = ali.get(ym, np.nan)
    precursor[ym] = (q * a) if (q == q and a == a) else np.nan
prec = pd.Series(precursor)

# vol 报警 (加分对照): |池日收益| 20日均值 -> 分位
pool_day_ret = pd.DataFrame({'d': P.didx, 'ret': P.ret}).groupby('d')['ret'].median()
vol_alarm_daily = pool_day_ret.abs().rolling(20, min_periods=10).mean().rank(pct=True)
valarm = {}
for ym in all_ym:
    d = last_day_of_ym.get(ym)
    valarm[ym] = vol_alarm_daily.get(d, np.nan)
valarm = pd.Series(valarm)

ym_arr = np.array(all_ym)
res1 = {}
for lag in (1, 2):
    pre_crisis, pre_non = [], []
    for ym in all_ym:
        t = prec.get(ym, np.nan)
        if t != t:
            continue
        nxt = ym_arr[ym_arr.searchsorted(ym) + lag] if ym_arr.searchsorted(ym) + lag < len(ym_arr) else None
        if nxt is None:
            continue
        (pre_crisis if nxt in crisis_set else pre_non).append(t)
    u, pv = mw_pvalue(pre_crisis, pre_non)
    res1[f'lag{lag}'] = dict(n_crisis=len(pre_crisis), n_non=len(pre_non),
                             mean_crisis=float(np.mean(pre_crisis)), mean_non=float(np.mean(pre_non)),
                             ratio=float(np.mean(pre_crisis) / np.mean(pre_non)),
                             mw_u=float(u), p=float(pv))
    print(f"题一 lag{lag}: 危机前均值 {np.mean(pre_crisis):.4f} vs 非危机 {np.mean(pre_non):.4f} "
          f"比 {np.mean(pre_crisis)/np.mean(pre_non):.3f} p={pv:.4g}")

# 加分项: 信号先行月数 = 危机前信号首次进 top 三分位的提前月数 中位
def lead_months(sig):
    leads = []
    for cy in crisis:
        i = ym_arr.searchsorted(cy)
        cnt = None
        for k in (1, 2, 3, 4, 5, 6):
            j = i - k
            if j < 0:
                break
            v = sig.get(ym_arr[j], np.nan)
            if v == v and v > 2 / 3:
                cnt = k
        leads.append(cnt)
    valid = [x for x in leads if x is not None]
    return dict(hit=len(valid) / len(leads), med_lead=float(np.median(valid)) if valid else None)

lead_prec = lead_months(prec)
lead_vol = lead_months(valarm)
print('题一加分: 先行量 lead', lead_prec, '| vol报警 lead', lead_vol)

# ---------- 题二 触板腿 ----------
# 触板 = |pct_chg|>=9.5 (20cm 板少计, 代理申报). 月触板率 = 池内日均触板比例.
ret = P.ret
touch = (np.abs(ret) >= 0.095).astype(np.float32)
dfr2 = pd.DataFrame({'ym': P.yearmonth[P.didx], 't': touch})
touch_rate = dfr2.groupby('ym')['t'].mean()
touch_rate = touch_rate.reindex(all_ym)
hot_thr = touch_rate.quantile(0.9)
hot = touch_rate >= hot_thr
res2 = {}
for w in (1, 3, 6):
    hit, tot = 0, 0
    for i, ym in enumerate(all_ym):
        if not hot.get(ym, False):
            continue
        nxt = ym_arr[i + 1: i + 1 + w]
        if len(nxt) == 0:
            continue
        tot += 1
        if any(y in crisis_set for y in nxt):
            hit += 1
    res2[f'w{w}'] = dict(n_hot=tot, freq=hit / tot if tot else None, base=base_rate,
                         ratio=(hit / tot) / base_rate if tot else None)
    print(f"题二 触板后{w}月危机频率 {hit}/{tot} = {hit/tot if tot else float('nan'):.3f} "
          f"(基率 {base_rate:.3f}, 比 {((hit/tot)/base_rate) if tot else float('nan'):.2f})")

# ---------- 题三 自由度坍缩 K_eff ----------
# 月内日收益矩阵 (只取当月天数>=17的股), 800只随机子样×3, 相关矩阵特征值.
sub = pd.DataFrame({'c': P.cidx, 'ym': P.yearmonth[P.didx], 'ret': P.ret, 'd': P.didx})
sub = sub[sub.groupby(['c', 'ym'])['ret'].transform('size') >= 17]
keff = {}
for ym, grp in sub.groupby('ym'):
    m = grp.pivot(index='d', columns='c', values='ret').to_numpy()
    n_st = m.shape[1]
    if n_st < 100:
        continue
    vals = []
    for _ in range(3):
        cols = rng.choice(n_st, size=min(800, n_st), replace=False)
        mm = m[:, cols]
        mm = mm[:, ~np.isnan(mm).any(axis=0)]
        if mm.shape[1] < 100:
            continue
        mm = (mm - mm.mean(0)) / (mm.std(0) + 1e-12)
        try:
            ev = np.linalg.eigvalsh(np.corrcoef(mm.T))
        except Exception:
            continue
        ev = np.clip(ev, 0, None)
        vals.append((ev.sum() ** 2) / (ev ** 2).sum())
    if vals:
        keff[ym] = float(np.mean(vals))
ks = pd.Series(keff)
kc = ks[ks.index.isin(crisis_set)]
kn = ks[~ks.index.isin(crisis_set)]
u3, p3 = mw_pvalue(kc.values, kn.values)
print(f'题三 K_eff: 危机月均值 {kc.mean():.1f} vs 常态 {kn.mean():.1f} '
      f'(n={len(kc)}/{len(kn)}) p={p3:.4g}')

save_json('q123', dict(
    crisis_months=[int(x) for x in crisis], base_rate=base_rate, n_months=len(all_ym),
    q1=res1, q1_lead=dict(precursor=lead_prec, vol_alarm=lead_vol),
    q2=res2,
    q3=dict(keff_crisis_mean=float(kc.mean()), keff_non_mean=float(kn.mean()),
            ratio=float(kc.mean() / kn.mean()), p=float(p3), n_crisis=int(len(kc)), n_non=int(len(kn)),
            keff_by_month={int(k): v for k, v in ks.items()})))
print('Q123 DONE')
