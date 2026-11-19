# -*- coding: utf-8 -*-
"""考卷二 题四/题九 (两融调入调出自然实验). 预注册: 悟道附录@488.
题四 调入: 入两融标的后 +5/+20/+60 棒 相对配对对照无正超额 (调入≠利好) 且死亡谱不变差.
题九 调出: 两融缺席>=3月而股仍在交易 -> 短窗 5-20 棒负超额 ∧ 长窗 60 棒回归对照, 两面齐才 PASS.
配对: 同月/同板/规模十分位±1/前60棒涨幅十分位±1, 每事件<=5 对照. 事件锚=事件月月末."""
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from common import Pool, board, save_json

P = Pool()
mon = P.month_frame()
c_of_code = {code: i for i, code in enumerate(P.codes)}

# 全股月特征
pr60 = np.full(len(mon), np.nan)
sz = np.full(len(mon), np.nan)
for r in mon.itertuples():
    c, irow = int(r.c), int(r.irow)
    s = P.start[c]
    if irow - 60 >= s:
        pr60[r.Index] = P.close[irow] / P.close[irow - 60] - 1.0
    if irow - 20 >= s:
        sz[r.Index] = np.log(P.amount[irow - 20:irow].mean() + 1.0)
mon['pr60'], mon['sz'] = pr60, sz
mon['code'] = P.codes[mon['c'].values]
mon['board'] = [board(x) for x in mon['code']]

pool = mon.dropna(subset=['pr60', 'sz']).copy()
pool['sz_dec'] = pool.groupby('ym')['sz'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
pool['pr_dec'] = pool.groupby('ym')['pr60'].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
# 桶: ym -> (board, sz_dec, pr_dec) -> [(code, irow)]
buckets = defaultdict(lambda: defaultdict(list))
for r in pool.itertuples():
    buckets[int(r.ym)][(r.board, int(r.sz_dec), int(r.pr_dec))].append((r.code, int(r.irow)))
irow_of = {(r.code, int(r.ym)): r for r in pool.itertuples()}  # (code,ym) -> 命名元组

# ---- 事件定义 (margin_daily 在册月份) ----
present = {}
for f in sorted(glob.glob('/home/muller/git/quant_trade-main/data/margin_daily/*.csv')):
    code = f.split('/')[-1][:-4]
    try:
        s = pd.read_csv(f, usecols=['trade_date'])
        present[code] = sorted(set((s['trade_date'].astype(str).str[:6]).astype(int)))
    except Exception:
        continue
print(f'margin_daily 在册股 {len(present)}')

in_ev, out_ev = [], []
for code, yms in present.items():
    c = c_of_code.get(code)
    if c is None:
        continue
    s, e = P.start[c], P.end[c]
    all_ym = sorted(set(P.yearmonth[P.didx[s:e]].tolist()))
    allset = set(all_ym)
    for i, ym in enumerate(yms):
        if i == 0:
            prev = [y for y in all_ym if y < ym]
            gap_ok = len(prev) >= 3
        else:
            prev = [y for y in all_ym if yms[i - 1] < y < ym]
            gap_ok = len(prev) >= 3
        if gap_ok and ym in allset:
            in_ev.append((code, c, ym))
    for i, ym in enumerate(yms):
        nxt = yms[i + 1] if i + 1 < len(yms) else None
        after = [y for y in all_ym if y > ym and (nxt is None or y < nxt)]
        if len(after) >= 4 and ym in allset:
            out_ev.append((code, c, ym))
print(f'调入事件 {len(in_ev)}, 调出事件 {len(out_ev)}')


def rel(r0, c, bars):
    r1 = min(r0 + bars, P.end[c] - 1)
    return P.close[r1] / P.close[r0] - 1.0 if r1 > r0 else np.nan


def me_feat(code, ym):
    b = buckets.get(int(ym))
    me = irow_of.get((code, int(ym)))
    if b is None or me is None:
        return None
    return b, me


def study(events, label):
    recs = []
    for code, c, ym in events:
        got = me_feat(code, ym)
        if got is None:
            continue
        b, me = got
        cand = []
        for dsz in (-1, 0, 1):
            for dpr in (-1, 0, 1):
                cand += b.get((me.board, int(me.sz_dec) + dsz, int(me.pr_dec) + dpr), [])
        cand = [x for x in cand if x[0] != code][:5]
        if len(cand) < 2:
            continue
        r0 = int(me.irow)
        rec = dict(code=code, ym=int(ym))
        for bb in (5, 20, 60):
            rec[f'r{bb}'] = rel(r0, c, bb)
            rec[f'c{bb}'] = np.nanmedian([rel(ir, c_of_code[cc], bb) for cc, ir in cand])
        fwd = rel(r0, c, 60)
        cfwds = [rel(ir, c_of_code[cc], 60) for cc, ir in cand]
        if fwd == fwd:
            rec['death'] = int(fwd <= -0.30)
            rec['cdeath'] = float(np.mean([x <= -0.30 for x in cfwds if x == x]))
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty or len(df) < 50:
        print(f'{label}: 样本不足 ({len(df)})')
        return dict(n=len(df))
    out = dict(n=len(df), ym_span=[int(df.ym.min()), int(df.ym.max())])
    for bb in (5, 20, 60):
        rr = df[f'r{bb}'] - df[f'c{bb}']
        rr = rr.dropna()
        out[f'rel{bb}'] = dict(mean=float(rr.mean()), med=float(rr.median()),
                               pos_share=float((rr > 0).mean()), n=int(len(rr)))
    d = df.dropna(subset=['death'])
    out['death'] = dict(event=float(d.death.mean()), ctrl=float(d.cdeath.mean()),
                        ratio=float(d.death.mean() / max(d.cdeath.mean(), 1e-9)), n=int(len(d)))
    print(f"{label} n={len(df)}: rel5 {out['rel5']['mean']:+.4f} rel20 {out['rel20']['mean']:+.4f} "
          f"rel60 {out['rel60']['mean']:+.4f} | 死亡 {out['death']['event']:.3%} vs 对照 {out['death']['ctrl']:.3%}")
    return out


res_in = study(in_ev, '题四 调入')
res_out = study(out_ev, '题九 调出')
save_json('q49', dict(q4=res_in, q9=res_out))
print('Q49 DONE')
