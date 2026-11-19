# -*- coding: utf-8 -*-
"""考卷二公共底座: 池加载/月框架/危机月定义/工具. 预注册: papers/study_wudao_...md 附录@488."""
import os, glob, json
import numpy as np
import pandas as pd

DATA = '/home/muller/git/quant_trade-main/data'
DIRS = [DATA + '/daily_data2', DATA + '/daily_data3', DATA + '/daily_data2_delisted']
CACHE = os.path.dirname(os.path.abspath(__file__)) + '/cache_pool.npz'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/out'
os.makedirs(OUT, exist_ok=True)

USECOLS = ['ts_code', 'trade_date', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount']


def build_cache(verbose=True):
    """全池一次读入 -> 长表数组缓存 (code_idx/date_idx/close/high/low/ret/vol/amount).
    d2 优先, d3/delisted 只补 d2 没有的 code. 幸存者偏差: delisted 在池."""
    have = set()
    frames = []
    for d in DIRS:
        files = sorted(glob.glob(d + '/*.csv'))
        n_new = 0
        for f in files:
            code = os.path.basename(f)[:-4]
            if code in have:
                continue
            try:
                df = pd.read_csv(f, usecols=USECOLS)
            except Exception:
                continue
            if df.empty or 'trade_date' not in df.columns:
                continue
            frames.append(df)
            have.add(code)
            n_new += 1
        if verbose:
            print(f'{os.path.basename(d)}: +{n_new} (cum {len(have)})', flush=True)
    big = pd.concat(frames, ignore_index=True)
    del frames
    for c in ('trade_date', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount'):
        big[c] = pd.to_numeric(big[c], errors='coerce')
    big['trade_date'] = big['trade_date'].astype('int64')
    big = big.dropna(subset=['close']).sort_values(['ts_code', 'trade_date'], kind='stable')
    # ret: pct_chg 优先, 缺则 close/pre_close via 组内 shift
    ret = big['pct_chg'] / 100.0
    fallback = big.groupby('ts_code')['close'].transform(lambda s: s / s.shift(1) - 1.0)
    ret = ret.where(ret.notna() & (ret.abs() < 0.95), fallback)
    big['ret'] = ret
    big = big.dropna(subset=['ret'])
    codes = np.array(sorted(big['ts_code'].unique()))
    cidx = pd.Categorical(big['ts_code'], categories=codes).codes.astype(np.int32)
    dates = np.array(sorted(big['trade_date'].unique()))
    didx = np.searchsorted(dates, big['trade_date'].values).astype(np.int32)
    np.savez(CACHE,
             codes=codes.astype(str), dates=dates,
             cidx=cidx, didx=didx,
             close=big['close'].values.astype(np.float32),
             high=big['high'].values.astype(np.float32),
             low=big['low'].values.astype(np.float32),
             ret=big['ret'].values.astype(np.float32),
             vol=np.nan_to_num(big['vol'].values.astype(np.float64)),
             amount=np.nan_to_num(big['amount'].values.astype(np.float64)))
    if verbose:
        print(f'cache: {len(codes)} stocks, {len(dates)} days, {len(big)} rows -> {CACHE}', flush=True)
    return big


class Pool:
    """缓存读取 + 常用派生: 每股切片/月框架/危机月."""

    def __init__(self):
        z = np.load(CACHE, allow_pickle=False)
        self.codes = z['codes']
        self.dates = z['dates']
        self.cidx, self.didx = z['cidx'], z['didx']
        self.close, self.high, self.low = z['close'], z['high'], z['low']
        self.ret, self.vol, self.amount = z['ret'], z['vol'], z['amount']
        self.n = len(self.cidx)
        self.yearmonth = (self.dates // 100).astype(np.int64)  # YYYYMM
        # 每股切片 (cidx 已排序)
        self.start = np.searchsorted(self.cidx, np.arange(len(self.codes)), 'left')
        self.end = np.searchsorted(self.cidx, np.arange(len(self.codes)), 'right')

    def stock_slice(self, i):
        return slice(self.start[i], self.end[i])

    def month_frame(self):
        """每股每月: 月收益(简单)/月内天数/月末行号. 全池月表: 等权月收益."""
        ym = self.yearmonth[self.didx]
        df = pd.DataFrame({'c': self.cidx, 'ym': ym, 'ret': self.ret,
                           'close': self.close, 'high': self.high})
        df['lr'] = np.log1p(df['ret'].clip(-0.9, 5))
        df['row'] = np.arange(len(df))
        g = df.groupby(['c', 'ym'], sort=True)
        mon = g.agg(lr=('lr', 'sum'), ndays=('ret', 'size'))
        last = df.groupby(['c', 'ym'])['row'].max().rename('irow')
        mon = mon.join(last)
        mon['mret'] = np.expm1(mon['lr'])
        return mon.reset_index()  # c, ym, lr, ndays, irow, mret

    def crisis_months(self, mon, min_stocks=500, q=0.10):
        """危机月 = 全池等权月收益最差 10% 分位 (透明定义), 月覆盖>=500 只."""
        pool_mon = mon[mon['ndays'] >= 10].groupby('ym')['mret'].agg(['mean', 'size'])
        pool_mon = pool_mon[pool_mon['size'] >= min_stocks]
        thr = pool_mon['mean'].quantile(q)
        crisis = sorted(pool_mon.index[pool_mon['mean'] <= thr].tolist())
        return pool_mon, crisis


def mw_pvalue(x, y):
    """Mann-Whitney U 双侧 p (正态近似, 无并列修正够用)."""
    from scipy.stats import norm
    x, y = np.asarray(x, float), np.asarray(y, float)
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank().values
    nx = len(x)
    U = ranks[:nx].sum() - nx * (nx + 1) / 2.0
    n1, n2 = nx, len(y)
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd == 0:
        return np.nan, np.nan
    z = (U - mu) / sd
    return U / (n1 * n2), 2 * norm.sf(abs(z))


def board(code):
    p = code[:2]
    if p in ('60',):
        return 'SH_MB'
    if p == '68':
        return 'SH_KC'
    if p == '00':
        return 'SZ_MB'
    if p == '30':
        return 'SZ_CY'
    return 'BJ_OTH'


def save_json(name, obj):
    p = f'{OUT}/{name}.json'
    with open(p, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    print(f'artifact -> {p}', flush=True)


if __name__ == '__main__':
    build_cache()
