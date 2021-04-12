# -*- coding: utf-8 -*-

__author__ = "Gabriel Feng"

import pandas as pd
import numpy as np
import llquant.numba_recursive as nbr
import gc
import llquant.utilities as lut
import scipy.stats as scs


def chi2(x, y):
    """
    Simple method to evaluate the distance between a and b
    the results range from 0 to 1, standing for closest to farest
    """
    with np.errstate(divide='ignore'):
        a = x - y
        b = x + y
        return np.divide(a, b, out=np.zeros(a.shape), where=b != 0)


def calc_midprice(target_quote):
    price_lv1 = target_quote[['ap1', 'bp1']]
    midprice = price_lv1.mean(1)
    midprice[(price_lv1 == 0).any(1)] = price_lv1.max(1)
    midprice[(price_lv1 == 0).all(1)] = target_quote['new_price'].shift()
    return midprice.values


def calc_weight(askbid, curpri, th):
    # calculate weight about ask/bid distance from prices
    return np.exp(-th * abs(1 - np.divide(askbid, curpri)))


def calc_wb(target_quote, th=600):
    # prices_arr = calc_midprice(target_quote).reshape(-1, 1)
    prices_arr = target_quote['new_price'].values.reshape(-1, 1)
    ask_p = target_quote.filter(regex=r'ap\d').values
    ask_v = target_quote.filter(regex=r'av\d').values
    bid_p = target_quote.filter(regex=r'bp\d').values
    bid_v = target_quote.filter(regex=r'bv\d').values

    # calculate volume based on weights
    wbuyamt = calc_weight(bid_p, prices_arr, th) * bid_v
    wsellamt = calc_weight(ask_p, prices_arr, th) * ask_v

    wbuy_amt_sum = wbuyamt.sum(1)  # order long
    wsell_amt_sum = wsellamt.sum(1)  # order short
    wb = chi2(wbuy_amt_sum, wsell_amt_sum)

    return wb


def tba(target_quote, alpha=0.05):
    # ba_ratio = np.log(target_quote['bv1'] / target_quote['av1'])
    bv1, av1 = target_quote[['bv1', 'av1']].values.astype(np.float).T
    ba_ratio = np.sqrt(np.divide(bv1, av1, out=np.full_like(bv1, np.nan), where=av1 != 0))
    ba_ratio_ema = nbr.numba_ewma(ba_ratio, alpha=alpha, state=None, adjust=True, ignore_na=True, minp=1)[0]
    return ba_ratio_ema


def calc_tba_express(code, date):
    # TODO to be tested
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, tba(df, alpha=0.05)[-1]
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def dbook(target_quote, n=1, alpha=0.05):
    bv1, av1 = target_quote[['bv1', 'av1']].values.astype(np.float).T
    diff_bv1 = np.insert(np.sign(np.diff(bv1, n=n)), [0] * n, 0)
    diff_av1 = np.insert(np.sign(np.diff(av1, n=n)), [0] * n, 0)
    rs = nbr.numba_ewma((diff_bv1 - diff_av1), alpha=alpha, state=None, adjust=True, ignore_na=True, minp=1)[0]
    return rs


def calc_dbook_express(code, date):
    # TODO to be tested
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, dbook(df, n=1, alpha=0.05)[-1]
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def approx_price(target_quote, limit_flag=1):
    """volume-weighted price a.k.a wpr

    Args:
        target_quote (TYPE): Description
        limit_flag (int, optional): Description

    Returns:
        TYPE: Description
    """
    ap1, bp1, av1, bv1 = target_quote[['ap1', 'bp1', 'av1', 'bv1']].values.T
    x = ap1 * bv1 + bp1 * av1
    y = bv1 + av1
    smp = np.divide(x, y, out=np.full_like(x, np.nan), where=y!=0)

    if limit_flag == 1:
        is_ap_eq_0 = target_quote['ap1'].values == 0
        is_bp_eq_0 = target_quote['bp1'].values == 0
        smp[is_ap_eq_0] = target_quote.loc[is_ap_eq_0, 'bp1'].values
        smp[is_bp_eq_0] = target_quote.loc[is_bp_eq_0, 'ap1'].values
        smp[is_ap_eq_0 & is_bp_eq_0] = target_quote.loc[is_ap_eq_0 & is_bp_eq_0, 'new_price'].values

    if limit_flag == 2:
        touched_limit = np.any(target_quote[['ap1', 'bp1']] == 0, 1)
        smp[touched_limit] = target_quote.loc[touched_limit, 'new_price'].values

    smp = pd.Series(smp).ffill().values
    # smp = self.numpy_shift(smp, n=1)
    return smp


def calc_absr(df, window):
    """
    Estimate the active trading volume (absolute buy sell ratio).
    ask, bid volume and amount must be float.

    """
    ask, bid, volume, amount = df[['ap1', 'bp1', 'volume', 'turnover']].values.T

    vol = volume.copy().astype(float)
    np.place(vol, vol == 0, np.nan)
    avgcost = np.divide(amount, vol)
    spread = ask - bid

    # Seperate
    avgc_over_ask = avgcost[1:] > ask[:-1]
    avgc_below_bid = avgcost[1:] < bid[:-1]
    avgc_between = (~avgc_over_ask) & (~avgc_below_bid)
    vol_1 = vol[1:]

    # calculate active buyer
    actibuy = vol_1 * avgc_over_ask
    ab_ratios = np.divide((avgcost[1:] - bid[:-1]), spread[:-1])
    actibuy[avgc_between] = vol_1[avgc_between] * ab_ratios[avgc_between]

    # calculate active seller
    actisell = vol_1 * avgc_below_bid
    as_ratios = 1 - ab_ratios
    actisell[avgc_between] = vol_1[avgc_between] * as_ratios[avgc_between]

    actibuy_end = nbr.numba_ewma(actibuy, alpha=0.05)[0][-1]
    actisell_end = np.max([nbr.numba_ewma(actisell, alpha=0.05)[0][-1], 1])
    out = actibuy_end / actisell_end
    # active_volume = np.vstack([actibuy, actisell]).T.astype('float64')
    # active_volume = np.nan_to_num(active_volume)
    # active_volume_arr = np.full((len(vol), 2), np.nan)
    # active_volume_rolsum = np.sum(lut.sliding_window_view(active_volume.T, window), axis=2).T
    # active_volume_arr[window:] = active_volume_rolsum
    # out = np.log(np.divide(active_volume_arr[:, 0], active_volume_arr[:, 1], out=np.full(len(active_volume_arr), np.nan), where=active_volume_arr[:, 1]!=0))
    return out


def calc_absr_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_absr(df, 10)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def calc_factor1(df):
    # 基于订单簿的筹码倾向：以价格为标尺
    _obp = np.ravel(df.filter(regex=r'[ab]p\d+'))
    _obv = np.ravel(df.filter(regex=r'[ab]v\d+'))
    # ob_agg = pd.Series({p: np.mean(_obv[_obp == p]) for p in np.unique(_obp) if p != 0})
    ob_agg = pd.Series({p: nbr.numba_ewma(_obv[_obp == p], alpha=0.05)[-1][0] for p in np.unique(_obp) if p != 0})
    _mid = df['new_price'].iloc[-1]
    over = ob_agg[ob_agg.index > _mid].copy()
    wgt_over_sum = np.sum(over * over.index / _mid)
    under = ob_agg[ob_agg.index < _mid].copy()
    # wgt_under_sum = np.sum(under * _mid / under.index)
    wgt_under_sum = np.max([np.sum(under * _mid / under.index), 1])
    order_ratio = wgt_over_sum / wgt_under_sum
    return order_ratio


def calc_factor1_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_factor1(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def calc_factor2(df):
    # 基于订单簿的筹码倾向：以位置为标尺
    _obp = df.filter(regex=r'[ab]p\d+')
    argsort = np.argsort(_obp.iloc[0, :])
    _obp = _obp.iloc[:, argsort]
    _obv = df.filter(regex=r'[ab]v\d+')
    _obv = _obv.iloc[:, argsort]
    _wgt_obv = pd.DataFrame(columns=_obv.columns, index=_obv.index)
    _wgt_obv.iloc[:, :10] = _obv.iloc[:, :10] * df[['new_price']].values / _obp.iloc[:, :10].values
    _wgt_obv.iloc[:, 10:] = _obv.iloc[:, 10:] * _obp.iloc[:, 10:].values / df[['new_price']].values
    obv_ema = _wgt_obv.ewm(alpha=0.1).mean().iloc[-1]
    order_ratio = obv_ema.iloc[:10].sum() / obv_ema.iloc[10:].sum()
    return order_ratio


def calc_factor2_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_factor2(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def calc_factor3(df):
    # 基于逐笔成交的筹码倾向：以价格为标尺
    deal_mp = df.groupby('price')['volume'].sum()
    _mid = df['price'].iloc[-1]
    over = deal_mp[deal_mp.index > _mid].copy()
    wgt_over_sum = np.sum(over * over.index / _mid)
    under = deal_mp[deal_mp.index < _mid].copy()
    # wgt_under_sum = np.sum(under * _mid / under.index)
    wgt_under_sum = np.max([np.sum(under * _mid / under.index), 1])
    order_ratio = wgt_over_sum / wgt_under_sum

    # avg = pd.Series(volume, index=turnover / volume)
    # over =  avg[avg > _mid].copy()
    # under = avg[avg < _mid].copy()
    # wgt_over_sum = np.sum(over * over.index / _mid)
    # wgt_under_sum = np.max([np.sum(under * _mid / under.index), 1])
    # order_ratio = wgt_over_sum / wgt_under_sum
    return order_ratio


def calc_factor3_express(code, date):
    df = lut.load_deal_data(code=code, date=date, verbose=0)
    if df.shape[0] > 0:
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_factor3(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def calc_factor4(df):
    # 基于逐笔成交的当日相对大单
    df['timerank'] = df['timeint'].rank(method='dense')
    df['timeweight'] = np.exp(df['timerank'] / df['timerank'].nunique()) / np.exp(1)
    df['amount'] = df['price'] * df['volume'] * 100 / 1e6
    df['amount1'] = df['amount'] * df['timeweight']
    sale_amount1 = df.groupby('saleorderid')['amount'].sum()
    s_qt = sale_amount1.quantile(0.9)
    buy_amount1 = df.groupby('buyorderid')['amount'].sum()
    b_qt = buy_amount1.quantile(0.9)
    bsr = b_qt / s_qt
    return bsr


def calc_factor4_express(code, date):
    df = lut.load_deal_data(code=code, date=date, verbose=0)
    if df.shape[0] > 0:
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_factor4(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def calc_factor5(df):
    # 基于逐笔成交的当日最大单
    df['timerank'] = df['timeint'].rank(method='dense')
    df['timeweight'] = np.exp(df['timerank'] / df['timerank'].nunique()) / np.exp(1)
    df['amount'] = df['price'] * df['volume'] * 100 / 1e6
    df['amount1'] = df['amount'] * df['timeweight']
    sale_amount1 = df.groupby('saleorderid')['amount'].sum()
    s_qt = sale_amount1.max()
    buy_amount1 = df.groupby('buyorderid')['amount'].sum()
    b_qt = buy_amount1.max()
    bsr = b_qt / s_qt
    return bsr


def calc_factor5_express(code, date):
    df = lut.load_deal_data(code=code, date=date, verbose=0)
    if df.shape[0] > 0:
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, calc_factor5(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def price_volatility_updown(df):
    tgrouper = df.groupby(pd.Grouper(key='datetime', freq='60s', closed='right', label='right', convention='e'))
    avg_price = np.array((tgrouper['turnover'].sum() / tgrouper['volume'].sum()).dropna())
    logret = np.diff(np.log(avg_price))
    ups = logret[logret > 0]
    up_vol = np.nan if len(ups) == 0 else np.var(ups)
    downs = logret[logret < 0]
    down_vol = np.nan if len(downs) == 0 else np.var(downs)
    price_volatility_updown_value = up_vol - down_vol
    return price_volatility_updown_value


def price_volatility_updown_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, price_volatility_updown(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def volume_distance(df):
    tgrouper = df.groupby(pd.Grouper(key='datetime', freq='30min', closed='right', label='right', convention='e'))
    volume_ratio = tgrouper['volume'].sum() / df['volume'].sum()
    volume_ratio = volume_ratio[volume_ratio.index.hour != 12]
    volume_ratio.index =  volume_ratio.index.strftime('%H%M')
    return volume_ratio


def volume_distance_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, volume_distance(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def price_volume_corr(df):
    tgrouper = df.groupby(pd.Grouper(key='datetime', freq='60s', closed='right', label='right', convention='e'))
    sum_volume = tgrouper['volume'].sum()
    avg_price = tgrouper['turnover'].sum() / sum_volume
    idx_nan = np.isnan(avg_price) | np.isnan(sum_volume)
    avg_price_arr = avg_price[~idx_nan].values
    sum_volume_arr = sum_volume[~idx_nan].values
    corr_value = nbr.numba_pearsonr(avg_price_arr, sum_volume_arr)
    return corr_value


def price_volume_corr_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, price_volume_corr(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def flowinratio(df):
    op, cl = df['new_price'].iloc[[0, -1]].values
    sign = 0 if cl == op else (cl - op) / np.abs(cl - op)
    flowinratio_value = np.sum(df['volume']) * cl * sign / np.sum(df['turnover'])
    return flowinratio_value


def flowinratio_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, flowinratio(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv


def trendstrength(df):
    touch_high_limit = np.all(df['ap1'] == 0)
    touch_low_limit = np.all(df['bp1'] == 0)
    if touch_high_limit:
        trendstrength_value = np.log(1.1)
    elif touch_low_limit:
        trendstrength_value = np.log(0.9)
    else:
        displacement = np.log(df['new_price'].iloc[[0, -1]]).diff().iloc[-1]
        circumference = np.sum(np.abs(np.diff(np.log(df['new_price']))))
        trendstrength_value = displacement / circumference
    return trendstrength_value


def trendstrength_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=0)
    is_abnormal = np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0)
    if (df.shape[0] > 0) and (~is_abnormal):
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, fatv = date, code, trendstrength(df)
    else:
        dt, co, fatv = date, code, np.nan
    del df
    gc.collect()
    return dt, co, fatv

# --------------------------------------------------------------------------------
# return
# --------------------------------------------------------------------------------

def calc_hmo(df):
    if np.all(df['ap1'] == df['bp1']) | np.all(df['ap1'] == 0) | np.all(df['bp1'] == 0):
        hmo = np.nan
    else:
        sorted_price = df.sort_values('datetime')['new_price']
        hmo = np.log(sorted_price.iloc[-1] / sorted_price.iloc[0])
    return hmo


def calc_hmo_express(code, date):
    df = lut.load_orderbook(code=code, date=date, verbose=10)
    if df.shape[0] > 0:
        df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
        dt, co, hmo = date, code, calc_hmo(df)
    else:
        dt, co, hmo = date, code, np.nan
    del df
    gc.collect()
    return date, code, hmo
