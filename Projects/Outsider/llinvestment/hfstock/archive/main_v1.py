# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-15 10:13:22
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-15 11:30:40


import pandas as pd
import numpy as np
import scipy.stats as scs
import llquant.utilities as lut
import llquant.numba_recursive as nbr
from tqdm import tqdm
from joblib import Parallel, delayed


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
    ask_p = target_quote.filter(regex='ap\d').values
    ask_v = target_quote.filter(regex='av\d').values
    bid_p = target_quote.filter(regex='bp\d').values
    bid_v = target_quote.filter(regex='bv\d').values

    # calculate volume based on weights
    wbuyamt = calc_weight(bid_p, prices_arr, th) * bid_v
    wsellamt = calc_weight(ask_p, prices_arr, th) * ask_v

    wbuy_amt_sum = wbuyamt.sum(1)  # order long
    wsell_amt_sum = wsellamt.sum(1)  # order short
    wb = chi2(wbuy_amt_sum, wsell_amt_sum)

    return wb


def tba(target_quote, alpha=0.05, bounds=(-5, 5)):
    ba_ratio = np.log(target_quote['bv1'] / target_quote['av1'])
    clipped_ba_ratio = np.clip(ba_ratio, bounds[0], bounds[1]).values
    ba_ratio_ema = nbr.numba_ewma(clipped_ba_ratio, alpha=alpha, state=None, adjust=True, ignore_na=True, minp=1)[0]
    return ba_ratio_ema


def dbook(target_quote, n=1, alpha=0.05):
    bv1, av1 = target_quote[['bv1', 'av1']].values.T
    diff_bv1 = np.insert(np.sign(np.diff(bv1, n=n)), [0] * n, 0)
    diff_av1 = np.insert(np.sign(np.diff(av1, n=n)), [0] * n, 0)
    rs = nbr.numba_ewma((diff_bv1 - diff_av1), alpha=alpha, state=None, adjust=True, ignore_na=True, minp=1)[0]
    return rs


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


def calc_absr(target_quote, window):
    """
    Estimate the active trading volume (absolute buy sell ratio).
    ask, bid volume and amount must be float.

    """
    ask, bid, volume, amount = target_quote[['ap1', 'bp1', 'volume', 'turnover']].values.T

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

    active_volume = np.vstack([actibuy, actisell]).T.astype('float64')
    active_volume = np.nan_to_num(active_volume)
    active_volume_arr = np.full((len(vol), 2), np.nan)
    active_volume_rolsum = np.sum(lut.sliding_window_view(active_volume.T, window), axis=2).T
    active_volume_arr[window:] = active_volume_rolsum
    out = np.log(np.divide(active_volume_arr[:, 0], active_volume_arr[:, 1], out=np.full(len(active_volume_arr), np.nan), where=active_volume_arr[:, 1]!=0))
    return out


def get_training_data():
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # load data
    # --------------------------------------------------------------------------------
    all_trading_dates = lut.get_trading_dates()[-21:]
    # raw_codes = lut.get_all_stock_codes()
    # selected_codes = raw_codes[:30]
    hs300 = pd.read_csv('../data/SH000300.csv', header=None).astype(str)[0]
    hs300a = hs300.str.split('\t', expand=True)[0].str.split('.', expand=True)
    hs300a[2] = hs300a[0].astype(int)
    code_type_map = hs300a.set_index(2)[0].to_dict()
    selected_codes = list(hs300a[1].str.lower() + hs300a[0])

    # all_trading_dates = all_trading_dates[-5:]
    # selected_codes = selected_codes[-5:]

    result_list = Parallel(n_jobs=3, verbose=10)(delayed(lut.load_orderbook)(code=co, date=d) for d in all_trading_dates for co in selected_codes)
    rawdata = pd.concat(result_list)
    rawdata['code'] = rawdata['code'].map(code_type_map)
    rawdata = rawdata.infer_objects()
    rawdata['datetime'] = pd.to_datetime(rawdata['datetime'])
    rawdata['timeint'] = rawdata['datetime'].dt.strftime('%H%M%S').astype(int)
    gooddata = rawdata[(rawdata['timeint'] >= 93000) & (rawdata['timeint'] <= 145700)].copy()

    input_data = gooddata.copy()
    cycle_list = [5, 20, 100, 300, 600]
    for cyc in cycle_list:
        logret_piv = np.log(input_data.pivot('datetime', 'code', 'new_price')).ffill().diff(periods=cyc)
        logret_piv_s = logret_piv.shift(periods=-cyc)
        input_data = input_data.merge(logret_piv_s.stack().rename(str(cyc)).reset_index(), on=['datetime', 'code'])

    # --------------------------------------------------------------------------------
    # compute factors
    # --------------------------------------------------------------------------------
    # wb

    input_data['factor'] = calc_wb(input_data, th=600)
    factor_piv = input_data.pivot('datetime', 'code', 'factor').ffill()

    signal_piv_arr = np.stack([lut._generate_signal(ss, flag=1) for _, ss in factor_piv.iteritems()], axis=1)
    ret_dict = {}
    for cyc in cycle_list:
        logret_piv_s = input_data.pivot('datetime', 'code', str(cyc)).ffill()
        logret_piv_s_good = logret_piv_s.values
        logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
        ret = np.sum(signal_piv_arr * logret_piv_s_good, axis=0) / cyc
        ret_dict[cyc] = ret

    pd.DataFrame(ret_dict, index=logret_piv_s.columns).mean()


    # input_data = input_data.merge(logret_piv1.stack().rename('logret1').reset_index(), on=['datetime', 'code'])
    # input_data = input_data.dropna()
    # factor, logret1 = input_data[['factor', 'logret1']].values.T
    # total_ic = scs.spearmanr(factor, logret1).correlation
    # daily_ic = input_data.groupby('code').apply(lambda g: g.groupby(pd.Grouper(key='datetime', freq='1d')).apply(lambda x: scs.spearmanr(x['factor'].values, x['logret1'].values).correlation)).T
    # daily_ic.dropna().describe()
    # total_ic

    # # smf
    # result = gooddata[['datetime', 'new_price']].copy()
    # factor = tba(gooddata)
    # result['factor'] = factor
    # result['logret'] = np.log(gooddata['new_price']).diff()
    # result['logret1'] = result['logret'].shift(-1)
    # result = result.dropna()
    # factor, logret1 = result[['factor', 'logret1']].values.T
    # total_ic = scs.spearmanr(factor, logret1).correlation
    # daily_ic = result.groupby(pd.Grouper(key='datetime', freq='1d')).apply(lambda x: scs.spearmanr(x['factor'].values, x['logret1'].values).correlation)
    # daily_ic.dropna().describe()
    # total_ic
    #
    #
    # # dbook
    # result = gooddata[['datetime', 'new_price']].copy()
    # factor = dbook(gooddata, 1, 0.05)
    # result['factor'] = factor
    # result['logret'] = np.log(gooddata['new_price']).diff()
    # result['logret1'] = result['logret'].shift(-1)
    # result = result.dropna()
    # factor, logret1 = result[['factor', 'logret1']].values.T
    # total_ic = scs.spearmanr(factor, logret1).correlation
    # daily_ic = result.groupby(pd.Grouper(key='datetime', freq='1d')).apply(lambda x: scs.spearmanr(x['factor'].values, x['logret1'].values).correlation)
    # daily_ic.dropna().describe()
    # total_ic
    #
    # # approx_price
    # result = gooddata[['datetime', 'new_price']].copy()
    # factor = approx_price(gooddata)
    # result['factor'] = factor
    # result['logret'] = np.log(gooddata['new_price']).diff()
    # result['logret1'] = result['logret'].shift(-1)
    # result = result.dropna()
    # factor, logret1 = result[['factor', 'logret1']].values.T
    # total_ic = scs.spearmanr(factor, logret1).correlation
    # daily_ic = result.groupby(pd.Grouper(key='datetime', freq='1d')).apply(lambda x: scs.spearmanr(x['factor'].values, x['logret1'].values).correlation)
    # daily_ic.dropna().describe()
    # total_ic

    # --------------------------------------------------------------------------------
    # evalution data
    # --------------------------------------------------------------------------------
    import llquant.evaluation as leval
    from importlib import reload
    reload(leval)

    ft = leval.factor_tester(input_data, split_num=5, verbose=True)
    g = ft.plot_res_between_groups(2)
    g.savefig('./res_between_groups.jpg')

    g = ft.plot_res_within_group(flag=2, calc_all=True)
    g.savefig('./res_within_group.jpg')

    g = ft.plot_sectional_return(2, 1)
    g.savefig('./sectional_return.jpg')

    exit()