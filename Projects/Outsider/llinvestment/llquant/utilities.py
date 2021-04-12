# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-15 11:25:10
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-17 15:42:41

import os
import pandas as pd
import numpy as np
# import akshare as ak
import scipy.stats as scs
from numba import jit


try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    print("sliding_window_view is not imported.")

    def sliding_window_view(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def load_orderbook(code, date, verbose=0):
    _year = str(date)[:4]

    raw_columns_names = ['时间', '代码', '市场', '最新价', '最高价', '最低价', '总量', '总金额', '挂买价1', '挂买量1',
       '挂买价2', '挂买量2', '挂买价3', '挂买量3', '挂买价4', '挂买量4', '挂买价5', '挂买量5', '挂买价6',
       '挂买量6', '挂买价7', '挂买量7', '挂买价8', '挂买量8', '挂买价9', '挂买量9', '挂买价10',
       '挂买量10', '挂卖价1', '挂卖量1', '挂卖价2', '挂卖量2', '挂卖价3', '挂卖量3', '挂卖价4', '挂卖量4',
       '挂卖价5', '挂卖量5', '挂卖价6', '挂卖量6', '挂卖价7', '挂卖量7', '挂卖价8', '挂卖量8', '挂卖价9',
       '挂卖量9', '挂卖价10', '挂卖量10', '总成交笔数', 'IOPV', '昨收', '开盘价', '涨停价', '跌停价']

    col1 = ['datetime', 'code', 'market', 'new_price', 'high', 'low', 'volume', 'turnover']
    # col2 = np.hstack([[f'bp{i}', f'bv{i}'] for i in range(1, 11)])
    col2 = ['bp1', 'bv1', 'bp2', 'bv2', 'bp3', 'bv3', 'bp4', 'bv4', 'bp5', 'bv5', 'bp6', 'bv6', 'bp7', 'bv7', 'bp8',
            'bv8', 'bp9', 'bv9', 'bp10', 'bv10']
    # col3 = np.hstack([[f'ap{i}', f'av{i}'] for i in range(1, 11)])
    col3 = ['ap1', 'av1', 'ap2', 'av2', 'ap3', 'av3', 'ap4', 'av4', 'ap5', 'av5', 'ap6', 'av6', 'ap7', 'av7', 'ap8',
            'av8', 'ap9', 'av9', 'ap10', 'av10']
    col4 = ['trade_num', 'IOPV', 'pre_close', 'open', 'high_limit', 'low_limit']
    column_names = col1 + col2 + col3 + col4

    name_map = {r: n for r, n in zip(raw_columns_names, column_names)}
    try:
        csv_filen_name = f"\\\\dbserver07\\Stk_Tick10_{_year}\\{date}\\{code}.csv"
        df = pd.read_csv(csv_filen_name, encoding='GB18030')
        df.rename(columns=name_map, inplace=True)
    except FileNotFoundError as e:
        if verbose > 0:
            print(e)
        df = pd.DataFrame(columns=column_names)

    df = df.infer_objects()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['dateint'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    df['timeint'] = df['datetime'].dt.strftime('%H%M%S').astype(int)
    return df


def load_deal_data(code, date, verbose=0):
    if isinstance(date, int):
        dstr = str(date)
        assert len(dstr) == 8, 'date length could not be processed.'
        date_str = '-'.join([dstr[:4], dstr[4:6], dstr[6:]])
    elif isinstance(date, str):
        assert ('-' in date) and (len(date) == 10), 'date format is incorrect.'
        date_str = str(date)
    else:
        raise Exception()

    column_names = ['TranID', 'Time', 'Price', 'Volume', 'SaleOrderVolume', 'BuyOrderVolume', 'Type', 'SaleOrderID', 'SaleOrderPrice', 'BuyOrderID', 'BuyOrderPrice']
    directories = ['逐笔成交', '逐笔成交2', '逐笔成交CurDay']
    for di in directories:
        try:
            df = pd.read_csv(f"\\\\dbserver07\\{di}\\{date_str}\\{code}.csv")
            break
        except FileNotFoundError as e:
            if verbose > 0:
                print(e)
            df = pd.DataFrame(columns=column_names)
    df['code'] = code
    df['Time'] = pd.to_datetime(date_str + ' ' + df['Time'])
    df.columns = df.columns.str.lower()
    df.rename(columns={'time': 'datetime'}, inplace=True)

    df = df.infer_objects()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['dateint'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
    df['timeint'] = df['datetime'].dt.strftime('%H%M%S').astype(int)
    return df


def get_trading_dates():
    all_trading_dates = np.hstack([[f for f in os.listdir(f"\\\\dbserver07\\Stk_Tick10_{y}") if f.startswith('2')] for y in [2018, 2019, 2020, 2021]])
    return all_trading_dates


def get_all_stock_codes():
    for f in os.listdir(f"\\\\dbserver07\\Stk_Tick10_2021"):
        if f.startswith('2'):
            ff = f
    all_listed_stock_codes = [co[:-4] for co in os.listdir(f"\\\\dbserver07\\Stk_Tick10_2021\\{ff}" )]
    return all_listed_stock_codes


def split_data(data, ncut, flag=1, nan_policy='omit', return_bins=False):
    """Summary

    Split data into nq blocks.

    Args:
        data (array): factors on first column is required.
        ncut (int or list): number of segmentations for flag 1 and 2.
            Otherwis, list of percentiles following the format in np.percentiles
        flag (int, optional):
            1, split by equal distance;
            2, split by equal numbers;
            3, split with custom percentiles;

    Returns:
        TYPE: Description
    """
    tf_ind_nan = np.isnan(data)
    nan_in_rawdata = any(tf_ind_nan)
    if nan_in_rawdata:
        if nan_policy == 'omit':
            data = data[~tf_ind_nan].copy()
        elif nan_policy == 'raise':
            raise ValueError('The input contains nan values')
        else:
            return np.nan

    if flag == 1:
        slices = np.linspace(data.min(), data.max(), ncut + 1)
    elif flag == 2:
        segs = np.linspace(0, 100, int(ncut + 1))
        slices = np.percentile(data, segs)
    elif flag == 3:
        slices = np.percentile(data, ncut)
    else:
        raise ValueError('flag should be 1, 2 or 3')

    slices[0] = slices[0] - 1
    slices[-1] = slices[-1] + 1
    try:
        labels = np.digitize(data, slices) - 1
    except ValueError as e:
        idx_dup = np.hstack([False, np.isclose(np.diff(slices), 1e-15)])
        fixed_sp = slices.copy()
        val = fixed_sp[idx_dup] + np.arange(1, np.sum(idx_dup) + 1) * 1e-15
        np.place(fixed_sp, idx_dup, val)
        labels = np.digitize(data, fixed_sp) - 1

    if nan_in_rawdata and nan_policy == 'omit':
        idx_loc = np.where(tf_ind_nan)[0] - np.arange(sum(tf_ind_nan))
        labels = np.insert(labels.astype(float), idx_loc, np.nan)

    if return_bins:
        slices[0] = slices[0] + 1
        slices[-1] = slices[-1] - 1
        return np.array(labels), list(zip(slices[:-1], slices[1:]))
    else:
        return np.array(labels)


def calc_ic(x, y):
    return scs.spearmanr(x, y).correlation


def _generate_signal(y_pred, signal_ref=1024, q_lower=0.2, q_upper=0.8, flag=1, min_periods=None):
    if len(y_pred) <= signal_ref:
        raise ValueError("length({}) is smaller than window({}).".format(len(y_pred), signal_ref))
        # print("[WARNING] length({}) is smaller than window({}).".format(len(y_pred), n))

    upper = pd.Series(y_pred).rolling(signal_ref, min_periods=min_periods).quantile(q_upper).values
    lower = pd.Series(y_pred).rolling(signal_ref, min_periods=min_periods).quantile(q_lower).values

    if flag == 1:
        # long/short
        pass
    elif flag == 2:
        # long only
        lower = -np.inf
    else:
        raise ValueError("flag must be 1 (long/short) or 2(long only)")

    signal = np.zeros_like(y_pred)
    signal[y_pred > upper] = 1
    signal[y_pred < lower] = -1

    # assert (np.sum(signal < 0) == 0) and (flag == 2), "Strategy Error."
    return signal



@jit(nopython=True)
def sigmoid(x, ymin=0, ymax=1, x50L=-1, x50U=1, e=2):
    """
    Map the x into (ymin, ymax), as S-curve, with 50% of the values
    inside (x50L, x50U)

    Default is normal S-curve

    Reference:
    https://stats.stackexchange.com/questions/265266/adjusting-s-curves-sigmoid-functions-with-hyperparameters
    """
    a = (x50L + x50U) / e
    b = e / (x50L - x50U)
    c = ymin
    d = ymax - c
    y = c + (d / (1.0 + np.exp(b * (x - a))))
    return y



def t_stat(a, b):
    # scs.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate', alternative="two-sided")
    assert len(a) == len(b), 'lengths are not equal.'
    a_var = np.var(a, ddof=1)
    b_var = np.var(b, ddof=1)
    a_mu = np.mean(a)
    b_mu = np.mean(b)
    s = np.sqrt(np.mean([a_var, b_var]))
    t = (a_mu - b_mu) / (s * np.sqrt(2 / len(a)))
    return t