# from __context__ import *

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
# import option_numba as optnb
from numba import jit, float64, boolean
import math

from IPython.display import display
import sys
import logging as log
import logging.handlers
import json
import os.path
import re
import ipykernel
import requests
from requests.compat import urljoin
from notebook.notebookapp import list_running_servers
from multiprocessing import Pool
from functools import partial
import option_numba as optnb

# init
# define db


# def __init__():
#     pass


# ------------------------------------------------------------------------------- #
# 历史函数
# ------------------------------------------------------------------------------- #
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


def select_slice_time(df, time_slices, s=True, e=False):
    for ts_s, ts_e in time_slices:
        df_s = df.between_time(ts_s, ts_e, include_start=s, include_end=e)
        yield df_s


@jit(nopython=True)
def get_last_valid_value(sequence):
    for i, x in enumerate(sequence[::-1]):
        if not math.isnan(x):
            break
    return x, i


@jit((float64[:], float64, boolean, boolean), nopython=True, nogil=True)
def _numba_ema(X, alpha, adjust, ignore_na):
    """Exponentialy weighted moving average specified by a decay ``alpha``

    Reference:
    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm

    Example:
        >>> ignore_na = True     # or False
        >>> adjust = True     # or False
        >>> myema = _numba_ema_adjusted(X, alpha=alpha, ignore_na=ignore_na)
        >>> pdema = pd.Series(X).ewm(alpha=alpha, adjust=adjust, ignore_na=ignore_na).mean().values
        >>> print(np.allclose(myema, pdema, equal_nan=True))
        True

    Args:
        X (array): raw data
        alpha (float): decay factor
        adjust (boolean):
            True for assuming infinite history via the recursive form
            False for assuming finite history via the recursive form
        ignore_na (boolean): True for decaying by relative location, False for absolute location

    Returns:
        TYPE: Description
    """
    ewma = np.empty_like(X, dtype=float64)
    offset = 1
    w = 1
    for i, x in enumerate(X):
        if i == 0:
            ewma[i] = x
            ewma_old = x
        else:
            is_ewma_nan = math.isnan(ewma[i - 1])
            is_x_nan = math.isnan(x)
            if is_ewma_nan and is_x_nan:
                ewma[i] = np.nan
            elif is_ewma_nan:
                ewma[i] = x
                ewma_old = x
            elif is_x_nan:
                offset += 1
                ewma[i] = ewma[i - 1]
            else:
                if ignore_na:
                    if adjust:
                        w = w * (1 - alpha) + 1
                        ewma_old = ewma_old * (1 - alpha) + x
                        ewma[i] = ewma_old / w
                    else:
                        ewma[i] = ewma[i - 1] * (1 - alpha) + x * alpha
                else:
                    if adjust:
                        w = w * (1 - alpha) ** offset + 1
                        ewma_old = ewma_old * (1 - alpha) ** offset + x
                        ewma[i] = ewma_old / w
                    else:
                        ewma[i] = (ewma[i - 1] * (1 - alpha) ** offset + x * alpha) / ((1 - alpha) ** offset + alpha)
                    offset = 1
    return ewma


@jit((float64[:], float64, float64, boolean, boolean), nopython=True, nogil=True)
def numba_ema(X, alpha=0.05, start=np.nan, adjust=False, ignore_na=False):
    """Exponentialy weighted moving average specified by a decay ``alpha``

    Args:
        X (array): raw data
        alpha (float): decay factor
        adjust (boolean):
            True for assuming infinite history via the recursive form
            False for assuming finite history via the recursive form
        ignore_na (boolean): True for decaying by relative location, False for absolute location

    Returns:
        TYPE: Description
    """
    if math.isnan(start):
        out = _numba_ema(X, alpha=alpha, adjust=adjust, ignore_na=ignore_na)
    else:
        X = np.array([start] + list(X))
        out = _numba_ema(X, alpha=alpha, adjust=adjust, ignore_na=ignore_na)
        out = out[1:]
    return out


@jit((float64[:], float64[:]), nopython=True)
def numba_ols_beta(x, y):
    sum_xy = 0
    sum_x_sq = 0
    for x0, y0 in zip(x, y):
        sum_xy += x0 * y0
        sum_x_sq += x0 * x0
    return sum_xy / sum_x_sq


def numpy_shift(a, shift=1, axis=None):
    shifted_a = np.roll(a, shift, axis)
    shifted_a[:shift] = np.nan
    return shifted_a

# ------------------------------------------------------------------------------- #
# 估值与相关helper functions
# ------------------------------------------------------------------------------- #


@jit(nopython=True)
def numba_rollreg(x, y, w, min_valid_sample):
    """Summary

    Args:
        x (numpy.array): Description
        y (numpy.array): Description
        w (int): Description
        min_valid_sample (int): Description

    Returns:
        list: Description
    """
    betas = []
    if w < min_valid_sample:
        min_valid_sample = w

    for i in range(len(x)):
        if i < min_valid_sample:
            beta = np.nan
        elif i < w:
            x0, y0 = x[:i], y[:i]
            idx_good = ~(np.isnan(x0) | np.isnan(y0))
            if np.sum(idx_good) < min_valid_sample:
                beta = np.nan
            else:
                beta = numba_ols_beta(x0[idx_good], y0[idx_good])
        else:
            x0, y0 = x[i - w:i], y[i - w:i]
            idx_good = ~(np.isnan(x0) | np.isnan(y0))
            if np.sum(idx_good) < min_valid_sample:
                beta = np.nan
            else:
                beta = numba_ols_beta(x0[idx_good], y0[idx_good])
        betas.append(beta)
    return np.array(betas)


def rollreg_ts(x, y, ts, tfreq, min_valid_sample, padding=np.nan):
    N = len(x)
    j0 = 0
    betas = []
    is_1d = x.ndim == 1
    for i in range(N):
        for j in range(j0, N):
            if (ts[j] - ts[i]) > tfreq:
                x_ = x[i:j]
                y_ = y[i:j]
                if is_1d:
                    idx_nan = np.isnan(x_) | np.isnan(y_)
                else:
                    idx_nan = np.any(np.isnan(x_), 1) | np.isnan(y_)
                num_nan = np.sum(idx_nan)
                if num_nan < len(x_) - min_valid_sample:
                    beta = numba_ols_beta(x_[~idx_nan].reshape(-1, 1), y_[~idx_nan])
                    betas.append(beta)
                else:
                    betas.append(np.nan)
                j0 = j + 1
                break
        if j0 == N:
            break
    num_padding = len(x) - len(betas)
    result = np.hstack([np.full(num_padding, padding), np.hstack(betas)])
    return result


@jit(nopython=True)
def numba_rollreg_ts(x, y, ts, tfreq, min_valid_sample):
    """Summary

    Args:
        x (numpy.array): nump
        y (numpy.array): Description
        ts (numpy.datetime): Description
        tfreq (numpy.timedelta): Description
        min_valid_sample (int): Description

    Returns:
        list: Description
    """
    assert len(x) == len(y) == len(ts)
    N = len(x)
    betas = []
    i = 0
    for j in range(N):
        while (ts[j] - ts[i]) > tfreq:
            i += 1
        x_ = x[i:j]
        y_ = y[i:j]
        idx_good = ~(np.isnan(x_) | np.isnan(y_))
        if np.sum(idx_good) < min_valid_sample:
            beta = np.nan
        else:
            beta = numba_ols_beta(x_[idx_good], y_[idx_good])
        betas.append(beta)
    return np.array(betas)


def convert_timestamp_to_int(timestamp_series, reverse=False):
    """
    reverse == True:
        20181101150003000 ---> 2018-11-01 15:00:03
        20181101150003300 ---> 2018-11-01 15:00:03.300
    reverse == False:
        2018-11-01 15:00:03 ---> 20181101150003000
        2018-11-01 15:00:03.300 ---> 20181101150003300
    """
    if reverse:
        return pd.to_datetime(timestamp_series, format='%Y%m%d%H%M%S%f')
    else:
        return timestamp_series.dt.strftime('%Y%m%d%H%M%S%f').str[:-3].astype(int)


def get_timeUntilExp(target_dates, expire_dates, fmt='%Y%m%d%H%M%S%f', return_unit='s'):
    tdiff = pd.to_datetime(expire_dates, format=fmt) - pd.to_datetime(target_dates, format=fmt)
    return tdiff / np.timedelta64(1, return_unit)


def coral_get_dtm(df):
    tdiff = pd.to_datetime(df['expire_date'].astype(int).astype(str)) - \
        pd.to_datetime(df['date'].astype(int).astype(str))
    return tdiff.dt.days / 365


def check_option_value_status(S, K, cp_flag=None, target_status='itm'):
    if target_status == 'itm':
        if cp_flag is None:
            raise ValueError('cp_flag must be defined')
        is_call_itm = (cp_flag == 1) & (K < S)
        is_put_itm = (cp_flag == 2) & (S < K)
        result = np.array(is_call_itm | is_put_itm)
    elif target_status == 'otm':
        if cp_flag is None:
            raise ValueError('cp_flag must be defined')
        is_call_otm = (cp_flag == 1) & (K > S)
        is_put_otm = (cp_flag == 2) & (S > K)
        result = np.array(is_call_otm | is_put_otm)
    elif target_status == 'atm':
        result = np.array(S == K)
    else:
        result = np.full(max(map(np.shape, (S, K, cp_flag))), True)
    return result


def coral_get_dominant_contract(domcode, date):
    df = db.getBar(domcode, date, fields='code', beginTime=93000000, endTime=150000000).toDataFrame()
    return df.iloc[0, :]['code']


def coral_check_rollover_contract(domcode, current_date, verbose=False):
    date_df = db.getTradingDays(coral_get_pre_trading_day(current_date), current_date).toDataFrame()
    datearr = np.squeeze(date_df.values)
    previous_info = db.getBar(domcode, datearr[-2], fields='code',
                              beginTime=93000000, endTime=150000000).toDataFrame().iloc[0].values
    current_info = db.getBar(domcode, datearr[-1], fields='code',
                             beginTime=93000000, endTime=150000000).toDataFrame().iloc[0].values
    if verbose:
        print('previous date:\t', previous_info, '\ncurrent date:', current_info)
    return previous_info[1] != current_info[1]


def coral_get_good_sample(curdate, underlying_code, min_volume=5000, min_T=(0, 'D'), max_T=(90, 'D'), moneyness_status='itm', max_sample=None):
    """Summary

    Args:
        curdate (int): date
        underlying_code (str): underlying code
        min_volume (int, optional): minimum volume
        min_T (tuple, optional): minimum days until expiry, (num of unit, unit)
        max_T (tuple, optional): maximum days until expiry, (num of unit, unit)
        moneyness_status (str, optional): moneyness of the option, itm, otm or atm.
        max_sample (None, optional): maximum number of sample should be returned. (num of sample, sorted columns)

    Returns:
        TYPE: Description
    """
    option_info = db.getCode(
        '期权', date=curdate, fields='code,date,underlying_code,new_price,exercise_price,sum_volume,cp_flag,expire_date').toDataFrame().reset_index(drop=True)
    multiple_df = db_multiple.getCode('期权', date=curdate, fields='code,volume_multiple').toDataFrame(
    ).reset_index(drop=True)[['code', 'volume_multiple']]
    option_info = pd.merge(option_info, multiple_df, on='code', how='left', validate='1:1')
    spot_close = db.getBar(underlying_code, beginDate=curdate, fields='new_price',
                           cycle=pycoraldb.D).toDataFrame()['new_price'].values[0]
    is_target_underlying = option_info['underlying_code'] == underlying_code
    is_target_value_status = check_option_value_status(
        spot_close, option_info['exercise_price'].values, option_info['cp_flag'].values, target_status=moneyness_status)

    is_active = option_info['sum_volume'] > min_volume

    if (min_T[0] == 0) or (min_T is None):
        is_far_from_expire = np.full(option_info.shape[0], True)
    else:
        is_far_from_expire = get_timeUntilExp(
            option_info['date'], option_info['expire_date'], fmt='%Y%m%d', return_unit=min_T[1]) >= min_T[0]

    if (max_T[0] == 0) or (max_T is None):
        is_close_to_expire = np.full(option_info.shape[0], True)
    else:
        is_close_to_expire = get_timeUntilExp(
            option_info['date'], option_info['expire_date'], fmt='%Y%m%d', return_unit=max_T[1]) <= max_T[0]

    good_sample = option_info[is_target_underlying & is_target_value_status &
                              is_active & is_far_from_expire & is_close_to_expire]
    if max_sample is not None:
        good_sample = good_sample.sort_values(max_sample[1]).iloc[:max_sample[0], :].reset_index()

    return good_sample


def calc_midprice(target_quote):
    is_ap_eq_0 = target_quote['ap1'].values == 0
    is_bp_eq_0 = target_quote['bp1'].values == 0
    midprice = target_quote[['ap1', 'bp1']].mean(1)
    midprice[is_ap_eq_0] = target_quote.loc[is_ap_eq_0, 'bp1']
    midprice[is_bp_eq_0] = target_quote.loc[is_bp_eq_0, 'ap1']
    midprice[midprice == 0] = target_quote['new_price']
    return midprice.values


def estimate_option_price(cp_flag, optpri, S, S_, K, alpha, start=None):
    cp = np.unique(cp_flag)
    if len(cp) > 1:
        raise ValueError('error')
    else:
        cp = cp[0]
    if cp == 1:
        instapnl = numba_ema(S - K - optpri, alpha, start, False, False)
        optpri_ = S_ - K - instapnl
    elif cp == 2:
        instapnl = numba_ema(K - S - optpri, alpha, start, False, False)
        optpri_ = K - S_ - instapnl
    return optpri_, instapnl


def coral_check_if_etf_regular(etflistinfo, threshold=0.0002):
    """Summary

    Args:
        etflistinfo (TYPE): dataframe from coraldb for only one date
                etflistinfo = db1.getEtfListInfo('510050.SH', 20190221).toDataFrame()    # regular date
                etflistinfo = db1.getEtfListInfo('510050.SH', 20181203).toDataFrame()    # dividend date
        threshold (float, optional): threshold for deviation

    Returns:
        TYPE: Description
    """
    stocklistinfo = pd.DataFrame(etflistinfo['stocks'][0])

    tfind_abnormal = (stocklistinfo['cash_replace_flag'] == 1) | (stocklistinfo['cash_replace_flag'] == 3)
    stocklist = stocklistinfo.loc[tfind_abnormal, 'code'].tolist()
    stockvolumelist = stocklistinfo.loc[tfind_abnormal, 'volume'].tolist()
    cash_replace = stocklistinfo.loc[~tfind_abnormal, 'cash_replace_amount'].sum()
    unit_volume = etflistinfo['unit_volume'][0]
    estimate_cash = etflistinfo['estimate_cash'][0]

    # 读取清单股票行情————————————————————————
    target_quota = db.getBar(
        ','.join(stocklist), etflistinfo['date'][0], cycle=pycoraldb.D, fields='code,stamp,pre_close,new_price').toDataFrame()

    # 计算IOPV————————————————————————
    pre_nav = etflistinfo['pre_unit_iopv'].values[0] / unit_volume
    iopv = (np.dot(stockvolumelist, target_quota['pre_close'].values) + cash_replace + estimate_cash) / unit_volume
    return abs(pre_nav / iopv - 1) < threshold


def nancmp(a, b, method, fillna):
    """
    - fillna must be boolean
    - return False if any nan
    """
    assert isinstance(fillna, bool), 'fillna must be boolean'
    out = np.full_like(a, fillna, dtype='bool')
    where = ~(np.isnan(a) | np.isnan(b))
    return eval('np.{}(a, b, out=out, where=where)'.format(method, out, where))


def modify_numpy_timedelta(timedelta, unit='s', dtype=str):
    tfloat = timedelta / np.timedelta64(1, unit)
    if dtype == float:
        out = tfloat
    elif dtype == str:
        if tfloat != int(tfloat):
            raise ValueError('float is not equal to its own integer.')
        out = str(int(tfloat)) + unit
    elif dtype == int:
        if tfloat != int(tfloat):
            raise ValueError('float is not equal to its own integer.')
        out = int(tfloat)
    return out


def np_polyfit_predict(x, paras):
    """make prediction from polyfit

    Args:
        x (array): r-by-1
        paras (array): r-by-c

    Returns:
        TYPE: predictions
    """
    y = np.zeros_like(x)
    n = paras.shape[1]
    for deg, p in enumerate(paras.T):
        y += p * x ** (n - deg - 1)
    return y


def delay_data(df, columns=None, delay=(1, 's')):
    delay_tindex = df.index + np.timedelta64(*delay)
    if len(df.shape) == 1:
        out = pd.Series(df.values, index=delay_tindex, name=df.name)
    else:
        if columns is None:
            columns = df.columns
        out = pd.DataFrame(df[columns].values, index=delay_tindex, columns=columns)
    return out

# ------------------------------------------------------------------------------- #
# 数据清洗及观察 #
# ------------------------------------------------------------------------------- #


def extract_string(str_list, substr_list):
    """Summary

    Eg.
    >>> a = ['volume', 'vega', 'vix']
    >>> aa = ['wvolume_1T', 'wvega_1T', 'cwvolume_1T', 'cwvega_1T', 'vix_1T', 'spot_price', 'rv_1T_21']
    >>> print(extract_string(aa, a))
    ['wvolume_1T', 'wvega_1T', 'cwvolume_1T', 'cwvega_1T', 'vix_1T']

    Args:
        str_list (str): list of strings
        substr_list (str): list of substrings

    Returns:
        str: Description
    """
    return [s for s in str_list if any(ss in s for ss in substr_list)]


def fix_duplicated_index(timeindex, increament=np.timedelta64(-1, 'ms')):
    """fix duplicated timeindex

    Args:
        timeindex (datetime-like): Description
        increament (np.timedelta64, optional): Description

    Returns:
        TYPE: Description
    """
    dummy_td = 0 * increament

    if increament >= dummy_td:
        timeindex = timeindex.copy()
    else:
        timeindex = timeindex[::-1].copy()

    timediff = np.hstack([increament, np.diff(timeindex)])
    while np.any(timediff == dummy_td):
        timeindex = timeindex + np.where(timediff == dummy_td, increament, 0)
        timediff = np.hstack([increament, np.diff(timeindex)])

    if increament >= dummy_td:
        timeindex = timeindex.copy()
    else:
        timeindex = timeindex[::-1].copy()

    return timeindex


def coral_fix_duplicated_index(target_quote, base=np.timedelta64(0, 'ms'), increament=np.timedelta64(-1, 'ms')):
    df = target_quote.copy()
    df.index = fix_duplicated_index(df.index + base, increament=increament)
    return df


def align_time(target_quote, signal_wt_time, fillna=True, method='ffill'):
    """Align index between two objects with same type of index

    Args:
        target_quote (TYPE): Description
        signal_wt_time (TYPE): Description
        fillna (bool, optional): Description
        method (str, optional): Description

    Returns:
        TYPE: Description
    """
    # arr = pd.merge(pd.DataFrame(target_quote), pd.DataFrame(signal_wt_time), how='outer', left_index=True, right_index=True).values
    if not fillna:
        num_nan = signal_wt_time.isna().sum()
        signal_wt_time_filled = signal_wt_time.fillna('dummy')
    else:
        signal_wt_time_filled = signal_wt_time.copy()

    arr = pd.merge(target_quote, pd.DataFrame(signal_wt_time_filled),
                   how='outer', left_index=True, right_index=True, validate='m:1').values
    idx_nan_arr = pd.isnull(arr)
    idx_good_a = np.all(idx_nan_arr[:, :-1], 1)

    out = pd.Series(arr[:, -1]).fillna(method=method).values
    out = out[~idx_good_a]

    if not fillna:
        if num_nan > 0:
            np.place(out, out == 'dummy', np.nan)
    return out.astype(signal_wt_time.dtype)


def coral_get_pre_trading_day(date, n=1):
    if n < 1:
        raise ValueError('n must be greater than 1.')
    date_arr = np.squeeze(db.getTradingDays(date - 10000, date).values)
    pre_date = date_arr[-n - 1] if date_arr[-1] == date else date_arr[-n]
    return pre_date


def coral_get_next_trading_day(date, n=1):
    return np.squeeze(db.getTradingDays(date, date + 10000).values)[n]


def coral_get_trading_day(date, n=0):
    """get specific date

    n<0: get date that n-day before date
    n>0: get date that n-day after date
    n=0: the nearest next date

    Args:
        date (TYPE): Description
        n (int, optional): Description

    Returns:
        TYPE: Description

    Example:
        20190102 <= coral_get_trading_day(20190101, 0)
        20190103 <= coral_get_trading_day(20190101, 1)
        20181228 <= coral_get_trading_day(20190101, -1)
        20190103 <= coral_get_trading_day(20190103, 0)
        20190104 <= coral_get_trading_day(20190103, 1)
        20190102 <= coral_get_trading_day(20190103, -1)
    """
    if n < 0:
        date_arr = np.squeeze(db.getTradingDays(date - 10000, date).values)
        pre_date = date_arr[n - 1] if date_arr[-1] == date else date_arr[n]
    else:
        date_arr = np.squeeze(db.getTradingDays(date, date + 10000).values)
        pre_date = date_arr[n]
    return pre_date


def get_trading_period(tradingrecord_list):
    for i, df in enumerate(tradingrecord_list):
        n = df.shape[0]
        if n > 0:
            open_ts = df.timestamp[df.holdings != 0].values.reshape(-1, 1)
            close_ts = df.timestamp[df.holdings == 0].values.reshape(-1, 1)
            code = np.repeat(df.loc[0, ['date', 'code']].values.reshape(1, -1), repeats=n / 2, axis=0)
            yield np.hstack([code, open_ts, close_ts])


def get_coexist_contract(df_grp):
    for gn, gp in tqdm(df_grp):
        earliest_open = str(gp['t0'].min() // 1000)
        if len(earliest_open) == 5:
            earliest_open = '0' + earliest_open
        elif len(earliest_open) == 6:
            pass
        else:
            raise ValueError('earliest_open')

        lastest_close = str(gp['t1'].max() // 1000)
        if len(lastest_close) == 5:
            lastest_close = '0' + lastest_close
        elif len(lastest_close) == 6:
            pass
        else:
            raise ValueError('lastest_close')

        trange = np.hstack([
            pd.date_range(str(gn) + earliest_open, str(gn) + '113000', freq='1s'),
            pd.date_range(str(gn) + '130000', str(gn) + lastest_close, freq='s')
        ])
        for tt in trange:
            tfind = (tt > gp['dt0']) & (tt < gp['dt1'])
            codes = gp.loc[tfind, 'code'].tolist()
            nct = sum(tfind)
            yield [tt, nct, codes]


def calc_holding_period(tradingrecord_df):
    curdate_str = tradingrecord_df['date'].astype(str)[0]
    zero_pos = np.array(tradingrecord_df['holdings'] == 0)
    datetime_str = tradingrecord_df['date'].astype(str) + tradingrecord_df['timestamp'].astype(str)
    noon_close = convert_timestamp_to_int(curdate_str + '113000000', True)
    noon_open = convert_timestamp_to_int(curdate_str + '130000000', True)
    datetime_ts = convert_timestamp_to_int(datetime_str, True)
    is_cross_noon_period = np.array((datetime_ts[~zero_pos] <= noon_close).values & (
        datetime_ts[zero_pos] >= noon_open).values)
    hp = datetime_ts.values[zero_pos] - datetime_ts.values[~zero_pos]
    hp[is_cross_noon_period] = hp[is_cross_noon_period] - np.timedelta64('90', 'm')
    return hp


def calc_avg_holding_period(tradingrecord_df):
    return np.mean(calc_holding_period(tradingrecord_df))


def fix_50etf_multiple(exercise_price, multiple):
    return exercise_price * multiple / 10000


def get_head_tail(df, coln, num=5, ascending=True, dropna=False):
    if dropna:
        sorted_values = df.dropna(subset=[coln]).sort_values(coln, ascending=ascending)
    else:
        sorted_values = df.sort_values(coln, ascending=ascending)

    if df.shape[0] >= num * 2:
        out = sorted_values.iloc[np.hstack([np.arange(0, num), np.arange(-num, 0)]), :]
    else:
        out = sorted_values
    return out


def get_result_by_date(df, date, sortby, show=True):
    idx_target_date = df['date'] == date
    trunc_df = df.loc[idx_target_date, :].sort_values(sortby)
    if trunc_df.shape[0] == 0:
        print('No trades occur on {}'.format(date))
        return trunc_df
    if show:
        show_rs = trunc_df[['netpnl', 'trades', 'total_period']].sum()
        show_rs['daily_avg_hp'] = np.array(show_rs['total_period'] / np.timedelta64(1, 's') / show_rs['trades'])
        show_rs['num_contracts'] = len(trunc_df)
        show_rs.name = date
        display(show_rs.to_frame().T)
    return trunc_df


def coral_select_trading_time(df, timeslots, on_key='stamp', on_index=False):
    """Select data within specific timeslots

    Args:
        df (pandas.dataframe): Description
        timeslots (list): list of timeslots
        on_key (str, optional): if on_index is False, on_key will be used to compare
        on_index (bool, optional): if on_index is True, index will be used to compare

    Returns:
        pandas.dataframe: selected dataframe
    """
    if on_index:
        whole_timestamp = df.index.strftime('%Y%m%d%H%M%S%f')
        key_ss = whole_timestamp.str[8:-3].astype(int)
    else:
        key_ss = df[on_key]

    tfind = np.full(df.shape[0], False)
    for t0, t1 in timeslots:
        tfind |= (key_ss >= t0) & (key_ss < t1)
    return df[tfind]


def _get_periodical_index(df, coln, s0, s1):
    idx = (df[coln] >= s0) & (df[coln] <= s1)
    if sum(idx) < 2:
        idx = pd.Series(False, index=df['knockid'])
    else:
        if df[idx]['holdings'].iloc[0] == 0:
            idx.iloc[np.where(idx)[0][0]] = False

        if df[idx]['holdings'].iloc[-1] != 0:
            idx.iloc[np.where(idx)[0][-1]] = False

        idx.index = df['knockid']
    return idx


def pnl_anatomy(all_tradingrecord_df, column, slots, tc, verbose=False):
    all_trec_df_grp_by_date_and_code = all_tradingrecord_df.groupby(['date', 'code'])
    results = []
    if verbose:
        print(column, ':')
    for t0, t1 in slots:
        # idx = all_trec_df_grp_by_date_and_code.apply(lambda g: _get_periodical_index(g, column, t0, t1))
        # semi_df = pd.concat([all_tradingrecord_df.set_index(['date', 'code', 'knockid']), idx.to_frame('idx_target')], 1)
        # pnl0 = semi_df.loc[semi_df['idx_target'], 'price'].sum() - sum(semi_df['idx_target']) * tc / 2
        # pnl1 = semi_df.loc[~semi_df['idx_target'], 'price'].sum() - sum(~semi_df['idx_target']) * tc / 2
        idx = all_trec_df_grp_by_date_and_code.apply(lambda g: _get_periodical_index(g, column, t0, t1)).values
        # semi_df = pd.concat([all_tradingrecord_df.set_index(['date', 'code', 'knockid']), idx.to_frame('idx_target')], 1)
        pnl0 = all_tradingrecord_df.loc[idx, 'price'].sum() - sum(idx) * tc / 2
        pnl1 = all_tradingrecord_df.loc[~idx, 'price'].sum() - sum(~idx) * tc / 2
        result = t0, t1, pnl0 + pnl1, len(idx) / 2, pnl0, sum(idx) / 2, pnl1, sum(~idx) / 2
        results.append(result)
        if verbose:
            print('\t{} ~ {}: \t{:.4f} ({}) = {:.4f} ({}) + {:.4f} ({})'.format(*result))
    return results


def coral_get_spread(date, code_a, code_b, multiple, period=1000):
    price_a = db.getBar(code_a, date, fields='new_price', cycle=period).toDataFrame()['new_price']
    price_b = db.getBar(code_b, date, fields='new_price', cycle=period).toDataFrame()['new_price']
    return price_a - price_b * multiple


def cprint(df, rows=5, max_info_cols=10000):
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_frame()
        except Exception as e:
            raise ValueError('object cannot be coerced to df')

    print('-' * 79)
    print('dataframe information')
    print('-' * 79)
    print(df.head(rows))
    print('-' * 79)
    print(df.tail(rows))
    print('-' * 50)
    print(df.info(max_cols=max_info_cols))
    print('-' * 79)
    print()


def describe(df, percentiles=None, include=None, exclude=None):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df.reshape(-1, 1))
    if len(df.shape) == 2:
        stat_df = df.describe(percentiles=percentiles, include=include, exclude=exclude)
        stat_df.loc['sum', :] = df.sum()
        stat_df.loc['skew', :] = df.skew()
        stat_df.loc['kurt', :] = df.kurt()
    elif len(df.shape) == 1:
        stat_df = df.describe(percentiles=percentiles, include=include, exclude=exclude)
        stat_df['sum'] = df.sum()
        stat_df['skew'] = df.skew()
        stat_df['kurt'] = df.kurt()
    return stat_df


def profit_explained(open_var, close_var, constants):
    S10, S20, sigma10, sigma20 = open_var
    S11, S21, sigma11, sigma21 = close_var
    cp_flag, bs_flag, K, r, T, Q = constants

    V_s0_sig0 = optnb.get_option_value(cp_flag, S10, K, r, T, sigma10, Q)
    V_s0_sig1 = optnb.get_option_value(cp_flag, S10, K, r, T, sigma11, Q)
    V_s1_sig0 = optnb.get_option_value(cp_flag, S11, K, r, T, sigma10, Q)
    V_s1_sig1 = optnb.get_option_value(cp_flag, S11, K, r, T, sigma11, Q)

    V_es0_esig0 = optnb.get_option_value(cp_flag, S20, K, r, T, sigma20, Q)
    V_es0_esig1 = optnb.get_option_value(cp_flag, S20, K, r, T, sigma21, Q)
    V_es1_esig0 = optnb.get_option_value(cp_flag, S21, K, r, T, sigma20, Q)
    V_es1_esig1 = optnb.get_option_value(cp_flag, S21, K, r, T, sigma21, Q)

    V_s0_esig0 = optnb.get_option_value(cp_flag, S10, K, r, T, sigma20, Q)
    V_s0_esig1 = optnb.get_option_value(cp_flag, S10, K, r, T, sigma21, Q)
    V_s1_esig0 = optnb.get_option_value(cp_flag, S11, K, r, T, sigma20, Q)
    V_s1_esig1 = optnb.get_option_value(cp_flag, S11, K, r, T, sigma21, Q)

    V_es0_sig0 = optnb.get_option_value(cp_flag, S20, K, r, T, sigma10, Q)
    V_es0_sig1 = optnb.get_option_value(cp_flag, S20, K, r, T, sigma11, Q)
    V_es1_sig0 = optnb.get_option_value(cp_flag, S21, K, r, T, sigma10, Q)
    V_es1_sig1 = optnb.get_option_value(cp_flag, S21, K, r, T, sigma11, Q)

    # Total
    Dv = V_s1_sig1 - V_s0_sig0    # pnl0
    Dv[bs_flag == 2] = -Dv[bs_flag == 2]    # 统一方向

    # Unexpected
    DvDu = V_es1_esig1 - V_es0_esig0
    DvDu[bs_flag == 2] = -DvDu[bs_flag == 2]    # 统一方向
    # unexpceted from underlying and sigma1
    DvDus = V_es1_sig0 - V_es0_sig0
    DvDus[bs_flag == 2] = -DvDus[bs_flag == 2]    # 统一方向
    DvDusig = V_s0_esig1 - V_s0_esig0
    DvDusig[bs_flag == 2] = -DvDusig[bs_flag == 2]    # 统一方向

    # Expected
    DvDt1 = V_s1_sig1 - V_es1_esig1
    DvDt1[bs_flag == 2] = -DvDt1[bs_flag == 2]    # 统一方向
    DvDt0 = V_s0_sig0 - V_es0_esig0
    DvDt0[bs_flag == 2] = -DvDt0[bs_flag == 2]    # 统一方向
    DvDe = DvDt1 - DvDt0
    # expected from underlying and sigma1 at t1 and t0
    DvDes1 = V_s1_sig1 - V_es1_sig1
    DvDes1[bs_flag == 2] = -DvDes1[bs_flag == 2]    # 统一方向
    DvDesig1 = V_s1_sig1 - V_s1_esig1
    DvDesig1[bs_flag == 2] = -DvDesig1[bs_flag == 2]    # 统一方向

    DvDes0 = V_s0_sig0 - V_es0_sig0
    DvDes0[bs_flag == 2] = -DvDes0[bs_flag == 2]    # 统一方向
    DvDesig0 = V_s0_sig0 - V_s0_esig0
    DvDesig0[bs_flag == 2] = -DvDesig0[bs_flag == 2]    # 统一方向

    # subtotal: t1 - t0
    DvDes = DvDes1 - DvDes0
    DvDesig = DvDesig1 - DvDesig0

    DvD = np.vstack([cp_flag, Dv, DvDu, DvDe, DvDus, DvDusig, DvDes, DvDesig]).T
    DvD_df = pd.DataFrame(DvD, columns=['cp_flag', 'Dv', 'DvDu', 'DvDe', 'DvDus', 'DvDusig', 'DvDes', 'DvDesig'])
    return DvD_df


def coral_getCode(name, date):
    return db.getCode(name, date, fields='code').toDataFrame()['code'].values


def coral_getCBar(code, beginDate, endDate, fields, cycle):
    return db.getBar(code, beginDate, endDate, fields=fields, cycle=cycle).toDataFrame()


def coral_getMBar(name, beginDate, endDate, fields, cycle, processes=8):
    """多进程提取一段时间内板块周期数据（根据这段时间内最新的代码）"""
    codes = coral_getCode(name, endDate)
    with Pool(processes=processes) as p:
        f_partial = partial(coral_getCBar, beginDate=beginDate, endDate=endDate, fields=fields, cycle=cycle)
        rs = p.map(f_partial, codes)
    return pd.concat(rs, sort=False).sort_values('timestamp').reset_index(drop=True)


def coral_getMBar1(name, beginDate, endDate, fields, cycle):
    """提取一段时间内板块周期数据（根据这段时间内最新的代码）"""
    rs = [coral_getCBar(co, beginDate, endDate, fields, cycle) for co in coral_getCode(name, endDate)]
    return pd.concat(rs, sort=False).sort_values('timestamp').reset_index(drop=True)


def coral_getMBarDaily(name, date, fields, cycle, processes=8):
    """多进程提取单日板块周期数据"""
    codes = coral_getCode(name, date)
    f_partial = partial(coral_getCBar, beginDate=date, endDate=date, fields=fields, cycle=cycle)
    with Pool(processes=processes) as p:
        rs = p.map(f_partial, codes)
    return pd.concat(rs, sort=False).sort_values('timestamp').reset_index(drop=True)


def coral_getMBar2(name, beginDate, endDate, fields, cycle, processes=8):
    """多进程提取一段时间内板块周期数据（根据当天更新存在代码）"""
    datearr = db.getTradingDays(beginDate, endDate).toDataFrame().values.squeeze()
    rs = pd.concat([coral_getMBarDaily(name, d, fields, cycle, processes=8) for d in datearr], sort=False)
    return rs.sort_values('timestamp').reset_index(drop=True)


# ------------------------------------------------------------------------------- #
# 交易部分
# ------------------------------------------------------------------------------- #
def get_open_signal(target_quota_option, th):
    open_signal = pd.DataFrame(np.zeros([target_quota_option.shape[0], 4]), columns=[
                               'long_open', 'long_close', 'short_open', 'short_close'])

    flag_long_open = np.array(nancmp(target_quota_option.est_option_price -
                                     target_quota_option.ap1, th['th_long_open'], 'greater', False))
    flag_short_close = np.array(nancmp(target_quota_option.est_option_price -
                                       target_quota_option.bp1, th['th_short_close'], 'less', False))
    flag_short_open = np.array(nancmp(target_quota_option.est_option_price -
                                      target_quota_option.bp1, th['th_short_open'], 'less', False))
    flag_long_close = np.array(nancmp(target_quota_option.est_option_price -
                                      target_quota_option.ap1, th['th_long_close'], 'greater', False))

    flag0 = np.array((target_quota_option['ap1'] > target_quota_option['bp1']) & (target_quota_option['bp1'] > 0))
    flag1_long = np.array(target_quota_option['ap1'].diff().shift(-1) <= 0)    # buyable
    flag1_short = np.array(target_quota_option['bp1'].diff().shift(-1) >= 0)    # sellable
    flag2 = np.array(target_quota_option['timestamp'].diff().shift(-1) > np.timedelta64(500, 'ms'))

    flag3 = np.array(target_quota_option['risk_factor'] < th['risk_on_vol'])    # chg2
    flag3[np.isnan(target_quota_option['spot_price']) | np.isnan(target_quota_option['iv_raw'])] = False    # chg2

    open_signal.loc[flag_long_open & flag0 & (flag1_long | flag2) & flag3, 'long_open'] = 1
    open_signal.loc[flag_short_close & flag0 & (flag1_short | flag2), 'short_close'] = 1
    open_signal.loc[flag_short_open & flag0 & (flag1_short | flag2) & flag3, 'short_open'] = 1
    open_signal.loc[flag_long_close & flag0 & (flag1_long | flag2), 'long_close'] = 1

    return open_signal


def get_trading_record(target_quota_option, open_signal):
    # 交易撮合
    open_flag = 0  # 状态，0表示空仓，1表示有仓位
    holdings = 0
    open_signal_tfarr = open_signal[['long_open', 'short_close', 'short_open', 'long_close']].values == 1
    my_endtime_idx = np.where(target_quota_option['stamp'] < target_quota_option['stamp'].max() - 300000)[0][-1]
    tradingrecord = {'date': [], 'timestamp': [], 'knockid': [], 'code': [], 'bs_flag': [], 'price': [],
                     'volume': [], 'holdings': [], 'open_flag': [], 'use_model': [], 'iv_raw': [], 'iv_pred': [],
                     'spot_price': [], 'est_spot_price': [], 'option_price': [], 'est_option_price': [], }
    for i, (lo, sc, so, lc) in enumerate(open_signal_tfarr):
        # long open
        if (open_flag == 0) & (i < my_endtime_idx) & lo:
            open_flag = 1
            tradingrecord['date'].append(target_quota_option.date[i])
            tradingrecord['code'].append(target_quota_option.code[i])
            tradingrecord['timestamp'].append(target_quota_option.stamp[i])
            tradingrecord['knockid'].append(i)
            tradingrecord['bs_flag'].append(1)
            tradingrecord['price'].append(-target_quota_option.ap1[i])
            knock_volume = 1
            tradingrecord['volume'].append(knock_volume)
            holdings = holdings + knock_volume
            tradingrecord['holdings'].append(holdings)
            tradingrecord['open_flag'].append(open_flag)
            tradingrecord['use_model'].append(target_quota_option.use_model[i])
            tradingrecord['iv_raw'].append(target_quota_option.iv_raw[i])
            tradingrecord['iv_pred'].append(target_quota_option.iv_pred[i])
            tradingrecord['spot_price'].append(target_quota_option.spot_price[i])
            tradingrecord['est_spot_price'].append(target_quota_option.est_spot_price[i])
            tradingrecord['option_price'].append(target_quota_option.option_price[i])
            tradingrecord['est_option_price'].append(target_quota_option.est_option_price[i])
        # short close
        elif (open_flag == 1) & (sc | (i == my_endtime_idx)):
            open_flag = 0
            tradingrecord['date'].append(target_quota_option.date[i])
            tradingrecord['code'].append(target_quota_option.code[i])
            tradingrecord['timestamp'].append(target_quota_option.stamp[i])
            tradingrecord['knockid'].append(i)
            tradingrecord['bs_flag'].append(2)
            tradingrecord['price'].append(target_quota_option.bp1[i])
            tradingrecord['volume'].append(holdings)
            tradingrecord['open_flag'].append(open_flag)
            tradingrecord['use_model'].append(target_quota_option.use_model[i])
            holdings = 0
            tradingrecord['holdings'].append(holdings)
            tradingrecord['iv_raw'].append(target_quota_option.iv_raw[i])
            tradingrecord['iv_pred'].append(target_quota_option.iv_pred[i])
            tradingrecord['spot_price'].append(target_quota_option.spot_price[i])
            tradingrecord['est_spot_price'].append(target_quota_option.est_spot_price[i])
            tradingrecord['option_price'].append(target_quota_option.option_price[i])
            tradingrecord['est_option_price'].append(target_quota_option.est_option_price[i])
        # short open
        if (open_flag == 0) & (i < my_endtime_idx) & so:
            open_flag = -1
            tradingrecord['date'].append(target_quota_option.date[i])
            tradingrecord['code'].append(target_quota_option.code[i])
            tradingrecord['timestamp'].append(target_quota_option.stamp[i])
            tradingrecord['knockid'].append(i)
            tradingrecord['bs_flag'].append(2)
            tradingrecord['price'].append(target_quota_option.bp1[i])
            knock_volume = 1
            tradingrecord['volume'].append(knock_volume)
            holdings = holdings - knock_volume
            tradingrecord['holdings'].append(holdings)
            tradingrecord['open_flag'].append(open_flag)
            tradingrecord['use_model'].append(target_quota_option.use_model[i])
            tradingrecord['iv_raw'].append(target_quota_option.iv_raw[i])
            tradingrecord['iv_pred'].append(target_quota_option.iv_pred[i])
            tradingrecord['spot_price'].append(target_quota_option.spot_price[i])
            tradingrecord['est_spot_price'].append(target_quota_option.est_spot_price[i])
            tradingrecord['option_price'].append(target_quota_option.option_price[i])
            tradingrecord['est_option_price'].append(target_quota_option.est_option_price[i])
        # long close
        elif (open_flag == -1) & (lc | (i == my_endtime_idx)):
            open_flag = 0
            tradingrecord['date'].append(target_quota_option.date[i])
            tradingrecord['code'].append(target_quota_option.code[i])
            tradingrecord['timestamp'].append(target_quota_option.stamp[i])
            tradingrecord['knockid'].append(i)
            tradingrecord['bs_flag'].append(1)
            tradingrecord['price'].append(-target_quota_option.ap1[i])
            tradingrecord['volume'].append(abs(holdings))
            tradingrecord['open_flag'].append(open_flag)
            tradingrecord['use_model'].append(target_quota_option.use_model[i])
            holdings = 0
            tradingrecord['holdings'].append(holdings)
            tradingrecord['iv_raw'].append(target_quota_option.iv_raw[i])
            tradingrecord['iv_pred'].append(target_quota_option.iv_pred[i])
            tradingrecord['spot_price'].append(target_quota_option.spot_price[i])
            tradingrecord['est_spot_price'].append(target_quota_option.est_spot_price[i])
            tradingrecord['option_price'].append(target_quota_option.option_price[i])
            tradingrecord['est_option_price'].append(target_quota_option.est_option_price[i])
    pdtradingrecord = pd.DataFrame(tradingrecord)
    return pdtradingrecord


# ------------------------------------------------------------------------------- #
# 其他
# ------------------------------------------------------------------------------- #
def make_line_sep(fill='-', n=39):
    return '\n#' + '-' * n + fill + '-' * n + '#\n'


def make_block_sep(fill='-', n=39):
    line_sep = make_line_sep(fill='', n=n + round(len(fill) / 2))
    return line_sep + '#' + '-' * n + fill + '-' * n + '#' + line_sep


def init_log(filename, level=log.DEBUG, nbprint=False):
    log.addLevelName(log.DEBUG, 'DEBUG')
    log.addLevelName(log.INFO, 'INFO')
    log.addLevelName(log.WARNING, 'WARN')
    log.addLevelName(log.ERROR, 'ERROR')
    log.addLevelName(log.CRITICAL, 'FATAL')
    pattern = '[%(asctime)s.%(msecs)-3d][%(levelname)-5s] %(lineno)d - %(message)s'
    formatter = log.Formatter(pattern, datefmt='%Y-%m-%d %H:%M:%S')
    log.basicConfig(level=level, format=pattern, datefmt='%Y-%m-%d %H:%M:%S', )
    del logging.getLogger().handlers[:]

    if nbprint:
        handler = log.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        log.getLogger().addHandler(handler)

    fh = log.handlers.RotatingFileHandler(filename, maxBytes=10 << 20, backupCount=5)
    fh.setFormatter(formatter)
    fh.setLevel(level)
    log.getLogger().addHandler(fh)
    return log


def get_notebook_name():
    """
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.join(ss['notebook_dir'], relative_path)


def iStop(txt='Stop on demand.'):
    raise Exception(txt)


def rough_50etf_expire_date(d0, d1):
    dates = pd.date_range(str(d0), str(d1))
    values = dates.groupby(dates.month).values()
    expire_dates = []
    for d in values:
        if np.any(sum(d.dayofweek == 3) > 3):
            expire_dates.append(int(d[d.dayofweek == 2][3].strftime('%Y%m%d')))
    return expire_dates


def list_method(instance):
    return print(dir(instance))
