# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-13 18:03:38
# @Last Modified by:   gabrielfeng
# @Last Modified time: 2020-05-13 18:03:50

#############################
# # Numpy Rolling Compute # #
#############################

import numpy as np
import pandas as pd
from numba import jit
import gqp.utilities.numba_recursive as nbr
import gqp.utilities as gut
try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    print("sliding_window_view is not imported.")

    def sliding_window_view(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def __init__():
    pass


def rolling_window(arr, window, axis=0):
    """
    Usage:
    a = np.random.rand(30, 5)
    for 2d array:
        roll aling axis=0: rolling_window(a.T, 3).transpose(1, 2, 0)
        roll along axis=1: rolling_window(a, 3).transpose(1, 0, 2)
    for 3d array:
        roll along height(axis=0): rolling_window(a.transpose(2, 1, 0), 3).transpose(2, 3, 1, 0)
        roll along width(axis=1): rolling_window(a, 3).transpose(2, 0, 1, 3)
        roll along depth(axis=2): rolling_window(a.transpose(0, 2, 1), 3).transpose(3, 0, 2, 1)
    """

    def _rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    if arr.ndim == 1:
        return _rolling_window(arr, window)
    elif arr.ndim == 2:
        if axis == 0:
            return _rolling_window(arr.T, window).transpose(1, 2, 0)
        elif axis == 1:
            return _rolling_window(arr, window).transpose(1, 0, 2)
        else:
            raise Exception(
                'AxisError: axis {} is out of bounds for array of dimension {}'.format(axis, arr.ndim))
    elif arr.ndim == 3:
        if axis == 0:
            return _rolling_window(arr.transpose(0, 2, 1), window).transpose(3, 0, 2, 1)
        elif axis == 1:
            return _rolling_window(arr, window).transpose(2, 0, 1, 3)
        elif axis == 2:
            return _rolling_window(arr.transpose(2, 1, 0), window).transpose(2, 3, 1, 0)
        else:
            raise Exception(
                'AxisError: axis {} is out of bounds for array of dimension {}'.format(axis, arr.ndim))
    else:
        return _rolling_window(arr, window)


def rolling_apply(func, x, window, forward=True):
    """Summary

    Args:
        func (function): Description
        x (TYPE): Description
        window (int): Description
        forward (bool, optional): Description

    Returns:
        TYPE: Description
    """
    if forward:
        arr = np.array(x)
    else:
        arr = np.array(x)[::-1]

    res = map(func, rolling_window(arr, window))
    try:
        return np.array(res)
    except Exception:
        return list(res)


def numpy_rolling_apply(func, a, w, f_axis=1):
    row = a.shape[0]
    out = np.full(row, np.nan)
    out[w-1:] = func(sliding_window_view(a, window_shape=w), axis=f_axis)
    return out


def rolling_extend(func, x, forward=True):
    """
    Apply specific function by rolling forward or backward.

    :param func: function to be applied
    :param x: variables
    :param forward: Apply with forward value if ture. Default is true.
    :return:
    """
    if forward:
        arr = np.array(x)
    else:
        arr = np.array(x)[::-1]

    res = (np.nan,) + tuple(func(arr[:i]) for i in range(2, len(arr) + 1))
    return np.array(res)


def rollreg(x, y, w, min_valid_sample, padding=np.nan):
    x_w = rolling_window(x, w)
    y_w = rolling_window(y, w)
    idx_good = ~(np.isnan(x_w) | np.isnan(y_w))
    is_valid = idx_good.sum(1) > min_valid_sample
    rs = [nbr.numba_ols_beta(x0[idx].reshape(-1, 1), y0[idx].reshape(-1, 1)) if tf else np.nan for x0,
          y0, idx, tf in zip(x_w, y_w, idx_good, is_valid)]
    if padding is not None:
        rs = np.hstack([np.full(w - 1, padding), np.squeeze(rs)])
    return rs


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
                    beta = nbr.numba_ols_beta(x_[~idx_nan].reshape(-1, 1), y_[~idx_nan])
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
def rolling_window_ts(x, ts, tfreq):
    N = len(x)
    j0 = 0
    windows = []
    for i in range(N):
        for j in range(j0, N):
            if (ts[j] - ts[i]) > tfreq:
                windows.append(x[i:j])
                j0 = j + 1
                break
        if j0 == N:
            break
    return windows


@jit(nopython=True)
def rolling_diff_ts(x, ts, tfreq):
    N = len(x)
    j0 = 0
    diffs = []
    for i in range(N):
        for j in range(j0, N):
            if (ts[j] - ts[i]) > tfreq:
                diffs.append(x[j] - x[i])
                j0 = j + 1
                break
        if j0 == N:
            break
    return diffs


@jit(nopython=True)
def rolling_anynan_ts(x, ts, tfreq):
    N = len(x)
    j0 = 0
    isnans = []
    for i in range(N):
        for j in range(j0, N):
            if (ts[j] - ts[i]) > tfreq:
                isnans.append(nbr.anynan(x[i:j]))
                j0 = j + 1
                break
        if j0 == N:
            break
    return isnans


def rolling_combine(func, x, win, lag):
    arr = np.array(x)
    res = np.zeros(len(arr), dtype='O')
    res[win - 1:] = [func(arr[i:i + win]) for i in range(len(arr) - win + 1)]

    if isinstance(x, pd.core.series.Series):
        return pd.Series(res, index=x.index, name=x.name).shift(lag)
    elif lag < 0:
        return np.roll(res, lag)[:lag]
    elif lag > 0:
        return np.roll(res, lag)[lag:]
    else:
        return res


def get_func_argnames(func):
    """
    Retrieve function's input arguments

    :param func:
    :return: a tuple of names of arguments
    """
    argcnt = func.__code__.co_argcount
    argvars = func.__code__.co_varnames
    return argvars[:argcnt]


def run_argtup(func, argvalues):
    """
    Execute any functions with their input arguments in tuples.

    :param func:
    :param argvalues:
    :return: results from assigned function
    """
    argnames = get_func_argnames(func)
    if len(argnames) != len(argvalues):
        raise ValueError("Length of args doens't match.")
    for argn, argv in zip(argnames, argvalues):
        exec('{}=argv'.format(argn))
        exec('{}={}'.format(argn, argv))
    return eval('func(%s, %s)' % argnames)


def rolling_df(func, df, win, apply_colns=None):
    """

    :param func:
    :param df: the orders of df.columns should be the same as function input
    :param win: windows
    :param apply_colns: optional.
    :return:
    """
    rolrang = range(df.shape[0] - win + 1)
    vals = [run_argtup(func, tuple(df[i:i + win].values.T)) for i in rolrang]
    results = pd.DataFrame(vals, columns=apply_colns, index=df.index[win - 1:])
    return results


