# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-13 18:04:19
# @Last Modified by:   Gabriel
# @Last Modified time: 2020-11-30 20:43:33


#####################
# # Compute Tools # #
#####################

import numpy as np
import scipy.stats as scs
import pandas as pd
import gqp.utilities.rolling as roll
import statsmodels.api as sm
from numba import jit
from copy import copy

def __init__():
    pass


def date_lag(date_list, target, shift):
    return list(date_list)[list(date_list).index(target) + shift]


def get_diff_trading_days(date_list, date1, date2):
    dd = date_list.index(date1) - date_list.index(date2)
    return np.abs(dd)


def get_pre_trading_date(date, n, ref_trade_dates):
    d1 = np.array(ref_trade_dates)[list(ref_trade_dates).index(date) - n]
    return d1


def scale(a, range_in, range_out=(-1, 1)):
    assert len(range_in) == len(range_out)
    in0, in1 = range_in
    out0, out1 = range_out
    b = (a - in0) / (in1 - in0)
    c = b * (out1 - out0) + out0
    return c


def numpy_diff(arr, n=1, fill_values=np.nan, axis=-1):
    """Mimic pd.DataFrame(a).diff(n=shift, axis=axis).fillna(np.nan) in pandas

    Args:
        arr (TYPE): Description
        n (int, optional): Description
        fill_values (TYPE, optional): Description
        axis (TYPE, optional): Description

    Returns:
        TYPE: Description
    """
    return np.insert(np.diff(arr, n=n, axis=axis), [0] * n, fill_values, axis=axis)


def rescale_by_rank(x, scale=(0, 1)):
    drang = float(scale[1]) - scale[0]
    out = scale[0] + drang * scs.rankdata(x) / len(x)
    if isinstance(x, pd.core.series.Series):
        return pd.Series(out, index=x.index, name=x.name)

    return out


def findLocalMaxima(ar):
    # find local maxima of array, including centers of repeating elements
    maxInd = np.zeros_like(ar)
    peakVar = -np.inf
    i = -1
    while i < len(ar) - 1:
        i += 1
        if peakVar < ar[i]:
            peakVar = ar[i]
            for j in range(i, len(ar)):
                if peakVar < ar[j]:
                    break
                elif peakVar == ar[j]:
                    continue
                elif peakVar > ar[j]:
                    peakInd = i + np.floor(abs(i - j) / 2)
                    maxInd[peakInd.astype(int)] = 1
                    i = j
                    break
        peakVar = ar[i]
    maxInd = np.where(maxInd)[0]
    return maxInd


def local_extrema(data, flag, **kwargs):
    from scipy import signal
    globals().update(kwargs)
    if flag == 1:
        inds_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
        inds_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max
        inds = [inds_min, inds_max]
    elif flag == 2:
        inds_min = signal.find_peaks_cwt(-data, np.ones(5) * 3)
        inds_max = signal.find_peaks_cwt(data, np.ones(5) * 3)
        inds = [inds_min, inds_max]
    elif flag == 3:
        inds_min = signal.argrelextrema(data, np.less)[0]
        inds_max = signal.argrelextrema(data, np.greater)[0]
        inds = [inds_min, inds_max]
    elif flag == 4:
        # find local maxima of array, including centers of repeating elements
        inds_min = findLocalMaxima(-data)
        inds_max = findLocalMaxima(data)
        inds = [inds_min, inds_max]
    else:
        inds = []
    return inds


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


def min_max_scaling(a, axis=0):
    return (a - np.nanmin(a, axis=axis, keepdims=True)) / (np.nanmax(a, axis=axis, keepdims=True) - np.nanmin(a, axis=axis, keepdims=True))

# def min_max_scaling(a, axis=0):
#     a_min = np.nanmin(a, axis=axis, out=np.full_like(a, np.nan))
#     a_max = np.nanmax(a, axis=axis, out=np.full_like(a, np.nan))
#     return (a - a_min) / (a_max - a_min)


def standardize(a, axis=0):
    # Difference in default, numpy.std(ddof=0) vs. pandas.Series.std(ddof=1)
    return (a - np.nanmean(a, axis=axis, keepdims=True)) / np.nanstd(a, axis=axis, keepdims=True)


def rolling_minmaxscale(arr, period):
    rollarr = roll.rolling_window(arr, period)
    rolmin = np.insert(np.nanmin(rollarr, 1), [0] * (period - 1), np.nan)
    rolmax = np.insert(np.nanmax(rollarr, 1), [0] * (period - 1), np.nan)
    rs = (arr - rolmin) / (rolmax - rolmin) - 0.5
    return rs


def rolling_standardize(arr, period):
    rollarr = roll.rolling_window(arr, period)
    rollmean = rollarr.mean(1)
    rollstd = rollarr.std(1)
    normals = (arr[period - 1:] - rollmean) / rollstd
    return normals


def demean(a, axis=0):
    return a - np.nanmean(a, axis=axis, keepdims=True)


def compute_resid(y, x):
    X = sm.add_constant(x)
    m = sm.OLS(y, X)
    return m.fit().resid


def neutralize(X, B):
    """Neutralize factors according to specific dimension such as market cap and industry
    
    Args:
        X (Nx1 array): factor values
        B (2-d array): specific dimensions
    
    Returns:
        TYPE: Description
    
    Raises:
        ValueError: Description
    """
    if np.any(np.isnan(X)):
        raise ValueError('X cannot contain nan values.')
    X_neutral = np.empty_like(X)
    for i in range(X.shape[1]):
        X_neutral[:, i] = compute_resid(X[:, i], B)
    return X_neutral


def convex_mapping(x):
    return np.exp(-(x * np.exp(1))**2)


def keep_less_correlated(corrmat, min_corr, keep='high', absolute=True, nan_policy='keep'):
    corrmat = copy(corrmat)
    np.fill_diagonal(corrmat, 0.)

    if absolute:
        corrmat = np.abs(corrmat)

    if nan_policy == 'keep':
        corrmat[np.isnan(corrmat)] = 1.

    if keep == 'high':
        _m = min
    elif keep == 'low':
        _m = max
    else:
        raise ValueError('unknown keep.', keep)

    components = list(range(len(corrmat)))
    indices = list(range(len(corrmat)))
    # Iteratively remove least fit individual of most correlated pair
    while np.max(corrmat) >= min_corr:
        most_correlated = np.unravel_index(np.argmax(corrmat), corrmat.shape)

        # The correlation matrix is sorted by fitness, so identifying
        # the least fit of the pair is simply getting the higher index
        worst = _m(most_correlated)
        components.pop(worst)
        indices.remove(worst)
        corrmat = corrmat[:, indices][indices, :]
        indices = list(range(len(components)))
    return corrmat, indices