import re

import numpy as np


# from numba import jit, float64


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_str(string):
    list_str = re.findall('[a-zA-Z]*', string)
    return [ss for ss in list_str if len(ss) > 0]


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


def min_max_scaling(a, axis=0):
    return (a - np.nanmin(a, axis=axis, keepdims=True)) / (
                np.nanmax(a, axis=axis, keepdims=True) - np.nanmin(a, axis=axis, keepdims=True))


def standardize(a, axis=0):
    # Difference in default, numpy.std(ddof=0) vs. pandas.Series.std(ddof=1)
    return (a - np.nanmean(a, axis=axis, keepdims=True)) / np.nanstd(a, axis=axis, keepdims=True)


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
    return np.exp(-(x * np.exp(1)) ** 2)


def numpy_ols_beta(x, y):
    try:
        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))[0][0]
    except np.linalg.LinAlgError as e:
        beta = np.nan
    return beta


def numpy_ols_resid(x, y):
    return np.sum(y - numpy_ols_beta(x, y) * x)


class ieval():
    def __init__(self, global_context, local_context):
        self.globals = global_context
        self.locals = local_context

    def eval(self, string):
        return eval(string, self.globals, self.locals)


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
