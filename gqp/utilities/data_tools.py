# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-13 18:18:07
# @Last Modified by:   Gabriel
# @Last Modified time: 2020-10-27 16:00:43

###################
# # Data Tools # #
###################

import numpy as np
import pandas as pd
from functools import reduce
import re
import os


def __init__():
    pass


def is_distinct(g):
    # fastest method!
    s = set()
    for x in g:
        if x in s:
            return False
        s.add(x)
    return True


def union1d(_list):
    """
    get union1d set of all lists withitn list

    :param      _list:  The list
    :type       _list:  { type_description }
    """
    return reduce(np.union1d, _list)


def intercept(_list):
    """
    get intercept set of all lists withitn list

    :param      _list:  The list
    :type       _list:  { type_description }
    """
    return reduce(np.intersect1d, _list)


def checkEqualIvo(lst):
    """
    check if all lists within list have same length

    :param      lst:  The list
    :type       lst:  { type_description }
    """
    return not lst or lst.count(lst[0]) == len(lst)


def is_all_equal(x):
    return np.all(x == x[0])


def first_true(mylist):
    """Summary
    locate first ture value in the list.

    Args:
        mylist (TYPE): iterable type

    Returns:
        TYPE: int
    """
    return next((i for i, x in enumerate(mylist) if x), None)


def drop_nan_and_inf(arr, out=None):
    idx_tf = np.isnan(arr) | np.isinf(arr)
    if out == 'index':
        return idx_tf
    return arr[~idx_tf]


def select_slice_time(df, time_slices, s=True, e=False):
    for ts_s, ts_e in time_slices:
        df_s = df.between_time(ts_s, ts_e, include_start=s, include_end=e)
        yield df_s


def get_str(string):
    list_str = re.findall('[a-zA-Z]*', string)
    return [ss for ss in list_str if len(ss) > 0]


def get_digit(string):
    list_digit = re.findall('[0-9]*', string)
    return [ss for ss in list_digit if len(ss) > 0]


def get_sign(data):
    """
    Get signs of the data.
    :param data:
    :return: element-wise signs
    """
    return abs(data) / data


def get_lengths(sequences):
    return list(map(np.shape, sequences))


def dropna(data):
    """
    Drop NaN values from one-dimensional input.

    Args:
        data (TYPE): Description
    """
    return data[~np.isnan(data)]


def np_clip_and_replace(a, a_min, a_max, value):
    # enhancing numpy.clip()
    arr = a.copy()
    np.place(arr, (arr < a_min) | (arr > a_max), value)
    return arr


def pd_clip_and_replace(df, a_min, a_max, value):
    out_df = df.copy()
    mask = (df < a_min) | (df > a_max)
    out_df[mask] = value
    return out_df


def compress_dataframe(df):
    dtypes = df.dtypes
    newdf = df.copy()
    for coln, dtyp in dtypes.iteritems():
        if 'object' in dtyp.name:
            newdf[coln] = pd.Categorical(newdf[coln])
        elif 'float' in dtyp.name:
            newdf[coln] = pd.to_numeric(newdf[coln], downcast='float')
        elif 'int' in dtyp.name:
            newdf[coln] = pd.to_numeric(newdf[coln], downcast='integer')
    return newdf


def binarize(x, th=0, flag=0, type_out=float):
    """
    flag:
    0: upside is including threshold
    1: upside is not including threshold
    2: triplet

    return
:    0 or -1: represent x < th
    1: upside
    """
    arr = np.array(x)
    if flag == 0:
        res = (arr >= th).astype(float) - (arr < th).astype(float)
    elif flag == 1:
        res = (arr > th).astype(float) - (arr <= th).astype(float)
    elif flag == 2:
        res = (arr > th).astype(float) - (arr < th).astype(float)
    else:
        raise ValueError('flag {} is not defined.'.format(flag))

    if isinstance(x, pd.core.series.Series):
        return pd.Series(res, index=x.index, name=x.name, dtype=type_out)

    return res.astype(type_out)


def get_flatten_index_by_side(arr, target_value, on_side):
    """
    提取在目标值左边或右边的index
    如果on_side == 'left'：提取每行中最后一个目标值(如果有多个)，所有左边的索引
    如果on_side == 'right'：提取每行中首个目标值(如果有多个)，所有右边的索引

    Args:
        arr (TYPE): 2-d numpy array
        target_value (TYPE): target value
        on_side (string): 'left' ot 'right'

    Returns:
        TYPE: Description
    """
    arr = arr.copy()
    rows, colns = arr.shape
    flatten_n = colns * np.arange(rows)

    if on_side == 'left':
        idx_end = (arr != target_value).cumsum(1).argmax(1)
        idx_flatten = np.hstack([np.arange(e + 1, arr.shape[1]) + d for e, d in zip(idx_end, flatten_n)])
    elif on_side == 'right':
        idx_begin = (arr != target_value).argmax(1)
        idx_flatten = np.hstack([np.arange(0, b) + d for b, d in zip(idx_begin, flatten_n)])

    return idx_flatten


def replace_side_on_target(target_value, replace_to, arr, on_side):
    """
    替换目标值左边或右边的值
    如果on_side == 'left'：替换每行中最后一个目标值(如果有多个)，所有左边的值
    如果on_side == 'right'：提取每行中首个目标值(如果有多个)，所有右边的值

    Args:
        target_value (TYPE): Description
        replace_to (TYPE): Description
        arr (TYPE): Description
        on_side (TYPE): Description

    Returns:
        TYPE: Description
    """
    arr = arr.copy()
    flatten_index = get_flatten_index_by_side(arr, target_value, on_side)
    if len(flatten_index) > 0:
        np.put(arr, flatten_index, replace_to)
    return arr


def label_timeindex(timeindex, period):
    return timeindex.searchsorted(timeindex - pd.Timedelta(seconds=period))


def scale_index(index, n, form='%Y-%m-%d %H:%M:%S', dtype='str'):
    """
    generate time index as strings for plotting when time index have "ugly" gap.

    :param      index:  The index
    :type       index:  { type_description }
    :param      n:      { parameter_description }
    :type       n:      { type_description }
    :param      form:   The form
    :type       form:   string
    :param      dtype:  The dtype
    :type       dtype:  string

    :returns:   { description_of_the_return_value }
    :rtype:     { return_type_description }
    """
    xaxis_idx = map(int, np.linspace(0, len(index) - 1, n))
    if isinstance(index, pd.core.indexes.datetimes.DatetimeIndex):
        xaxis_val = index.strftime(form).astype(dtype)
    else:
        xaxis_val = index.astype(dtype)
    return xaxis_idx, xaxis_val[xaxis_idx]


def numpy_shift(a, shift, axis=0):
    """Mimic pd.DataFrame(a).shift(shift=shift, axis=axis) in pandas

    Args:
        a (TYPE): Description
        shift (TYPE): Description
        axis (int, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if axis == 0:
        shifted_a = np.roll(a, shift, axis=0)
        if shift > 0:
            shifted_a[:shift] = np.nan
        else:
            shifted_a[shift:] = np.nan
    elif axis == 1:
        shifted_a = np.roll(a.T, shift, axis=0)
        if shift > 0:
            shifted_a[:shift] = np.nan
        else:
            shifted_a[shift:] = np.nan
        shifted_a = shifted_a.T
    else:
        raise ValueError('axis can only be 0 or 1.')
    return shifted_a


def numpy_fillna_plus(arr, fill_value=None, method='left'):
    """Summary
    Fill nan values in array.

    Args:
        arr (numpy.array): an array
        method (str):
            - left: fill all nan value on the left of the first valid values
            - right: fill all nan value on the right of the first valid values
            - between: fill all nan value between two valid values at extreme locations.
            - np_fill: numpy forward fill from left to right
            - others: same as methods in pd.fillna()
    Returns:
        TYPE: numpy.array
    """

    arr = arr.astype(float).copy()

    if method in ['right', 'left', 'between']:
        if fill_value is None:
            raise ValueError('fill_value is not assigned.')

        idx_arr_notnull = ~np.isnan(arr)
        mask_range = np.repeat(np.arange(arr.shape[1])[
                               None, :], arr.shape[0], axis=0)

        if method == 'right':
            idx_end = idx_arr_notnull.cumsum(1).argmax(1)
            mask = mask_range >= (idx_end[:, None] + 1)
        elif method == 'left':
            idx_begin = idx_arr_notnull.argmax(1)
            mask = mask_range <= (idx_begin[:, None] - 1)
        elif method == 'between':
            idx_end = idx_arr_notnull.cumsum(1).argmax(1)
            idx_begin = idx_arr_notnull.argmax(1)
            mask = (mask_range >= idx_begin[:, None]) & (
                mask_range <= idx_end[:, None]) & np.isnan(arr)

        masked_arr = np.ma.array(arr, mask=mask, fill_value=0, copy=True)
        out = masked_arr.filled(fill_value)

    elif method == 'np_ffill':
        '''Solution provided by Divakar.'''
        # forward-fill start frome left to right
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]

    else:
        df = pd.DataFrame(arr.copy())
        df.fillna(method=method, axis=0, inplace=True)
        out = df.values.reshape(arr.shape)

    return out


def nancmp(a, b, method, fillna):
    """
    - fillna must be boolean
    - return False if any nan
    """
    assert isinstance(fillna, bool), 'fillna must be boolean'
    out = np.full_like(a, fillna, dtype='bool')
    where = ~(np.isnan(a) | np.isnan(b))
    return eval('np.{}(a, b, out=out, where=where)'.format(method, out, where))


def get_values_between(series, lb, ub, qtile=True, return_type=None):
    if qtile:
        ub = np.percentile(series[~np.isnan(series)], ub)
        lb = np.percentile(series[~np.isnan(series)], lb)

    tf = (series > lb) & (series < ub)
    if return_type == bool:
        return tf
    else:
        return series[tf]


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


def groupby(df, freq, key=None, level=None):
    timegrouper = pd.Grouper(key=key, level=level, freq=freq, closed='left', label='right')
    return df.groupby(timegrouper)


def numpy_ffillna_vector(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]
    return out


def numpy_ffillna_matrix(arr, axis=0):
    assert (axis == 0) or (axis == 1), 'can only handle two dimensions.'
    if axis == 0:
        arr = arr.T

    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    arr = arr[np.arange(idx.shape[0])[:, None], idx]

    if axis == 0:
        arr = arr.T
    return arr


def init_data_directory(filepaths=[f'dataset/data', f'dataset/result']):
    for fp in filepaths:
        try:
            os.listdir(fp)
            print('[Info] 已存在:', fp)
        except FileNotFoundError as e:
            print(e)
            os.makedirs(fp, exist_ok=False)
            print('[Success] 已创建:', fp)


