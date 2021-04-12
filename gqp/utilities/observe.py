# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-13 18:04:02
# @Last Modified by:   gabrielfeng
# @Last Modified time: 2020-05-13 18:07:52


#####################
# # Observe Tools # #
#####################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import functools

import gqp.utilities.data_tools as utdat


def __init__():
    pass


def observe_factor(arr, bins=50, fixna=False):
    arr = arr.copy()
    if fixna:
        arr = utdat.numpy_ffillna_matrix(arr, method='ffill')
        arr = arr[~np.isnan(arr)]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(arr)
    axes[0].grid(True)
    axes[1].hist(arr, bins=bins)
    axes[1].grid(True)


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


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Function {func.__name__} comsumed {(end - start) * 1000:.4f} ms')
        return res

    return wrapper


