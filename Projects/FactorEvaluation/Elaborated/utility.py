# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:23:56 2016

@author: Gabriel F.
"""

import pandas as pd
import numpy as np
import scipy.stats as scs


def split_data(data, nq, flag=1):
    """Summary

    Split data into nq blocks.

    Args:
        data (array): factors on first column is required.
        nq (int): number of segmentation

    Returns:
        TYPE: Description
    """
    if flag == 1:
        quantiles = np.linspace(data.min(), data.max(), nq + 1)
    elif flag == 2:
        segs = np.linspace(0, 100, nq + 1)
        quantiles = np.percentile(data, segs)
    quantiles[0] = quantiles[0] - 1e-15
    quantiles[-1] = quantiles[-1] + 1e-15
    grp_names = np.digitize(data, quantiles) - 1
    return grp_names


def calc_corr(data):
    return list(scs.pearsonr(data[:, 0], data[:, 1]))


def calc_ic(data):
    """Summary

    Calculate information coefficient

        Args:
        data (array): Description

    Returns:
        tuple: spearman coefficient and p value
    """
    return scs.spearmanr(data[:, 0], data[:, 1]).correlation


# def calc_mic(data):
#     """Summary

#     Calculate maximum informaiton coefficient

#     Args:
#         data (array): Description

#     Returns:
#         TYPE: Description
#     """
#     m = MINE()
#     m.compute_score(data[:, 0], data[:, 1])
#     return m.mic()


def calc_mse(data, ax=0):
    """Summary

    Calculate mean square error

    Args:
        data (array): Description
        ax (int, optional): Description

    Returns:
        TYPE: Description
    """
    return ((data[:, 0] - data[:, 1]) ** 2).mean(axis=ax)


def calc_granger_caulity(data):
    """Summary

    Args:
        data (array): array without nan values

    Returns:
        TYPE: Description
    """
    from statsmodels.tsa.api import VAR
    # data_dropna = data[~np.isnan(data).any(1)]
    try:
        model = VAR(data)
        res = model.fit(verbose=False)
        out = res.test_causality(0, 1, verbose=False)['statistic']
    except ValueError as e:
        # print 'calc_granger_caulity: ', 'most factors are zeros.'
        out = np.nan
    return out


def scale_index(index, n, form='%Y-%m-%d %H:%M:%S', dtype='str'):
    xaxis_idx = map(int, np.linspace(0, len(index) - 1, n))
    if isinstance(index, pd.core.indexes.datetimes.DatetimeIndex):
        xaxis_val = index.strftime(form).astype(dtype)
    else:
        xaxis_val = index.astype(dtype)
    return xaxis_idx, xaxis_val[xaxis_idx]


def detect_outlier(seq, n_std):
    """Summary

    Label outlier.

    Args:
        seq (array): Description
        n_std (int or float): Description

    Returns:
        array of boolean: Description
    """
    med = np.median(seq)
    amp = n_std * np.std(seq)
    over_cap = seq > (med + amp)
    under_cap = seq < (med - amp)
    return over_cap | under_cap


def update_std(sumsq_x, sum_x, cnt):
    var = (sumsq_x - sum_x**2 / cnt) / cnt
    return np.sqrt(var)


def __revert_compressed_val(comp_val, seed=False):
    # comp_arr = np.array(comp_val, dtype=[('mu', float), ('std', float), ('cnt', int)])
    for mu, sd, ct in comp_val:
        if ct > 1:
            if seed:
                np.random.seed(seed)
            yield scs.norm.rvs(loc=mu, scale=sd, size=int(ct))
        elif ct == 1:
            yield mu


def revert_compressed_val(comp_val):
    return np.hstack(__revert_compressed_val(comp_val))


def update_maxminscale(stats_on_target, lastest_minmax):
    """
    online maxminscale normalization using key basic statistics.
    Return: sumsq, sum and count

    stats_on_target: recorded stats in the past which is going to be updated. i.e. [sum of square, sum, count] of last
    lastest_minmax: current status i.e. [min, max] of current
    """
    target_xss, target_xs, target_xct = stats_on_target
    xmn, xmx = lastest_minmax

    zss = (target_xss - 2 * xmn * target_xs + target_xct * xmn**2) / (xmx - xmn)**2
    zs = (target_xs - target_xct * xmn) / (xmx - xmn)
    zct = target_xct

    return zss, zs, zct


def update_zscore(stats_on_target, lastest_mustd):
    """
    online zscore normalization using key basic statistics.

    Args:
        stats_on_target (TYPE): recorded stats in the past which is going to be updated. i.e. [sum of square, sum, count] of last
        lastest_mustd (TYPE): current status i.e. [mu, std] of current

    Returns:
        TYPE: sumsq, sum and count
    """
    target_xss, target_xs, target_xct = stats_on_target
    xmu, xstd = lastest_mustd

    zss = (target_xss - 2 * target_xs * xmu + target_xct * xmu ** 2) / xstd ** 2
    zs = (target_xs - target_xct * xmu) / xstd
    zct = target_xct

    return zss, zs, zct
