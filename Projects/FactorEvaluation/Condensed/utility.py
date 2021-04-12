# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:23:56 2016

@author: Gabriel F.
"""


import pandas as pd
import numpy as np
import scipy.stats as scs
# from minepy import MINE


# def split_data(data, nq, flag=1):
#     """Summary

#     Split data into nq blocks.

#     Args:
#         data (array): factors on first column is required.
#         nq (int): number of segmentation

#     Returns:
#         TYPE: Description
#     """
#     subset_data = data[:, 0].astype(float)
#     if flag == 1:
#         quantiles = np.linspace(subset_data.min() - 1e-15, subset_data.max() + 1e-15, nq)
#     elif flag == 2:
#         segs = np.linspace(0, 1, nq + 1)
#         quantiles = np.percentile(subset_data, segs)
#     grp_bounds = zip(quantiles, quantiles[1:])
#     grp_names = np.digitize(subset_data, quantiles) - 1
#     grp_data = [data[grp_names == i, :] for i in np.unique(grp_names)]
#     return grp_data, grp_bounds, grp_names


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
        quantiles = np.linspace(data.min() - 1e-15, data.max() + 1e-15, nq)
    elif flag == 2:
        segs = np.linspace(0, 1, nq + 1)
        quantiles = np.percentile(data, segs)
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
