# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-15 10:14:31
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-16 09:24:46

import numpy as np
import math
# import pandas as pd
from numba import jit, float64
# from numba import jit, float64, int64, boolean
# import math
# import matplotlib.pyplot as plt

############################
# # Advanced Computation # #
############################

@jit(nopython=True, nogil=True)
def numba_ewma(X, alpha, state=None, adjust=True, ignore_na=True, minp=1):
    """
    Compute exponentially-weighted moving average using center-of-mass.
    Parameters
    ----------
    X : ndarray (float64 type)
    alpha : float64
    state (None or array): value describe last state, [weighted_avg, old_wt]
    adjust: boolean
    ignore_na: boolean
    minp: min period
    Returns
    -------
    y : ndarray


    By default, it runs with the following parameters:
        _numba_ewma(x, alpha, adjust=True, ignore_na=True, minp=1)
    """

    N = len(X)
    if N == 0:
        output = np.empty(N, dtype=float64)
        output_state = state
    else:
        # np.put(X, np.isinf(X), np.nan)
        X[np.isinf(X)] = np.nan

        if state is None:
            old_wt = 1.
            drop_first = False
        else:
            X = np.array([state[0]] + list(X))
            old_wt = state[1]
            N += 1
            drop_first = True

        minp = max(minp, 1)
        old_wt_factor = 1. - alpha
        new_wt = 1. if adjust else alpha
        output = np.empty(N, dtype=float64)

        weighted_avg = X[0]
        is_observation = (weighted_avg == weighted_avg)
        nobs = int(is_observation)
        output[0] = weighted_avg if (nobs >= minp) else np.nan

        for i in range(1, N):
            cur = X[i]
            is_observation = (cur == cur)
            nobs += int(is_observation)
            if weighted_avg == weighted_avg:

                if is_observation or (not ignore_na):

                    old_wt *= old_wt_factor
                    if is_observation:

                        # avoid numerical errors on constant series
                        if weighted_avg != cur:
                            weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt)
                        if adjust:
                            old_wt += new_wt
                        else:
                            old_wt = 1.
            elif is_observation:
                weighted_avg = cur

            output[i] = weighted_avg if (nobs >= minp) else np.nan

        output = output[1:] if drop_first else output
        output_state = np.array([weighted_avg, old_wt])
    return output, output_state


@jit(nopython=True, nogil=True)
def _numba_ewcov(input_x, input_y, alpha, state, adjust, ignore_na, minp, bias):
    """
    Compute exponentially-weighted moving variance using decay factor.
    Parameters
    ----------
    input_x : ndarray (float64 type)
    input_y : ndarray (float64 type)
    alpha : float64
    state (None or array): value describe last state, [mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt]
    adjust: boolean
    ignore_na: boolean
    minp: min period
    bias: boolean
    Returns
    -------
    y : ndarray

    By default, it runs with the following parameters:
        _numba_ewcov(input_x, input_y, alpha, adjust=True, ignore_na=True, minp=1, bias=False)
    """

    N = len(input_x)
    if N == 0:
        output = np.empty(N, dtype=float64)
        output_state = state
    else:
        # np.put(input_x, np.isinf(input_x), np.nan)
        # np.put(input_y, np.isinf(input_y), np.nan)

        input_x[np.isinf(input_x)] = np.nan
        input_y[np.isinf(input_y)] = np.nan

        if state is None:
            cov, sum_wt, sum_wt2, old_wt = 0., 1., 1., 1.
            drop_first = False
        else:
            input_x = np.array([state[0]] + list(input_x))
            input_y = np.array([state[1]] + list(input_y))
            cov, sum_wt, sum_wt2, old_wt = state[2:]
            N += 1
            drop_first = True

        output = np.empty(N, dtype=float64)
        minp = max(minp, 1)
        old_wt_factor = 1. - alpha
        new_wt = 1. if adjust else alpha

        mean_x, mean_y = input_x[0], input_y[0]
        is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
        nobs = int(is_observation)
        if not is_observation:
            mean_x, mean_y = np.nan, np.nan
        output[0] = (0. if bias else np.nan) if (nobs >= minp) else np.nan

        for i in range(1, N):
            cur_x, cur_y = input_x[i], input_y[i]
            is_observation = ((cur_x == cur_x) and (cur_y == cur_y))
            nobs += int(is_observation)
            if mean_x == mean_x:
                if is_observation or (not ignore_na):
                    sum_wt *= old_wt_factor
                    sum_wt2 *= (old_wt_factor * old_wt_factor)
                    old_wt *= old_wt_factor
                    if is_observation:
                        old_mean_x = mean_x
                        old_mean_y = mean_y

                        # avoid numerical errors on constant series
                        if mean_x != cur_x:
                            mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt)

                        # avoid numerical errors on constant series
                        if mean_y != cur_y:
                            mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt)
                        cov = ((old_wt * (cov + ((old_mean_x - mean_x) *
                                                 (old_mean_y - mean_y)))) +
                               (new_wt * ((cur_x - mean_x) *
                                          (cur_y - mean_y)))) / (old_wt + new_wt)
                        sum_wt += new_wt
                        sum_wt2 += (new_wt * new_wt)
                        old_wt += new_wt

                        if not adjust:
                            sum_wt /= old_wt
                            sum_wt2 /= (old_wt * old_wt)
                            old_wt = 1.

            elif is_observation:
                mean_x = cur_x
                mean_y = cur_y

            if nobs >= minp:
                if not bias:
                    numerator = sum_wt * sum_wt
                    denominator = numerator - sum_wt2
                    if (denominator > 0.):
                        output[i] = ((numerator / denominator) * cov)
                    else:
                        output[i] = np.nan
                else:
                    output[i] = cov
            else:
                output[i] = np.nan

        output = output[1:] if drop_first else output
        output_state = np.array([mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt])
    return output, output_state


@jit(nopython=True, nogil=True)
def numba_ewvar(X, alpha, state=None, adjust=True, ignore_na=False, minp=1, bias=False):
    """Exponentialy weighted moving varaiance specified by a decay ``alpha``

    Args:
        X (array): raw data
        alpha (float): decay factor
        state (None or array): value describe last state, [mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt]
        adjust (boolean):
            True for assuming infinite history via the recursive form
            False for assuming finite history via the recursive form
        ignore_na (boolean): True for decaying by relative location, False for absolute location
        minp (int): min periods
        bias (boolean) : keep bias. default is False
        return_state (boolean). if only return state value

    Returns:
        TYPE: Description

    By default, it runs with the following parameters:
        numba_ewcov(x, alpha adjust=True, state=None, ignore_na=True, minp=1, bias=False, return_state=False)
    """
    output_ew, output_state = _numba_ewcov(X, X, alpha=alpha, adjust=adjust, state=state, ignore_na=ignore_na, minp=minp, bias=bias)
    return output_ew, output_state


@jit(nopython=True, nogil=True)
def numba_ewstd(X, alpha, state=None, adjust=True, ignore_na=False, minp=1, bias=False):
    """Exponentialy weighted moving varaiance specified by a decay ``alpha``

    Args:
        X (array): raw data
        alpha (float): decay factor
        state (None or array): value describe last state, [mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt]
        adjust (boolean):
            True for assuming infinite history via the recursive form
            False for assuming finite history via the recursive form
        ignore_na (boolean): True for decaying by relative location, False for absolute location
        minp (int): min periods
        bias (boolean) : keep bias. default is False
        return_state (boolean). if only return state value

    Returns:
        TYPE: Description

    By default, it runs with the following parameters:
        numba_ewstd(x, alpha adjust=True, state=None, ignore_na=True, minp=1, bias=False, return_state=False)
    """
    output_ew, output_state = _numba_ewcov(X, X, alpha=alpha, adjust=adjust, state=state, ignore_na=ignore_na, minp=minp, bias=bias)
    return np.sqrt(output_ew), output_state


@jit(nopython=True, nogil=True)
def numba_ols_beta(x, y):
    sum_xy = 0
    sum_x_sq = 0
    for x0, y0 in zip(x, y):
        sum_xy += x0 * y0
        sum_x_sq += x0 * x0
    if sum_x_sq == 0:
        out = np.nan
    else:
        out = sum_xy / sum_x_sq
    return out


@jit(nopython=True, nogil=True)
def numba_ols_resid(x, y):
    # no intercept
    resid = y - numba_ols_beta(x, y) * x
    return np.mean(resid)


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


@jit(nopython=True)
def numba_rollresid(x, y, w, min_valid_sample):
    """Summary

    Args:
        x (numpy.array): Description
        y (numpy.array): Description
        w (int): Description
        min_valid_sample (int): Description

    Returns:
        list: Description
    """
    resids = []
    if w < min_valid_sample:
        min_valid_sample = w

    for i in range(len(x)):
        if i < min_valid_sample:
            resid = np.nan
        elif i < w:
            x0, y0 = x[:i], y[:i]
            idx_good = ~(np.isnan(x0) | np.isnan(y0))
            if np.sum(idx_good) < min_valid_sample:
                resid = np.nan
            else:
                resid = numba_ols_resid(x0[idx_good], y0[idx_good])
        else:
            x0, y0 = x[i - w:i], y[i - w:i]
            idx_good = ~(np.isnan(x0) | np.isnan(y0))
            if np.sum(idx_good) < min_valid_sample:
                resid = np.nan
            else:
                resid = numba_ols_resid(x0[idx_good], y0[idx_good])
        resids.append(resid)
    return np.array(resids)
    

@jit(nopython=True)
def brent(f, x0, x1, max_iter=100, tolerance=1e-6):
    """Summary
    https://nickcdryan.com/2017/09/13/root-finding-algorithms-in-python-line-search-bisection-secant-newton-raphson-boydens-inverse-quadratic-interpolation-brents/

    Args:
        f (TYPE): function decorated by @numba.jit(nopython=True)
        x0 (TYPE): minimum point
        x1 (TYPE): maximum point
        max_iter (int, optional): maximum iteration
        tolerance (float, optional): Description

    Returns:
        TYPE: Description
    """
    fx0 = f(x0)
    fx1 = f(x1)

    steps_taken = 0
    if fx0 * fx1 <= 0:
        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
            fx0, fx1 = fx1, fx0

        x2, fx2 = x0, fx0

        mflag = True
        d, x2 = x2, x1
        while steps_taken < max_iter and abs(x1 - x0) > tolerance:
            fx0 = f(x0)
            fx1 = f(x1)
            fx2 = f(x2)

            if fx0 != fx2 and fx1 != fx2:
                L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
                L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
                L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
                new = L0 + L1 + L2
            else:
                new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

            cond1 = (new < ((3 * x0 + x1) / 4) or new > x1)
            cond2 = (mflag and (abs(new - x1)) >= (abs(x1 - x2) / 2))
            cond3 = ((not mflag) and (abs(new - x1)) >= (abs(x2 - d) / 2))
            cond4 = (mflag and (abs(x1 - x2)) < tolerance)
            cond5 = ((not mflag) and (abs(x2 - d)) < tolerance)
            if (cond1 or cond2 or cond3 or cond4 or cond5):
                new = (x0 + x1) / 2
                mflag = True
            else:
                mflag = False

            fnew = f(new)
            d, x2 = x2, x1

            if (fx0 * fnew) < 0:
                x1 = new
            else:
                x0 = new

            if abs(fx0) < abs(fx1):
                x0, x1 = x1, x0

            steps_taken += 1
    else:
        # Root not bracketed
        x1 = np.nan

    return x1, steps_taken


@jit(nopython=True, nogil=True)
def numba_pearsonr(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    sum_x, sum_y = np.sum(x), np.sum(y)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.
    # if (sum_x == 0) or (sum_y == 0):
    #     return np.nan
    else:
        avg_x = float(sum_x) / len(x)
        avg_y = float(sum_y) / len(y)
        diffprod = 0
        xdiff2 = 0
        ydiff2 = 0
        for idx in range(n):
            xdiff = x[idx] - avg_x
            ydiff = y[idx] - avg_y
            diffprod += xdiff * ydiff
            xdiff2 += xdiff * xdiff
            ydiff2 += ydiff * ydiff
        return diffprod / math.sqrt(xdiff2 * ydiff2)


@jit(nopython=True, nogil=True)
def numba_pairwise_pearsonr(arr1, arr2):
    col = arr1.shape[1]
    rho = np.full(col, np.nan)
    for i, (a, b) in enumerate(zip(arr1.T, arr2.T)):
        rho[i] = numba_pearsonr(a, b)
    return rho


@jit(nopython=True, nogil=True)
def numba_rolling_pearsonr(a, b, n):
    # rolling
    assert a.shape == b.shape
    row = a.shape[0]
    rho = np.full(row, np.nan)
    for i in range(n - 1, row):
        k1, k2 = i - n + 1, i + 1
        rho[i] = numba_pearsonr(a[k1:k2], b[k1:k2])
    return rho


@jit(nopython=True, nogil=True)
def numba_pairwise_rolling_pearsonr(a, b, n):
    # pairwise rolling
    assert a.shape == b.shape
    row, col = a.shape
    rho = np.full((row, col), np.nan)
    for i in range(n - 1, row):
        for j in range(col):
            k1, k2 = i - n + 1, i + 1
            rho[i, j] = numba_pearsonr(a[k1:k2, j], b[k1:k2, j])
    return rho


@jit(nopython=True, nogil=True)
def numba_spearmanr(a, b):
    rank_a = numba_rankdata_vector(a, method='min')
    rank_b = numba_rankdata_vector(b, method='min')
    corr = numba_pearsonr(rank_a, rank_b)
    return corr


@jit(nopython=True, nogil=True)
def numba_pairwise_spearmanr(arr1, arr2):
    col = arr1.shape[1]
    rho = np.full(col, np.nan)
    for i, (a, b) in enumerate(zip(arr1.T, arr2.T)):
        rho[i] = numba_spearmanr(a, b)
    return rho


@jit(nopython=True, nogil=True)
def numba_rolling_spearmanr(a, b, n):
    # rolling
    row = a.shape[0]
    rho = np.full(row, np.nan)
    for i in range(n - 1, row):
        k1, k2 = i - n + 1, i + 1
        rho[i] = numba_spearmanr(a[k1:k2], b[k1:k2])
    return rho


@jit(nopython=True, nogil=True)
def numba_pairwise_rolling_spearmanr(a, b, n):
    # pairwise rolling
    assert a.shape == b.shape
    row, col = a.shape
    rho = np.full((row, col), np.nan)
    for i in range(n - 1, row):
        for j in range(col):
            k1, k2 = i - n + 1, i + 1
            rho[i, j] = numba_spearmanr(a[k1:k2, j], b[k1:k2, j])
    return rho



@jit(nopython=True, nogil=True)
def numba_standardize_vector(a, nan_policy='omit'):
    if len(a.shape) == 1:
        if nan_policy == 'omit':
            sd = np.nanstd(a)
            if sd == 0:
                out = np.full_like(a, np.nan)
            else:
                out = (a - np.nanmean(a)) / sd
        else:
            sd = np.std(a)
            if sd == 0:
                out = np.full_like(a, np.nan)
            else:
                out = (a - np.mean(a)) / sd
    else:
        raise ValueError('use numba_standardize_matrix for 2d array')
    return out


@jit(nopython=True, nogil=True)
def numba_standardize_matrix(arr, nan_policy='omit'):
    out = np.full_like(arr, np.nan)
    for i, a in enumerate(arr.T):
        out[:, i] = numba_standardize_vector(a, nan_policy=nan_policy)
    return out


@jit(nopython=True, nogil=True)
def numba_rolling_standardize_vector(a, n, nan_policy='omit'):
    # rolling
    if len(a.shape) == 1:
        row = len(a)
        assert row > n, 'len(a) is smaller than n.'
        rho = np.full(row, np.nan)
        for i in range(n - 1, row):
            k1, k2 = i - n + 1, i + 1
            rho[i] = numba_standardize_vector(a[k1:k2], nan_policy=nan_policy)[-1]
    else:
        raise ValueError('use numba_rolling_standardize_matrix for 2d array')
    return rho


@jit(nopython=True, nogil=True)
def numba_rolling_standardize_matrix(arr, n, nan_policy='omit'):
    # rolling
    if len(arr.shape) == 2:
        row, col = arr.shape
        rho = np.full((row, col), np.nan)
        for i in range(n - 1, row):
            for j in range(col):
                k1, k2 = i - n + 1, i + 1
                rho[i, j] = numba_standardize_vector(arr[k1:k2, j], nan_policy=nan_policy)[-1]
    else:
        raise ValueError('use numba_rolling_standardize_vector for 1d array')
    return rho



@jit(nopython=True, nogil=True)
def numba_minmaxscale_vector(a, nan_policy='omit'):
    if len(a.shape) == 1:
        if np.all(a[0] == a):
            out = np.full_like(a, 0.5)
        else:
            if nan_policy == 'omit':
                _a = a[~np.isnan(a)]
                mn = np.min(_a)
                mx = np.max(_a)
                out = (_a - mn) / (mx - mn)
            else:
                mn = np.min(a)
                mx = np.max(a)
                out = (a - mn) / (mx - mn)
    else:
        raise ValueError('use numba_minmaxscale_matrix for 2d array')
    return out


@jit(nopython=True, nogil=True)
def numba_minmaxscale_matrix(arr, nan_policy='omit'):
    out = np.full_like(arr, np.nan)
    for i, a in enumerate(arr.T):
        out[:, i] = numba_minmaxscale_vector(a, nan_policy=nan_policy)
    return out


@jit(nopython=True, nogil=True)
def numba_rolling_minmaxscale_vector(a, n, nan_policy='omit'):
    # rolling
    if len(a.shape) == 1:
        row = len(a)
        assert row > n, 'len(a) is smaller than n.'
        rho = np.full(row, np.nan)
        for i in range(n - 1, row):
            k1, k2 = i - n + 1, i + 1
            rho[i] = numba_minmaxscale_vector(a[k1:k2], nan_policy=nan_policy)[-1]
    else:
        raise ValueError('use numba_rolling_minmaxscale_matrix for 2d array')
    return rho


@jit(nopython=True, nogil=True)
def numba_rolling_minmaxscale_matrix(arr, n, nan_policy='omit'):
    # rolling
    if len(arr.shape) == 2:
        row, col = arr.shape
        rho = np.full((row, col), np.nan)
        for i in range(n - 1, row):
            for j in range(col):
                k1, k2 = i - n + 1, i + 1
                rho[i, j] = numba_minmaxscale_vector(arr[k1:k2, j], nan_policy=nan_policy)[-1]
    else:
        raise ValueError('use numba_rolling_minmaxscale_vector for 1d array')
    return rho



@jit(nopython=True, nogil=True)
def numba_rankdata_vector(a, method):
    if method not in ('min', 'max', 'dense'):
        # raise ValueError('unknown method "{0}"'.format(method))
        raise ValueError('got unknown method')

    sorter = np.argsort(a, kind='quicksort')
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    a = a[sorter]
    obs = np.append([True], a[1:] != a[:-1])
    dense = np.cumsum(obs)[inv]

    if method == 'dense':
        out = dense
    # # cumulative counts of each unique value
    count = np.append(np.nonzero(obs)[0], [len(obs)])
    if method == 'max':
        out = count[dense]

    if method == 'min':
        out = count[dense - 1] + 1

    # if method == 'average':
    #     out = 0.5 * np.array(count[dense] + count[dense - 1] + 1)
    return out


@jit(nopython=True, nogil=True)
def numba_rankdata_matrix(arr, method):
    out = np.full_like(arr, np.nan)
    for i, a in enumerate(arr.T):
        out[:, i] = numba_rankdata_vector(a, method=method)
    return out


@jit(nopython=True, nogil=True)
def numba_rolling_rankdata_vector(a, n, method):
    # rolling
    if len(a.shape) == 1:
        row = len(a)
        assert row > n, 'len(a) is smaller than n.'
        rho = np.full(row, np.nan)
        for i in range(n - 1, row):
            k1, k2 = i - n + 1, i + 1
            rho[i] = numba_rankdata_vector(a[k1:k2], method=method)[-1]
    else:
        raise ValueError('use numba_rolling_rankdata_matrix for 2d array')
    return rho


@jit(nopython=True, nogil=True)
def numba_rolling_rankdata_matrix(arr, n, method):
    # rolling
    if len(arr.shape) == 2:
        row, col = arr.shape
        rho = np.full((row, col), np.nan)
        for i in range(n - 1, row):
            for j in range(col):
                k1, k2 = i - n + 1, i + 1
                rho[i, j] = numba_rankdata_vector(arr[k1:k2, j], method=method)[-1]
    else:
        raise ValueError('use numba_rolling_rankdata_vector for 1d array')
    return rho

##################
# # Data Tools # #
##################

@jit(nopython=True)
def check_equal_length(*args):
    m = len(args[0])
    for n in args:
        if len(n) != m:
            return False
    return True


@jit(nopython=True)
def anynan(array):
    for a in array:
        if math.isnan(a):
            return True
    return False


@jit(nopython=True)
def anynan_along_axis(x, axis=0):
    rs = []
    if axis == 0:
        for i in range(x.shape[axis]):
            rs.append(anynan(x[i, :]))
    elif axis == 1:
        for i in range(x.shape[axis]):
            rs.append(anynan(x[:, i]))
    return np.array(rs)


@jit(nopython=True)
def nnan(array):
    i = 0
    for a in array:
        if math.isnan(a):
            i += 1
    return i


@jit(nopython=True)
def get_last_valid_value(sequence):
    for i, x in enumerate(sequence[::-1]):
        if not math.isnan(x):
            break
    return x, i



@jit(nopython=True, nogil=True)
def split(data, ncut, flag):
    if flag == 1:
        sorted_slices = np.linspace(data.min(), data.max(), ncut + 1)
    elif flag == 2:
        sorted_slices_pct = np.linspace(0, 100, int(ncut + 1))
        sorted_slices = np.percentile(data, sorted_slices_pct)
    else:
        raise ValueError('unknown flag')

    sorted_slices[0] = sorted_slices[0] - 1e-8    # include left
    sorted_slices[-1] = sorted_slices[-1] + 1e-8    # include right
    indices = np.searchsorted(sorted_slices, data, side='left')
    return indices


@jit(nopython=True, nogil=True)
def nansplit(data, ncut, flag):
    idx_nan = np.isnan(data)
    data = data[~idx_nan]
    indices = split(data, ncut, flag)
    out = np.full(len(idx_nan), np.nan)
    out[~idx_nan] = indices
    return out

##################
# # Basic Calc # #
##################

@jit(nopython=True)
def nb_divide(a, b, out, where):
    assert len(a) == len(b) == len(out) == len(where)
    for i in range(len(a)):
        if where[i]:
            out[i] = a[i] / b[i]
    return out

