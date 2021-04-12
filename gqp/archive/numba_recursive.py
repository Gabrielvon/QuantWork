import numpy as np
# import pandas as pd
from numba import jit, float64
# from numba import jit, float64, int64, boolean
# import math
# import matplotlib.pyplot as plt


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
        return np.empty(N, dtype=float64), state

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
                        weighted_avg = ((old_wt * weighted_avg) +
                                        (new_wt * cur)) / (old_wt + new_wt)
                    if adjust:
                        old_wt += new_wt
                    else:
                        old_wt = 1.
        elif is_observation:
            weighted_avg = cur

        output[i] = weighted_avg if (nobs >= minp) else np.nan

    state = np.array([weighted_avg, old_wt])
    output = output[1:] if drop_first else output
    return output, state


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
        return np.empty(N, dtype=float64), state

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

    state = np.array([mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt])
    output = output[1:] if drop_first else output
    return output, state


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
    ew, state = _numba_ewcov(X, X, alpha=alpha, adjust=adjust, state=state, ignore_na=ignore_na, minp=minp, bias=bias)
    return ew, state


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
    ew, state = _numba_ewcov(X, X, alpha=alpha, adjust=adjust, state=state, ignore_na=ignore_na, minp=minp, bias=bias)
    return np.sqrt(ew), state


@jit(nopython=True, nogil=True)
def numba_ols_beta(x, y):
    sum_xy = 0
    sum_x_sq = 0
    for x0, y0 in zip(x, y):
        sum_xy += x0 * y0
        sum_x_sq += x0 * x0
    return sum_xy / sum_x_sq


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
