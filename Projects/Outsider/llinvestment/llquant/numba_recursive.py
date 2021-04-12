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
def numba_pearsonr(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    sum_x, sum_y = np.sum(x), np.sum(y)
    if (sum_x == 0) or (sum_y == 0):
        return np.nan
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
def numba_spearmanr(a, b):
    rank_a = a.argsort().argsort()
    rank_b = b.argsort().argsort()
    corr = numba_pearsonr(rank_a, rank_b)
    return corr


@jit(nopython=True, nogil=True)
def numba_rolling_spearmanr(a, b, n):
    # rolling
    row = a.shape[0]
    rho = np.full(row, np.nan)
    for i in range(n - 1, row):
        k1, k2 = i - n + 1, i + 1
        rho[i] = numba_spearmanr(a[k1:k2], b[k1:k2])
    return rho
