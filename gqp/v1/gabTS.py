# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from itertools import permutations


def hurst(ts, minLags=2, maxLags=20):  # FaRst
    # Create a range of lag values
    lags = np.arange(minLags, maxLags)
    # Calculate variance array of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Slow: Use a linear fit to estimate
    #poly = np.polyfit(np.log(lags), np.log(tau), 1)[0]

    # Fast
    # From: Derek M. Tishler - dmtishler@gmail.com
    # 1st degree polynomial approximation for speed
    # source: http://stackoverflow.com/questions/28237428/fully-vectorise-np-polyfit
    n = len(lags)
    x = np.log(lags)
    y = np.log(tau)
    poly = (n * (x * y).sum() - x.sum() * y.sum()) / \
        (n * (x * x).sum() - x.sum() * x.sum())

    # Return the Hurst exponent
    hurst_exp = poly * 2.0
    return hurst_exp


def regress(y, x):
    import statsmodels.api as sm
    X = sm.add_constant(x)
    result = sm.OLS(y, X).fit()
    return result


def half_life(theta):
    """
    theta is the mean reversion coefficient. It can be obtained from OU process.

    :param theta:
    :return:
    """
    return np.log(2) / theta


# def half_life(resid):
#     dz = np.diff(resid)
#     prevz = np.roll(dz, 1)[1:]
#     result = regress(dz[1:], prevz - np.mean(prevz))
#     theta = result.params[1]
#     if theta < 0:
#         hl = -np.log(2) / theta
#     else:
#         hl = np.nan
#         raise Warning('theta cannot be negtive')
#     return hl


def lbtest(data, lags=None, boxpierce=False):
    """
    Ljung-box test and Box-pierce(optional) test for no autocorrelation.
    H0: no autocorrelation

    For how to determine lags, and use and interpret the result, please read the references:
    https://stats.stackexchange.com/questions/200267/interpreting-ljung-box-test-results-from-statsmodels-stats-diagnostic-acorr-lju

    :param data:
    :param lags:
    :param boxpierce:
    :return:
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox as aclb
    df = pd.DataFrame(list(aclb(data, lags, boxpierce))).T
    df.index = df.index + 1
    df.index.name = 'lag'
    df.columns = ['lbstat', 'lbpval', 'bpstat', 'bppval'][:df.shape[1]]
    return df


def adftest(data, maxlag=None, regres=None):
    """
    Null Hypothesis (H0): If accepted, it suggests the time series has a unit
    root, meaning it is non-stationary. It has some time dependent structure.
    Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests
    the time series does not have a unit root, meaning it is stationary. It
    does not have time-dependent structure.
    """
    from statsmodels.tsa.stattools import adfuller
    if isinstance(data, float):
        return np.nan
    teststat, pval, lag, nobs, cv, icbest = adfuller(data, maxlag=maxlag, regression=regres)
    results = {'adf': teststat, 'pvalue': pval, 'usedlag': lag, 'nobs': nobs, 'best_ic': icbest}
    cv = {'crt_val(' + k + ')': v for k, v in cv.items()}
    results.update(cv)
    return pd.Series(results)


def eg_coint(data, tr='c', method='aeg', maxlag=None, autolag='aic', return_results=None):
    """
    Null Hypothesis (H0): If accepted, it suggests the time series has a unit
    root, meaning it is non-stationary. It has some time dependent structure.
    Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests
    the time series does not have a unit root, meaning it is stationary. It
    does not have time-dependent structure.
    """
    from statsmodels.tsa.stattools import coint
    if isinstance(data, float):
        return np.nan
    elif isinstance(data, pd.core.frame.DataFrame):
        y0, y1 = data.values.T
    else:
        y0, y1 = data

    teststat, pval, cv = coint(y0, y1, tr, method, maxlag, autolag, None)
    results = {'stat': teststat, 'pvalue': pval}
    cv_keys = ['1%', '5%', '10%']
    cv = {'crt_val(' + k + ')': v for k, v in zip(cv_keys, cv)}
    results.update(cv)
    if isinstance(data, pd.core.frame.DataFrame):
        return pd.Series(results, name='~'.join(data.columns))
    else:
        return pd.Series(results)


def arima(ts, order, verbose=False):
    from statsmodels.tsa.arima_model import ARIMA
    try:
        model = ARIMA(ts, order=order)
        m_fit = model.fit(disp=True)
        res = [order, m_fit.aic, m_fit.bic, m_fit.hqic, m_fit]
    except Exception as e:
        if verbose:
            print('\n', order, '\n', e, '\n')
            print(e)
        res = [order, np.nan, np.nan, np.nan, np.nan]

    if verbose:
        print(res)

    return res


def auto_arima(ts, order_tuples, n_jobs=4, verbose=False):
    """
    future improvement:
        1. add max order for attempts
        2. add a condition for stop attempts
    """

    pool = ThreadPool(processes=n_jobs)
    rec = pool.map(lambda x: arima(ts, x), order_tuples)
    df = pd.DataFrame(rec, columns=['orders', 'aic', 'bic', 'hqic', 'model'])

    return df


def stationarity_test(ts, params):
    #    orders = permute3(range(0,6), range(0,2), range(0,6))
    p, d, q = params
    orders = permutations(range(0, p + 1), range(0, d + 1), range(0, q + 1))
    arima_res = auto_arima(ts, order_tuples=orders, n_jobs=4, verbose=False)
    idx = arima_res['aic'].idxmax()
    if np.isnan(idx):
        best_aic = np.nan
        hl = np.nan
    else:
        best_m = arima_res.loc[idx, :]
        best_aic = best_m['aic']
        hl = half_life(best_m['model'].resid)
    out = pd.Series([adftest(ts)[1], hurst(ts), best_aic, hl],
                    index=['adf', 'hurst', 'aic', 'half-life'],
                    name=ts.index[0].date())
    return out


# def rebalance_position(y, x):
#     result = regress(y, x)
#     beta = result.params[1]
#     lookback = int(half_life(result.resid))
#     if lookback <= 0:
#         ret = 0.0
#     else:
#         spread = y - beta * x
#         spread.ffill(inplace=True)
#         ma = spread.rolling(lookback).mean()
#         mstd = spread.rolling(lookback).std()
#         zscore = (spread - ma) / mstd

#         positions = pd.concat([-zscore, zscore], 1)
#         stkp = pd.concat([y,x],axis=1)
#         positions.columns = [y.name, x.name]
#         pnl = np.sum(positions.shift(1) * stkp.pct_change(), 1)
#         ret = pnl / np.sum(abs(positions.shift(1)), 1)

#     return pd.Series({'ret':np.sum(ret), 'lookback':lookback, 'tcost'})


def test_gc(data, maxlag=None, signif=0.05, verbose=False):
    """Summary
    Apply granger causaulity test into permutation of all columns

    Args:
        data (TYPE): Description
        maxlag (None, optional): Description
        signif (float, optional): Description
        verbose (bool, optional): Description

    Returns:
        TYPE: dataframe
    """
    from statsmodels.tsa.api import VAR
    if isinstance(data, pd.core.frame.DataFrame):
        colns = data.columns
        arr = data.values
    else:
        arr = np.array(data)

    model = VAR(arr)
    if maxlag:
        res = model.fit(maxlag, verbose=verbose)
    else:
        res = model.fit(verbose=verbose)
    gc_test = []
    obs_name = res.names
    for c1, c2 in permutations(obs_name, 2):
        gc_res = res.test_causality(c1, c2, signif=signif, verbose=verbose)
        coln1, coln2 = colns[[obs_name.index(c1), obs_name.index(c2)]]
        gc_res = pd.Series(gc_res, name=(coln1, coln2))
        gc_res['H0'] = "'{}' do not Granger-cause '{}'".format(coln2, coln1)
        gc_test.append(gc_res)
    results = pd.DataFrame(gc_test)
    results['VAR'] = model
    results['best_order'] = (len(model.exog_names) - 1) / data.shape[1]
    return results


def test_gc2(data, gc_format, maxlag=None, signif=0.05, verbose=False):
    from statsmodels.tsa.api import VAR
    model = VAR(data)
    if maxlag:
        res = model.fit(maxlag, verbose=verbose)
    else:
        res = model.fit(verbose=verbose)
    gc_res = res.test_causality(gc_format[0], gc_format[1],
                                signif=signif, verbose=verbose)
    # results = pd.Seires({k: v for k, v in gc_res.iteritems() if k in ['conclusion','pvalue']})

    results = pd.Series(gc_res)
    results['H0'] = "'{}' do not Granger-cause '{}'".format(
        gc_format[1], gc_format[0])
    results['VAR'] = res
    results['best_order'] = (len(model.exog_names) - 1) / data.shape[1]
    return results


def test_gc3(data, maxlag=10, addconst=True, verbose=False):
    from statsmodels.tsa.stattools import grangercausalitytests
    if isinstance(data, pd.core.frame.DataFrame):
        # colns = data.columns
        arr = data.values
    else:
        arr = np.array(data)

    res = grangercausalitytests(arr, maxlag, addconst, verbose)
    res_dict = dict()
    for k, v in res.iteritems():
        vtests = {k0: [vi for i, vi in enumerate(
            v0)][:2] for k0, v0 in v[0].items() if k0 != 'params_ftest'}
        models = pd.Series(v[1][:2], index=['restri', 'unrestri'])
        res_dict[k] = pd.Series(vtests).append(models)
    return pd.DataFrame(res_dict).T

