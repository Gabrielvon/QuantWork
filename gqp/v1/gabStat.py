# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:16:54 2017

@author: Gabriel F.
"""
import pandas as pd
import numpy as np
import scipy.stats as scs
from statsmodels.stats.diagnostic import lillifors
import matplotlib.pyplot as plt
import random
import itertools


def unstack_freq(arr, add_e=None):
    if isinstance(arr, pd.core.frame.DataFrame):
        arr = arr.values
    if add_e:
        unstacked = np.hstack(
            [list(itertools.repeat(x, int(f))) +
             np.random.random(size=int(f)) * add_e for x, f in arr]
        )
    else:
        unstacked = np.hstack(
            [list(itertools.repeat(x, int(f))) for x, f in arr])
    return unstacked


# Normality Test
def check_normality(testData, plot=False, sampling=False,
                    is_freq=False, add_e=False, verbose=False):
    # 20<样本数<50用normal test算法检验正态分布性
    if is_freq:
        testData = unstack_freq(testData, add_e=add_e)

    if sampling:
        testData = random.sample(testData, sampling)

    if len(testData) <= 50:
        p_value = scs.shapiro(testData)[1]
        if verbose:
            print("use shapiro:")

    if 50 < len(testData) <= 300:
        p_value = lillifors(testData)[1]
        if verbose:
            print("use lillifors:")

    if len(testData) > 300:
        mu, sigma = scs.norm.fit(testData)
        norm_fit = scs.norm(loc=mu, scale=sigma)
        p_value = scs.kstest(testData, norm_fit.cdf)[1]
        if verbose:
            print("use kstest:")

    if plot:
        norm_fit = scs.norm(loc=np.mean(testData), scale=np.std(testData))
        uni_data = np.unique(testData)
        num_of_bins = min(len(uni_data), 200)
        sorted_uni_data = np.sort(uni_data)
        min_diff = scs.mode(np.diff(sorted_uni_data)).mode
        fitted_dist_x = np.arange(
            sorted_uni_data[0], sorted_uni_data[-1] + min_diff, min_diff)

        _, axes = plt.subplots(2, 1)
        axes[0].hist(testData, num_of_bins)
        axes[1] = scs.probplot(testData, plot=plt)
        axes[0].plot(fitted_dist_x, 350 * norm_fit.pdf(fitted_dist_x))
        plt.show()
    return p_value


def check_poisson(testData, plot=False):
    mu = np.mean(testData)
    meanFreq = pd.Series(testData).value_counts().mean()
    p_value = scs.kstest(testData, scs.poisson(meanFreq, loc=mu).cdf)
    if plot:
        _, axes = plt.subplots(2, 1)
        axes[0].hist(testData)
        poiDa = scs.poisson.rvs(meanFreq, loc=mu, size=len(testData))
        axes[1] = scs.probplot(poiDa, plot=plt)
    return p_value


def generate_shape_data(series):
    import itertools
    scsd_stats = scs.describe(series)
    features = pd.Series(list(itertools.chain(scsd_stats[1], scsd_stats[2:])),
                         index=['min', 'max', 'mean', 'var',
                                'skewness', 'kurtosis'])
    qs = pd.Series([series.quantile(qi) for qi in np.arange(0.1, 1, 0.1)],
                   index=['q[' + str(qi) + ']' for qi in np.arange(0.1, 1, 0.1)])

    return features.append(qs)


def select_high_corr_pair(df, min_abs_corr, plot_heatmap=False):
    df_corr = df.corr()
    np.fill_diagonal(df_corr.values, 0)

    if plot_heatmap:
        import seaborn as sns
        sns.heatmap(df_corr)

    pair_prod_tf = df_corr.abs() > min_abs_corr
    pair_prod_tf_dict = pair_prod_tf.to_dict(orient='dict')
    pair_corr = [[k1, k2, df_corr.loc[k1, k2]]
                 for k1, v1 in pair_prod_tf_dict.items() for k2, v2 in v1.items() if v2]
    pair_corr = pd.DataFrame(pair_corr, columns=['prod1', 'prod2', 'corr'])

    return pair_corr


def detect_outlier(seq, n_std):
    med = np.median(seq)
    amp = n_std * np.std(seq)
    over_cap = seq > (med + amp)
    under_cap = seq < (med - amp)
    return over_cap | under_cap


def remove_outlier(seq, n_std, return_outlier=False):
    outlier_mask = detect_outlier(seq, n_std)
    normal_data = seq[~outlier_mask]
    outliers = seq[outlier_mask]
    if return_outlier:
        return normal_data, outliers
    else:
        return normal_data


def quantile_loc(series, value, value_included=False,
                 return_rank=False, verbose=False):
    """
    Find the rank/quantile location of given value in a list

    :param myl: a list of numeric values which must be sorted in advanced.
    :param value: given value for location
    :param value_included: if true, myl includes value
    :param rank: if true , return the rank instead of quantile values
    """
    if not isinstance(series, pd.core.series.Series):
        series = pd.Series(series)

    if not value_included:
        series = series.append(pd.Series(value))

    series.index = series
    val_rank = series.rank(method='average')
    val_pct = series.rank(method='average', pct=True)

    try:
        v_r = val_rank[value].drop_duplicates().tolist()[0]
        v_p = val_pct[value].drop_duplicates().tolist()[0]
    except AttributeError:
        if verbose:
            print(Warning('Input series has duplicated values.\n'))
        v_r = val_rank[value]
        v_p = val_pct[value]

    if return_rank is True:
        return v_r
    else:
        return v_r


def idescribe(series):
    sta = scs.describe(series)
    mi, ma = sta.minmax
    ss = pd.Series(sta._asdict()).drop('minmax')
    ss = ss.append(pd.Series({'min': mi, 'max': ma}))
    return ss


def multi_collinearity_test(data, flag=0):
    """
    When flag=0:
    Return values of the determinant of eigenvalues of df.
    The value describe the multi collinearity level.
    The larger value the more unlikely multi collinearity.
    * 0 = perfect collinearity, 1 = no collinearity
    When flag=1:
    When flag=2:
    Return variance inflation factor.
    It quantifies the severity of multicollinearity in an ordinary least squares regression analysis
    A rule of thumb for interpreting the variance inflation factor:
    * 1 = not correlated.
    * Between 1 and 5 = moderately correlated.
    * Greater than 5 = highly correlated
    """
    corr = np.corrcoef(data, rowvar=0)
    if flag == 0:
        res = np.linalg.det(corr)
    elif flag == 1:
        res = np.linalg.eig(corr)
    elif flag == 2:
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
        if isinstance(data, pd.core.frame.DataFrame):
            res = {c: vif(data.values, data.columns.get_loc(c)) for c in data.columns}
        else:
            res = {i: vif(data, i) for i, v in enumerate(data.T)}
    else:
        raise ValueError('flag {} is not defined.'.format(flag))
    return res


def compute_confint(data, alpha=0.95):
    """
    The underlying assumptions for both are that the sample (array a)
    was drawn independently from a normal distribution with unknown
    standard deviation

    Reference:
    http://mathworld.wolfram.com/Studentst-Distribution.html
    https://en.wikipedia.org/wiki/Student's_t-distribution

    :param data: data
    :param alpha: confident level
    :return:
    """
    import statsmodels.stats.api as smsa
    return smsa.DescrStatsW(data).tconfint_mean(alpha)


def fibs(n, seq=False):
    """
    Generate nth fibonacci number
    References:
        http://mathworld.wolfram.com/FibonacciNumber.html
        http://mortada.net/fibonacci-numbers-in-python.html
    :param n: the number of digits in the fibonacci series.
    :param seq: Default is False, whihc return the nth digits; True will return a series
    of n number starting from 0.
    :return:
    """
    a = 1 + np.sqrt(5)
    b = 1 - np.sqrt(5)
    c = (a ** n - b ** n) / (2 ** n * np.sqrt(5))
    if seq:
        return [fibs(i) for i in range(n + 1)]
    else:
        return round(c)


def generate_fibs_ratios(th=10, n=5, standardized=False):
    """
    This functions is created for generating ratios of fibonacci retracement.

    :param th:
    :param n:
    :return:
    """
    if standardized:
        u = (generate_fibs_ratios(n=4) + 1)[-5:]
        m = generate_fibs_ratios(n=2)[:2]
        d = -(generate_fibs_ratios(n=4) + 1)[-5:]
        return np.sort(np.hstack([u, 0, m, 1, d]))
    else:
        fb = fibs(th * 2, seq=True)
        return np.divide(fb[th - n:th + n], fb[th])


def sigmoid(x, ymin=0, ymax=1, x50L=-1, x50U=1, e=2):
    """
    Map the x into (ymin, ymax), as S-curve, with 50% of the values
    inside (x50L, x50U)

    Default is normal S-curve
    #Reference: https://stats.stackexchange.com/questions/265266/adjusting-s-curves-sigmoid-functions-with-hyperparameters
    """
    a = (x50L + x50U) / e
    b = e / (x50L - x50U)
    c = ymin
    d = ymax - c
    y = c + (d / (1.0 + np.exp(b * (x - a))))
    return y


def Distance(a, b, preprocess=None):
    """
    Simple method to evaluate the distance between a and b
    the results range from 0 to 1, standing for closest to farest

    Property:
    when a and b have different signs, the abs
    """
    if preprocess:
        x, y = map(preprocess, [a, b])
    else:
        x, y = a, b
    return (x - y) / (x + y)


def zscore(x, axis=0):
    return (x - np.mean(x)) / np.std(x)


def true_pct(x, denom_adj=0):
    x = x[~np.isnan(x)]  # remove nan value
    x = list(map(np.float, x))  # convert to float
    return np.sum(x) / (len(x) - denom_adj)


def revreldist(x, y, kind='exp'):
    """
    Reversed Relative Distance between two points.
    x and y should be positive.
    """
    if kind == 'exp':
        return 2 ** (-abs(x - y))
    elif kind == 'abs':
        return (x + y + 10 ** (-20)) / (abs(x - y) + 10 ** (-20))
    else:
        raise ValueError('kind can only be exp or abs.')


def permute3(rang1st, rang2nd, rang3rd):
    # permutation based on specfic values
    return [(p, d, q) for p in rang1st for d in rang2nd for q in rang3rd]


