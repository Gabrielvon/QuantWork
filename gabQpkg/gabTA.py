import numpy as np
import pandas as pd
import warnings
import gabFunc as gfc
import gabperformance as gperf
import gabStat as gstat


def momentum(c, win, method='pct'):
    """Momentum

    [description]

    Arguments:
        c {[type]} -- prices
        win {[type]} -- time window
    """
    arr = np.array(c)
    if method == 'pct':
        mom = arr[win:] / arr[:-win] - 1

    if method == 'log':
        mom = np.log(arr[win:] / arr[:-win])

    return mom


def momentum2(c, win):
    return [p / c[i:win + i].mean() - 1 for i, p in enumerate(c[win:])]


def slope(c, win):
    import scipy.stats as scs
    steps = range(win)
    return np.array([scs.linregress(steps, c[i:win + i]).slope for i in range(c.shape[0] - win)])


def trades_ratio(traded_v, circulating_v):
    """[summary]

    [description]

    Arguments:
        traded_v {[type]} -- trading volume
        circulating_v {[type]} -- circulating volume

    Returns:
        [type] -- [description]
    """
    return np.cumsum(traded_v) / circulating_v


def trades_ratio_change(tr):
    return np.diff(tr)


def variance_change(var):
    return np.diff(var)


def fluctuation(c):
    return (max(c) - min(c)) / (c[0] + c[-1])


def calc_illiq(c, trade_amt, win):
    """Illiquidity

    https://uqer.io/community/share/57c6730b228e5b6d227c7314

    Arguments:
        c {[type]} -- prices
        trade_amt {[type]} -- trading amount
        win {[type]} -- window

    Returns:
        [type] -- [description]
    """
    assert len(c) == len(trade_amt)
    v = np.array(trade_amt)
    amp = momentum(c, win)
    illiq = np.divide(amp, np.log(v[win:]))
    # illiq = gfc.rolling_apply(np.sum, illiq_i, win) / win
    return illiq


def calc_smart_money(c, trade_v, win):
    """Smart money

    # https://uqer.io/community/share/578f04e0228e5b3b9b5f1ab7
    Maybe dig into the reason using sqrt(trade_v)

    Arguments:
        c {[type]} -- price
        trade_v {[type]} -- trading volume
        win {[type]} -- window
    """

    assert len(c) == len(trade_v)
    v = np.array(trade_v)
    amp = momentum(c, win)
    S = amp / np.log(v[win:])
    return S


def calc_fibs_retra(h, l, fb_ratios):
    """
    Calculate fibonacci retracement based on given bounds.

    :param h: upper band
    :param l: lower band
    :param fb_ratios: user-defined ratios
    :return: series
    """
    out = dict()
    fbs_rev = 1 - fb_ratios[fb_ratios > 1]
    fbs = np.sort(np.hstack([fb_ratios, fbs_rev, 0]))
    for r in fbs:
        out[np.round(r, 2)] = l + (h - l) * r
    return pd.Series(out)


def calc_bavr(new_v, new_ask_v, new_bid_v, win):
    """Summary
    Calculate active buy ratio

    Args:
        new_v (TYPE): traded volume
        new_ask_v (TYPE): traded ask volume (active buy)
        new_bid_v (TYPE): traded bid volume (active sell)
    """
    new_v_mu = momentum2(new_v, win)
    return (new_ask_v[win:] - new_bid_v[win:]) / new_v_mu


def loc_within_retra(x, series):
    """
    To get the location of one value among different regimes.

    :param x: one value that can compare to series.
    :param series: one series comprising of values.
    :return: tuple of range.
    """
    vals = np.array(series)
    if isinstance(series, pd.core.series.Series):
        idx = series.index
    else:
        idx = np.arange(len(vals))
    i, v = 0, vals[0]
    while x > v:
        i += 1
        if i >= len(vals):
            return idx[i - 1], np.inf
        else:
            v = vals[i]
    else:
        if i == 0:
            return -np.inf, idx[i]

        return idx[i - 1], idx[i]


def calc_keltner(hi, lo, cl, ema_win, atr_win, amps=[1]):
    # ema = numpy_ewm_vectorized(cl, ema_win)
    ema = numpy_ewm_alpha(cl, 0.05, ema_win)
    atr = calc_atr(hi, lo, cl, atr_win)
    chann = [{'upper_%d' % a: ema + a * atr, 'lower_%d' %
              a: ema - a * atr} for a in amps]
    channel = {k: v for ch in chann for k, v in ch.items()}
    channel.update({'middle': cl})
    return channel


def calc_atr(hi, lo, cl, window):
    """
    Calculate ATR with fixed alpha of 0.05

    :param hi:
    :param lo:
    :param cl:
    :param n:
    :return:
    """
    if not gfc.checkEqualIvo(list(map(len, [hi, lo, cl]))):
        raise ValueError('hi, lo and cl should have identical size.')

    true_range_list = np.zeros_like(hi)
    true_range_list[0] = np.nan
    for i in range(len(hi) - 1):
        max_hi_cl = max(hi[i + 1], cl[i])
        min_lo_cl = min(lo[i + 1], cl[i])
        true_range_list[i + 1] = max_hi_cl - min_lo_cl
    atr = numpy_ewm_alpha(true_range_list, 0.05, window)
    return atr


def numpy_ewm_vectorized(data, com=None, span=None, halflife=None, alpha=None):
    """Summary
    Calculate ema with automatically-generated alpha. Weight of past effect
    decreases as the length of window increasing.

    Args:
        data (TYPE): Description
        com (float, optional): Specify decay in terms of center of mass, alpha=1/(1+com), for com>=0
        span (float, optional): Specify decay in terms of span, alpha=2/(span+1), for span>=1
        halflife (float, optional): Specify decay in terms of half-life, alpha=1-exp(log(0.5)/halflife), for halflife>0
        alpha (float, optional): Specify smoothing factor alpha directly, 0<alpha<=1

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    n_input = map(bool, [com, span, halflife, alpha])
    if n_input != 1:
        raise ValueError('com, span, halflife, and alpha are mutually exclusive')

    if com:
        alpha = 1 / (1 + com)
    elif span:
        alpha = 2 / (span + 1.0)
    elif halflife:
        alpha = 1 - np.exp(np.log(0.5) / halflife)
    else:
        pass

    alpha_rev = 1 - alpha
    nrow = data.shape[0]

    pows = alpha_rev**(np.arange(nrow + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev**(nrow - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def numpy_ewm_alpha(data, alpha, window):
    """
    Calculate ema with fixed alpha. Weight of past effect keep constant
    no matter how the length of window change

    Note: the weights within the window are still varied by distance.
    References:
    https://stats.stackexchange.com/questions/5290/moving-return-of-exponential-moving-average-choice-of-alpha

    :param data: 1d numpy array
    :param alpha: weight of past effect
    :param window:
    :return:
    """
    wghts = (1 - alpha)**np.arange(window)
    wghts /= wghts.sum()
    out = np.full(data.shape, np.nan)
    out[window - 1:] = np.convolve(data, wghts, 'valid')
    # out = out[:data.size]
    return out


def calc_cmo(price, win='forward'):
    """Summary
    Calculcate Chande Momentum Oscillator in vectorized way

    Args:
        price (TYPE): Description
        win (None, int): Description

    Returns:
        numpy.array:

    Raises:
        ValueError: Description
    """
    def cmo(ud):
        SoU = sum(ud > 0)
        SoD = sum(ud < 0)
        if SoU + SoD == 0:
            return 0
        return 100 * float(SoU - SoD) / float(SoU + SoD)

    pdiff = np.diff(price)

    if isinstance(win, int):
        res = gfc.rolling_apply(cmo, pdiff, win)
    elif win.lower() == 'forward':
        res = gfc.rolling_extend(cmo, pdiff)
    else:
        res = cmo(pdiff)

    return res


def calc_sortino(price, mar_th, win=None, forward=None):
    """Summary
    Calculate Sortino Ratio in vectorized way

    Args:
        price (TYPE): Description
        mar_th (float): threshold of minimum accepted return
        win (int, optional): Description
        forward (bool, optional): Description

    Returns:
        numpy.array: Description

    Raises:
        ValueError: Description
    """
    def __sortino(x):
        return gperf.sortino(x, mar_th)

    rtns = np.divide(price[1:], price[:-1]) - 1
    if mar_th == 'avg':
        mar_th = np.mean(rtns)
    if bool(win) != bool(forward):
        if win:
            res = gfc.rolling_apply(__sortino, rtns, win)
            res[np.isinf(res)] = np.nan
            res = gfc.pandas_fill(res, method='ffill')
        elif forward:
            res = gfc.rolling_extend(__sortino, rtns)
    elif not any([None, None]):
        res = gperf.sortino(rtns, mar_th)
    else:
        raise ValueError('Can only assign value to either win or forward.')

    return res


def calc_weight(askbid, curpri, th):
    # calculate weight about ask/bid distance from prices
    return np.exp(-th * abs(1 - askbid / curpri))



