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


def calc_absr(ask, bid, volume, amount, multiplier, window, suppress=True):
    """
    Estimate the active trading volume (absolute buy sell ratio).
    ask, bid volume and amount must be float.

    :param ask: ask price
    :param bid: bid price
    :param volume: trading volume
    :param amount: trading amount
    :param multiplier: in case of leverage
    :return:
    """

    if suppress:
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in greater")
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in less")

    vol = volume.copy().astype(float)
    np.place(vol, vol == 0, np.nan)
    avgcost = np.divide(amount, vol)
    avgcost = np.divide(avgcost, multiplier)
    spread = ask - bid

    # Seperate
    avgc_over_ask = avgcost[1:] > ask[:-1]
    avgc_below_bid = avgcost[1:] < bid[:-1]
    avgc_between = (~avgc_over_ask) & (~avgc_below_bid)
    vol_1 = vol[1:]

    # calculate active buyer
    actibuy = vol_1 * avgc_over_ask
    ab_ratios = np.divide((avgcost[1:] - bid[:-1]), spread[:-1])
    actibuy[avgc_between] = vol_1[avgc_between] * ab_ratios[avgc_between]

    # calculate active seller
    actisell = vol_1 * avgc_below_bid
    as_ratios = 1 - ab_ratios
    actisell[avgc_between] = vol_1[avgc_between] * as_ratios[avgc_between]
    active_volume = np.vstack([actibuy, actisell]).T.astype('float64')
    active_volume = np.nan_to_num(active_volume)
    active_volume = np.sum(gfc.rolling_window(
        active_volume.T, window), axis=2).T
    out = np.log(np.divide(active_volume[:, 0], active_volume[:, 1]))

    if suppress:
        warnings.filterwarnings('default')

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


def calc_open_interest_intensity(volume, open_interest, window):
    """Summary
    Calculate intensity level based on open interest.

    Args:
        volume (pandas.series): trading volume
        open_interest (pandas.series): accumulated open interest

    Returns:
        TYPE: intensity level
    """
    open_interest_diff = open_interest.diff()
    increase = (volume + open_interest_diff) / 2
    decrease = (volume - open_interest_diff) / 2
    acc_incre = increase.rolling(window).sum()
    acc_decre = decrease.rolling(window).sum()
    intensity = gfc.sigmoid(np.log(np.divide(acc_incre, acc_decre)))
    return intensity
