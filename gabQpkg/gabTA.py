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


# def calc_active_volume(ask, bid, volume, amount, multiplier=1):
#     """
#     Estimate the active trading volume.

#     :param ask: ask price
#     :param bid: bid price
#     :param volume: trading volume
#     :param amount: trading amount
#     :param multiplier: in case of leverage
#     :return:
#     """
#     vol = volume.replace(0, np.nan)
#     avgcost = np.divide(amount, vol)
#     avgcost = np.divide(avgcost, multiplier)
#     spread = ask - bid

#     # Seperate
#     avgc_over_ask = avgcost > ask.shift(1)
#     avgc_below_bid = avgcost < bid.shift(1)
#     avgc_between = (~avgc_over_ask) & (~avgc_below_bid)

#     # calculate active buyer
#     actibuy = vol * avgc_over_ask
#     ab_ratios = np.divide((avgcost - bid.shift(1)), spread.shift(1))
#     actibuy[avgc_between] = vol[avgc_between] * ab_ratios[avgc_between]

#     # calculate active seller
#     actisell = vol * avgc_below_bid
#     as_ratios = 1 - ab_ratios
#     actisell[avgc_between] = vol[avgc_between] * as_ratios[avgc_between]

#     df = pd.concat([actibuy, actisell], 1).fillna(0)
#     df.columns = ['activebuyer', 'activeseller']
#     return df

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


# def calc_active_volume(ask, bid, volume, amount, multiplier=1, suppress=True):
#     """
#     Estimate the active trading volume.
#     ask, bid volume and amount must be float.

#     :param ask: ask price
#     :param bid: bid price
#     :param volume: trading volume
#     :param amount: trading amount
#     :param multiplier: in case of leverage
#     :return:
#     """
#     if suppress:
#         warnings.filterwarnings(
#             "ignore", message="invalid value encountered in greater")
#         warnings.filterwarnings(
#             "ignore", message="invalid value encountered in less")

#     vol = volume.copy()
#     np.place(vol, vol == 0, np.nan)
#     avgcost = np.divide(amount, vol)
#     avgcost = np.divide(avgcost, multiplier)
#     spread = ask - bid

#     # Seperate
#     avgc_over_ask = avgcost[1:] > ask[:-1]
#     avgc_below_bid = avgcost[1:] < bid[:-1]
#     avgc_between = (~avgc_over_ask) & (~avgc_below_bid)
#     vol_1 = vol[1:]

#     # calculate active buyer
#     actibuy = vol_1 * avgc_over_ask
#     ab_ratios = np.divide((avgcost[1:] - bid[:-1]), spread[:-1])
#     actibuy[avgc_between] = vol_1[avgc_between] * ab_ratios[avgc_between]

#     # calculate active seller
#     actisell = vol_1 * avgc_below_bid
#     as_ratios = 1 - ab_ratios
#     actisell[avgc_between] = vol_1[avgc_between] * as_ratios[avgc_between]
#     active_volume = np.vstack([actibuy, actisell]).T.astype('float64')

#     if suppress:
#         warnings.filterwarnings('default')

#     return np.nan_to_num(active_volume)


# def calc_absr(active_long, active_short, window):
#     acc_avol = np.sum(rolling_window(activolume.T, duration), axis=2).T
#     return np.log(np.divide(acc_avol[:, 0], acc_avol[:, 1]))


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


def calc_weight(askbid, curpri, th):
    # calculate weight about ask/bid distance from prices
    return np.exp(-th * abs(1 - askbid / curpri))


def calc_wb(prices, ask_p, bid_p, ask_v, bid_v, th):
    prices = prices.reshape(-1, 1)
    # calculate volume based on weights
    wbuyamt = calc_weight(bid_p, prices, th) * bid_v
    wsellamt = calc_weight(ask_p, prices, th) * ask_v

    wbuy_amt_sum = wbuyamt.sum(1)  # order long
    wsell_amt_sum = wsellamt.sum(1)  # order short
    wb = gfc.Distance(wbuy_amt_sum, wsell_amt_sum)

    return wb


def calc_wb2(prices, ask_p, bid_p, ask_v, bid_v, new_bid_v, new_ask_v, th):
    prices = prices.reshape(-1, 1)
    # calculate volume based on weights
    wbuyamt = calc_weight(bid_p, bid_v, th) * bid_v  # order long
    wsellamt = calc_weight(ask_p, ask_v, th) * ask_v  # order short

    wbuy_amt_sum = new_bid_v + wbuyamt.sum(1)
    wsell_amt_sum = new_ask_v + wsellamt.sum(1)
    wb = gstat.Distance(wbuy_amt_sum, wsell_amt_sum)
    return wb


def smooth_variate(order_p, order_v, direc, camp):
    order_v.columns = order_p.columns
    ts_pv = pd.concat([order_p.stack(), order_v.stack()], 1, keys=['p', 'v']).reset_index().drop('level_1', 1)
    volm = ts_pv.set_index(['timestamp', 'p'])['v'].unstack('p')
    arr = volm.values.copy()
    arr = np.nan_to_num(arr)
    flatten_n = arr.shape[1] * np.arange(arr.shape[0])

    if camp == 'ask':
        idx_end = volm.notnull().values.cumsum(1).argmax(1)
        idx_flatten = [np.arange(e, arr.shape[1]) + d for e, d in zip(idx_end, flatten_n)]
    elif camp == 'bid':
        idx_begin = volm.notnull().values.argmax(1)
        idx_flatten = [np.arange(0, b) + d for b, d in zip(idx_begin, flatten_n)]

    np.put(arr, np.hstack(idx_flatten), np.nan)
    volm = pd.DataFrame(arr, index=volm.index, columns=volm.columns)

    if direc == 'inc':
        volm_vary = volm.diff().fillna(0)
        volm_vary[volm_vary < 0] = 0
    elif direc == 'dec':
        volm_vary = -volm.diff().fillna(0)
        volm_vary[volm_vary < 0] = 0

    volm_vary_rollm = volm_vary.rolling(10).mean()

    return volm_vary_rollm


def calc_sizeIncre(prices, ask_p, bid_p, ask_v, bid_v, th):
    prices_arr = prices.values.reshape(len(prices), 1)

    bv_inc = smooth_variate(bid_p, bid_v, direc='inc', camp='bid')
    bp_expand = np.repeat(bv_inc.columns.values.reshape(1, bv_inc.shape[1]), bv_inc.shape[0], 0)
    wbuyamt = calc_weight(bp_expand, prices_arr, th) * bv_inc

    av_inc = smooth_variate(ask_p, ask_v, direc='inc', camp='ask')
    ap_expand = np.repeat(av_inc.columns.values.reshape(1, av_inc.shape[1]), av_inc.shape[0], 0)
    wsellamt = calc_weight(ap_expand, prices_arr, th) * av_inc

    wbuy_amt_sum = wbuyamt.sum(1)  # order long
    wsell_amt_sum = wsellamt.sum(1)  # order short
    sizeInc = gstat.Distance(wbuy_amt_sum, wsell_amt_sum)
    return sizeInc


def calc_sizeDecre(prices, ask_p, bid_p, ask_v, bid_v, th):
    prices_arr = prices.values.reshape(len(prices), 1)

    bv_dec = smooth_variate(bid_p, bid_v, direc='dec', camp='bid')
    bp_expand = np.repeat(bv_dec.columns.values.reshape(1, bv_dec.shape[1]), bv_dec.shape[0], 0)
    wbuyamt = calc_weight(bp_expand, prices_arr, th) * bv_dec

    av_dec = smooth_variate(ask_p, ask_v, direc='dec', camp='ask')
    ap_expand = np.repeat(av_dec.columns.values.reshape(1, av_dec.shape[1]), av_dec.shape[0], 0)
    wsellamt = calc_weight(ap_expand, prices_arr, th) * av_dec

    wbuy_amt_sum = wbuyamt.sum(1)  # order long
    wsell_amt_sum = wsellamt.sum(1)  # order short
    sizeDec = gstat.Distance(wbuy_amt_sum, wsell_amt_sum)
    return sizeDec


def sectional_avg_knock_amount(new_amount, new_knock_count, window):
    return new_amount
