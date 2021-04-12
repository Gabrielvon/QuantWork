
import gabFunc as gfc
import numpy as np


def calc_return(pnl, investment):
    # Cash yield
    return np.divide(float(pnl), investment)


def calc_txn_yield(rtn):
    # Trades yield
    return np.mean(rtn)


def calc_return_from_price(prices, flag='pchg'):
    if flag == 'pchg':
        rtn = np.diff(prices) / prices[:-1]
    elif flag == 'log':
        rtn = np.log(np.diff(prices))
    else:
        raise ValueError('flag {} is not defined.'.format(flag))

    return rtn


def calc_period_return(rtn, window, forward=True):
    period_return = rtn.rolling(window).sum()
    if forward:
        return period_return
    return period_return.shift(-window)


def calc_side_return(rtn, th, upside=True):
    if upside:
        return np.mean(rtn[rtn > th])
    return np.mean(rtn[rtn < th])


def drawdown(arr):
    """Drawdown

    input must be position integer or float

    Arguments:
        arr {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    rolmax = gfc.rolling_extend(np.nanmax, arr, forward=True)
    return abs(1 - arr / rolmax)


def max_drawdown(arr):
    """Max Drawdown

    input must be position integer or float

    Arguments:
        arr {[type]} -- [description]
    """
    i = np.argmax((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr))  # end of the period
    j = np.argmax(arr[:i])  # start of period
    return (1 - arr[i] / arr[j])


def calc_ic(x, y, correlated=True):
    x_direc = gfc.binarize(np.diff(x), flag=2)
    y_direc = gfc.binarize(np.diff(y), flag=2)
    ic = 2.0 * np.sum(x_direc == y_direc) / len(x) - 1
    if correlated:
        ic = ic / np.sqrt(2 / (1 + np.corrcoef(x, y)))
    return ic[0, 1]


def sharpe(r, benchmark):
    """Sharpe ratio

    [description]

    Arguments:
        r {[type]} -- [description]
        benchmark {[type]} -- [description]
    """
    return (r - benchmark) / np.std(r)


def calmar(r, md):
    return r / max_drawdown


def sortino(r, mar=None):
    """Sortino ratio

    [description]

    Arguments:
        r {[type]} -- [description]
        mar_th {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    downsides = r[r < mar]
    if len(downsides) > 1:
        out = (np.sum(r) - mar) / np.std(downsides)
    else:
        out = np.inf
    return out


def calc_ir(r, benchmark):
    # needed double-check
    resid = r - benchmark
    ir = np.mean(resid) / np.std(resid)
    return ir
