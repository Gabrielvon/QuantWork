# -*- coding: utf-8 -*-
"""
@author: Gabriel Feng
"""
def __init__():
    pass

def calc_init_cap(price_in, n, underlying):
    """
    Calculate initial capital of investment with leverage if possible.

    :param price_in:
    :param n:
    :param underlying:
    :return:
    """
    if isinstance(underlying, (int, float)):
        pct = float(underlying)
        underly = 'any'
    else:
        pct = 0  # dummy
        underly = underlying

    calc_initCap = {'stock': n * price_in,
                    # SHFE
                    'rb': n * price_in * 10 * .20,
                    'hc': n * price_in * 10 * .20,
                    'au': n * price_in * 1000 * .20,
                    'al': n * price_in * 5 * .20,
                    'ru': n * price_in * 10 * .20,
                    'cu': n * price_in * 10 * .30,
                    'ni': n * price_in * 1 * .30,
                    'zn': n * price_in * 5 * .30,

                    # CFFEX
                    'if': n * price_in * 300 * .25,
                    'ic': n * price_in * 200 * .40,
                    'ih': n * price_in * 300 * .25,
                    # DCE
                    'c': n * price_in * 10 * .30,
                    'i': n * price_in * 100 * .30,
                    'j': n * price_in * 100 * .30,
                    'jm': n * price_in * 60 * .30,
                    'jd': n * price_in * 5 * .30,
                    'm': n * price_in * 10 * .30,
                    'cs': n * price_in * 10 * .17,
                    'l': n * price_in * 5 * .30,
                    # CZCE
                    'oi': n * price_in * 10 * .17,
                    'rm': n * price_in * 10 * .16,
                    'cf': n * price_in * 5 * .17,
                    # Other
                    'any': n * price_in * pct
                    }
    return calc_initCap[underly.lower()]


def calc_pnl(price_in, price_out, n, underlying):
    """
    Calculate profit and loss with leverage if possible.

    :param price_in:
    :param price_out:
    :param n:
    :param underlying:
    :return:
    """
    if isinstance(underlying, (int, float)):
        pct = float(underlying)
        underly = 'any'
    else:
        pct = 1.0  # dummy
        underly = underlying

    calcPnL = {'stock': n * (price_out - price_in),
               # SHFE
               'rb': n * (price_out - price_in) * 10,
               'hc': n * (price_out - price_in) * 10,
               'au': n * (price_out - price_in) * 1000,
               'al': n * (price_out - price_in) * 5,
               'ru': n * (price_out - price_in) * 10,
               'cu': n * (price_out - price_in) * 10,
               'ni': n * (price_out - price_in) * 1,
               'zn': n * (price_out - price_in) * 5,
               # CFFEX
               'if': n * (price_out - price_in) * 300,
               'ic': n * (price_out - price_in) * 200,
               'ih': n * (price_out - price_in) * 300,
               # DCE
               'c': n * (price_out - price_in) * 10,
               'i': n * (price_out - price_in) * 100,
               'j': n * (price_out - price_in) * 100,
               'jm': n * (price_out - price_in) * 60,
               'jd': n * (price_out - price_in) * 5,
               'm': n * (price_out - price_in) * 10,
               'cs': n * (price_out - price_in) * 10,
               'l': n * (price_out - price_in) * 5,
               # CZCE
               'oi': n * (price_out - price_in) * 10,
               'rm': n * (price_out - price_in) * 10,
               'cf': n * (price_out - price_in) * 5,
               # Other
               'any': n * (price_out - price_in) * pct
               }
    return calcPnL[underly.lower()]


def calc_transation_cost(price_in, price_out, n, underlying):
    """
    Calculate transation cost considering margin, leverage and different underlying.

    :param price_in:
    :param price_out:
    :param n:
    :param underlying:
    :return:
    """
    if isinstance(underlying, (int, float)):
        pct = float(underlying)
        underly = 'any'
    else:
        pct = 0  # dummy
        underly = underlying

    calcTCost = {'stock': n * (price_in * .0002 + price_out * .0012),
                 # SHFE
                 'rb': n * (price_in + price_out) * 10 * .0001,
                 'hc': n * (price_in + price_out) * 10 * .0001,
                 'au': n * 10.0 * 2,
                 'al': n * 3.0 * 2,
                 'ru': n * (price_in + price_out) * 10 * 0.000045,
                 'cu': n * (price_in + price_out) * 10 * 0.000050,
                 'ni': n * 6.0 * 2,
                 'zn': n * 3.0 * 2,
                 # CFFEX
                 'if': n * (price_in + price_out) * 300 * .000023,
                 'ic': n * (price_in + price_out) * 200 * .000023,
                 'ih': n * (price_in + price_out) * 300 * .000023,
                 # DCE
                 'c': n * 1.2 * 2,
                 'i': n * .60 * 2,
                 'j': n * .60 * 2,
                 'jm': n * .60 * 2,
                 'jd': n * (price_in + price_out) * 10 * .000150,
                 'm': n * 10.0 * 2,
                 'cs': n * 3.0 * 2,
                 'l': n * 2.0 * 2,
                 # CZCE
                 'oi': n * 2.0 * 2,
                 'rm': n * 2.5 * 2,
                 'cf': n * 6.0 * 2,
                 # Others
                 'any': n * (price_in + price_out) * pct
                 }
    return calcTCost[underly.lower()]


def market_at_day(exchange):
    exc = exchange.lower()
    ts_exc0 = ['sh', 'sz', 'cffex', 'stock']
    ts_exc1 = ['shfe', 'czce', 'dce', 'future']

    if exc in ts_exc0:
        ts = [['09:30:00', '11:30:00'],
              ['13:00:00', '15:00:00']]

    elif exc in ts_exc1:
        ts = [['09:00:00', '10:15:00'],
              ['10:30:00', '11:30:00'],
              ['13:30:00', '15:00:00']]
    else:
        raise ValueError('{} is not included'.format(exchange.lower()))
    return ts


def market_at_night(underlying):
    und = underlying.lower()
    ts_und0 = ['rm', 'oi', 'ta', 'cf', 'fg', 'sr', 'zc',
               'MA', 'a', 'b', 'm', 'y', 'j', 'i', 'jm', 'p']
    ts_und1 = ['cu', 'al', 'ni', 'sn', 'zn', 'pb']
    ts_und2 = ['hc', 'bu', 'rb', 'ru']
    ts_und3 = ['au', 'ag']
    if und in ts_und0:
        ts = [['21:00:00', '23:30:00']]
    elif und in ts_und1:
        ts = [['21:00:00', '01:00:00']]
    elif und in ts_und2:
        ts = [['21:00:00', '23:00:00']]
    elif und in ts_und3:
        ts = [['21:00:00', '02:30:00']]
    else:
        raise ValueError('{} is not included'.format(underlying))
    return ts
