# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-13 18:02:31
# @Last Modified by:   Gabriel
# @Last Modified time: 2020-10-30 18:23:08

###################
# # Transformer # #
###################

import numpy as np
import pandas as pd
import datetime

def __init__():
    pass


def convert_return_type(arr, flag=1):
    """
    flag
    1: from log return to percentage return
    2: from percentage return to log return
    """
    if flag == 1:
        rs = np.exp(arr) - 1
    elif flag == 2:
        rs = np.log(arr + 1)
    return rs


def calc_ew_alpha(**args):
    if 'com' in args:
        alpha = 1 / (1 + args['com'])
    elif 'span' in args:
        alpha = 2 / (args['span'] + 1.0)
    elif 'halflife' in args:
        alpha = 1 - np.exp(np.log(0.5) / args['halflife'])
    return alpha


def multi_reindex(df):
    indices = df.index.to_frame(index=False)
    indices_merged = indices.apply(
        lambda x: '|'.join(x.astype('str').tolist()), axis=1)
    try:
        return pd.DataFrame(df.values, index=indices_merged, columns=df.columns)
    except AttributeError:
        return pd.Series(df.values, index=indices_merged, name=df.name)



def dateint_datestr(d, reverse=False):
    if not reverse:
        out = datetime.datetime.strptime(str(d), '%Y%m%d').strftime('%Y-%m-%d')
    else:
        out = int(pd.to_datetime(d).strftime('%Y%m%d'))
    return out


def fuzzy_trading_date(date, ref_trading_dates, flag=0):
    """Summary
    
    Find the closest trading dates to input date.

    Args:
        date (TYPE): Description
        ref_trading_dates (TYPE): Description
        flag (int):
            0, closest for both sides;
            1, closest on left;
            2, closest on right;
    
    Returns:
        TYPE: Description
    """
    if flag == 0:
        out = ref_trading_dates[np.argmin(np.abs(ref_trading_dates - date))]
    elif flag == 1:
        out = ref_trading_dates[ref_trading_dates <= date].max()
    elif flag == 2:
        out = ref_trading_dates[ref_trading_dates >= date].min()

    return out


def convet_log_simple_return(R, flag=0):
    """

    :param R:
    :param flag: 0, convert log-return to percentage-return; 1, reverse
    :return:
    """
    if flag == 0:
        out = np.exp(R) - 1
    elif flag == 1:
        out = np.log(1 + R)
    else:
        raise ValueError('unknown flag ', flag)

    return out