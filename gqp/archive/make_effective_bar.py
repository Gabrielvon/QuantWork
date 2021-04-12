import pandas as pd
import numpy as np


def make_volume_weighted_market_data(df, batch_volume=1e6, verbose=False):
    """Summary

    df must have columns: ['timestamp', 'new_volume', 'ap1', 'av1', 'new_price']

    Args:
        df (TYPE): Description
        batch_volume (float, optional): Description
        verbose (bool, optional): Description

    Returns:
        TYPE: Description
    """
    pre_num_batch = 0
    refresh = True
    results = []
    for _, row in df.iterrows():
        ts, new_volume = row[['timestamp', 'new_volume']].values
        if refresh:
            sum_volume = 0
            refresh = False
            rows = []
        sum_volume += new_volume
        num_batch = np.floor(sum_volume / batch_volume)
        rows.append(row)
        if num_batch > pre_num_batch:
            if verbose:
                print(num_batch, pre_num_batch, len(rows), sum_volume)
            rs = pd.concat(rows, axis=1).T
            ap1 = np.sum(rs['ap1'] * rs['av1']) / np.sum(rs['av1'])
            bp1 = np.sum(rs['bp1'] * rs['av1']) / np.sum(rs['av1'])
            new_price = np.sum(rs['new_price'] *
                               rs['new_volume']) / np.sum(rs['new_volume'])
            results.append(row.append(
                pd.Series({'vw_ap1': ap1, 'vw_bp1': bp1, 'vw_new_price': new_price})))
            pre_num_batch = num_batch
            refresh = True
    return pd.concat(results, axis=1).T


def meta_label(arr, th_upper, th_lower):
    """Summary

    Args:
        arr (TYPE): asset return array
        th_upper (TYPE): upper barrier
        th_lower (TYPE): lower barrier

    Returns:
        TYPE: Description
    """
    # th_upper_ir, th_lower_ir = thresholds
    log_ret = np.diff(np.log(arr))
    for v in log_ret:
        if v > th_upper:
            return 1

        if v < th_lower:
            return -1

    if np.all(np.isnan(log_ret)):
        return np.nan
    else:
        return -1
