# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:23:56 2016

@author: Gabriel F.
"""
import pandas as pd
import numpy as np
# from datetime import timedelta


class Factors_class():
    def __init__(self):
        pass

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def calc_cmo(self, arr, win, method='concat'):
        """
        calculate ups and downs for each batches split by defined length.

        Args:
            arr (TYPE): Description
            win (TYPE): defined length for each window
            method (str, optional): for batches integration

        Returns:
            float: cmo value.
        """
        is_last_sufficient = (len(arr) % win) / float(win) > .5
        if method == 'concat':
            if is_last_sufficient:
                batches = np.split(arr[:(len(arr) / win) * win], len(arr) / win) + [arr[:len(arr) % win]]
            else:
                batches = np.split(arr[:(len(arr) / win) * win], len(arr) / win)
            signs = map(lambda x: np.sign(x[-1] - x[0]), batches)
        elif method == 'rolling':
            batches = self.rolling_window(arr, win)
            signs = np.full_like(arr, np.nan)
            signs[win - 1:] = map(lambda x: np.sign(x[-1] - x[0]), batches)

        ud_counts = pd.value_counts(signs)
        values_existed = ud_counts.index
        SoU = ud_counts[1] if 1 in values_existed else 0
        SoD = ud_counts[-1] if -1 in values_existed else 0
        num_changing_tick = SoU + SoD
        rate = (0 if num_changing_tick == 0 else float(SoU - SoD) / num_changing_tick)

        return rate

    def calc_rolling_cmo(self, target_quota, external_win, internal_win):
        """

        Args:
            target_quota (TYPE): regular format from pycoraldb
            external_win (TYPE): window length for rolling original series
            internal_win (TYPE): window length for each cmo stat

        Returns:
            TYPE: Description
        """
        cmos = np.full_like(target_quota['midprice'].values, 0.5)
        values_in_windows = self.rolling_window(target_quota['midprice'].values, external_win)
        cmos[external_win - 1:] = map(lambda x: self.calc_cmo(x, internal_win), values_in_windows)
        return cmos

