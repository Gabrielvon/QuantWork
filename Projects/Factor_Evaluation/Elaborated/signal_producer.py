import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:23:56 2016

@author: Gabriel F.
"""

class signal_producer():
    def __init__(self):
        pass

    def crude_signal(self, input_df):
        return input_df

    def calc_midprice(self, input_df):
        df = input_df.copy()
        idx_ap1_eq_0 = df['ap1'].values == 0
        idx_bp1_eq_0 = df['bp1'].values == 0
        df['ap1'].values[idx_ap1_eq_0] = df['bp1'].values[idx_ap1_eq_0]
        df['bp1'].values[idx_bp1_eq_0] = df['ap1'].values[idx_bp1_eq_0]
        return df[['ap1', 'bp1']].mean(1)

    def _drop_limit(self, df, len_before, len_after):
        from itertools import chain
        df = df.reset_index()
        remove_colns = [cn for cn in df.columns if cn not in ['code', 'ts']]
        idx_orig = pd.Series(df.index, index=df['ts'].values)
        abn_ts = df['ts'][df['status'] != 0]
        if len(abn_ts) > 0:
            n_b = np.timedelta64(len_before, 's')
            n_a = np.timedelta64(len_after, 's')
            temp = [idx_orig[b:a] for b, a in zip(abn_ts - n_b, abn_ts + n_a)]
            full_idx = np.unique(list(chain.from_iterable(temp)))
            valid_idx = full_idx[full_idx <= df.index[-1]]
            df.loc[valid_idx, remove_colns] = np.nan
        return df

    def drop_limit_by_return(self, df, len_before, len_after):
        def __groupapply(x):
            xgrp = x.groupby(pd.Grouper(level='ts', freq='1d'))
            return xgrp.apply(lambda x: self._drop_limit(x, len_before, c))

        df = df.set_index(['code', 'ts'])
        for c in len_after:
            temp_df = df[['status', c]]
            grp = temp_df.groupby('code')
            res = grp.apply(__groupapply).set_index(['code', 'ts'])[c]
            df.loc[:, c] = res
        return df.reset_index()

    def drop_open_close(self, data_df, len_open, len_close):
        assert np.datetime64(data_df['ts'].values[0], 'D') == np.datetime64(data_df['ts'].values[-1], 'D')
        n_open = np.timedelta64(9 * 3600 + 30 * 60, 's') + np.timedelta64(len_open, 's') + \
            np.datetime64(data_df['ts'].values[0], 'D')
        n_close = np.timedelta64(15 * 3600, 's') - np.timedelta64(len_close, 's') + \
            np.datetime64(data_df['ts'].values[0], 'D')
        idx = np.logical_and(data_df['ts'].values[:] > n_open, data_df['ts'].values[:] < n_close)
        return data_df.iloc[idx, :]

    def add_cap(self, data_df, upper_limit, lower_limit):
        factor_arr = data_df['factor'].values
        factor_arr[factor_arr[:] > upper_limit] = upper_limit
        factor_arr[factor_arr[:] < lower_limit] = lower_limit
        return factor_arr

    def normalize(self, data_df, negative1_to_1=True):
        factor_arr = data_df['factor'].values
        mx = factor_arr.max()
        mn = factor_arr.min()
        factor_arr = (factor_arr - mn) / (mx - mn)
        if negative1_to_1:
            factor_arr = 2 * factor_arr - 1
        return factor_arr

    def normalize_with_cap(self, data_df, n_cap):
        factor_arr = data_df['factor'].values
        factor_arr[factor_arr[:] > n_cap] = n_cap
        factor_arr[factor_arr[:] < -n_cap] = -n_cap
        factor_arr = factor_arr / n_cap
        return factor_arr

    def get_cyc_return(self, cycle_tlist, input_df, unit='s'):
        df = pd.Series(input_df['midprice'].values, index=input_df['ts'].values)

        def get_return(df, cyc_n):
            w = str(cyc_n) + unit
            rtn = df.rolling(w, closed='both').apply(lambda x: x[-1] / x[0] - 1)

            n_a = np.timedelta64(cyc_n, 's')
            invalid_n = df[:rtn.index[0] + n_a].shape[0]
            rtn_shifted = rtn.shift(-invalid_n)
            rtn_shifted.name = cyc_n
            return rtn_shifted

        return pd.concat([get_return(df, cyc_n) for cyc_n in cycle_tlist], 1)

    def get_cyc_return2(self, cycle_tlist, input_df):
        """Summary
        if rolling by tick instead of time period, this method would be 10+ times
        faster comparing to the original one

        Args:
            cycle_tlist (TYPE): Description
            input_df (TYPE): Description

        Returns:
            TYPE: Description
        """
        midpri_arr = input_df['midprice'].values
        cyc_ret_list = [tuple(midpri_arr[cyc:] / midpri_arr[:-cyc] - 1) + (np.nan,) * cyc for cyc in cycle_tlist]
        cyc_ret_arr = np.vstack(cyc_ret_list)
        return pd.DataFrame(cyc_ret_arr, columns=cycle_tlist, index=input_df.index)
