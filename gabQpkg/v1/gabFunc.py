import pandas as pd
import numpy as np
import scipy.stats as scs
from settings import market_at_day, market_at_night


class timegrouper():

    def __init__(self, agg=None):
        self.__agg_method(agg)

    def __agg_method(self, agg=None):
        if agg:
            self.agg = agg
        else:
            self.agg = {
                np.sum: ['new_volume', 'new_amount', 'open_interest',
                         'new_bid_volume', 'new_bid_amount', 'new_ask_volume', 'new_ask_amount',
                         'bid_order_volume', 'bid_order_amount', 'bid_cancel_volume', 'bid_cancel_amount',
                         'ask_order_volume', 'ask_order_amount', 'ask_cancel_volume', 'ask_cancel_amount',
                         'new_knock_count', 'v'],
                np.max: ['high'],
                np.min: ['low'],
                self.__get_first: ['open'],
                self.__get_last: ['timestamp', 'dtype', 'date', 'stamp', 'code', 'name', 'market', 'type', 'status', 'new_price', 'sum_volume', 'sum_amount',
                                  'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8', 'bp9', 'bp10',
                                  'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'bv6', 'bv7', 'bv8', 'bv9', 'bv10',
                                  'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'ap9', 'ap10',
                                  'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9', 'av10',
                                  'upper_limit', 'lower_limit', 'pre_close', 'close', 'pre_price', 'new_bs_flag',
                                  'sum_bid_volume', 'sum_bid_amount', 'sum_ask_volume', 'sum_ask_amount',
                                  'sum_knock_count', 'volume_multiple', 'price_tick', 'create_date', 'list_date',
                                  'expire_date', 'start_settle_date', 'end_settle_date', 'exercise_date',
                                  'exercise_price', 'cp_flag', 'underlying_code', 'underlying_type',
                                  'weighted_avg_bid_price', 'weighted_avg_ask_price', 'ytm', 'action_date',
                                  'c', 'ask', 'bid']
            }

    def __get_first(self, x):
        if isinstance(x, np.ndarray):
            return x[0]
        return x.iloc[0]

    def __get_last(self, x):
        if isinstance(x, np.ndarray):
            return x[-1]
        return x.iloc[-1]

    def __calc_within_groups(self, grp, coln):
        for f, cn in self.agg.iteritems():
            cns = [c for c in coln if c in cn]
            res = (f(df[cns]).to_frame(dt) for dt, df in grp if len(df) > 0)
            yield pd.concat(res, 1)

    def __calc_rolling(self, data, window):
        for f, cn in self.agg.iteritems():
            for c in data.columns:
                if c in cn:
                    res = data[c].rolling(window).apply(f)
                    yield res

    def __calc_group(self, df):
        for f, cn in self.agg.iteritems():
            cns = [c for c in df.columns if c in cn]
            if len(cns) > 0:
                res = f(df[cns])
                yield res

    def reformat(self, data, freq=None):
        if freq is None:
            res = pd.concat(self.__calc_group(data))
            res.name = data.iloc[-1, :].name
        else:
            coln = data.columns
            grp = data.groupby(pd.Grouper(
                freq=freq, label='right', closed='right'))
            new_data = pd.concat(self.__calc_within_groups(grp, coln), 0).T
            res = new_data.infer_objects()
        return res

    def rolling_reformat(self, data, window):
        res = pd.concat(self.__calc_rolling(data, window), 1)
        if isinstance(window, str):
            valid_start = data.iloc[0].name + pd.to_timedelta(window)
        else:
            valid_start = window
        return res[valid_start:]


class clean_rdata():

    """Summary

    Clean-up functionality:
        * convert index format into DatetimeIndex
        * drop duplicated column of 'timestamp' like 'timestamp.1'
        * rename specfic columns
        * remove data pushed at weird datetime
        * fix date starting at weird datetime in dates_odds

    Attributes:
        clean_df (TYPE): Description
        exchange (TYPE): Description
        raw_df (TYPE): Description
        underlying (TYPE): Description
    """

    def __init__(self, df):
        self.raw_df = df
        self.clean_df = df

    def remove_by_time(self, exchange=True, underlying=False):
        self.exchange = exchange
        self.underlying = underlying
        time_slices = []
        if self.exchange:
            time_slices.extend(market_at_day(self.exchange))

        if self.underlying:
            time_slices.extend(market_at_night(self.underlying))

        slice_dfs = select_slice_time(self.clean_df, time_slices)
        self.clean_df = pd.concat(slice_dfs)

    def fix_items(self, drop_items=True, rename_items=False):
        # Initialize
        df = self.clean_df.copy()
        if not isinstance(df.index,
                          pd.core.indexes.datetimes.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])

        if isinstance(drop_items, bool):
            if drop_items:
                drop_items = ['timestamp', 'timestamp.1']
            else:
                drop_items = []
        elif isinstance(drop_items, list):
            drop_items.append(['timestamp', 'timestamp.1'])

        if isinstance(rename_items, bool):
            if rename_items:
                rename_items = {'new_price': 'c',
                                'new_volume': 'v',
                                'ap1': 'ask',
                                'bp1': 'bid'}
            else:
                rename_items = {}

        # Fix dataset
        valid_drop_items = [itm for itm in drop_items if itm in df.columns]
        df.drop(valid_drop_items, axis=1, inplace=True)

        renamed = {k: v for k, v in rename_items.items() if k in df.columns}
        df.rename(columns=renamed, inplace=True)
        self.clean_df = df.copy()

    def fix_duplicated_index(self, s=None):
        """Summary
        Dealing with abnormal data with duplicated indices in DatatimeIndex
        format. It simply adds user_defined values to those indices in level
        of seconds.

        Args:
            p (DataFrame): dataframe
            s (list): values added to duplicated index. At default, s = [0.3, 0.45, 0.6, 0.75, 0.9]

        Returns:
            Dataframe: dataframe with fixed indices.
        """
        p = self.clean_df.copy()
        if s is None:
            s = pd.to_timedelta(np.linspace(0.3, 0.9, 5), unit='s')

        p_idx = p.index
        is_dup = p_idx.duplicated(keep=False)
        if np.any(is_dup):
            fixing_idx = p_idx[is_dup]
            s_idx_grp = pd.Series(fixing_idx, index=fixing_idx).groupby(pd.Grouper(freq='1s'))
            fixed_idx = (a + b for _, g in s_idx_grp for a, b in zip(g.tolist(), s) if g.shape[0] > 0)
            new_idx = np.array(p_idx)
            new_idx[is_dup] = list(fixed_idx)
            p.index = new_idx
        self.clean_df = p.copy()

    def scale_frequency(self, freq, rolling=False):
        tg = timegrouper()
        if rolling:
            return tg.rolling_reformat(self.clean_df, freq)
        else:
            return tg.reformat(self.clean_df, freq)


class get_directory():
    def __init__(self, path):
        self.path = path

    def ls(self, path=None, suffix=None, fullpath=False):
        from os import listdir
        pwd = self.path if path is None else path
        filenames = listdir(pwd)
        if suffix is None:
            fns = filenames
        else:
            fns = [fn for fn in filenames if '.' + suffix in fn]
        if fullpath:
            out = [pwd + fn for fn in fns]
        else:
            out = fns
        return out

    def load_all_csv(self, index_col, parse_dates=True, n_jobs=None):

        def load_data(fn):
            df = pd.read_csv(fn, index_col=index_col, parse_dates=True)
            df['from'] = fn.split('.')[0]
            return df

        fns = self.ls(suffix='csv', fullpath=True)
        if n_jobs:
            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(processes=n_jobs)
            updated_df = pd.concat(pool.map(load_data, fns))
        else:
            from tqdm import tqdm
            updated_df = pd.concat([load_data(fn) for fn in tqdm(fns)])
        return updated_df


def select_slice_time(df, time_slices, s=False, e=False):
    for ts_s, ts_e in time_slices:
        df_s = df.between_time(ts_s, ts_e, include_start=s, include_end=e)
        yield df_s


def splist(seq, size=10, fillna=True):
    """
    To split a list evenly
    :param seq: sequence of list
    :param size:  size of each group
    Example: new_list = list(splist(range(100), 5))
    """
    from itertools import izip_longest
    new_list = list(izip_longest(*[iter(seq)] * size))

    if not fillna:
        new_list[-1] = [i for i in new_list[-1] if i]

    return new_list


def checkEqualIvo(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def timed(fun, ntimes=1, *args):
    from time import time
    s = time()
    for _ in range(ntimes):
        r = fun(*args)
    st = time() - s
    print('{} execution took {} seconds.'.format(fun.__name__, st))
    print('{} loops, {} seconds per loop.'.format(ntimes, st / float(ntimes)))
    return r, st


def widen_pandas_display(maxrow=False, maxcol=False, width=False):
    if any([maxrow, maxcol, width]):
        pd.set_option('display.max_rows', maxrow)
        pd.set_option('display.max_columns', maxcol)
        pd.set_option('display.width', width)
    else:
        pd.reset_option('display')


def widen_numpy_display(precision=None, threshold=None, edgeitems=None,
                        linewidth=None, suppress=None, nanstr=None,
                        infstr=None, formatter=None):
    np.set_printoptions(precision=precision, threshold=threshold,
                        edgeitems=edgeitems, linewidth=linewidth,
                        suppress=suppress, nanstr=nanstr,
                        infstr=infstr, formatter=formatter)


def first_true(mylist):
    """Summary
    locate first ture value in the list.

    Args:
        mylist (TYPE): iterable type

    Returns:
        TYPE: int
    """
    return next((i for i, x in enumerate(mylist) if x), None)


def multi_reindex(df):
    indices = df.index.to_frame(index=False)
    indices_merged = indices.apply(
        lambda x: '|'.join(x.astype('str').tolist()), axis=1)
    try:
        return pd.DataFrame(df.values, index=indices_merged, columns=df.columns)
    except AttributeError:
        return pd.Series(df.values, index=indices_merged, name=df.name)


def fillna(arr, method='ffill', axis=0):
    """Summary
    Fill nan values in arr wrapped up by pandas.

    Args:
        arr (numpy.array): an array
        method (str): same as method in pd.fillna()
        axis (int): 0 is fill along columns, 1 is along rows.

    Returns:
        TYPE: numpy.array
    """
    df = pd.DataFrame(arr.copy())
    df.fillna(method=method, axis=axis, inplace=True)
    out = df.values.reshape(arr.shape)
    return out


def get_values_between(series, lb, ub, qtile=True, return_type=None):
    if qtile:
        ub = np.percentile(series[~np.isnan(series)], ub)
        lb = np.percentile(series[~np.isnan(series)], lb)

    tf = (series > lb) & (series < ub)
    if return_type == bool:
        return tf
    else:
        return series[tf]


def rescale_by_rank(x, scale=(0, 1)):
    drang = float(scale[1]) - scale[0]
    out = scale[0] + drang * scs.rankdata(x) / len(x)
    if isinstance(x, pd.core.series.Series):
        return pd.Series(out, index=x.index, name=x.name)

    return out


def binarize(x, th=0, flag=0, type_out=float):
    """
    flag:
    0: upside is including threshold
    1: upside is not including threshold
    2: triplet

    return
:    0 or -1: represent x < th
    1: upside
    """
    arr = np.array(x)
    if flag == 0:
        res = (arr >= th).astype(float) - (arr < th).astype(float)
    elif flag == 1:
        res = (arr > th).astype(float) - (arr <= th).astype(float)
    elif flag == 2:
        res = (arr > th).astype(float) - (arr < th).astype(float)
    else:
        raise ValueError('flag {} is not defined.'.format(flag))

    if isinstance(x, pd.core.series.Series):
        return pd.Series(res, index=x.index, name=x.name, dtype=type_out)

    return res.astype(type_out)


def get_str(string):
    import re
    list_str = re.findall('[a-zA-Z]*', string)
    return [ss for ss in list_str if len(ss) > 0]


def get_digit(string):
    import re
    list_digit = re.findall('[0-9]*', string)
    return [ss for ss in list_digit if len(ss) > 0]


def get_sign(data):
    """
    Get signs of the data.
    :param data:
    :return: element-wise signs
    """
    return abs(data) / data


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_apply(func, x, window, forward=True):
    """Summary

    Args:
        func (function): Description
        x (TYPE): Description
        window (int): Description
        forward (bool, optional): Description

    Returns:
        TYPE: Description
    """
    if forward:
        arr = np.array(x)
    else:
        arr = np.array(x)[::-1]

    res = map(func, rolling_window(arr, window))
    try:
        return np.array(res)
    except Exception:
        return list(res)


def rolling_extend(func, x, forward=True):
    """
    Apply specific function by rolling forward or backward.

    :param func: function to be applied
    :param x: variables
    :param forward: Apply with forward value if ture. Default is true.
    :return:
    """
    if forward:
        arr = np.array(x)
    else:
        arr = np.array(x)[::-1]

    res = (np.nan,) + tuple(func(arr[:i]) for i in range(2, len(arr) + 1))
    return np.array(res)


def rolling_combine(func, x, win, lag):
    arr = np.array(x)
    res = np.zeros(len(arr), dtype='O')
    res[win - 1:] = [func(arr[i:i + win]) for i in range(len(arr) - win + 1)]

    if isinstance(x, pd.core.series.Series):
        return pd.Series(res, index=x.index, name=x.name).shift(lag)
    elif lag < 0:
        return np.roll(res, lag)[:lag]
    elif lag > 0:
        return np.roll(res, lag)[lag:]
    else:
        return res


def get_func_argnames(func):
    """
    Retrieve function's input arguments

    :param func:
    :return: a tuple of names of arguments
    """
    argcnt = func.__code__.co_argcount
    argvars = func.__code__.co_varnames
    return argvars[:argcnt]


def run_argtup(func, argvalues):
    """
    Execute any functions with their input arguments in tuples.

    :param func:
    :param argvalues:
    :return: results from assigned function
    """
    argnames = get_func_argnames(func)
    if len(argnames) != len(argvalues):
        raise ValueError("Length of args doens't match.")
    for argn, argv in zip(argnames, argvalues):
        exec('{}=argv'.format(argn))
    return eval('func(%s, %s)' % argnames)


def rolling_df(func, df, win, apply_colns=None):
    """

    :param func:
    :param df: the orders of df.columns should be the same as function input
    :param win: windows
    :param apply_colns: optional.
    :return:
    """
    rolrang = range(df.shape[0] - win + 1)
    vals = [run_argtup(func, tuple(df[i:i + win].values.T)) for i in rolrang]
    results = pd.DataFrame(vals, columns=apply_colns, index=df.index[win - 1:])
    return results


def main():
    pass


if __name__ == '__main__':
    main()
