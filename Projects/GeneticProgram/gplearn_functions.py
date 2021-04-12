import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.functions import make_function, _function_map
from tqdm import tqdm

import utilities as ut

warnings.filterwarnings("ignore")

import talib
from sklearn.metrics import accuracy_score

data_frequency = 'day'

# Base
def __init__(self):
    pass


def _sigmoid(data):
    return 1 / (1 + np.exp(-1 * data)).astype(float)


def _divide(data1, data2):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        # print("\t[DEBUG] - _divide", type(x1), type(x2), x1[:5], x2[:5])
        return np.divide(data1, data2, out=np.zeros_like(data2), where=data2 != 0).astype(float)


def _sqrt(data):
    """Closure of square root for negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        # if np.all(x1 <= 0):
        #     value = np.zeros_like(x1)
        # else:
        #     value = np.sqrt(x1)
        return np.sqrt(data, out=np.zeros_like(data), where=data >= 0).astype(float)


def _log(data):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        # print("\t[DEBUG] - _log", type(x1), len(x1), x1[:5])
        # if np.all(x1 <= 0):
        #     value = np.zeros_like(x1)
        # else:
        #     value = np.log(x1)
        return np.log(data, out=np.zeros_like(data), where=data > 0).astype(float)


def _inverse(data):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(data) != 0., 1. / data, 0.).astype(float)


def _exp(data):
    """Closure of log for zero arguments."""
    return np.exp(data).astype(float)


def _square(data):
    return np.power(data, 2).astype(float)


def _cube(data):
    return np.power(data, 3).astype(float)


def _cbrt(data):
    return np.cbrt(data).astype(float)


def _compare(data1, data2):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    return (np.array(data1) > np.array(data2)).astype(float)


# Time Series Functions
def _roll_sum(data, n):
    ss = pd.Series(data).rolling(n).sum()
    ss = ss.fillna(ss.ewm(span=n).mean()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _roll_mul(data, n):
    ss = pd.Series(data).rolling(n).apply(np.prod)
    ss = ss.fillna(ss.ewm(span=n).mean()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _delta(data, n):
    value = np.diff(data, n=n)
    value = np.concatenate([[0] * n, value])
    return value.astype(float)


def _pct_change(data, n):
    ss = pd.Series(data).pct_change(n)
    ss = ss.fillna(ss.ewm(span=n).mean()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _delay(data, n):
    ss = pd.Series(data).shift(n)
    ss = ss.fillna(ss.ewm(span=n).mean()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _roll_sma(data, n):
    ss = pd.Series(data).rolling(n).mean()
    ss = ss.fillna(ss.ewm(span=n).mean()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _roll_std(data, n):
    ss = pd.Series(data).rolling(n).std()
    ss = ss.fillna(ss.ewm(span=n).std()).fillna(0.)
    value = ss.values
    return value.astype(float)


def _ewma(data, n):
    ss = pd.Series(data).ewm(span=n).mean()
    ss = ss.ffill().fillna(0.)
    value = ss.values

    # val, _ = numba_ewma(data, alpha=2 / (n+1), state = None, adjust = True, ignore_na = True, minp = 1)
    # ss = pd.Series(val).ffill().fillna(0.)
    # value = ss.values
    return value.astype(float)


def _ewsd(data, n):
    ss = pd.Series(data).ewm(span=n).std()
    ss = ss.ffill().fillna(0.)
    value = ss.values

    # val, _ = numba_ewstd(data, alpha=2 / (n+1), adjust=True)
    # ss = pd.Series(val).ffill().fillna(0.)
    # value = ss.values
    return value.astype(float)


def _roll_min(data, n):
    ss = pd.Series(data).rolling(n).min()
    ss = ss.ffill().fillna(0.)
    value = ss.values
    return value.astype(float)


def _roll_max(data, n):
    ss = pd.Series(data).rolling(n).max()
    ss = ss.ffill().fillna(0.)
    value = ss.values
    return value.astype(float)


def _roll_corr(data1, data2, n):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    na_pct = np.isnan(np.stack([data1, data2], axis=1)).any(1).sum() / len(data1)
    if na_pct > 0.618:
        value = np.zeros_like(data1)
    else:
        corr = [np.corrcoef(data1[i - n:i], data2[i - n:i])[1, 0] for i in range(n, len(data1))]
        full_corr = np.concatenate([[0.] * n, corr])
        ss = pd.Series(full_corr).ffill().fillna(0.)
        value = ss.values
    return value.astype(float)


def _roll_rank(data, n):
    # if np.any(np.isnan(data)):
    #     raise ValueError('data contains NA.')
    arr = np.array(data)
    rk = [1 + np.argsort(arr[i - n:i])[-1] for i in range(n, len(arr))]
    value = np.concatenate([[len(arr) / 2] * n, rk])
    return value.astype(float)


def _roll_minmaxscale(data, n):
    arr = np.array(data)
    value = [ut.min_max_scaling(arr[i - n:i])[-1] for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_standardize(data, n):
    arr = np.array(data)
    value = [ut.standardize(arr[i - n:i])[-1] for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_convex_mapping(data, n):
    arr = np.array(data)
    value = [ut.convex_mapping(arr[i - n:i])[-1] for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_ols_beta(data1, data2, n):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    x = np.squeeze(data1)[:, None]
    y = np.squeeze(data2)[:, None]
    value = [ut.numpy_ols_beta(x[i - n:i, :], y[i - n:i, :]) for i in range(n, len(x))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_ols_resid(data1, data2, n):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    x = np.squeeze(data1)[:, None]
    y = np.squeeze(data2)[:, None]
    value = [ut.numpy_ols_resid(x[i - n:i, :], y[i - n:i, :]) for i in range(n, len(x))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


# unexplainable
def _scale(data):
    arr = np.array(data)
    c = np.abs(arr).sum()
    if c != 0:
        value = arr / c
    else:
        value = arr
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_argmin(data, n):
    arr = np.array(data)
    value = [np.argmin(arr[i - n:i]) for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_argmax(data, n):
    arr = np.array(data)
    value = [np.argmax(arr[i - n:i]) for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _roll_argmaxmin(data, n):
    arr = np.array(data)
    value = [np.argmax(arr[i - n:i]) - np.argmin(arr[i - n:i]) for i in range(n, len(arr))]
    value = np.concatenate([[0] * n, value])
    ss = pd.Series(value).ffill().fillna(0)
    value = ss.values
    return value.astype(float)


def _talib_HT_DCPHASE(data):
    try:
        value = talib.HT_DCPHASE(data)
        ss = pd.Series(value).ffill().fillna(0)
        value = ss.values
    except Exception as e:
        raise Exception("[_talib_HT_DCPHASE]", e)
        # print("[WARNING] _talib_HT_DCPHASE: {}".format(e.args[0]))
        value = np.zeros_like(data)
    return value.astype(float)


def _talib_KAMA(data, n):
    if n <= 1:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.KAMA(data, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_KAMA]", e)
            # print("[WARNING] _talib_KAMA: {}".format(e.args[0]))
            value = np.zeros_like(data)
    return value.astype(float)


def _talib_MIDPOINT(data, n):
    if n <= 1:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.MIDPOINT(data, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_MIDPOINT]", e)
            # print("[WARNING] _talib_MIDPOINT: {}".format(e.args[0]))
            value = np.zeros_like(data)
    return value.astype(float)


def _talib_LINEARREG_ANGLE(data, n):
    if n <= 1:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.LINEARREG_ANGLE(data, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_LINEARREG_ANGLE]", e)
            # print("[WARNING] _talib_LINEARREG_ANGLE: {}".format(e.args[0]))
            value = np.zeros_like(data)
    return value.astype(float)


def _talib_LINEARREG_INTERCEPT(data, n):
    if n <= 1:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.LINEARREG_INTERCEPT(data, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_LINEARREG_INTERCEPT]", e)
            # print("[WARNING] _talib_LINEARREG_INTERCEPT: {}".format(e.args[0]))
            value = np.zeros_like(data)
    return value.astype(float)


def _talib_LINEARREG_SLOPE(data, n):
    if n <= 1:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.LINEARREG_SLOPE(data, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_LINEARREG_SLOPE]", e)
            # print("[WARNING] _talib_LINEARREG_SLOPE: {}".format(e.args[0]))
            value = np.zeros_like(data)
    return value.astype(float)


def _talib_stream_BETA(data1, data2, n):
    # if len(data1) != len(data2):
    #     return np.zeros_like(max(len(data1), len(data2)))
    if n <= 0:
        value = np.zeros_like(data)
    else:
        try:
            value = talib.BETA(data1, data2, timeperiod=n)
            ss = pd.Series(value).ffill().fillna(0)
            value = ss.values
        except Exception as e:
            raise Exception("[_talib_stream_BETA]", e)
            # print("[WARNING] _talib_stream_BETA: {}".format(e.args[0]))
            value = np.zeros_like(data1)
    return value.astype(float)


# Make base functions
_base_function_params1 = ['add', 'sub', 'mul', 'abs', 'neg', 'max', 'min', 'sin', 'cos', 'tan']
_base_function_params2 = {
    # function_name: (function, arity)
    'div': (_divide, 2),
    'sqrt': (_sqrt, 1),
    'log': (_log, 1),
    'inv': (_inverse, 1),
    'exp': (_exp, 1),
    'sig': (_sigmoid, 1),
    'square': (_square, 1),
    'cube': (_cube, 1),
    'compare': (_compare, 2),
    'scale': (_scale, 1),
    'talib_HT_DCPHASE': (_talib_HT_DCPHASE, 1),
}
base_function_dict = {}
for fn in _base_function_params1:
    base_function_dict[fn] = _function_map[fn]

for fn, (f, a) in _base_function_params2.items():
    base_function_dict[fn] = deepcopy(make_function(function=f, name=fn, arity=a))

# Make time series functions
rolling_periods = {
    'minute': np.array([1, 5, 15, 30, 60]),
    'day': np.array([1, 3, 5, 10, 20]),
}

# annualized_factor = {
#     'minute': np.sqrt(240 * 252),
#     'day': np.sqrt(252)
# }


ts_function_params = {
    # function_name: (function, arity, window_iterator)
    # arity == 1
    'roll_sum': (_roll_sum, 1, rolling_periods[data_frequency][1:]),
    'roll_mul': (_roll_mul, 1, rolling_periods[data_frequency][1:]),
    'delta': (_delta, 1, rolling_periods[data_frequency]),
    'pct_change': (_pct_change, 1, rolling_periods[data_frequency]),
    'delay': (_delay, 1, rolling_periods[data_frequency]),
    'roll_min': (_roll_min, 1, rolling_periods[data_frequency][1:]),
    'roll_max': (_roll_max, 1, rolling_periods[data_frequency][1:]),
    'roll_sma': (_roll_sma, 1, rolling_periods[data_frequency][1:]),
    'roll_std': (_roll_std, 1, rolling_periods[data_frequency][1:]),
    'roll_rank': (_roll_rank, 1, rolling_periods[data_frequency][1:]),
    'roll_minmaxscale': (_roll_minmaxscale, 1, rolling_periods[data_frequency][1:]),
    'roll_standardize': (_roll_standardize, 1, rolling_periods[data_frequency][1:]),
    'roll_convex_mapping': (_roll_convex_mapping, 1, rolling_periods[data_frequency][1:]),
    'roll_argmin': (_roll_argmin, 1, rolling_periods[data_frequency][1:]),
    'roll_argmax': (_roll_argmax, 1, rolling_periods[data_frequency][1:]),
    'roll_argmaxmin': (_roll_argmaxmin, 1, rolling_periods[data_frequency][1:]),
    'ewma': (_ewma, 1, rolling_periods[data_frequency][1:]),
    'ewsd': (_ewsd, 1, rolling_periods[data_frequency][1:]),
    'talib_KAMA': (_talib_KAMA, 1, rolling_periods[data_frequency][1:]),
    'talib_MIDPOINT': (_talib_MIDPOINT, 1, rolling_periods[data_frequency][1:]),
    'talib_LINEARREG_ANGLE': (_talib_LINEARREG_ANGLE, 1, rolling_periods[data_frequency][1:]),
    'talib_LINEARREG_INTERCEPT': (_talib_LINEARREG_INTERCEPT, 1, rolling_periods[data_frequency][1:]),
    'talib_LINEARREG_SLOPE': (_talib_LINEARREG_SLOPE, 1, rolling_periods[data_frequency][1:]),

    # arity == 2
    'roll_corr': (_roll_corr, 2, rolling_periods[data_frequency][1:]),
    'roll_ols_beta': (_roll_ols_beta, 2, rolling_periods[data_frequency][1:]),
    'roll_ols_resid': (_roll_ols_resid, 2, rolling_periods[data_frequency][1:]),
    'talib_stream_BETA': (_talib_stream_BETA, 2, rolling_periods[data_frequency][1:]),
}

# ts_function_multi_dict = {}
# for n1 in np.arange(3, 21, 3):
#     for n2 in np.arange(10, 63, 5):
#         if n1 < n2:
#             name = '{}_{}_{}'.format('chaikin_oscillator', n1, n2)
#             gp_func = make_function(lambda x: _chaikin_oscillator(x, n1=n1, n2=n2), name=name, arity=1, valid=False)
#             ts_function_multi_dict['name'] = deepcopy(gp_func)

ts_function_dict = {}
for fn, (f, a, ws) in ts_function_params.items():
    for w in ws:
        name = '{}_{}'.format(fn, w)
        if a == 1:
            gp_func = make_function(lambda x: f(x, n=w), name=name, arity=a)
        elif a == 2:
            gp_func = make_function(lambda x, y: f(x, y, n=w), name=name, arity=a)
        ts_function_dict[name] = deepcopy(gp_func)

function_set_dict = {}
function_set_dict.update(base_function_dict)
function_set_dict.update(ts_function_dict)
function_set = list(function_set_dict.values())


def _generate_signal(y_pred, n=1024, q_lower=0.2, q_upper=0.8, flag=1):
    # if len(y_pred) <= n:
    #     raise ValueError("length({}) is smaller than window({}).".format(len(y_pred), n))
    #     # print("[WARNING] length({}) is smaller than window({}).".format(len(y_pred), n))

    if flag == 1:
        # long/short
        pass
    elif flag == 2:
        # long only
        lower = 0.0
    else:
        raise ValueError("flag must be 1 (long/short) or 2(long only)")

    upper = pd.Series(y_pred).rolling(n).quantile(q_upper).values
    lower = pd.Series(y_pred).rolling(n).quantile(q_lower).values
    signal = np.zeros_like(y_pred)
    signal[y_pred > upper] = 1
    signal[y_pred < lower] = -1
    return signal


def _logret(y, y_pred, w=None):
    signal = _generate_signal(y_pred, n=1024, q_lower=0.2, q_upper=0.8)
    daily_ret = signal * y
    totret = np.sum(daily_ret)
    return totret


def _sharpe(y, y_pred, w=None):
    signal = _generate_signal(y_pred, n=1024, q_lower=0.2, q_upper=0.8)

    daily_ret = signal * y
    totret = np.sum(daily_ret)
    if totret == 0:
        sp = 0.0
    else:
        # annual_std = np.sqrt(240 * 252) * np.nanstd(daily_ret)
        # annual_std = annualized_factor[data_frequency] * np.nanstd(daily_ret)
        std = np.nanstd(daily_ret)  # not annualized
        if std == 0:
            sp = 0
        else:
            sp = totret / std
    return sp


def _accuracy_score(y, y_pred, w=None):
    y_digi = np.digitize(y, [-0.05, 0.05]) - 1
    y_pred_digi = np.digitize(y_pred, [-0.05, 0.05]) - 1
    return accuracy_score(y_digi, y_pred_digi)


gp_sharpe = make_fitness(_sharpe, greater_is_better=True)
gp_accuracy_score = make_fitness(_accuracy_score, greater_is_better=True)


def clean_gplearn_programs(gplearn_programs, verbose=0):
    all_programs_info_list = []
    if verbose > 0:
        iterobj = tqdm(enumerate(gplearn_programs))
    else:
        iterobj = enumerate(gplearn_programs)

    for gen_i, gen in iterobj:
        for prog_i, prog in enumerate(gen):
            if prog is not None:
                _fitness = prog.fitness_
                _depth = prog.depth_
                _length = prog.length_
            else:
                _fitness = np.nan
                _depth = np.nan
                _length = np.nan
            all_programs_info_list.append([gen_i, prog_i, _fitness, str(prog), _depth, _length])

    all_programs_info_df = pd.DataFrame.from_records(all_programs_info_list,
                                                     columns=['generation', 'program', 'fitness', 'expression', 'depth',
                                                              'length'])
    organized_info_df = all_programs_info_df.dropna().drop_duplicates(subset=['expression']).sort_values(
        ['fitness', 'generation'], ascending=False)
    return organized_info_df


def evaluate_logret(arr):
    # totret, mdd, sharpe, calmar, num_trade, size
    _cumret = np.cumsum(arr)
    _totret = _cumret[-1]
    _dd = (1 + _cumret) / np.maximum.accumulate(1 + _cumret)
    _mdd = _dd[-1]
    _sharpe = _totret / np.std(arr)  # not annualized
    _calmar = _totret / _mdd
    _size = len(arr)
    return [_totret, _mdd, _sharpe, _calmar, _size]
