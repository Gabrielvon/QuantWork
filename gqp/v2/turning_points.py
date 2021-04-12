import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def rolling_maxchg(series, window):
    return series.rolling(window).agg({
        'from_top': lambda x: 1 - float(x[-1]) / x.max(),
        'from_bott': lambda x: float(x[-1]) / x.min() - 1})


def locate_turning(series, window, maxchg, plot=False):
    # Initialize
    if isinstance(series, pd.core.frame.DataFrame):
        series = pd.Series(series.iloc[:, 0], index=series.index)

    if isinstance(window, str):
        window = int(pd.to_timedelta(window) /
                     (series.index[1] - series.index[0]))
    else:
        window = window

    rolling_mm = series.rolling(window).agg(['max', 'min'])

    se_arr = np.array(series)
    se_idx = series.index
    uptp, dntp = [], []
    for i in range(len(se_arr) - window + 1):
        uptp.append(se_idx[i:i + window][se_arr[i:i + window].argmax()])
        dntp.append(se_idx[i:i + window][se_arr[i:i + window].argmin()])

    uptp = pd.Series(uptp, index=se_idx[window - 1:], name='rolling_argmax')
    dntp = pd.Series(dntp, index=se_idx[window - 1:], name='rolling_argmin')

    rolmaxchg = rolling_maxchg(series, window)

    # Top or Bottom
    is_convex = (series.shift(window - 1) <
                 rolling_mm['max']) & (series < rolling_mm['max'])
    is_concave = (series.shift(window - 1) >
                  rolling_mm['min']) & (series > rolling_mm['min'])


    # Satisfying Windows
    upt = (rolmaxchg['from_top'] > maxchg) & is_concave
    dnt = (rolmaxchg['from_bott'] > maxchg) & is_convex

    # Locates the turning points
    # Up
    uptp_idx = uptp[upt[window - 1:]]
    uptp = uptp_idx.to_frame('loc')
    uptp['type'] = 1

    # Down
    dntp_idx = dntp[dnt[window - 1:]]
    dntp = dntp_idx.to_frame('loc')
    dntp['type'] = -1

    # Remove redundant points
    idx_mat = pd.concat([uptp, dntp]).sort_values('loc').drop_duplicates()
    idx_mat = idx_mat[idx_mat['type'].diff() != 0]

    # Plot
    if plot:
        # Extract location along with corresponding prices
        series = series.reset_index().drop_duplicates().set_index(series.index.name)
        series.name = 'Price'
        uptps = series[series.index.isin(idx_mat.loc[idx_mat['type'] == 1, 'loc'])]
        uptps.name = 'Top'
        dntps = series[series.index.isin(idx_mat.loc[idx_mat['type'] == -1, 'loc'])]
        dntps.name = 'Bottom'

        fig, ax = plt.subplots()
        series.plot.line(linewidth=0.5, ax=ax)
        uptps.plot(style='v', ax=ax)
        dntps.plot(style='^', ax=ax)
        ax.legend(['Price', 'Top', 'Bottom'])

    return idx_mat.reset_index(drop=True)


# Turning Identifier 2
def locate_turning2(series, window, maxchg, plot=False):

        # init
    rolmean_win = window

    ma_pctchg_diff = series.rolling(rolmean_win).mean().pct_change()
    # ma_pctchg_diff = series.rolling(rolmean_win).mean().pct_change()
    # from sklearn.preprocessing import Binarizer
    # dire = Binarizer(0).fit_transform(ma_pctchg_diff.to_frame().dropna())
    # ma_pctchg_diff = series.rolling(30).apply(lambda x: x.ewm(span=30))
    uptps = series[ma_pctchg_diff > ma_pctchg_diff.quantile(0.99)]
    uptps.name = 'up_turning'
    dntps = series[ma_pctchg_diff < ma_pctchg_diff.quantile(0.01)]
    dntps.name = 'dn_turning'

    if plot:
        _, ax = plt.subplots()
        series.plot.line(linewidth=0.5, ax=ax)
        ax.plot(uptps, marker='x', linewidth=0.01, color='red')
        ax.plot(dntps, marker='x', linewidth=0.01, color='green')
        ax.legend(['Price', 'Top', 'Bottom'])

    return pd.concat([uptps, dntps], axis=1)
