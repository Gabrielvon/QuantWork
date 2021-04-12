from settings import *
import gabFunc as gfc

import pandas as pd
import numpy as np


def add_lagged_features(df, name, lags, rolnums):
    out_df = df.copy()

    if len(lags) > 0:
        for lag in lags:
            out_df['{}_lag{}'.format(name, lag)] = out_df[name].shift(lag)

    if len(rolnums) > 0:
        for rolnum in rolnums:
            out_df['{}_rolmean{}'.format(
                name, rolnum)] = out_df[name].rolling(rolnum).mean()

    return out_df


def num_of_crossing(s1, s2, details=False):
    '''
    Count the number of times when s1 and s2 are crossing each other
    '''
    tf1 = isinstance(s1, pd.core.series.Series)
    tf2 = isinstance(s2, pd.core.series.Series)
    if tf1:
        index = s1.index
    if tf2:
        index = s2.index
    s1 = np.array(s1)
    s2 = np.array(s2)
    out = np.diff((s1 > s2).astype(float)) != 0
    out = np.append(False, out)
    if not details:
        out = out.sum()

    return pd.Series(out, index=index)


def num_of_double_sized(series, ub, lb, details=False):
    x_ub = num_of_crossing(series, ub, details=True)
    x_lb = num_of_crossing(lb, series, details=True)
    X_ub = x_ub[x_ub].replace(True, 1)
    X_lb = x_lb[x_lb].replace(True, 0)
    Xing = pd.concat([X_ub, X_lb], 0).sort_index().diff()
    out = Xing.dropna()[Xing != 0]
    if details:
        return out
    else:
        return out.shape[0] / 2


def calc_type_stat(df, groupby_col, analyze_col):
    type_stat = df.groupby(groupby_col)[analyze_col].describe()
    out = pd.DataFrame()
    for k, v in type_stat.to_dict().items():
        out.loc[k[1], k[0]] = v
    return out


def prob_of_conseq_occ_predict(df, coln_x, coln_y, n, tol=0,
                               direction='uu', detail=False, plot=False):
    """
    Compute the probability of consequently accumulated days of
    coln_x, which is leading coln_y

    :param df: dataframe
    :param coln_x: a column within df that must be binary values, [-1, 0 , 1]
    :param coln_y: a column within df that must be numeric values
    :param n: rolling number
    :param tol: tolerance for less strictly occurences.
    :param direction: uu for predict up trend based on up, dd for predict down trend based on down

    Example:
    In [226]: prob_of_conseq_occ_predict(Res, 'InitPoC_trend', 'intraday_trend', 3, 1, direction='uu')
    Out[226]: 0.375
    过去3天内，如果有两天或大于两天, 每天IB的PoC都大于前一天IB的PoC, 那么这一天today's close
    大于today's open的概率为62.5%。换言之, 这一天today's close小于today's open的概率为37.5%

    In [225]: prob_of_conseq_occ_predict(Res, 'InitPoC_trend', 'intraday_trend', 3, 1, reverse='dd')
    Out[225]: 0.5555555555555556
    过去3天内，如果有两天或大于两天, 每天IB的PoC都小于前一天IB的PoC,, 那么这一天today's close
    小于today's open的概率为55.55%。换言之, 这一天today's close大于today's open的概率为55.45%
    """
    # the status of coln_y when the cummulate sum of n consequently latest
    # coln_x satisfies the condition ( greater than n-tol)
    yhat = pd.Series([0], index=df.index, name=direction)

    if direction == 'dd':
        idx = df[coln_x].rolling(n).sum() <= tol - n
        yhat[idx] = df.loc[idx, coln_y]
        occ = (yhat < 0).sum()
    elif direction == 'uu':
        idx = df[coln_x].rolling(n).sum() >= n - tol
        yhat[idx] = df.loc[idx, coln_y]
        occ = (yhat > 0).sum()
    else:
        raise ValueError("direction must be 'uu' or 'dd'.")

    tot_attempts = sum(yhat != 0)

    if detail:
        return yhat

    if plot:
        sg = pd.concat([yhat, df[coln_y]], axis=1)
        sg.plot.line()

    return np.divide(float(occ), tot_attempts)


def sample_stats(df_y, prob_of_conseq_occ_predict_res, percentage=False):
    """
    Robust test for results of prob_of_conseq_occ_predict.
    This function inspect the numbers of samples.

    """
    res = prob_of_conseq_occ_predict_res.copy()
    stats_container = dict()
    for ns in res.columns.tolist():
        for coln in res.index:
            nms = coln.split('|')
            yhat = prob_of_conseq_occ_predict(
                df_y, nms[1], nms[0], ns[0], ns[1], detail=True)
            # numbers of entire samples
            n_samples0 = yhat.size
            # numbers of predictions
            n_samples1 = sum(yhat != 0)
            if n_samples1 == 0:
                n_samples1 = np.nan
            sg = pd.concat([df_y[nms[0]], yhat], axis=1)
            sg_drop0 = sg.loc[sg[nms[0]] != 0, :]
            # numbers of accurate prediction
            n_samples2 = sum(sg_drop0.iloc[:, 0] == sg_drop0.iloc[:, 1])
            stat = (n_samples0, n_samples1, n_samples2)
            if percentage:
                stat = tuple(pd.Series(stat).pct_change().iloc[1:] + 1) + (n_samples0,)

            stats_container[(coln, ns)] = stat
    return pd.DataFrame(stats_container, index=['n_all', 'n_attempts', 'n_samples1']).T


def calc_trailing_stop_along_price(df):
    stop_loss_bd = pd.concat(
        [gfc.rolling_extend(max, df['High']),
         gfc.rolling_extend(min, df['Low'])],
        axis=1)
    stop_loss_bd.columns = ['High', 'Low']
    stop_loss_bd['High'] = stop_loss_bd['High'] - df['eor'] - 0.01
    stop_loss_bd['Low'] = stop_loss_bd['Low'] + df['eor'] + 0.01
    stop_loss_bd.ix[stop_loss_bd['High'] < df['lb'] - 0.01, 'High'] = df['lb'] - 0.01
    stop_loss_bd.ix[stop_loss_bd['Low'] > df['ub'] + 0.01, 'Low'] = df['ub'] + 0.01
    stop_loss_bd.rename(columns={'High': 1, 'Low': -1}, inplace=True)
    return stop_loss_bd


def mpBT_wCost(tradingRec, underlying):
    EvalMat, tradingRec = gbt.BT_wCost(tradingRec, underlying)
    EvalMat['TotTrades(#Valid_ES==2)'] = gfc.true_pct(
        tradingRec['Valid_ES'] == 2)
    EvalMat['TotTrades(#Valid_ES==1)'] = gfc.true_pct(
        tradingRec['Valid_ES'] == 1)
    EvalMat['TotTrades(#Valid_ES==0)'] = gfc.true_pct(
        tradingRec['Valid_ES'] == 0)
    return EvalMat, tradingRec


def generate_trec1(long, short):
    """
    Generate trading record from raw signals.
    It will deal with long and short signals separately. Take long signal as
    example, it will cover long only and if only when neutral signal occurs(
    neither long nor short).

    :param long:
    :param short:
    :return:
    """
    long = long.astype(float)
    short = short.astype(float)
    # Long Spread
    if long.sum() > 0:
        enter_long = long.diff() == 1
        enter_long = enter_long[enter_long].reset_index()
        enter_long.columns = ['dt_Enter', 'Type_Enter']
        cover_long = long.diff() == -1
        cover_long = cover_long[cover_long].reset_index()
        cover_long.columns = ['dt_Exit', 'Type_Exit']
        long_rec = pd.concat([enter_long, cover_long], axis=1)
        if np.isnan(long_rec.iat[-1, -1]):
            long_rec.iat[-1, -2] = long.index[-1]
            long_rec.iat[-1, -1] = True

        long_rec['Type_Enter'] = long_rec['Type_Enter'] * 1.0
        long_rec['Type_Exit'] = long_rec['Type_Exit'] * -1.0
        long_rec['Type2'] = 'longspread'
    else:
        long_rec = pd.DataFrame(
            columns=['dt_Enter', 'Type_Enter', 'dt_Exit', 'Type_Exit', 'Type2'])

    # Short Spread
    if short.sum() > 0:
        enter_short = short.diff() == 1
        enter_short = enter_short[enter_short].reset_index()
        enter_short.columns = ['dt_Enter', 'Type_Enter']
        cover_short = short.diff() == -1
        cover_short = cover_short[cover_short].reset_index()
        cover_short.columns = ['dt_Exit', 'Type_Exit']
        short_rec = pd.concat([enter_short, cover_short], axis=1)
        if np.isnan(short_rec.iat[-1, -1]):
            short_rec.iat[-1, -2] = short.index[-1]
            short_rec.iat[-1, -1] = True

        short_rec['Type_Enter'] = short_rec['Type_Enter'] * -1.0
        short_rec['Type_Exit'] = short_rec['Type_Exit'] * 1.0
        short_rec['Type2'] = 'shortspread'
    else:
        short_rec = pd.DataFrame(
            columns=['dt_Enter', 'Type_Enter', 'dt_Exit', 'Type_Exit', 'Type2'])

    # Combine longSpread and ShortSpread
    signals = pd.concat([long_rec, short_rec])
    trarec = signals.set_index('dt_Enter', drop=False).sort_index()
    return trarec


def generate_trec2(long, short):
    """
    Generate trading record from raw signals.
    It will close the format position when it is opposite to the incoming signal,
    and open new position following the new signal as the same time.

    :param long:
    :param short:
    :return:
    """

    ll = long[long] * 1.0
    ss = short[short] * (-1.0)
    sig = pd.concat([ll, ss]).sort_index()
    sig = fix_signals(sig)
    trades = sig[sig != 0].to_frame('signals').reset_index()
    n = trades.shape[0]
    if n % 2 == 1:
        trades.loc[n, :] = pd.Series(dict(timestamp=long.index[-1],
                                          signals=-trades['signals'].sum()))
        n = trades.shape[0]
    trarec = pd.DataFrame(trades.values.reshape(int(n / 2), 4),
                          columns=['dt_Enter', 'Type_Enter', 'dt_Exit', 'Type_Exit'])
    trarec.index = trarec['dt_Enter']
    return trarec


def generate_trec3(df, actions, keep_balance=True, flag=1):
    """
    Repeat(Reshape) records with more than one actions.

    :param df:
    :param actions: actions
    :param keep_balance: if true, remove trades opened at the end.
    :return: flag 0: merely repeat and split following original format;
             flag 1: After doing flag 0, it returns trade-wise trading record.
    """
    data = df.copy()
    act = np.array(actions)
    data['dummy_act'] = np.abs(act) / act
    repeat_val = zip(data.values, np.abs(act))
    vv = [np.repeat(np.reshape(x, (1, len(x))), n, 0) for x, n in repeat_val]
    data = pd.DataFrame(np.concatenate(vv), columns=data.columns)
    data.rename(columns={'dummy_act': 'act'}, inplace=True)
    if keep_balance:
        if data['act'].sum() != 0:
            data = data.iloc[:-1, :]

    if flag == 0:
        tradingRec = data
    elif flag == 1:
        enter = data.iloc[range(0, len(data), 2), :].reset_index(drop=True)
        enter.columns = [s + '_Enter' for s in enter.columns]
        exit = data.iloc[range(1, len(data), 2), :].reset_index(drop=True)
        exit.columns = [s + '_Exit' for s in exit.columns]
        tradingRec = pd.concat([enter, exit], 1)

    return tradingRec


def BT_pairs(tradingRec, pair_nm):
    """
    Calculate the backtest results in the case of trading the spread within a pair.

    :param trec: dataframe with columns of dt, actions and prices in the pairs are required.
        dt: timestamp
        actions: trading direction with volume
        prices: prices of each underlying
    :param pair_nm: the names matching self_defined function for specific underlyings in gabBT.
        pair_nm should be in the order of spread = pair_nm[0] - pair_nm[1]
    :return: tuples with results in different aspects.
    """
    trec = tradingRec.copy()
    trec['capitalused'] = trec[pair_nm].apply(
        lambda x: calc_init_cap(x, 1, x.name)).sum(1)
    trec['prisprd'] = - \
        np.diff(trec[pair_nm].apply(lambda x: calc_init_cap(x, 1, x.name)), 1)
    trec['tcost'] = trec[pair_nm].apply(
        lambda x: calc_transation_cost(x / 2, x / 2, 1, x.name)).sum(1)
    tradingRec = generate_trec3(trec.reset_index()[['dt', 'prisprd', 'capitalused', 'tcost']],
                                trec['actions'], flag=1)
    tradingRec['PnL'] = tradingRec['act_Enter'] * (
        tradingRec['prisprd_Exit'] - tradingRec['prisprd_Enter'])
    tradingRec['CumPnL'] = tradingRec['PnL'].cumsum()
    tradingRec['tcost'] = tradingRec.filter(regex='tcost').sum(1)
    tradingRec['GrPnL'] = tradingRec['PnL'] - tradingRec['tcost']
    tradingRec['CumGrPnL'] = tradingRec['GrPnL'].cumsum()
    tradingRec['RReturn'] = tradingRec['CumGrPnL'] / tradingRec['capitalused_Enter']
    tradingRec['CumRet(RReturn)'] = tradingRec['RReturn'].cumsum()
    tradingRec['IOPV'] = np.cumprod(1 + tradingRec['RReturn'])
    tradingRec['HoldingPeriod'] = tradingRec['dt_Exit'] - tradingRec['dt_Enter']
    tradingRec['Drawdown'] = gperf.drawdown(tradingRec['IOPV'])
    trec1 = tradingRec.filter(regex='Exit|Enter')
    trec2 = pd.concat([trec1.filter(like='dt_'), tradingRec.loc[:, ~tradingRec.columns.isin(trec1.columns)]], 1)
    return trec, trec1, trec2


def N_extremum(series, n, n_max=False, n_min=False, remove_duplicates=False):
    if not (n_max ^ n_min):
        raise ValueError('Either n_max or n_min should be True')

    if not isinstance(series, pd.core.series.Series):
        series = pd.Series(series)

    if remove_duplicates:
        series = series.drop_duplicates()

    if n_max:
        out = series.sort_values(ascending=False)

    if n_min:
        out = series.sort_values()

    return out.iloc[:n]


def nanabs(x):
    if isinstance(x, (float, int, np.int64)):
        return abs(x)
    else:
        return np.nan


def gabMul(x, exog=0.0):
    x = list(x)
    res = 1.0
    for i in x:
        res *= i + exog
    return res


def isBetween(df, C, Cup, Cdn):
    """
    compare certain rows or columns in a matrix/frame.

    :param df: parent dataset
    :param C: column name for target
    :param Cup: column name for upper bound
    :param Cdn: column name for lower bound
    :return:
    """
    return (df[C] < df[Cup]) & (df[C] > df[Cdn])


def pair_plot(price_freq, scores, n=12):
    sorted_scores = scores.sort_values()
    try:
        sorted_scores.index = pd.to_datetime(sorted_scores.index)
    except ValueError:
        pass
    nlar = sorted_scores[-n:]
    nsma = sorted_scores[:n]

    grouped = price_freq.loc[nlar.index, :].groupby(level=0)
    nrows = len(grouped) / 3
    fig, axs = plt.subplots(3, nrows)
    fig.suptitle('Plots with {} largest values'.format(n))
    axs = axs.flatten()
    i = 0
    for _, pf in grouped:
        va, mp = compute_va(pf, 0.7, return_mp=True)
        axs[i].hist(unstack_freq(np.array(pf)),
                    bins=30, orientation='horizontal')
        axs[i].set_title(str(pf.index[0]) + ':' + '{0:.4g}'.format(nlar[i]))
        axs[i].axhline(y=va['VAH'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['VAL'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['PoC'], color='0.6', linestyle=':', lw=1.5)
        i += 1

    grouped = price_freq.loc[nsma.index, :].groupby(level=0)
    nrows = len(grouped) / 3
    fig, axs = plt.subplots(3, nrows)
    fig.suptitle('Plots with {} smallest values'.format(n))
    axs = axs.flatten()
    i = 0
    for _, pf in grouped:
        va, mp = compute_va(pf, 0.7, return_mp=True)
        axs[i].hist(unstack_freq(np.array(pf)),
                    bins=30, orientation='horizontal')
        axs[i].set_title(str(pf.index[0]) + ':' + '{0:.4g}'.format(nsma[i]))
        axs[i].axhline(y=va['VAH'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['VAL'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['PoC'], color='0.6', linestyle=':', lw=1.5)
        i += 1


def random_plot(freq_gp, selected_dates, n=10):
    try:
        freq_gp.index = pd.to_datetime(freq_gp.index)
        selected_dates = pd.to_datetime(selected_dates)
    except ValueError:
        pass

    pfreq = freq_gp.loc[selected_dates, :]
    pfreq_idx_uni = pfreq.index.unique()
    nsamp = min(len(pfreq_idx_uni), n)
    sample_dates = random.sample(pfreq_idx_uni, nsamp)
    grouped = pfreq.loc[sample_dates, :].groupby(level=0)
    nrows = len(grouped) / 3
    fig, axs = plt.subplots(3, nrows)
    axs = axs.flatten()
    i = 0
    for _, pf in grouped:
        va, mp = compute_va(pf, 0.7, return_mp=True)
        axs[i].hist(unstack_freq(np.array(pf)),
                    bins=30, orientation='horizontal')
        axs[i].set_title(str(pf.index[0]))
        axs[i].axhline(y=va['VAH'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['VAL'], color='0.6', linestyle='--', lw=1.5)
        axs[i].axhline(y=va['PoC'], color='0.6', linestyle=':', lw=1.5)
        i += 1


def random_plot_lines(data, selected_dates, n=10):
    try:
        data.index = pd.to_datetime(data.index)
        selected_dates = pd.to_datetime(selected_dates)
    except Exception:
        pass

    nsamp = min(len(selected_dates), n)
    sample_dates = random.sample(selected_dates, nsamp)

    fig, axs = plt.subplots(3, int(math.ceil(nsamp / 3.0)))
    myFmt = mdates.DateFormatter('%H')
    axs = axs.flatten()
    for i in range(nsamp):
        daily_p = data.get_group(sample_dates[i])[['c', 'v']]
        ib_daily_p = daily_p.between_time(init_tf[0], init_tf[1])
        vah, val, poc = compute_va(ib_daily_p, ci).values
        daily_p['c'].plot.line(ax=axs[i])
        axs[i].set_title(sample_dates[i])
        axs[i].xaxis.set_major_formatter(myFmt)
        axs[i].axhline(y=vah, color='green', lw='2', ls='--')
        axs[i].axhline(y=val, color='green', lw='2', ls='--')
        axs[i].axhline(y=poc, color='purple', lw='2', ls=':')
        axs[i].axvline(ib_daily_p.index[-1], color='grey', lw='0.5', ls='--')
