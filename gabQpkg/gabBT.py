# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:30:51 2017

@author: Gabriel Feng
"""

import pandas as pd
import numpy as np
import gabFunc as gfc
import gabStat as gstat
import gabperformance as gperf
from settings import calc_init_cap, calc_pnl, calc_transation_cost
# import scipy.stats as scs


def generate_trading_records(signals):
    """Summary


    Args:
        signals (dataframe): Must look like the following
            index: timestamp + signals (including ask and bid price)
            columns: underlyings

        underly                               rb       j      jm
                                signals
        2017-11-08 13:30:01.000 ask         3721    1832  1182.5
                                bid         3720  1830.5    1182
                                openlong   False   False   False
                                openshort   True    True    True
                                selllong    True    True    True
                                buycover   False   False   False
        2017-11-08 13:30:01.241 ask         3721  1832.5  1182.5
                                bid         3720    1832    1182
                                openlong   False   False   False
                                openshort   True    True    True
                                selllong    True    True    True
                                buycover   False   False   False
        2017-11-08 13:30:01.256 ask         3721  1832.5  1182.5
                                bid         3720    1832    1182
                                openlong   False   False   False
                                openshort   True    True    True
                                selllong    True    True    True
                                buycover   False   False   False

    Returns:
        TYPE: Description
    """
    nrows, ncols = signals.shape
    trsig_arr = np.array(signals).reshape(nrows / 6, 6, ncols)
    arr_colns = signals.columns
    ts_idx, sig_idx = signals.index.levels

    pos, logs = 0, []
    trasig = []
    for tr_ts, tr_data in zip(ts_idx, trsig_arr):
        ask, bid = tr_data[:2, :]
        tr_data[2:, :][pd.isnull(tr_data[2:, :])] = False
        oplo, opsh, selo, buco = tr_data[2:, :]
        log = [tr_ts, np.nan, np.nan, 0, 'Hold']
        if pos == 0:
            if oplo.any():
                # openlong
                co_i = np.where(oplo)[0][0]
                log = [tr_ts, arr_colns[co_i], ask[co_i], 1, 'Enter']
                pos += 1
                trasig.append(tr_data)
            elif opsh.any():
                # openshort
                co_i = np.where(opsh)[0][0]
                log = [tr_ts, arr_colns[co_i], bid[co_i], -1, 'Enter']
                pos -= 1
                trasig.append(tr_data)
        elif pos == 1:
            if selo[co_i]:
                # selllong
                log = [tr_ts, arr_colns[co_i], bid[co_i], -1, 'Exit']
                pos -= 1
                trasig.append(tr_data)
        elif pos == -1:
            if buco[co_i]:
                # buycover
                log = [tr_ts, arr_colns[co_i], ask[co_i], 1, 'Exit']
                pos += 1
                trasig.append(tr_data)
        logs.append(log)

    df_logs = pd.DataFrame(logs, columns=['dt', 'underly', 'Price', 'Type', 'Type2'])
    trec = df_logs[df_logs['Type'] != 0]
    if trec.shape[0] > 0:
        imbalance_und = trec.groupby('underly')['Type'].sum()
        if np.any(imbalance_und != 0):
            dt_coer = ts_idx[-1]
            imb_status = imbalance_und[imbalance_und != 0].to_dict()
            for imb_und, imb_pos in imb_status.items():
                last_mdta = signals.loc[(dt_coer, ['ask', 'bid']), imb_und].values
                coer_ty = -imb_pos
                coer_pri = last_mdta[{1: 0, -1: 1}[coer_ty]]
                coer_tr = pd.Series({'dt': dt_coer, 'underly': imb_und, 'Price': coer_pri,
                                     'Type': coer_ty, 'Type2': 'Exit'}, name=0)
                trec = trec.append(coer_tr)
        trec.sort_values('dt', inplace=True)
        tr = reformat_trec(trec, balance=False)
    else:
        tr = pd.DataFrame(columns=['dt_Enter', 'Price_Enter', 'Type_Enter', 'dt_Exit', 'Price_Exit', 'Type_Exit'])
    return tr


def BT_wCost(tradingRec):
    tRec = tradingRec.copy()

    tRec['PnL'] = tRec.apply(
        lambda x: calc_pnl(x['Price_Enter'], x['Price_Exit'],
                           x['Type_Enter'], x['Underlying']),
        axis=1)
    tRec['CumPnL'] = tRec['PnL'].cumsum()

    tRec['TCost'] = tRec.apply(
        lambda x: calc_transation_cost(x['Price_Enter'], x['Price_Exit'],
                                       abs(x['Type_Enter']), x['Underlying']),
        axis=1)

    tRec['initCap'] = tRec.apply(
        lambda x: np.abs(calc_init_cap(x['Price_Enter'], x['Type_Enter'], x['Underlying'])),
        axis=1)

    tRec['GrossPnL'] = tRec['PnL'] - tRec['TCost']
    tRec['CumGrPnL'] = tRec['GrossPnL'].cumsum()
    tRec['HoldingPeriod'] = tRec['dt_Exit'] - tRec['dt_Enter']

    tRec['RReturn'] = tRec.apply(lambda x: gperf.calc_return(x['GrossPnL'], x['initCap']), axis=1)
    tRec['CumRet(RReturn)'] = tRec['RReturn'].cumsum()
    tRec['CumRet(GrossPnL)'] = (tRec['initCap'].iloc[0] + tRec['GrossPnL']).cumsum().pct_change()
    tRec['IOPV'] = np.cumprod(1 + tRec['RReturn'])
    tRec['Drawdown'] = gperf.drawdown(tRec['IOPV'])

    eMat = tRec['GrossPnL'].astype('float').agg(
        ['count', 'sum', 'mean', 'median', 'max', 'min'])
    eMat.index = [nm + '|PnL' for nm in eMat.index]
    eMat['Underlyings'] = tRec['Underlying'].unique()
    eMat['TCost'] = tRec['TCost'].sum()
    # eMat['GrossPnL'] = tRec['GrossPnL'].sum()
    eMat['TotalReturn'] = tRec['RReturn'].sum()
    eMat['AvgReturn'] = tRec['RReturn'].sum() / eMat['count|PnL']

    eMat['MaxDD'] = tRec['Drawdown'].max()
    eMat['Calmar'] = np.divide(eMat['TotalReturn'], eMat['MaxDD'])
    eMat['AvgPeriod(h)'] = np.mean(tRec['HoldingPeriod']).total_seconds() / 3600

    win_idx = tRec['GrossPnL'] > 0
    loss_idx = ~win_idx
    eMat['count|Win'] = float(win_idx.sum())
    eMat['count|Loss'] = tRec.shape[0] - eMat['count|Win']
    eMat['WinLoss'] = np.divide(eMat['count|Win'], tRec.shape[0])
    eMat['AvgRet|Win'] = tRec.loc[win_idx, 'RReturn'].mean()
    eMat['AvgRet|Loss'] = tRec.loc[loss_idx, 'RReturn'].mean()

    for k, v in tRec['Type_Enter'].value_counts().items():
        k0 = {1: 'Long', -1: 'Short'}[k]
        eMat['pct|Enter:{}'.format(k0)] = float(v) / tRec.shape[0]
        tyen_idx = tRec['Type_Enter'] == k
        eMat['pct|Win|Enter:{}'.format(k0)] = gstat.true_pct(tyen_idx & win_idx)
        eMat['pct|Loss|Enter:{}'.format(k0)] = gstat.true_pct(tyen_idx & loss_idx)
        eMat['PnL|Enter:{}'.format(k0)] = tRec.loc[tyen_idx, 'GrossPnL'].sum()
        eMat['AvgRet|Enter:{}'.format(k0)] = tRec.loc[tyen_idx, 'RReturn'].mean()

    for k, v in tRec['Type_Exit'].value_counts().items():
        eMat['pct|Exit:{}'.format(k)] = float(v) / tRec.shape[0]
        tyex_idx = tRec['Type_Exit'] == k
        eMat['pct|Win|Exit:{}'.format(k)] = gstat.true_pct(tyex_idx & win_idx)
        eMat['pct|Loss|Exit:{}'.format(k)] = gstat.true_pct(tyex_idx & loss_idx)
        eMat['PnL|Exit:{}'.format(k)] = tRec.loc[tyex_idx, 'GrossPnL'].sum()
        eMat['RReturn|Exit:{}'.format(k)] = tRec.loc[tyex_idx, 'RReturn'].sum()
        eMat['AvgRet|Exit:{}'.format(k)] = tRec.loc[tyex_idx, 'RReturn'].mean()

    if tRec['Underlying'].nunique() == 1:
        eMat.name = tRec['Underlying'].iloc[0]
    return eMat, tRec


def fix_signals(signals):
    """
    Remove continuously signals which are in the same direction.

    :param signals:
    :return:
    """
    pos = np.zeros(signals.shape[0])
    for i, sig in enumerate(signals):
        if np.sum(pos) == 0:
            pos[i] = sig
        elif np.sum(pos[:i]) < 0 and sig == 1:
            pos[i] = sig
        elif np.sum(pos[:i]) > 0 and sig == -1:
            pos[i] = sig

    if np.sum(pos) != 0:
        if pos[-1] == 0:
            pos[-1] = - np.sum(pos)
        else:
            pos[-1] = 0

    if isinstance(signals, pd.core.series.Series):
        pos = pd.Series(pos, index=signals.index, name=signals.name)
    return pos


def refine_tradingRecord(TraRec):
    """
    Remove overlapping round-trip trades and keep the one with earlier entry.
    Requirement for the input dataframe:
        1. columns must follow this order
            ['dt_Enter', 'Price_Enter', 'Type_Enter', 'dt_Exit', 'Price_Exit', 'Type_Exit']
        2. TraRec must be sorted by 'dt_Enter' ascending.

    :param TraRec: dataframe
    :return:
    """
    TraRec.index = TraRec['dt_Enter']
    if TraRec.shape[0] < 2:
        return TraRec
    arr = TraRec.values
    updated_trades = np.zeros(arr.shape, dtype=arr.dtype)

    for i, vi in enumerate(arr):
        if i == 0:
            updated_trades[i, :] = vi
        elif all(vi[0] > arr[:i, 3]):
            updated_trades[i, :] = vi
        elif vi[1] + arr[i - 1, 1] == 0:
            updated_trades[i - 1, [3, 4, 5]] = (vi[0], 'stop_open', vi[2])
            updated_trades[i, :] = vi

    updated_trades = updated_trades[updated_trades[:, 0] != 0]
    new_df = pd.DataFrame(updated_trades, index=updated_trades[:, 0], columns=TraRec.columns).infer_objects()
    return new_df


def reformat_trec(tradingRec, id_coln='Type2', balance=False):
    """
    Reformat dataframe of trading records like the following:
                                           dt Type Type2
        0             2017-08-10 09:15:03    0  Hold
        1      2017-08-10 09:15:03.500000    0  Hold
        2             2017-08-10 09:15:04    0  Hold
        3      2017-08-10 09:15:04.500000    0  Hold
        4             2017-08-10 09:15:05    0  Hold
        5      2017-08-10 09:15:05.500000    0  Hold

    It must have a column with 'Enter' and 'Exit' elements in it.

    :param tradingRec: dataframe
    :param id_coln: the column with 'Enter' and 'Exit'
    :param balance: optional. if balance is True, the last row (if the length
                    is not even) will be remove.
    :return: dataframe
    """
    trec = tradingRec.copy()
    if balance:
        if trec['Type'].sum() != 0:
            trec = trec.iloc[:-1, :]
    trec = trec.set_index(id_coln, append=True).unstack(
        id_coln, fill_value=np.nan)
    trec.columns = [s0 + '_' + s1 for s0, s1 in trec.columns]
    en = trec.filter(like='Enter').sort_values(['dt_Enter', 'Type_Enter'])
    ex = trec.filter(like='Exit').sort_values(['dt_Exit', 'Type_Exit'])
    trec_arr = np.concatenate([en, ex], 1)
    trec_coln = np.append(en.columns, ex.columns)
    trec = pd.DataFrame(trec_arr, columns=trec_coln).dropna(how='all')
    return trec


def trigger_sim(tradingRec, tradescol, freq='1d', verbose=True):
    """Random trigger simulation

    Args:
        tradingRec (TYPE): Description
        tradescol (TYPE): Description
        freq (str, optional): Description
    """
    tr_col = tradingRec.groupby(pd.Grouper(freq=freq)).size()
    tr_col = tr_col[tr_col != 0]
    sim_trades = []
    for dc, n_tr in tr_col.items():
        date = dc.strftime('%Y%m%d')
        rawdata = DB.getBar(code=underly, beginDate=date, fields=FIELDS).toDataFrame()
        clean_data = gfc.clean_rdata(rawdata, exchange=mye)
        p = clean_data[['ask', 'bid', 'c']].between_time('09:00:00', COEREXT)

        real_tr = trarec0[date]
        time_offset = pd.to_timedelta(135, 'm') / n_tr
    #    time_offset = pd.to_timedelta(0)
        hp_th = np.mean(real_tr['dt_Exit'] - real_tr['dt_Enter'])
        empty_th = np.mean(real_tr['dt_Enter'] - real_tr['dt_Exit'].shift()) - time_offset

        cnt, t1 = 0, p.iloc[0].name
        sim_trade_container = []
        attempt_i = 0
        while cnt < n_tr:
            try:
                rest_of_sample = p[date].loc[t1:t1 + hp_th + empty_th, :].iloc[1:, :]
                sim_trade = rest_of_sample.sample(2).sort_index()
            except ValueError:
                #            print 'failed at {}: '.format(cnt), t0, t1, 'period too long, repeat...'
                #            print '\n\n'
                t0 = np.nan
                cnt, t1 = 0, p.iloc[0].name
                sim_trade_container = []
                attempt_i += 1
                continue

            en_tr, ex_tr = sim_trade.reset_index().values
            t0, t1 = en_tr[0], ex_tr[0]
            cnt += 1

            sim_direc = np.random.binomial(1, 0.5)
            sim_trade['Type'] = [sim_direc * 2 - 1, 1 - sim_direc * 2]
            sim_trade_container.append(sim_trade)
    #        print cnt, t0, t1, 'appended one, next...'
        sim_trades.append(pd.concat(sim_trade_container))
        if verbose:
            print date, ': ', attempt_i, 'attempts; ', n_tr, 'trades;'
            print '\n\n'
    return gbt.reformat_trec(sim_trades)
