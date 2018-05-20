import os

import gabFunc as gfc
import gabBT as gbt
import gabTA as gta
# import gabperformance as gperf

import pandas as pd
import numpy as np
import time
# plt.interactive(True)
gfc.widen_pandas_display(50, 50, 150)
gfc.widen_numpy_display(6, 50, 50, 120)


# import and clean data
rawdata = pd.read_csv('test_stock_data_intraday.csv', index_col='timestamp', parse_dates=True)
clean_df = gfc.clean_rdata(rawdata)
clean_df.remove_by_time('sz')
clean_df.fix_items('sz', rename_items=False)
clean_df.fix_duplicated_index()
scaled_df = clean_df.scale_frequency('5min')
df = scaled_df.copy()
df['midprice'] = df[['ap1', 'bp1']].replace(0, np.nan).mean(1)

# Factors
win = 5
enter_th = (50, -50)
exit_th = (0, 0)
myc = 'stock'
signal_values = gta.calc_cmo(df['midprice'], win=win)
signal_values = pd.Series(np.hstack([np.zeros(win), signal_values]), index=df.index)

# In[]
# Trading signals
openlong = (signal_values > enter_th[0]) & (signal_values.shift() < enter_th[0])
openshort = (signal_values < enter_th[1]) & (signal_values.shift() > enter_th[1])
selllong = (signal_values < exit_th[0]) & (signal_values.shift() > exit_th[0])
buycover = (signal_values > exit_th[1]) & (signal_values.shift() < exit_th[1])

# tradingsignals = pd.DataFrame([openlong, openshort, selllong, buycover],
#                               index=['openlong', 'openshort', 'selllong', 'buycover'],
#                               columns=df.index[win:]).T
tradingsignals = pd.concat([openlong, openshort, selllong, buycover], axis=1,
                           keys=['openlong', 'openshort', 'selllong', 'buycover'])

abprice = df[['ap1', 'bp1']]
daily_container = []
bt_data = pd.concat([abprice, tradingsignals], 1).ffill().dropna()
bt_data[tradingsignals.columns] = bt_data[tradingsignals.columns].shift(2)
bt_data.dropna(inplace=True)
bt_data = bt_data.rename(columns={'ap1': 'ask', 'bp1': 'bid'})
daily_container.append([myc, bt_data])

colns, dfs = zip(*daily_container)
tradingsignals_df = pd.concat(dfs, 1, keys=colns, names=['underly', 'signals'])
tradingsignals_df = tradingsignals_df.ffill().dropna().stack('signals')

# In[]
# Backtest results
trarec = gbt.generate_trading_records(tradingsignals_df)
trarec['Underlying'] = 'stock'
trarec.drop(['underly_Enter', 'underly_Exit'], axis=1, inplace=True)
cn_ord_parts = ['dt_Enter', 'Price_Enter', 'Type_Enter', 'dt_Exit', 'Price_Exit', 'Type_Exit']
trarec = trarec[cn_ord_parts + [cn for cn in trarec.columns if cn not in cn_ord_parts]]

emat, trec = gbt.BT_wCost(trarec)
print '\nOverview Performance: \n', emat,
print '\nTrading Logs: \n', trec

# In[]
# # Performance
# import RiskAdjustedReturnMetrics as rarm

# rr = trec['RReturn'].values
# mr = np.random.uniform(-1, 1, trec.shape[0])

# # Expected return
# e = np.mean(rr)
# # Risk free rate
# f = 0.06
# # Alpha
# alpha_a = 0.05
# alpha_b = 0.20  # prepared for rachev_ratio
# # Periods
# peri = 5  # prepared for sterling ratio and burke ratio

# print 'test_risk_metrics: ', rarm.test_risk_metrics(rr, mr)
# print 'test_risk_adjusted_metrics: ', rarm.test_risk_adjusted_metrics(rr, mr, e, f)
