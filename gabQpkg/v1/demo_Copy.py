import sys
for i in sys.path:
    if 'RefPkg' in i:
        sys.path.remove(i)
sys.path.append('E:\MyNutshell\FanCapitial\Frameworks')

import gabFunc as gfc
import gabBT as gbt
import gabpycoral as gcrl
import gabperformance as gperf
import gabTA as gta

import pandas as pd
import numpy as np

# plt.interactive(True)
gfc.widen_pandas_display(50, 50, 150)
gfc.widen_numpy_display(6, 50, 50, 120)


# In[2]:

# initialize
DB = gcrl.db('external')
BEGIN_DATE, END_DATE = 20171227, 20171230

# VALID_TD = DB.getTradingDays(BEGIN_DATE, END_DATE).values
# VALID_TD = [i[0] for i in VALID_TD]
FIELDS = 'open,high,low,new_price,new_volume,new_amount,open_interest,ap1,bp1'

DB.query_init({'begin': BEGIN_DATE, 'end': END_DATE, 'fields': FIELDS, 'dtype': 'tick'})
DB.query.update({'code': '600036.SH'})
rawdata = DB.getBar(4)


# clean data
clean_df = gfc.clean_rawdata(rawdata)
clean_df.remove_by_time('sh')
clean_df.fix_items('sh', rename_items=False)
clean_df.fix_duplicated_index('sh')
scaled_df = clean_df.scale_frequency('5min')
df = scaled_df.copy()
df['midprice'] = df[['ap1', 'bp1']].mean(1)

# init
daily_container = []

# Factors
win = 5
enter_th = (50, -50)
exit_th = (0, 0)
myc = DB.query['code']
signal_values = gta.calc_cmo(df['midprice'], win=win)
signal_values = pd.Series(np.hstack([np.zeros(win), signal_values]), index=df.index)
                
# Trading signals
openlong = (signal_values > enter_th[0]) & (signal_values.shift() < enter_th[0])
openshort = (signal_values < enter_th[1]) & (signal_values.shift() > enter_th[1])
selllong = (signal_values < exit_th[0]) & (signal_values.shift() > exit_th[0])
buycover = (signal_values > exit_th[1]) & (signal_values.shift() < exit_th[1])

tradingsignals = pd.DataFrame([openlong, openshort, selllong, buycover],
                              index=['openlong', 'openshort', 'selllong', 'buycover'],
                              columns=df.index[win:]).T

abprice = df[['ap1', 'bp1']]

bt_data = pd.concat([abprice, tradingsignals], 1).ffill().dropna()
bt_data[tradingsignals.columns] = bt_data[tradingsignals.columns].shift(2)
bt_data.dropna(inplace=True)
bt_data = bt_data.rename(columns={'ap1':'ask', 'bp1':'bid'})
daily_container.append([myc, bt_data])

colns, dfs = zip(*daily_container)
tradingsignals_df = pd.concat(dfs, 1, keys=colns, names=['underly', 'signals'])
tradingsignals_df = tradingsignals_df.ffill().dropna().stack('signals')

# Backtest results
trarec = gbt.generate_trading_records(tradingsignals_df)
trarec['Underlying'] = 'stock'
trarec.drop(['underly_Enter', 'underly_Exit'], axis=1, inplace=True)
cn_ord_parts = ['dt_Enter', 'Price_Enter', 'Type_Enter', 'dt_Exit', 'Price_Exit', 'Type_Exit']
trarec = trarec[cn_ord_parts + [cn for cn in trarec.columns if cn not in cn_ord_parts]]

trarec_res = gbt.BT_wCost(trarec)
