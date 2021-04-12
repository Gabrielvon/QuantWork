# -*- coding: utf-8 -*-

"""
基於`factor_update_full.py'的因子更新數據，進行對應時長的回測。

步驟如下
假設當下是t日，每n天更新一次。t-n日是上一次更新因子。
- 通過文件讀取t日更新的POOL
- 通過因子公式計算因子值
- 基於因子值得出買賣信號
- 循環
"""

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 有中文出现的情况，需要u'内容'

import os

os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np
import pandas as pd
import datetime
from copy import deepcopy
from tqdm import tqdm
import jqfunc as jqf
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from gplearn_functions import function_set_dict, _generate_signal


def get_stock_data(filepath):
    symbol = filepath.split('/')[-1]
    out = pd.read_parquet(filepath).assign(code='.'.join(symbol.split('.')[:2]))
    return out


class alpha_pool(object):

    def __init__(self, industry_symbol_arr, stock_symbol_arr, function_set_dict, verbose=0):
        self.industry_symbol_arr = industry_symbol_arr
        self.stock_symbol_arr = stock_symbol_arr
        self.industry_alpha_pool = {sym: [] for sym in industry_symbol_arr}
        self.stock_alpha_pool = {sym: [] for sym in stock_symbol_arr}
        self.function_set_dict = function_set_dict
        self.verbose = verbose

    def set_strategy(self, n, q_lower, q_upper, flag):
        self.strategy_kwargs = dict(
            n=n,
            q_lower=q_lower,
            q_upper=q_upper,
            flag=flag,
        )

    def _get_alpha_signal(self, expression, dataframe):
        function_set_dict = deepcopy(self.function_set_dict)
        factor_values = eval(expression, function_set_dict, dataframe.to_dict(orient="series"))
        signal = _generate_signal(factor_values, **self.strategy_kwargs)
        return signal

    def _get_alpha_avg_return(self, expression, dataframe, logret):
        function_set_dict = deepcopy(self.function_set_dict)
        factor_values = eval(expression, function_set_dict, dataframe.to_dict(orient="series"))
        signal = _generate_signal(factor_values, **self.strategy_kwargs)
        avglogret = np.mean(signal * logret)
        return avglogret

    def get_new_alpha(self, filepath):
        new_alpha = pd.read_csv(filepath).iloc[:, 1:]
        new_alpha['code'] = new_alpha['code'].astype(str)
        new_alpha = new_alpha.set_index('code').iloc[:, 6:].T.to_dict(orient='list')
        return new_alpha

    def update_industry_alpha_pool(self, dataset, new_industry_alpha, n_jobs=1):
        def _func(co):
            logret = np.log(dataset['close_' + co]).diff().shift(-1)
            unique_expressions = np.unique(self.industry_alpha_pool[co] + new_industry_alpha[co])
            result_list = []
            for expr in unique_expressions:
                try:
                    factor_return = self._get_alpha_avg_return(expr, dataset, logret)
                except Exception as e:
                    factor_return = np.nan
                    if self.verbose >= 10:
                        print('[WARNING] {} got invalid expression --- ${}$'.format(co, expr))
                result_list.append([co, expr, factor_return])
            return result_list

        rs = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='32M', verbose=self.verbose)(
            delayed(_func)(co) for co in self.industry_symbol_arr)
        return_list = np.vstack(rs)
        # return_list = np.vstack([_func(co) for co in self.industry_symbol_arr])
        industry_alpha = pd.DataFrame(return_list, columns=['code', 'expr', 'fr'])
        return industry_alpha

    def update_stock_alpha_pool(self, dataset, new_stock_alpha, n_jobs=1):

        def _func(co):
            _tmp_stock_data = dataset[dataset['code'] == co].copy()
            logret = np.log(_tmp_stock_data['close']).diff().shift(-1)
            unique_expressions = np.unique(self.stock_alpha_pool[co] + new_stock_alpha[co])
            result_list = []
            for expr in unique_expressions:
                try:
                    factor_return = self._get_alpha_avg_return(expr, _tmp_stock_data, logret)
                except Exception as e:
                    factor_return = np.nan
                    if self.verbose >= 10:
                        print('[WARNING] {} got invalid expression --- ${}$'.format(co, expr))
                result_list.append([co, expr, factor_return])
            return result_list

        rs = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='8M', verbose=self.verbose)(
            delayed(_func)(co) for co in self.stock_symbol_arr)
        return_list = np.vstack(rs)
        # return_list = np.vstack([_func(co) for co in self.stock_symbol_arr])
        stock_alpha = pd.DataFrame(return_list, columns=['code', 'expr', 'fr'])
        return stock_alpha

    def get_industry_signal(self, dataset):
        def _get_industry_signal(industry_alpha_pool):
            for co, expressions in industry_alpha_pool.items():
                for expr in expressions:
                    try:
                        signal = self._get_alpha_signal(expr, dataset)
                    except Exception as e:
                        signal = [0]
                        if self.verbose >= 10:
                            print('[WARNING] {} got invalid expression --- ${}$'.format(co, expr))

                    yield [co, expr, signal[-1]]

        industry_pred_info = pd.DataFrame(_get_industry_signal(self.industry_alpha_pool),
                                          columns=['code', 'expr', 'opinion'])
        industry_pred_info = industry_pred_info.rename(columns={c: 'industry_' + c for c in industry_pred_info.columns})
        industry_pred_info['industry_code'] = industry_pred_info['industry_code'].astype(str)
        return industry_pred_info

    def get_stock_signal(self, dataset):
        def _get_stock_signal(stock_alpha_pool):
            for co, expressions in stock_alpha_pool.items():
                _tmp_stock_data = dataset[dataset['code'] == co].copy()

                for expr in expressions:
                    try:
                        signal = self._get_alpha_signal(expr, _tmp_stock_data)
                    except Exception as e:
                        signal = [0]
                        if self.verbose >= 10:
                            print('[WARNING] {} got invalid expression --- ${}$'.format(co, expr))
                    yield [co, expr, signal[-1]]

        stock_pred_info = pd.DataFrame(_get_stock_signal(self.stock_alpha_pool), columns=['code', 'expr', 'opinion'])
        stock_pred_info = stock_pred_info.rename(columns={c: 'stock_' + c for c in stock_pred_info.columns})
        stock_pred_info['stock_code'] = stock_pred_info['stock_code'].astype(str)
        return stock_pred_info


start_date = '2016-02-01'
end_date = '2021-02-26'
result_type = 'sharpe'
strat_signal_ref = 60
strat_q_lower = 0.2
strat_q_upper = 0.8
strat_flag = 1
factor_filter = (10, 5)
industry_code = "sw_l1"
n_jobs = 50
update_lookback = 252

start_date_int = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
end_date_int = int(pd.to_datetime(end_date).strftime('%Y%m%d'))

industry_data_directory = f'./dataset/data/industry_{start_date}_{end_date}'
stock_data_directory = f'./dataset/data/hs300_{start_date}_{end_date}'
ga_result_direcotry_parent = f'ga_results/{result_type}'
ga_result_direcotry_name = f'ga_results_{start_date}_{end_date}'
ga_result_directory = f'dataset/{ga_result_direcotry_parent}/{ga_result_direcotry_name}'
ga_result_cummulative_direcotry_parent = f'ga_results/{result_type}'
ga_result_cummulative_direcotry_name = f'ga_result_cummulative_{start_date}_{end_date}'
ga_result_cummulative_directory = f'dataset/{ga_result_cummulative_direcotry_parent}/{ga_result_cummulative_direcotry_name}'
try:
    os.makedirs(ga_result_directory, exist_ok=False)
except FileExistsError as e:
    print(e)
    print('Direcotry [{}] is already existed with {} data files.'.format(ga_result_cummulative_directory,
                                                                         len(os.listdir(
                                                                             ga_result_cummulative_directory))))

print("Loading industry data...")
industry_data = pd.read_parquet(f'{industry_data_directory}/{industry_code}.parq')
industry_symbol_arr = np.array([c.split('_')[-1] for c in industry_data.filter(regex='volume').columns])

print("Loading stock data...")


stock_train_fields = ['open', 'high', 'low', 'close', 'volume', 'money', 'avg']
stock_train_fields2 = ['net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']
stock_data_filenames = [f'{stock_data_directory}/{fn}' for fn in os.listdir(stock_data_directory)]
stock_data_list = Parallel(n_jobs=8)(delayed(get_stock_data)(fn) for fn in stock_data_filenames)
stock_data = pd.concat(stock_data_list)
stock_data[stock_train_fields + stock_train_fields2] = stock_data[stock_train_fields + stock_train_fields2].ffill()
stock_data = stock_data.reset_index()
stock_data = stock_data[stock_data['date'].isin(industry_data.index)]
stock_symbol_arr = stock_data['code'].values

stock_industry_raw = jqf.get_industry(list(stock_symbol_arr))
stock_industry_list = [[co, int(v['sw_l1']['industry_code'])] if 'sw_l1' in v else [co, '000000'] for co, v in
                       stock_industry_raw.items()]
stock_industry = pd.DataFrame(stock_industry_list, columns=['stock_code', 'industry_code'], dtype=str)

ga_result_filenames = os.listdir(ga_result_directory)
if len(ga_result_filenames) % 2 != 0:
    print('[WARNING] ga_result_filenames may be missing.')
    file_pairs = np.reshape(np.concatenate([np.sort(ga_result_filenames), ['dummy']]),
                            (2, (len(ga_result_filenames) + 1) // 2)).T
    file_pairs = file_pairs[:-1, :]
else:
    file_pairs = np.reshape(np.sort(ga_result_filenames), (2, len(ga_result_filenames) // 2)).T
file_pairs = {int(fp[0].split('_')[-1].split('.')[0]): fp for fp in file_pairs}
all_trade_days = jqf.get_trade_days(start_date=start_date, end_date=end_date)
all_trade_days_int = np.array([int(dt.strftime("%Y%m%d")) for dt in all_trade_days])

# get opinion results
opinion_df_list = []
ap = alpha_pool(industry_symbol_arr, stock_symbol_arr, function_set_dict, verbose=0)
ap.set_strategy(n=strat_signal_ref, q_lower=strat_q_lower, q_upper=strat_q_upper, flag=strat_flag)
_opinion_start_dates_int = all_trade_days_int[all_trade_days_int >= min(file_pairs.keys())]
iterobj = tqdm(enumerate(zip(_opinion_start_dates_int, _opinion_start_dates_int[1:])),
               total=len(_opinion_start_dates_int) - 1, desc='BACKTEST', position=1)
for idt, (predt, dt) in iterobj:
    is_update_pool = predt in file_pairs
    iterobj.set_postfix({'runtime': datetime.datetime.now(), 'date_update': str(predt), 'date_pred': str(dt),
                         'update_pool': is_update_pool})

    # -------------------------------------------------------------------------------- #
    # Update alpha pool
    # -------------------------------------------------------------------------------- #
    if is_update_pool:
        industry_alpha_filepath, stock_alpha_filepath = file_pairs[predt]
        new_industry_alpha = ap.get_new_alpha(f'{ga_result_directory}\{industry_alpha_filepath}')
        new_stock_alpha = ap.get_new_alpha(f'{ga_result_directory}\{stock_alpha_filepath}')

        _source_dt = all_trade_days_int[list(all_trade_days_int).index(predt) - update_lookback]
        pre_industry_data = industry_data.loc[_source_dt:predt, :]
        pre_stock_data = stock_data[(stock_data['date'] >= _source_dt) & (stock_data['date'] <= predt)]

        # evaluate new batch of soldiers and update old batch
        # --------------------------------------------------------------------------------
        selected_industry_alpha = pd.read_csv(
            f'{ga_result_cummulative_directory}\industry_{start_date_int}_{predt}.csv', index_col='code')
        selected_industry_alpha = selected_industry_alpha.iloc[:, 0].map(eval).rename(predt)
        ap.industry_alpha_pool.update(deepcopy(selected_industry_alpha))

        # evaluate new batch of soldiers and update old batch
        # --------------------------------------------------------------------------------
        selected_stock_alpha = pd.read_csv(f'{ga_result_cummulative_directory}\stock_{start_date_int}_{predt}.csv',
                                           index_col='code')
        selected_stock_alpha = selected_stock_alpha.iloc[:, 0].map(eval).rename(predt)
        ap.stock_alpha_pool.update(deepcopy(selected_stock_alpha))

    industry_pool_size = len(np.hstack(ap.industry_alpha_pool.values()))
    stock_pool_size = len(np.hstack(ap.stock_alpha_pool.values()))
    if (industry_pool_size == 0) or (stock_pool_size == 0):
        continue

    # -------------------------------------------------------------------------------- #
    # Organizing and trade
    # -------------------------------------------------------------------------------- #
    filepath_opinions = f'{ga_result_cummulative_directory}\opinion_{start_date_int}_{predt}.csv'
    if os.path.basename(filepath_opinions) in os.listdir(os.path.dirname(filepath_opinions)):
        # get result from pre-computed files
        print(f'opinion_{start_date_int}_{predt}.csv existed, read from storage:\t', filepath_opinions)
        opinion_df = pd.read_csv(f'{ga_result_cummulative_directory}/opinion_{start_date_int}_{predt}.csv').iloc[:, 1:]
    else:
        # Make prediction
        _d0 = all_trade_days_int[list(all_trade_days_int).index(predt) - strat_signal_ref * 2]
        cur_industry_data = industry_data.loc[_d0:predt, :]
        cur_stock_data = stock_data[(stock_data['date'] >= _d0) & (stock_data['date'] <= predt)]
        industry_pred_info = ap.get_industry_signal(cur_industry_data)
        stock_pred_info = ap.get_stock_signal(cur_stock_data)

        # get result from computing
        industry_opinion = industry_pred_info.groupby('industry_code')['industry_opinion'].sum()
        stock_opinion = stock_pred_info.groupby('stock_code')['stock_opinion'].sum()
        opinion_df = pd.merge(industry_opinion, stock_industry, on='industry_code', how='outer')
        opinion_df = pd.merge(opinion_df, stock_opinion, on=['stock_code'], how='outer')
        opinion_df = opinion_df.dropna(subset=['industry_opinion', 'stock_opinion']).assign(date=dt)
        opinion_df.to_csv(f'{ga_result_cummulative_directory}/opinion_{start_date_int}_{predt}.csv')
    opinion_df_list.append(opinion_df)

print("from {} to {}: {} updates.".format(_opinion_start_dates_int[0], _opinion_start_dates_int[-1],
                                          len(opinion_df_list)))

all_opinion_df = pd.concat(opinion_df_list)
all_opinion_df['signal'] = np.nan
idx_long = (all_opinion_df['industry_opinion'] >= 3) & (all_opinion_df['stock_opinion'] >= 2)
idx_short = (all_opinion_df['industry_opinion'] <= -3) & (all_opinion_df['stock_opinion'] <= -2)
# idx_long = all_opinion_df['industry_opinion'] >= 4
# idx_short = all_opinion_df['industry_opinion'] <= -4
# idx_long = all_opinion_df['stock_opinion'] >= 1
# idx_short = all_opinion_df['stock_opinion'] <= -1
all_opinion_df.loc[idx_long, 'signal'] = 1
all_opinion_df.loc[idx_short, 'signal'] = -1
print(all_opinion_df['signal'].value_counts())

test_date_int = all_opinion_df['date'].min()
test_date = str(test_date_int)[:4] + '-' + str(test_date_int)[4:6] + '-' + str(test_date_int)[6:]

signal = all_opinion_df.pivot('date', 'stock_code', 'signal').shift()  # 前一天晚上出的信号
logret = np.log(stock_data.pivot('date', 'code', 'close')).diff()  # 前一天收盘价交易/当天收盘价交易

signal_each = signal.loc[test_date_int:, :].copy()
logret_each = logret.loc[test_date_int:, :].copy()

logret_bt_each = signal_each * logret_each
logret_bt = logret_bt_each.mean(axis=1).fillna(0)
cumret_bt = logret_bt.cumsum()

bm = jqf.get_price('000300.XSHG', start_date=test_date, end_date=end_date)
bm.index = bm.index.strftime('%Y%m%d').astype(int)
cumret_bm = np.log(bm['close']).diff().cumsum()

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(cumret_bt.values, marker='.')
axs[0].plot(cumret_bm.values, marker='.')
axs[0].plot(cumret_bt.values - cumret_bm.values, marker='.')
axs[0].grid(True)
axs[0].legend(['portfolio', 'benchmark', 'er'])
# signal_each.apply(pd.value_counts).T.hist()
axs[1].plot(np.sum(signal_each.values == 1, axis=1), marker='.')
axs[1].plot(np.sum(signal_each.values == -1, axis=1), marker='.')
axs[1].grid(True)
axs[1].legend(["long", "short"])

# presentation
strategy = logret_bt.to_frame('s1')
strategy['s1_cumsum'] = strategy['s1'].cumsum()
strategy['s1_cumsum_ma30'] = strategy['s1_cumsum'].rolling(30).mean()
strategy['s2'] = strategy['s1'].copy()
strategy.loc[strategy['s1_cumsum'] < strategy['s1_cumsum_ma30'], 's2'] = 0
# strategy.loc[(strategy['s1_cumsum'] < strategy['s1_cumsum_ma30']).shift().fillna(False), 's2'] = 0
strategy['s2_cumsum'] = strategy['s2'].cumsum()
strategy['bm300'] = cumret_bm
strategy['bm1'] = logret_each.mean(1).cumsum()
ax = strategy[['s1_cumsum', 's1_cumsum_ma30', 's2_cumsum', 'bm300', 'bm1']].reset_index(drop=True).plot.line(grid=True)
ax.legend(['原始策略累计收益', '原始策略累计收益均线', '原始策略（止损）累计收益', '沪深300累计收益', '选股累计收益均线'])

# final accuracy
daily_ret = (signal * logret)
non_zero_count = signal.notna().sum(axis=1)
win_ratio_daily = (daily_ret > 0).sum(axis=1) / non_zero_count
industry_stock_result = pd.concat([non_zero_count, win_ratio_daily, daily_ret.sum(1) / non_zero_count], axis=1,
                                  keys=['non_zero_cnt', 'win_pct', 'avgret'])
industry_stock_result = industry_stock_result.loc[test_date_int:].copy()
industry_stock_result['cumavgret'] = industry_stock_result['avgret'].cumsum()
avg_win_pct = industry_stock_result['win_pct'].mean()
industry_stock_result.reset_index(drop=True)[['non_zero_cnt', 'win_pct', 'cumavgret']].plot.line(subplots=True,
                                                                                                 grid=True, marker='.',
                                                                                                 title="avg win pct: {:.4f}".format(
                                                                                                     avg_win_pct))
