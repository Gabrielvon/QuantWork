# -*- coding: utf-8 -*-

"""
基於`factor_mining_mp.py`挖掘的因子，進行因子迭代更新，并且保存到本地

步驟如下
假設當下是t日，每n天更新一次。t-n日是上一次更新因子。
- 基於第t-n日獲得的因子構建現有因子庫(POOL)中；
- t日是獲得新因子，添加入POOL中；
- 基於t-252日到t日的數據對POOL中所有因子進行評估；
- 篩選出符合條件的因子組成新的股票池；
- 循環


"""
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import jqfunc as jqf
from gplearn_functions import _generate_signal
from gplearn_functions import function_set_dict

import os

os.environ['NUMEXPR_MAX_THREADS'] = '8'


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
# end_date = str(jqf.get_trade_days(count=1)[0])
result_type = 'sharpe'
strat_signal_ref = 60
strat_q_lower = 0.2
strat_q_upper = 0.8
strat_flag = 1
factor_filter = (10, 5)
industry_code = "sw_l1"
n_jobs = 32
update_lookback = 252
overwrite_result = True

start_date_int = int(pd.to_datetime(start_date).strftime('%Y%m%d'))
end_date_int = int(pd.to_datetime(end_date).strftime('%Y%m%d'))

industry_data_directory = f'dataset/data/industry_{start_date}_{end_date}'
stock_data_directory = f'dataset/data/hs300_{start_date}_{end_date}'
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
    print('Direcotry [{}] is already existed with {} data files.'.format(ga_result_directory,
                                                                         len(os.listdir(ga_result_directory))))
try:
    os.makedirs(ga_result_cummulative_directory, exist_ok=False)
except FileExistsError as e:
    print(e)
    print('Direcotry [{}] is already existed with {} data files.'.format(ga_result_cummulative_directory,
                                                                         len(os.listdir(
                                                                             ga_result_cummulative_directory))))

print("Loading industry data...")
industry_data = pd.read_parquet(f'{industry_data_directory}/{industry_code}.parq')
industry_symbol_arr = np.array([c.split('_')[-1] for c in industry_data.filter(regex='volume').columns])

print("Loading stock data...")


def _get_stock_data(filepath):
    symbol = filepath.split('/')[-1]
    out = pd.read_parquet(filepath).assign(code='.'.join(symbol.split('.')[:2]))
    return out


stock_train_fields = ['open', 'high', 'low', 'close', 'volume', 'money', 'avg']
stock_train_fields2 = ['net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']
stock_data_filenames = [f'{stock_data_directory}/{fn}' for fn in os.listdir(stock_data_directory)]
stock_data_list = Parallel(n_jobs=n_jobs)(delayed(_get_stock_data)(fn) for fn in stock_data_filenames)
stock_data = pd.concat(stock_data_list)
stock_data[stock_train_fields + stock_train_fields2] = stock_data[stock_train_fields + stock_train_fields2].ffill()
stock_data = stock_data.reset_index()
stock_data = stock_data[stock_data['date'].isin(industry_data.index)]
stock_symbol_arr = np.unique(stock_data['code'].values)

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

# get result
opinion_df_list = []
ap = alpha_pool(industry_symbol_arr, stock_symbol_arr, function_set_dict, verbose=0)
ap.set_strategy(n=strat_signal_ref, q_lower=strat_q_lower, q_upper=strat_q_upper, flag=strat_flag)
iterobj = tqdm(enumerate(file_pairs.items()), total=len(file_pairs), desc='FACTOR UPDATE', position=1)
for idt, (predt, fps) in iterobj:

    iterobj.set_postfix({'runtime': datetime.datetime.now(), 'date_update': str(predt)})

    # -------------------------------------------------------------------------------- #
    # Update alpha pool
    # -------------------------------------------------------------------------------- #

    # prepare
    industry_alpha_filepath, stock_alpha_filepath = fps
    new_industry_alpha = ap.get_new_alpha(f'{ga_result_directory}/{industry_alpha_filepath}')
    new_stock_alpha = ap.get_new_alpha(f'{ga_result_directory}/{stock_alpha_filepath}')

    _source_dt = all_trade_days_int[list(all_trade_days_int).index(predt) - update_lookback]
    pre_industry_data = industry_data.loc[_source_dt:predt, :]
    pre_stock_data = stock_data[(stock_data['date'] >= _source_dt) & (stock_data['date'] <= predt)]

    # evaluate new batch of soldiers and update old batch
    # --------------------------------------------------------------------------------
    print('\n')

    filepath_selected_industry_alpha = f'{ga_result_cummulative_directory}/industry_{start_date_int}_{predt}.csv'
    if os.path.basename(filepath_selected_industry_alpha) in os.listdir(
            os.path.dirname(filepath_selected_industry_alpha)):
        print('industry_alpha_pool existed, read from storage:\t', filepath_selected_industry_alpha)
        selected_industry_alpha = pd.read_csv(filepath_selected_industry_alpha, index_col='code')
        selected_industry_alpha = selected_industry_alpha.iloc[:, 0].map(eval).rename(predt)
        ap.industry_alpha_pool.update(deepcopy(selected_industry_alpha))
    else:
        t0 = datetime.datetime.now()
        industry_alpha = ap.update_industry_alpha_pool(pre_industry_data, new_industry_alpha, n_jobs=n_jobs)
        print('update_industry_alpha_pool:\t', datetime.datetime.now() - t0)

        selected_industry_alpha = industry_alpha.groupby('code').apply(
            lambda x: x.sort_values('fr').iloc[-factor_filter[1]:]['expr'].tolist())
        selected_industry_alpha.to_csv(filepath_selected_industry_alpha)
        ap.industry_alpha_pool.update(deepcopy(selected_industry_alpha))

    # evaluate new batch of soldiers and update old batch
    # --------------------------------------------------------------------------------
    filepath_selected_stock_alpha = f'{ga_result_cummulative_directory}/stock_{start_date_int}_{predt}.csv'

    if os.path.basename(filepath_selected_stock_alpha) in os.listdir(os.path.dirname(filepath_selected_stock_alpha)):
        print('stock_alpha_pool existed, read from storage:\t', filepath_selected_stock_alpha)
        selected_stock_alpha = pd.read_csv(filepath_selected_stock_alpha, index_col='code')
        selected_stock_alpha = selected_stock_alpha.iloc[:, 0].map(eval).rename(predt)
        ap.stock_alpha_pool.update(deepcopy(selected_stock_alpha))
    else:
        t0 = datetime.datetime.now()
        stock_alpha = ap.update_stock_alpha_pool(pre_stock_data, new_stock_alpha, n_jobs=n_jobs)
        print('update_stock_alpha_pool:\t', datetime.datetime.now() - t0)

        selected_stock_alpha = stock_alpha.groupby('code').apply(
            lambda x: x.sort_values('fr').iloc[-factor_filter[1]:]['expr'].tolist())
        selected_stock_alpha.to_csv(filepath_selected_stock_alpha)
        ap.stock_alpha_pool.update(deepcopy(selected_stock_alpha))

# -------------------------------------------------------------------------------- #
# Monitor replacement rate
# -------------------------------------------------------------------------------- #
# ap = alpha_pool(industry_symbol_arr, stock_symbol_arr, function_set_dict, verbose=0)
# ap.set_strategy(n=strat_signal_ref, q_lower=strat_q_lower, q_upper=strat_q_upper, flag=strat_flag)
industry_alpha_batch = []
stock_alpha_batch = []
industry_replacement_result_list = []
stock_replacement_result_list = []
iterobj = tqdm(enumerate(zip(all_trade_days_int, all_trade_days_int[1:])), total=len(all_trade_days_int) - 1,
               desc='BACKTEST', position=1)
for idt, (predt, dt) in iterobj:
    is_update_pool = predt in file_pairs
    iterobj.set_postfix({'runtime': datetime.datetime.now(), 'date_update': str(predt), 'date_pred': str(dt),
                         'update_pool': is_update_pool})

    # -------------------------------------------------------------------------------- #
    # Update alpha pool
    # -------------------------------------------------------------------------------- #
    if is_update_pool:
        selected_industry_alpha = pd.read_csv(
            f'{ga_result_cummulative_directory}/industry_{start_date_int}_{predt}.csv', index_col='code')
        selected_industry_alpha = selected_industry_alpha.iloc[:, 0].map(eval).rename(predt)
        industry_alpha_batch.append(selected_industry_alpha)
        if len(industry_alpha_batch) > 1:
            for co in selected_industry_alpha.index:
                a = set(industry_alpha_batch[-2][co])
                b = set(industry_alpha_batch[-1][co])
                industry_replacement_result_list.append([co, predt, len(set(a) & set(b)) / 5])

        selected_stock_alpha = pd.read_csv(f'{ga_result_cummulative_directory}/stock_{start_date_int}_{predt}.csv',
                                           index_col='code')
        selected_stock_alpha = selected_stock_alpha.iloc[:, 0].map(eval).rename(predt)
        stock_alpha_batch.append(selected_stock_alpha)
        if len(stock_alpha_batch) > 1:
            for co in selected_stock_alpha.index:
                a = set(stock_alpha_batch[-2][co])
                b = set(stock_alpha_batch[-1][co])
                stock_replacement_result_list.append([co, predt, len(set(a) & set(b)) / 5])

industry_replacement = pd.DataFrame(industry_replacement_result_list, columns=['code', 'predt', 'rate'])
industry_replacement_piv = industry_replacement.pivot('predt', 'code', 'rate')
industry_replacement_piv.mean(axis=1).plot.bar()

stock_replacement = pd.DataFrame(stock_replacement_result_list, columns=['code', 'predt', 'rate'])
stock_replacement_piv = stock_replacement.pivot('predt', 'code', 'rate')
stock_replacement_piv.mean(axis=1).plot.bar()
