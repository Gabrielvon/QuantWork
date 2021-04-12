# -*- coding: utf-8 -*-

"""
挖掘因子，並保存因子數據到`dataset/ga_results`中

原始特徵：个股自己+个股自己的资金流（不参考其他资金流）
"""
import os

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer

from gplearn_functions import function_set, function_set_dict, _generate_signal, clean_gplearn_programs
from gplearn_functions import gp_sharpe


def run_ga_industry(industry_sym, industry_code, data_directory, start_date_int, end_date_int, metric, signal_ref_data,
                    q_lower, q_upper, strat_flag, population_size, tournament_size, generations,
                    hall_of_fame, n_components, factor_filter, n_jobs=1, verbose=0):
    industry_data = pd.read_parquet(f'{data_directory}/{industry_code}.parq')
    industry_data = industry_data.loc[start_date_int:end_date_int, :].copy()
    data0 = industry_data.copy()
    data0['pct1'] = np.log(data0['close_' + industry_sym]).diff().shift(-1)
    dataset = data0.dropna()
    data = dataset.drop('pct1', axis=1).values
    ga_train_fields = dataset.drop('pct1', axis=1).columns

    target = dataset['pct1'].values
    test_size = 0.1
    test_num = int(len(data) * test_size)

    X_train = data[:-test_num].copy()
    X_train_df = dataset[ga_train_fields].iloc[:-test_num].copy()
    # X_train = ut.min_max_scaling(X_train)
    y_train = np.nan_to_num(target[:-test_num].copy())

    test_backward_i0 = test_num + signal_ref_data - 1
    # X_test = data[-test_backward_i0:].copy()
    X_test_df = dataset[ga_train_fields].iloc[-test_backward_i0:].copy()
    # X_test = ut.min_max_scaling(X_test)
    # y_test = np.nan_to_num(target[-test_backward_i0:].copy())

    # ================================================================================
    # Fitting
    # --------------------------------------------------------------------------------
    # SymbolicTransformer
    est_gp = SymbolicTransformer(
        population_size=population_size,  # 1000
        tournament_size=tournament_size,  # 20
        generations=generations,  # 20
        hall_of_fame=hall_of_fame,  # 100
        n_components=n_components,  # 10
        stopping_criteria=np.inf,  # 1.0
        const_range=None,  # (-1., 1.)
        # init_depth=(2, 6),    # (2, 6)
        # init_method='half and half',    # 'half and half'
        function_set=function_set,  # ('add', 'sub', 'mul', 'div')
        metric=metric,  # 'pearson'
        # metric=gp_sharpe,
        parsimony_coefficient=0.0001,  # 0.001
        # p_crossover=0.9,    # 0.9
        # p_subtree_mutation=0.01,    # 0.01
        # p_hoist_mutation=0.01,    # 0.01
        # p_point_mutation=0.01,    # 0.01
        # p_point_replace=0.05,    # 0.05
        max_samples=1.0,  # 1.0 || The fraction of samples to draw from X to evaluate each program on.
        feature_names=ga_train_fields,  # None
        # warm_start=False,    # False
        # low_memory=False,    # False
        n_jobs=n_jobs,  # 1
        verbose=verbose,  # 0
        random_state=10,  # None
    )
    est_gp.fit(X_train, y_train, sample_weight=None)

    # ================================================================================
    # Process programs
    # --------------------------------------------------------------------------------
    program_df = clean_gplearn_programs(est_gp._programs, verbose=0)
    function_expressions = program_df['expression'].values

    # ================================================================================
    # Backtest Overview
    # --------------------------------------------------------------------------------
    # train set
    logret = dataset.iloc[:-test_num]['pct1'].values

    alpha_train_overview_list = []
    for expr in function_expressions:
        factor_values = eval(expr, function_set_dict, X_train_df.to_dict(orient="series"))
        signal = _generate_signal(factor_values, n=signal_ref_data, q_lower=q_lower, q_upper=q_upper, flag=strat_flag)
        factor_return = np.sum(signal * logret)
        alpha_train_overview_list.append([expr, factor_return])

    train_ov = pd.DataFrame(alpha_train_overview_list, columns=['expression', 'totret_is']).set_index('expression')
    best_train_factor = train_ov.sort_values('totret_is').iloc[-factor_filter[0]:].index.tolist()

    # test set
    logret = dataset.iloc[-test_backward_i0:]['pct1'].values
    alpha_test_overview_list = []
    for expr in best_train_factor:
        factor_values = eval(expr, function_set_dict, X_test_df.to_dict(orient="series"))
        signal = _generate_signal(factor_values, n=signal_ref_data, q_lower=q_lower, q_upper=q_upper, flag=strat_flag)
        factor_return = np.sum(signal * logret)
        alpha_test_overview_list.append([expr, factor_return])

    test_ov = pd.DataFrame(alpha_test_overview_list, columns=['expression', 'totret_oos']).set_index('expression')
    best_factors = test_ov.sort_values("totret_oos").iloc[-factor_filter[1]:].index.tolist()

    _ref_data = industry_data.iloc[-signal_ref_data * 2:].to_dict(orient="series")
    best_opinions = [
        _generate_signal(eval(expr, function_set_dict, _ref_data), n=signal_ref_data, q_lower=q_lower, q_upper=q_upper,
                         flag=strat_flag)[-1] for expr in best_factors]
    ew_opinion = np.sum(best_opinions)
    output = [industry_sym, ew_opinion, best_opinions, best_factors]
    return output


def run_ga_stock(stock_sym, data_directory, raw_train_fields, start_date_int, end_date_int, metric, signal_ref_data,
                 q_lower, q_upper, strat_flag, population_size, tournament_size, generations,
                 hall_of_fame, n_components, factor_filter, n_jobs=1, verbose=0):
    stock_data = pd.read_parquet(f'{data_directory}/{stock_sym}.parq')
    stock_data = stock_data.loc[start_date_int:end_date_int, :].copy()
    dataset = stock_data.dropna()
    data = dataset[raw_train_fields].values
    ga_train_fields = dataset[raw_train_fields].columns

    target = dataset['pct1'].values
    test_size = 0.1
    test_num = int(len(data) * test_size)

    X_train = data[:-test_num].copy()
    X_train_df = dataset[ga_train_fields].iloc[:-test_num].copy()
    # X_train = ut.min_max_scaling(X_train)
    y_train = np.nan_to_num(target[:-test_num].copy())

    test_backward_i0 = test_num + signal_ref_data - 1
    # X_test = data[-test_backward_i0:].copy()
    X_test_df = dataset[ga_train_fields].iloc[-test_backward_i0:].copy()
    # X_test = ut.min_max_scaling(X_test)
    # y_test = np.nan_to_num(target[-test_backward_i0:].copy())

    # ================================================================================
    # Fitting
    # --------------------------------------------------------------------------------
    # SymbolicTransformer
    est_gp = SymbolicTransformer(
        population_size=population_size,  # 1000
        tournament_size=tournament_size,  # 20
        generations=generations,  # 20
        hall_of_fame=hall_of_fame,  # 100
        n_components=n_components,  # 10
        stopping_criteria=np.inf,  # 1.0
        const_range=None,  # (-1., 1.)
        # init_depth=(2, 6),    # (2, 6)
        # init_method='half and half',    # 'half and half'
        function_set=function_set,  # ('add', 'sub', 'mul', 'div')
        # metric='spearman',  # 'pearson'
        metric=metric,
        parsimony_coefficient=0.0001,  # 0.001
        # p_crossover=0.9,    # 0.9
        # p_subtree_mutation=0.01,    # 0.01
        # p_hoist_mutation=0.01,    # 0.01
        # p_point_mutation=0.01,    # 0.01
        # p_point_replace=0.05,    # 0.05
        max_samples=1.0,  # 1.0 || The fraction of samples to draw from X to evaluate each program on.
        feature_names=ga_train_fields,  # None
        # warm_start=False,    # False
        # low_memory=False,    # False
        n_jobs=n_jobs,  # 1
        verbose=verbose,  # 0
        random_state=10,  # None
    )
    try:
        est_gp.fit(X_train, y_train, sample_weight=None)
    except ValueError as e:
        print(stock_sym, " --- ", e)
    except TypeError as e:
        print(stock_sym, " --- ", e)
    finally:
        if len(X_train) < 60:
            return [stock_sym, np.nan, [], []]

    # ================================================================================
    # Process programs
    # --------------------------------------------------------------------------------
    program_df = clean_gplearn_programs(est_gp._programs, verbose=0)
    function_expressions = program_df['expression'].values

    # ================================================================================
    # Backtest Overview
    # --------------------------------------------------------------------------------

    # train set
    logret = dataset.iloc[:-test_num]['pct1'].values

    alpha_train_overview_list = []
    for expr in function_expressions:
        factor_values = eval(expr, function_set_dict, X_train_df.to_dict(orient="series"))
        signal = _generate_signal(factor_values, n=signal_ref_data, q_lower=q_lower, q_upper=q_upper, flag=strat_flag)
        factor_return = np.sum(signal * logret)
        alpha_train_overview_list.append([expr, factor_return])

    train_ov = pd.DataFrame(alpha_train_overview_list, columns=['expression', 'totret_is']).set_index('expression')
    best_train_factor = train_ov.sort_values('totret_is').iloc[-factor_filter[0]:].index.tolist()

    # test set
    logret = dataset.iloc[-test_backward_i0:]['pct1'].values
    alpha_test_overview_list = []
    for expr in best_train_factor:
        factor_values = eval(expr, function_set_dict, X_test_df.to_dict(orient="series"))
        signal = _generate_signal(factor_values, n=signal_ref_data, q_lower=q_lower, q_upper=q_upper, flag=strat_flag)
        factor_return = np.sum(signal * logret)
        alpha_test_overview_list.append([expr, factor_return])

    test_ov = pd.DataFrame(alpha_test_overview_list, columns=['expression', 'totret_oos']).set_index('expression')
    best_factors = test_ov.sort_values("totret_oos").iloc[-factor_filter[1]:].index.tolist()

    _ref_data = stock_data.iloc[-signal_ref_data * 2:].to_dict(orient="series")
    best_opinions = [
        _generate_signal(eval(expr, function_set_dict, _ref_data), n=signal_ref_data, q_lower=q_lower, q_upper=q_upper,
                         flag=strat_flag)[-1] for expr in best_factors]
    ew_opinion = np.sum(best_opinions)
    output = [stock_sym, ew_opinion, best_opinions, best_factors]
    return output


if __name__ == '__main__':

    from functools import partial
    from joblib import Parallel, delayed, parallel_backend
    import jqfunc as jqf
    from tqdm import tqdm

    start_date = '2016-02-01'
    end_date = '2021-02-26'
    result_type = 'sharpe'
    strat_signal_ref = 60
    q_lower = 0.2
    q_upper = 0.8
    strat_flag = 1
    factor_filter = (10, 5)
    industry_code = 'sw_l1'
    population_size = 1000
    generations = 5

    update_freq = 10
    update_len = 252

    if result_type == 'rankIC':
        my_metric = 'spearman'

    if result_type == 'sharpe':
        my_metric = gp_sharpe

    industry_data_directory = f'dataset/data/industry_{start_date}_{end_date}'
    stock_data_directory = f'dataset/data/hs300_{start_date}_{end_date}'
    ga_result_direcotry_parent = f'ga_results/{result_type}'
    ga_result_direcotry_name = f'ga_results_{start_date}_{end_date}'
    ga_result_directory = f'dataset/{ga_result_direcotry_parent}/{ga_result_direcotry_name}'
    try:
        os.makedirs(ga_result_directory, exist_ok=False)
    except FileExistsError as e:
        print(e)
        print('Direcotry [{}] is already existed with {} data files.'.format(ga_result_directory,
                                                                             len(os.listdir(ga_result_directory))))

    # get symbols for industries and stocks
    _tmp_data = pd.read_parquet(f'{industry_data_directory}/{industry_code}.parq')
    industry_symbol_arr = np.array([c.split('_')[-1] for c in _tmp_data.filter(regex='volume').columns])
    filenames = os.listdir(stock_data_directory)
    stock_symbol_arr = ['.'.join(fn.split('.')[:-1]) for fn in filenames if 'parq' in fn]

    # get biweek-dates
    all_trade_days = jqf.get_trade_days(start_date=start_date, end_date=end_date)
    all_trade_days_int = np.array([int(dt.strftime("%Y%m%d")) for dt in all_trade_days])
    biweek_dates = all_trade_days_int[-np.arange(-1, -len(all_trade_days), -update_freq)]
    biweek_date_start = all_trade_days_int[
        list(all_trade_days_int).index(int(''.join(end_date.split('-')))) - update_len]
    good_biweek_dates = biweek_dates[biweek_dates >= biweek_date_start]
    # dates_pairs = ((dt - 50000, dt) for dt in good_biweek_dates)
    dates_pairs = ((all_trade_days_int[0], dt) for dt in good_biweek_dates)

    for start_date_int, end_date_int in tqdm(dates_pairs, total=len(good_biweek_dates)):
        filename_industry_results = f'{ga_result_directory}/industry_{start_date_int}_{end_date_int}.csv'
        filename_stock_results = f'{ga_result_directory}/stock_{start_date_int}_{end_date_int}.csv'

        got_industry = filename_industry_results.split("\\")[-1] in os.listdir(ga_result_directory)
        got_stock = filename_stock_results.split("\\")[-1] in os.listdir(ga_result_directory)
        if got_industry and got_stock:
            print(filename_industry_results, "exists: \t{}".format(got_industry))
            print(filename_stock_results, "exists: \t{}".format(got_stock))
            continue
        else:
            print("Running for [{}] and [{}]".format(filename_industry_results, filename_stock_results))

        # industry
        # --------------------------------------------------------------------------------
        print('\nMining factors for industry...')

        run_ga_industry_kwargs = dict(
            industry_code=industry_code,
            data_directory=industry_data_directory,
            start_date_int=start_date_int,
            end_date_int=end_date_int,
            metric=my_metric,
            signal_ref_data=strat_signal_ref,
            q_lower=q_lower,
            q_upper=q_upper,
            strat_flag=strat_flag,
            population_size=population_size,
            tournament_size=int(0.1 * population_size),
            generations=generations,
            hall_of_fame=int(0.1 * population_size),
            n_components=int(0.1 * population_size),
            factor_filter=factor_filter,
            n_jobs=4,
            verbose=0,
        )
        run_ga_industry_partial = partial(run_ga_industry, **run_ga_industry_kwargs)
        with parallel_backend("loky", inner_max_num_threads=8):
            industry_opinion_result_list = Parallel(n_jobs=32, max_nbytes='32M', verbose=10)(
                delayed(run_ga_industry_partial)(x) for x in industry_symbol_arr)

        final_result = pd.DataFrame(
            [np.hstack([sym, ewo, ops, facts]) for sym, ewo, ops, facts in industry_opinion_result_list],
            columns=['code', 'ew_ops'] + ['op{}'.format(i) for i in range(5)] + ['f{}'.format(i) for i in range(5)]
        )
        final_result.to_csv(filename_industry_results)
        print(final_result)

        # stock
        # --------------------------------------------------------------------------------
        print('\nMining factors for stocks...')
        train_fields1 = ['open', 'high', 'low', 'close', 'volume', 'money', 'avg', ]
        train_fields2 = ['net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']
        raw_train_fields = train_fields1 + train_fields2

        run_ga_stock_kwargs = dict(
            data_directory=stock_data_directory,
            raw_train_fields=raw_train_fields,
            start_date_int=start_date_int,
            end_date_int=end_date_int,
            metric=my_metric,
            signal_ref_data=strat_signal_ref,
            q_lower=q_lower,
            q_upper=q_upper,
            strat_flag=strat_flag,
            population_size=population_size,
            tournament_size=int(0.1 * population_size),
            generations=generations,
            hall_of_fame=int(0.1 * population_size),
            n_components=int(0.1 * population_size),
            factor_filter=factor_filter,
            n_jobs=4,
            verbose=0,
        )
        run_ga_stock_partial = partial(run_ga_stock, **run_ga_stock_kwargs)
        with parallel_backend("loky", inner_max_num_threads=8):
            stock_opinion_result_list = Parallel(n_jobs=32, max_nbytes='32M', verbose=10)(
                delayed(run_ga_stock_partial)(x) for x in stock_symbol_arr)

        final_result = pd.DataFrame(
            [np.hstack([sym, ewo, ops, facts]) for sym, ewo, ops, facts in stock_opinion_result_list],
            columns=['code', 'ew_ops'] + ['op{}'.format(i) for i in range(5)] + ['f{}'.format(i) for i in range(5)]
        )
        final_result.to_csv(filename_stock_results)
        print(final_result)
