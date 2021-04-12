# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-15 10:13:22
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-15 11:30:40


import pandas as pd
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import llquant.utilities as lut
import llquant.numba_recursive as nbr
from joblib import Parallel, delayed
import llquant.factor_pool as lfp
from tqdm import tqdm
import warnings
warnings.filterwarnings('error')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # load data
    # --------------------------------------------------------------------------------
    all_trading_dates = lut.get_trading_dates()
    all_trading_dates_int = list(map(int, all_trading_dates))
    hs300 = pd.read_csv('../data/SH000300.csv', header=None).astype(str)[0]
    hs300a = hs300.str.split('\t', expand=True)[0].str.split('.', expand=True)
    hs300a[2] = hs300a[0].astype(int)
    hs300a[3] = list(hs300a[1].str.lower() + hs300a[0])
    code_type_map1 = hs300a.set_index(0)[3].to_dict()
    code_type_map2 = hs300a.set_index(2)[3].to_dict()

    test_slice = np.s_[-63:]
    hs300a = hs300a.iloc[test_slice].copy()
    all_trading_dates = all_trading_dates[test_slice]
    all_trading_dates_int = all_trading_dates_int[test_slice]

    selected_codes1 = hs300a[0].tolist()
    selected_codes2 = hs300a[2].tolist()
    selected_codes3 = hs300a[3].tolist()

    # --------------------------------------------------------------------------------
    # compute factors and target
    # TODO 加一些波动率相关的因子
    # TODO 因子值有点极端，是否同意缩放一下？
    # TODO
    # --------------------------------------------------------------------------------

    # for date in all_trading_dates_int:
    #     for code in selected_codes3:
    #         print(lfp.calc_hmo_express(code, date))
    #     break

    print('computing tba...')
    tba_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_tba_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    tba = pd.DataFrame(tba_list, columns=['date', 'code', 'tba'])
    # tba['code'] = tba['code'].map(code_type_map2)
    tba = tba.set_index(['date', 'code'])

    print('computing dbook...')
    dbook_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_dbook_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    dbook = pd.DataFrame(dbook_list, columns=['date', 'code', 'dbook'])
    # dbook['code'] = dbook['code'].map(code_type_map2)
    dbook = dbook.set_index(['date', 'code'])

    print('computing absr...')
    absr_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_absr_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    absr = pd.DataFrame(absr_list, columns=['date', 'code', 'absr'])
    # absr['code'] = absr['code'].map(code_type_map2)
    absr = absr.set_index(['date', 'code'])

    print('computing factor1...')
    factor1_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_factor1_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    factor1 = pd.DataFrame(factor1_list, columns=['date', 'code', 'factor1'])
    # factor1['code'] = factor1['code'].map(code_type_map2)
    factor1 = factor1.set_index(['date', 'code'])

    print('computing factor2...')
    factor2_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_factor2_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    factor2 = pd.DataFrame(factor2_list, columns=['date', 'code', 'factor2'])
    # factor2['code'] = factor2['code'].map(code_type_map2)
    factor2 = factor2.set_index(['date', 'code'])

    print('computing factor3...')
    factor3_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_factor3_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor3 = pd.DataFrame(factor3_list, columns=['date', 'code', 'factor3'])
    factor3['code'] = factor3['code'].map(code_type_map1)
    factor3 = factor3.set_index(['date', 'code'])

    print('computing factor4...')
    factor4_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_factor4_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor4 = pd.DataFrame(factor4_list, columns=['date', 'code', 'factor4'])
    factor4['code'] = factor4['code'].map(code_type_map1)
    factor4 = factor4.set_index(['date', 'code'])

    print('computing factor5...')
    factor5_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_factor5_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor5 = pd.DataFrame(factor5_list, columns=['date', 'code', 'factor5'])
    factor5['code'] = factor5['code'].map(code_type_map1)
    factor5 = factor5.set_index(['date', 'code'])

    print('computing day_logret...')
    day_logret_list = Parallel(n_jobs=22, verbose=3)(delayed(lfp.calc_hmo_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    day_logret = pd.DataFrame(day_logret_list, columns=['date', 'code', 'hmo'])
    day_logret = day_logret.set_index(['date', 'code'])
    day_logret_s1 = day_logret['hmo'].unstack('code').shift(-1).stack().to_frame('hmo_s1')

    # input_data = pd.concat([day_logret,  day_logret_s1, factor1, factor2, factor3, factor4, factor5], axis=1)
    # input_data = pd.concat([day_logret, day_logret_s1, factor1, factor3, factor4, factor5], axis=1, join='outer')
    # input_data = input_data.dropna()

    input_data = day_logret.copy()
    _more = [day_logret_s1, tba, dbook, absr, factor1, factor2, factor3, factor4, factor5]
    for m in _more:
        input_data = input_data.merge(m, on=['date', 'code'], how='inner')
    input_data = input_data.dropna()
    print('factor computation finished...')
    # input_data.to_parquet('input_data_252d_300c_7f.parq')

    all_factor_names = ['tba', 'dbook', 'absr', 'factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    selected_factor_names = all_factor_names
    # selected_factor_names = ['factor1', 'factor3']
    input_data[all_factor_names] = input_data.groupby('date')[all_factor_names].apply(lambda g: (g - g.min()) / (g.max() - g.min()))


    # --------------------------------------------------------------------------------
    # evaluate factors
    # --------------------------------------------------------------------------------
    for fatn in all_factor_names:
        if fatn not in input_data:
            print(fatn, 'is not inclunded.')
            continue
        print('Eval factor ---', fatn)
        ic_comb = scs.spearmanr(input_data['hmo_s1'].values, input_data[fatn].values)
        ic_xs = pd.Series({dt: scs.spearmanr(g['hmo_s1'].values, g[fatn].values)[0] for dt, g in input_data.groupby('date')})    # cross-sectional ic overtime


        plt.plot(np.array(ic_xs), marker='.')
        plt.axhline(np.mean(ic_xs), ls='--', c='r')
        plt.show()

        # 每只股票ic ---> 平均
        ic_ts = pd.Series({co: scs.spearmanr(g['hmo_s1'].values, g[fatn].values)[0] for co, g in input_data.groupby('code')})    # time-series ic by code

        from arch.unitroot import ADF
        # 每只股票时序ic显著性/ic平稳性 ---> 平均
        ic_ts_ttest = {}
        ic_ts_adf = {}
        for co, g in input_data.groupby('code'):
            ic_roll = nbr.numba_rolling_spearmanr(g['hmo_s1'].values, g[fatn].values, 21)
            ic_ttest = scs.ttest_1samp(ic_roll[~np.isnan(ic_roll)], 0.0, nan_policy='omit', alternative='two-sided')[0]
            ic_ts_ttest[co] = ic_ttest
            ic_adf = ADF(ic_roll[~np.isnan(ic_roll)]).stat
            ic_ts_adf[co] = ic_adf


        ic_stats = pd.concat([
            ic_xs.describe(),
            ic_ts.describe(),
            ic_xs[ic_xs > 0].describe(),
            ic_ts[ic_ts > 0].describe(),
            ic_xs[ic_xs < 0].describe(),
            ic_ts[ic_ts < 0].describe(),
        ], axis=1, keys=['xs', 'ts', 'xs_pos', 'ts_pos', 'xs_neg', 'ts_neg'])
        print(ic_stats.T)

    # --------------------------------------------------------------------------------
    # make strategy
    # --------------------------------------------------------------------------------
    strategy_ret = {}
    for fatn in all_factor_names:
        if fatn not in input_data:
            print(fatn, 'is not inclunded.')
            continue
        factor_piv = input_data[fatn].unstack('code')
        signal_piv_arr = np.stack([lut._generate_signal(ss, signal_ref=2, q_lower=0.2, q_upper=0.8, flag=1) for _, ss in factor_piv.iteritems()], axis=1)
        num_trade = np.sum(signal_piv_arr != 0)
        logret_piv_s_good = input_data['hmo_s1'].unstack('code').values
        logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
        num_trade = np.sum(signal_piv_arr != 0)
        ret = np.mean(signal_piv_arr * logret_piv_s_good, axis=1)
        strategy_ret[fatn] = ret
        print(fatn, num_trade, np.sum(ret), np.sum(ret) / num_trade)
    strategy_ret_df = pd.DataFrame(strategy_ret, index=factor_piv.index)

    # --------------------------------------------------------------------------------
    # modeling
    # --------------------------------------------------------------------------------

    test_num = 21
    input_date = input_data.index.get_level_values('date').unique()
    train_date = input_date[:-test_num]
    test_date = input_date[-test_num:]
    train_data = input_data.loc[train_date, ].copy()
    test_data = input_data.loc[test_date, ].copy()

    X = train_data[selected_factor_names].values
    y = train_data['hmo_s1'].values
    idx_good = ~(np.isnan(X).any(1) | np.isinf(X).any(1) | np.isnan(y) | np.isinf(y))
    X = X[idx_good]
    y = y[idx_good]

    X_test = test_data[selected_factor_names].values
    idx_test_good = ~(np.isnan(X_test).any(1) | np.isinf(X_test).any(1))
    y_test = test_data['hmo_s1'].values
    X_test = X_test[idx_test_good]
    y_test = y_test[idx_test_good]

    # # lasso
    # from sklearn.linear_model import Lasso
    # lasso = Lasso(alpha=0.001, fit_intercept=False)
    # model = lasso.fit(X, y)
    # y_pred = model.predict(X)
    # y_test_pred = model.predict(X_test)
    # print(model.score(X, y))
    # print(model.score(X_test, y_test))

    # # rf
    # from sklearn.ensemble import RandomForestRegressor
    # rfreg = RandomForestRegressor(criterion="mse", max_depth=4)
    # model = rfreg.fit(X, y)
    # y_pred = model.predict(X)
    # y_test_pred = model.predict(X_test)
    # print(model.score(X, y))
    # print(model.score(X_test, y_test))

    # # xgboost
    # import xgboost as xgb
    # xgb_model = xgb.XGBRFRegressor(random_state=42)
    # xgb_model.fit(X, y)
    # y_pred = xgb_model.predict(X)
    # y_test_pred = xgb_model.predict(X_test)
    # print(xgb_model.score(X, y))
    # print(xgb_model.score(X_test, y_test))

    # # auto-keras
    # import autokeras as ak
    # reg = ak.StructuredDataRegressor(loss="mean_squared_error", max_trials=3, seed=42, overwrite=True)
    # # Feed the structured data regressor with training data.
    # reg.fit(X, y, epochs=10)
    # y_pred = reg.predict(X).squeeze()
    # y_test_pred = reg.predict(X_test).squeeze()
    # print(reg.evaluate(X, y))
    # print(reg.evaluate(X_test, y_test))

    # TPOT
    from tpot import TPOTRegressor
    tpot = TPOTRegressor(
        # scoring=None, use_dask=True,
        generations=5, population_size=50, n_jobs=4, verbosity=2, random_state=42,
    )
    tpot.fit(X, y)
    y_pred = tpot.predict(X)
    y_test_pred = tpot.predict(X_test)
    print(tpot.score(X, y))
    print(tpot.score(X_test, y_test))

    y_pred_ss = pd.Series(y_pred, index=train_data.index[idx_good])
    strategy_train = train_data[selected_factor_names + ['hmo_s1']].assign(y_pred=y_pred_ss)
    y_test_pred_ss = pd.Series(y_test_pred, index=test_data.index[idx_test_good])
    strategy_test = test_data[selected_factor_names + ['hmo_s1']].assign(y_pred=y_test_pred_ss)
    test_date0 = test_date[0]

    strategy = pd.concat([strategy_train, strategy_test])
    # strategy = strategy_train.copy()
    # strategy = strategy_test.copy()
    factor_piv = strategy['y_pred'].unstack('code')
    signal_piv_arr = np.stack([lut._generate_signal(ss, signal_ref=2, q_lower=0.2, q_upper=0.8, flag=1) for _, ss in factor_piv.iteritems()], axis=1)
    num_trade = np.sum(signal_piv_arr != 0)
    logret_piv_s_good = strategy['hmo_s1'].unstack('code').values
    logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
    num_trade = np.sum(signal_piv_arr != 0)
    ret = np.mean(signal_piv_arr * logret_piv_s_good, axis=1)
    print(num_trade, np.sum(ret), np.sum(ret) / num_trade)
    strategy_ret_df['fitting'] = ret

    strategy_ret_df.cumsum().reset_index(drop=True).plot.line()
    plt.axvline(np.where(strategy_ret_df.index == test_date0)[0], ls='--')
    plt.grid(True)
    plt.show()

    strategy_ret_df[strategy_ret_df.index >= test_date0].cumsum().reset_index(drop=True).plot.line()
    plt.grid(True)
    plt.show()



