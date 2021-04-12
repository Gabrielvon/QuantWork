# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-17 16:15:21
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-17 16:17:52
# -*- coding: utf-8 -*-

__author__ = "Gabriel Feng"

import pandas as pd
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
from arch.unitroot import ADF
# import warnings
# warnings.filterwarnings('error')

import llquant.utilities as lut
import llquant.numba_recursive as nbr
from joblib import Parallel, delayed
import llquant.factor_pool as lfp
from tqdm import tqdm

import seaborn as sns

if __name__ == '__main__':

    raw_input_data = pd.read_parquet('input_data_252d_100c_21f.parq')
    input_data = raw_input_data.copy(deep=True)

    non_factor_columns = ['hmo', 'hmo_s1']
    all_factor_names = []
    for c in input_data:
        if c not in non_factor_columns:
            all_factor_names.append(c)

    selected_factor_names = all_factor_names
    # selected_factor_names = ['factor1', 'factor3']
    # input_data[all_factor_names] = input_data.groupby('date')[all_factor_names].apply(lambda g: (g - g.min()) / (g.max() - g.min()))

    # input_data[all_factor_names].hist()
    # plt.show()

    degs = [1, 2, 3]
    for fatn in all_factor_names:
        if fatn not in input_data:
            print(fatn, 'is not inclunded.')
            continue
        fig, axs = plt.subplots(2, 1)
        x, y = input_data[[fatn, 'hmo_s1']].sort_values(fatn).values.T
        xmn, xmx = np.percentile(x, [1, 99])
        x = np.clip(x, xmn, xmx)
        x_sim = np.linspace(x.min(), x.max(), 10000)
        axs[0].scatter(x, y, s=1, alpha=0.3)

        p_list = []
        for deg in degs:
            p = np.polyfit(x, y, deg=deg)
            pval = np.polyval(p, x_sim)
            p_list.append(''.join(['{:.6f},'.format(_p) for _p in p]))
            axs[0].plot(x_sim, pval)
            axs[0].grid(True)
        axs[0].legend(p_list)
        axs[0].set_title(fatn)
        axs[1].hist(x, bins=50)
        axs[1].grid(True)
        fig.tight_layout()


    # input_data_melted = input_data.melt(id_vars=['hmo_s1'])
    # # g = sns.lmplot(x="value", y="hmo_s1", hue='variable', data=input_data_melted)
    # g = sns.relplot(data=input_data_melted, x="value", y="hmo_s1", col="variable", col_wrap=6)
    # g.fig.show()


    # --------------------------------------------------------------------------------
    # evaluate factors
    # --------------------------------------------------------------------------------
    strategy_ret = {}
    factor_icir = {}
    for fatn in all_factor_names:
        if fatn not in input_data:
            print(fatn, 'is not inclunded.')
            continue
        print('Eval factor ---', fatn)
        ic_comb = scs.spearmanr(input_data['hmo_s1'].values, input_data[fatn].values)
        ic_xs = pd.Series({dt: scs.spearmanr(g['hmo_s1'].values, g[fatn].values)[0] for dt, g in input_data.groupby('date')})    # cross-sectional ic overtime
        icir_xs = ic_xs.mean() / ic_xs.std()

        # plt.plot(np.array(ic_xs), marker='.')
        # plt.axhline(np.mean(ic_xs), ls='--', c='r')
        # plt.show()

        # 每只股票ic ---> 平均
        ic_ts = pd.Series({co: scs.spearmanr(g['hmo_s1'].values, g[fatn].values)[0] for co, g in input_data.groupby('code')})    # time-series ic by code
        icir_ts = ic_ts.mean() / ic_ts.std()

        # 每只股票时序上的ic显著性和ic平稳性 ---> 平均
        test_result_list = []
        for co, g in input_data.groupby('code'):
            ic_roll = nbr.numba_rolling_spearmanr(g['hmo_s1'].values, g[fatn].values, 21)
            icir = np.nanmean(ic_roll) / np.nanstd(ic_roll)
            ic_tstat = scs.ttest_1samp(ic_roll[~np.isnan(ic_roll)], 0.0, nan_policy='omit', alternative='two-sided')[0]
            ic_adf = ADF(ic_roll[~np.isnan(ic_roll)]).stat
            test_result_list.append([co, ic_tstat, icir, ic_adf])
        test_result_df = pd.DataFrame(test_result_list, columns=['code', 'ic_tstat', 'ic_adf', 'icir'])
        test_desc = test_result_df.describe()
        test_tstat = test_result_df.set_index('code').apply(lambda x: scs.ttest_1samp(x[~np.isnan(x)], 0.0, nan_policy='omit', alternative='greater')[0])

        print('ic_comd', ic_comb)
        print('ic_xs', scs.ttest_1samp(ic_xs[~np.isnan(ic_xs)], 0.0, nan_policy='omit', alternative='two-sided'))
        print('icir_ts', icir_ts)
        print('ic by code')
        print(test_desc.append(test_tstat.to_frame('tstat').T).T[['count', 'mean', 'std', '50%', 'tstat']])

        # 单因子收益
        factor_piv = input_data[fatn].unstack('code')
        signal_piv_arr = np.stack([lut._generate_signal(ss, signal_ref=63, q_lower=0.2, q_upper=0.8, flag=1) for _, ss in factor_piv.iteritems()], axis=1)
        num_trade = np.sum(signal_piv_arr != 0)
        logret_piv_s_good = input_data['hmo_s1'].unstack('code').values
        logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
        num_trade = np.sum(signal_piv_arr != 0)
        ret = np.sum(signal_piv_arr * logret_piv_s_good, axis=1) / np.sum(signal_piv_arr != 0, axis=1)
        ret[np.isnan(ret)] = 0
        strategy_ret[fatn] = ret
        factor_icir[fatn] = test_result_df.set_index('code')['icir']
        print(fatn, num_trade, np.sum(ret), np.sum(ret) / num_trade)
        print('\n')

    strategy_ret_df = pd.DataFrame(strategy_ret, index=factor_piv.index)


    # --------------------------------------------------------------------------------
    # modeling
    # --------------------------------------------------------------------------------

    test_num = 21
    target_column = 'hmo_s1'
    input_date = input_data.index.get_level_values('date').unique()
    train_date = input_date[:-test_num]
    test_date = input_date[-test_num:]
    train_data = input_data.loc[train_date, :].copy()
    test_data = input_data.loc[test_date, :].copy()

    X = train_data[selected_factor_names].values
    X = lut.sigmoid(X)
    y = train_data['hmo_s1'].values
    idx_good = ~(np.isnan(X).any(1) | np.isinf(X).any(1) | np.isnan(y) | np.isinf(y))
    X = X[idx_good]
    y = y[idx_good]
    y = (y[idx_good] > 0).astype(int) - (y[idx_good] < 0).astype(int)

    X_test = test_data[selected_factor_names].values
    X_test = lut.sigmoid(X_test)
    y_test = test_data['hmo_s1'].values
    idx_test_good = ~(np.isnan(X_test).any(1) | np.isinf(X_test).any(1))
    X_test = X_test[idx_test_good]
    y_test = y_test[idx_test_good]
    y_test = (y_test[idx_test_good] > 0).astype(int) - (y_test[idx_test_good] < 0).astype(int)

    # # lasso
    # from sklearn.linear_model import Lasso
    # lasso = Lasso(alpha=0.001, fit_intercept=False)
    # lasso.fit(X, y)
    # y_pred = lasso.predict(X)
    # y_test_pred = lasso.predict(X_test)
    # print(lasso.score(X, y))
    # print(lasso.score(X_test, y_test))

    # # rf
    # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    # # rf_model = RandomForestRegressor(criterion="mse", max_depth=4, n_jobs=4, random_state=42)
    # rf_model = RandomForestClassifier(criterion="gini", max_depth=4, max_features=5, n_jobs=4, random_state=42)
    # # Q, _ = np.linalg.qr(X)
    # Q = X.copy()
    # rf_model.fit(Q, y)
    # y_pred = rf_model.predict(X)
    # y_test_pred = rf_model.predict(X_test)
    # print(rf_model.score(X, y))
    # print(rf_model.score(X_test, y_test))


    # xgboost
    import xgboost as xgb
    # xgb_model = xgb.XGBRFRegressor(max_depth=6, learning_rate=0.01, n_estimators=100,
    #              verbosity=3, objective=None, tree_method=None, n_jobs=4, random_state=42)
    xgb_model = xgb.XGBClassifier(max_depth=4, learning_rate=0.001, n_estimators=1000,
                 verbosity=1, objective=None, tree_method=None, n_jobs=8, random_state=42)
    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    y_test_pred = xgb_model.predict(X_test)
    print(xgb_model.score(X, y))
    print(xgb_model.score(X_test, y_test))

    # # auto-keras
    # import autokeras as ak
    #
    # def alpha_tstat(y_true, y_pred):
    #     # myret = lut._generate_signal(y_pred, signal_ref=2, q_lower=0.2, q_upper=0.8, flag=1)
    #     # long = [0] + list((tnp.diff(y_pred) > 0).astype(int))
    #     # short = [0] + list((tnp.diff(y_pred) < 0).astype(int))
    #     # signal = np.zeros_like(y_pred)
    #     # signal[long] = 1
    #     # signal[short] = -1
    #     # myret = signal * y_true
    #
    #     tstat = scs.ttest_ind(myret, y_true, equal_var=False, alternative='greater')[0]
    #     return tstat
    #
    # def ic_reverse(y_true, y_pred):
    #     ic_rev = scs.spearmanr(y_true, y_pred)[0]
    #     return tstat
    #
    # validnum = 63
    # _X_train = X[:-validnum]
    # _X_valid = X[-validnum:]
    # _y_train = y[:-validnum]
    # _y_valid = y[-validnum:]
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    #
    # # ak_model = ak.StructuredDataRegressor(
    # #     loss="mean_squared_error",
    # #     # objective=kerastuner.Objective('val_alpha_tstat', direction='min'),
    # #     # loss = spearman_correlation,
    # #     # Include it as one of the metrics.
    # #     # metrics=[get_spearman_rankcor],
    # #     max_trials=20, seed=42, overwrite=True)
    #
    # ak_model = ak.StructuredDataClassifier(
    #     loss="categorical_crossentropy",
    #     # objective=kerastuner.Objective('val_alpha_tstat', direction='min'),
    #     # loss = spearman_correlation,
    #     # Include it as one of the metrics.
    #     # metrics=[get_spearman_rankcor],
    #     max_trials=20, seed=42, overwrite=True)
    # # Feed the structured data regressor with training data.
    # ak_model.fit(_X_train, _y_train, validation_data=(_X_valid, _y_valid), epochs=20)
    #
    # y_pred = ak_model.predict(X).squeeze()
    # y_test_pred = ak_model.predict(X_test).squeeze()
    # print(ak_model.evaluate(X, y))
    # print(ak_model.evaluate(X_test, y_test))
    # ak_model_best1 = ak_model.export_model()

    # # TPOT
    # from tpot import TPOTRegressor
    # tpot = TPOTRegressor(
    #     # scoring=None, use_dask=True,
    #     generations=5, population_size=50, n_jobs=16, verbosity=2, random_state=42,
    # )
    # tpot.fit(X, y)
    # y_pred = tpot.predict(X)
    # y_test_pred = tpot.predict(X_test)
    # print(tpot.score(X, y))
    # print(tpot.score(X_test, y_test))

    # # Deep-forest
    # from sklearn.metrics import mean_squared_error, accuracy_score
    # from deepforest import CascadeForestRegressor, CascadeForestClassifier
    # # model = CascadeForestRegressor(n_estimators=50, max_depth=6, n_jobs=4)
    # model = CascadeForestClassifier(n_estimators=50, max_depth=6, n_jobs=4)
    # model.fit(X, y)
    # y_pred = model.predict(X).squeeze()
    # y_test_pred = model.predict(X_test).squeeze()
    # # print("\nTesting MSE: {:.3f}".format(mean_squared_error(y_pred, y)))
    # # print("\nTesting MSE: {:.3f}".format(mean_squared_error(y_test_pred, y_test)))
    # print("\nTesting accuracy_score: {:.3f}".format(accuracy_score(y_pred, y)))
    # print("\nTesting accuracy_score: {:.3f}".format(accuracy_score(y_test_pred, y_test)))
    #
    # 正交后通过时序上平均IC作为权重进行组合
    from sklearn.metrics import mean_squared_error, accuracy_score
    # Q, _ = np.linalg.qr(X)
    Q = X.copy()
    weight_list = []
    for v in Q.T:
        ic_ts = nbr.numba_rolling_spearmanr(v, y, 63)
        # icir = np.nanmean(ic_ts)
        icir = np.nanmean(ic_ts) / np.nanstd(ic_ts)
        weight_list.append(icir)
    weights = np.array(weight_list)
    weights = weights / weights.sum()
    y_pred = np.sum(weights * X, axis=1)
    y_test_pred = np.sum(weights * X_test, axis=1)
    print(mean_squared_error(y_pred, y))
    print(mean_squared_error(y_test_pred, y_test))

    # --------------------------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------------------------
    y_pred_ss = pd.Series(y_pred, index=train_data.index[idx_good])
    strategy_train = train_data[selected_factor_names + ['hmo_s1']].assign(y_pred=y_pred_ss)
    y_test_pred_ss = pd.Series(y_test_pred, index=test_data.index[idx_test_good])
    strategy_test = test_data[selected_factor_names + ['hmo_s1']].assign(y_pred=y_test_pred_ss)
    test_date0 = test_date[0]

    # # regression
    # strategy = pd.concat([strategy_train, strategy_test])
    # factor_piv = strategy['y_pred'].unstack('code')
    # signal_piv_arr = np.stack([lut._generate_signal(ss, signal_ref=63, q_lower=0.2, q_upper=0.8, flag=1, min_periods=21) for _, ss in factor_piv.iteritems()], axis=1)
    # num_trade = np.sum(signal_piv_arr != 0)
    # logret_piv_s_good = strategy['hmo_s1'].unstack('code').values
    # logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
    # num_trade = np.sum(signal_piv_arr != 0)
    # ret = np.sum(signal_piv_arr * logret_piv_s_good, axis=1) / np.sum(signal_piv_arr != 0, axis=1)
    # print(num_trade, np.sum(ret), np.sum(ret) / num_trade)
    # strategy_ret_df['fitting'] = ret
    #
    # strategy_ret_df.cumsum().reset_index(drop=True).plot.line(legend=False, ls='-.')
    # strategy_ret_df['fitting'].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-', c='b', marker='.')
    # plt.axvline(np.where(strategy_ret_df.index == test_date0)[0], ls='--')
    # plt.grid(True)
    # plt.show()
    #
    # strategy_ret_df[strategy_ret_df.index >= test_date0].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-.')
    # strategy_ret_df.loc[strategy_ret_df.index >= test_date0, 'fitting'].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-', c='b', marker='.')
    # plt.grid(True)
    # plt.show()

    # classification
    strategy = pd.concat([strategy_train, strategy_test])
    signal_piv_arr = strategy['y_pred'].unstack('code')
    num_trade = np.sum(signal_piv_arr != 0)
    logret_piv_s_good = strategy['hmo_s1'].unstack('code').values
    logret_piv_s_good[np.isinf(logret_piv_s_good) | np.isnan(logret_piv_s_good)] = 0
    num_trade = np.sum(strategy['y_pred'] != 0)
    ret = np.sum(signal_piv_arr * logret_piv_s_good, axis=1) / np.sum(signal_piv_arr != 0, axis=1)
    print(num_trade, np.sum(ret), np.sum(ret) / num_trade)
    strategy_ret_df['fitting'] = ret

    strategy_ret_df.cumsum().reset_index(drop=True).plot.line(legend=False, ls='-.')
    strategy_ret_df['fitting'].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-', c='b', marker='.')
    plt.axvline(np.where(strategy_ret_df.index == test_date0)[0], ls='--')
    plt.grid(True)
    plt.show()

    strategy_ret_df[strategy_ret_df.index >= test_date0].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-.')
    strategy_ret_df.loc[strategy_ret_df.index >= test_date0, 'fitting'].cumsum().reset_index(drop=True).plot.line(legend=False, ls='-', c='b', marker='.')
    plt.grid(True)
    plt.show()


