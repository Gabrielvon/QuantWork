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
    raw_hs300 = pd.read_csv('../data/SH000300.csv', header=None).astype(str)[0]
    hs300 = raw_hs300.str.split('\t', expand=True)
    hs300a = hs300[0].str.split('\t', expand=True)[0].str.split('.', expand=True)
    hs300a[2] = hs300a[0].astype(int)
    hs300a[3] = list(hs300a[1].str.lower() + hs300a[0])
    hs300a[4] = hs300[1].astype(int).copy()
    hs300a = hs300a.sort_values(4).reset_index(drop=True)
    code_type_map1 = hs300a.set_index(0)[3].to_dict()
    code_type_map2 = hs300a.set_index(2)[3].to_dict()

    hs300a = hs300a.iloc[:].copy()
    hs300a = hs300a[hs300a[4] < 20190101].copy()
    hs300b = hs300a.sample(100)
    # hs300b = hs300a.copy()
    test_slice = np.s_[-252:]
    all_trading_dates = all_trading_dates[test_slice]
    all_trading_dates_int = all_trading_dates_int[test_slice]

    selected_codes1 = hs300b[0].tolist()
    selected_codes2 = hs300b[2].tolist()
    selected_codes3 = hs300b[3].tolist()


    # --------------------------------------------------------------------------------
    # compute factors and target
    # --------------------------------------------------------------------------------

    for date in tqdm(all_trading_dates_int):
        for code in selected_codes3:
            print(lfp.volume_distance_express(code, date))
            df = lut.load_orderbook(code, date)
            df = df[(df['timeint'] > 93000) & (df['timeint'] < 145700)].copy()
            lfp.price_volume_corr_express(code, date)
            raise Exception()

    print('computing tba...')
    tba_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_tba_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    tba = pd.DataFrame(tba_list, columns=['date', 'code', 'tba'])
    # tba['code'] = tba['code'].map(code_type_map2)
    tba = tba.set_index(['date', 'code'])

    print('computing dbook...')
    dbook_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_dbook_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    dbook = pd.DataFrame(dbook_list, columns=['date', 'code', 'dbook'])
    # dbook['code'] = dbook['code'].map(code_type_map2)
    dbook = dbook.set_index(['date', 'code'])

    print('computing absr...')
    absr_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_absr_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    absr = pd.DataFrame(absr_list, columns=['date', 'code', 'absr'])
    # absr['code'] = absr['code'].map(code_type_map2)
    absr = absr.set_index(['date', 'code'])

    print('computing factor1...')
    factor1_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_factor1_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    factor1 = pd.DataFrame(factor1_list, columns=['date', 'code', 'factor1'])
    # factor1['code'] = factor1['code'].map(code_type_map2)
    factor1 = factor1.set_index(['date', 'code'])

    print('computing factor2...')
    factor2_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_factor2_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    factor2 = pd.DataFrame(factor2_list, columns=['date', 'code', 'factor2'])
    # factor2['code'] = factor2['code'].map(code_type_map2)
    factor2 = factor2.set_index(['date', 'code'])

    print('computing factor3...')
    factor3_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_factor3_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor3 = pd.DataFrame(factor3_list, columns=['date', 'code', 'factor3'])
    factor3['code'] = factor3['code'].map(code_type_map1)
    factor3 = factor3.set_index(['date', 'code'])

    print('computing factor4...')
    factor4_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_factor4_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor4 = pd.DataFrame(factor4_list, columns=['date', 'code', 'factor4'])
    factor4['code'] = factor4['code'].map(code_type_map1)
    factor4 = factor4.set_index(['date', 'code'])

    print('computing factor5...')
    factor5_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_factor5_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes1)
    factor5 = pd.DataFrame(factor5_list, columns=['date', 'code', 'factor5'])
    factor5['code'] = factor5['code'].map(code_type_map1)
    factor5 = factor5.set_index(['date', 'code'])

    # ----newly-added
    print('computing price_volatility_updown...')
    price_volatility_updown_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.price_volatility_updown_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    price_volatility_updown = pd.DataFrame(price_volatility_updown_list, columns=['date', 'code', 'price_volatility_updown'])
    # price_volatility_updown['code'] = price_volatility_updown['code'].map(code_type_map1)
    price_volatility_updown = price_volatility_updown.set_index(['date', 'code'])

    print('computing price_volume_corr...')
    price_volume_corr_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.price_volume_corr_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    price_volume_corr = pd.DataFrame(price_volume_corr_list, columns=['date', 'code', 'price_volume_corr'])
    # price_volume_corr['code'] = price_volume_corr['code'].map(code_type_map1)
    price_volume_corr = price_volume_corr.set_index(['date', 'code'])

    print('computing flowinratio...')
    flowinratio_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.flowinratio_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    flowinratio = pd.DataFrame(flowinratio_list, columns=['date', 'code', 'flowinratio'])
    # flowinratio['code'] = flowinratio['code'].map(code_type_map1)
    flowinratio = flowinratio.set_index(['date', 'code'])

    print('computing trendstrength...')
    trendstrength_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.trendstrength_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    trendstrength = pd.DataFrame(trendstrength_list, columns=['date', 'code', 'trendstrength'])
    # trendstrength['code'] = trendstrength['code'].map(code_type_map1)
    trendstrength = trendstrength.set_index(['date', 'code'])

    print('computing volume_distance...')
    volume_distance_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.volume_distance_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    volume_distance = pd.concat([pd.Series(dict(date=dt, code=co)).append(ss) if isinstance(ss, pd.core.series.Series) else pd.Series(dict(date=dt, code=co)) for dt, co, ss in volume_distance_list], axis=1).T
    volume_distance = volume_distance.set_index(['date', 'code'])
    volume_distance.columns = ['volume_distance_' + str(cn) for cn in volume_distance.columns]

    # ----
    print('computing day_logret...')
    day_logret_list = Parallel(n_jobs=16, verbose=3)(delayed(lfp.calc_hmo_express)(code=co, date=d) for d in all_trading_dates_int for co in selected_codes3)
    day_logret = pd.DataFrame(day_logret_list, columns=['date', 'code', 'hmo'])
    day_logret = day_logret.set_index(['date', 'code'])
    day_logret_s1 = day_logret['hmo'].unstack('code').shift(-1).stack().to_frame('hmo_s1')

    # input_data = pd.concat([day_logret,  day_logret_s1, factor1, factor2, factor3, factor4, factor5], axis=1)
    # input_data = pd.concat([day_logret, day_logret_s1, factor1, factor3, factor4, factor5], axis=1, join='outer')
    # input_data = input_data.dropna()

    input_data = day_logret.copy()
    _more = [day_logret_s1, tba, dbook, absr, factor1, factor2, factor3, factor4, factor5, price_volatility_updown, price_volume_corr, flowinratio, trendstrength, volume_distance]
    for m in _more:
        input_data = input_data.merge(m, on=['date', 'code'], how='outer')
    input_data = input_data.dropna()
    print('factor computation finished...')
    # num_date, num_code, num_data =  len(all_trading_dates), len(selected_codes1), input_data.shape[1] - 2
    num_date, num_code, num_data = len(set(all_trading_dates)), len(set(selected_codes1)), input_data.shape[1] - 2
    # input_data.to_csv('input_data_252d_300c_7f.csv')
    input_data.to_parquet('input_data_{}d_{}c_{}f.parq'.format(num_date, num_code, num_data))
