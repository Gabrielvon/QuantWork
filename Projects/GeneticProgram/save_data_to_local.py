# -*- coding: utf-8 -*-

"""
下载数据并保存至本地`dataset/data`文件夾下
"""

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import jqfunc as jqf


def save_industry_data_to_local(industry_code, start_date, end_date, directory):
    # industry_code="sw_l1"

    # industry_train_fields = [
    #     'open', 'high', 'low', 'close', 'volume', 'money',
    #     'net_amount_main', 'net_amount_xl', 'net_amount_l',
    #     'net_amount_m', 'net_amount_s'
    # ]

    price = jqf.get_sw_l1_industry_price(start_date=start_date, end_date=end_date)
    moneyflow = jqf.get_industry_daily_moneyflow(industry_code, start_date=start_date, end_date=end_date)
    raw_industry_data = pd.merge(price, moneyflow, on=['date', 'code'])
    industry_symbol_arr = price['code'].unique()
    _tmp_data = raw_industry_data.melt(['date', 'code', 'name'])
    _tmp_data['colname'] = _tmp_data['variable'] + '_' + _tmp_data['code']
    industry_data = _tmp_data.pivot('date', 'colname', 'value')
    industry_data.to_parquet(f'{directory}/{industry_code}.parq')
    return industry_data


def save_stock_data_to_local(stock_sym, start_date, end_date, fields, directory):
    is_data_fetched = False
    attempt_fetch_data = 0
    while not is_data_fetched:
        try:
            price = jqf.get_price(stock_sym, start_date=start_date, end_date=end_date, fq='post', frequency='1d',
                                  fields=fields, panel=False)
            moneyflow = jqf.get_money_flow(stock_sym, start_date=start_date, end_date=end_date)
            is_data_fetched = True
            attempt_fetch_data += 1
        except Exception as e:
            print("[WARNING] JQdata API error({}) Wait a minute and try again. Attempts: {}".format(';'.join(e.args),
                                                                                                    attempt_fetch_data))
            time.sleep(60)

    price.index.name = 'date'
    price.reset_index(inplace=True)
    stock_data = pd.merge(price, moneyflow, on=['date']).set_index('date')
    stock_data['pct1'] = np.log(stock_data['close']).diff().shift(-1)
    stock_data.index = stock_data.index.strftime("%Y%m%d").astype(int)
    stock_data = stock_data.ffill().dropna()
    stock_data.to_parquet(f'{directory}/{stock_sym}.parq')

    return stock_data


if __name__ == '__main__':

    start_date = '2016-02-01'
    end_date = '2021-02-26'
    overwrite = True

    # industry
    # --------------------------------------------------------------------------------
    data_directory = f'dataset/data/industry_{start_date}_{end_date}/'
    try:
        os.makedirs(data_directory, exist_ok=False)
        print('fetching industry data...')
        _ = save_industry_data_to_local(industry_code="sw_l1", start_date=start_date, end_date=end_date,
                                        directory=data_directory)
    except Exception as e:
        filenames = os.listdir(data_directory)
        print(e)
        print('Direcotry [{}] is already existed with {} data files.'.format(data_directory, len(filenames)))

    # stock
    # --------------------------------------------------------------------------------
    train_fields1 = ['open', 'high', 'low', 'close', 'volume', 'money', 'avg', ]
    # train_fields2 = ['net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']
    raw_train_fields = train_fields1
    data_directory = f'dataset/data/hs300_{start_date}_{end_date}/'

    try:
        os.makedirs(data_directory, exist_ok=False)
        filenames = os.listdir(data_directory)
    except Exception as e:
        filenames = os.listdir(data_directory)
        print(e)
        print('Direcotry [{}] is already existed with {} data files.'.format(data_directory, len(filenames)))

    existed_stock_symbol_arr = ['.'.join(fn.split('.')[:-1]) for fn in filenames if 'parq' in fn]
    print('fetching stock data...')
    index_weights = jqf.get_stock_index_weights("000300.XSHG", date=end_date, raw=True)
    stock_symbol_arr = index_weights['code'].values
    for symbol in tqdm(stock_symbol_arr, desc='FetchingData'):
        if overwrite | (symbol not in existed_stock_symbol_arr):
            _ = save_stock_data_to_local(stock_sym=symbol, start_date=start_date, end_date=end_date,
                                         fields=raw_train_fields, directory=data_directory)
