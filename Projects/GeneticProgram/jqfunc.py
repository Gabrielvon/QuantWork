# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-01 17:56:56
# @Last Modified by:   Gabriel
# @Last Modified time: 2021-04-12 11:42:02

import math
import pandas as pd
import numpy as np
from jqdatasdk import *  # run at local
from retry import retry
from sqlalchemy.sql import func

auth("*", "*")

_retry = 3
_delay = 3

db_NaN = -1
db_NaT_floor = pd.to_datetime('1970-01-01').date()
db_NaT_ceil = pd.to_datetime('2200-01-01').date()


def compress_price_data(df):
    dtypes_mapping = {
        'date': 'int64',
        'stamp': 'int64',
        'code': 'str',
        'exchange_code': 'str',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int64',
        'money': 'float64',
        'pre_close': 'float32',
        'avg': 'float32',
        'paused': 'int8',
        'factor': 'float32',
        'open_interest': 'int64',  # future and option
        'deal_number': 'int64',  # cbond_price
    }

    dtypes_raw = df.dtypes
    for k, v in dtypes_mapping.items():
        if k in dtypes_raw.keys():
            if v != dtypes_raw[k]:
                df[k] = df[k].fillna(db_NaN).astype(v)

    return df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_unlimited_data(f_query, query_args, filters, step=3000, verbose=0):
    """
    Gets the unlimited data.

    Example:
        query_args = (bond.BOND_BASIC_INFO,)
        filters = (
            bond.BOND_BASIC_INFO.list_date <= date,
            bond.BOND_BASIC_INFO.delist_Date >= date,
        )
        bond_info = get_unlimited_data(bond.run_query, query_args, filters, verbose=2)

    :param      f_query:      The f query
    :type       f_query:      { type_description }
    :param      query_args:   The query arguments
    :type       query_args:   { type_description }
    :param      filters:  The filter arguments
    :type       filters:  { type_description }
    :param      step:         The step
    :type       step:         number
    :param      verbose:      The verbose
    :type       verbose:      number

    :returns:   The unlimited data.
    :rtype:     { return_type_description }
    """
    q = query(func.count('*')).filter(*filters)

    sum_count = f_query(q).iloc[0, 0]  # 先查询总共有多少条数据
    if verbose > 0:
        print('总共有{}条数据, 需要获取{}次'.format(
            sum_count, int(math.ceil(sum_count / step))))

    if sum_count == 0:
        sum_count = 1

    rs_list = []
    if verbose > 0:
        from tqdm import tqdm
        iterobj = tqdm(range(0, sum_count, step))
    else:
        iterobj = range(0, sum_count, step)

    for i in iterobj:
        q = query(*query_args).filter(*filters).limit(step).offset(i)  # 自第i条数据之后进行获取
        rs = f_query(q)
        rs_list.append(rs)
        # if pd.concat(rs_list).duplicated().sum() > 0:
        #     raise Exception()
        if verbose > 1:
            print(i, rs.shape)
    return pd.concat(rs_list)


@retry(TimeoutError, tries=_retry, delay=_delay)
def getTradingDates(count=int(1e4)):
    """Get most recent trading dates
    
    Args:
        count (TYPE, optional): Description
    
    Returns:
        TYPE: Description
    """
    if count is not None:
        all_trade_days = get_trade_days(count=count)
    else:
        all_trade_days = get_trade_days()
    all_trade_days = list(map(lambda x: int(datetime.datetime.strftime(x, '%Y%m%d')), all_trade_days))
    return np.array(all_trade_days)


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_sw_l1_industry_price(codes=None, start_date='1970-01-01', end_date='2200-01-01'):
    query_args = (finance.SW1_DAILY_PRICE,)
    filters = (
        finance.SW1_DAILY_PRICE.date >= start_date,
        finance.SW1_DAILY_PRICE.date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.SW1_DAILY_PRICE.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.SW1_DAILY_PRICE.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    industry_price_df = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)
    industry_price_df = industry_price_df.drop(['id'], axis=1).reset_index(drop=True)
    industry_price_df.rename(columns={'day': 'date'}, inplace=True)
    industry_price_df['date'] = pd.to_datetime(industry_price_df['date']).dt.strftime('%Y%m%d').astype(int)
    return industry_price_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_industry_daily_moneyflow(industry, start_date='1970-01-01', end_date=None):
    """

    :param industry: 'zjh', 'sw_l1', 'sw_l2', 'sw_l3', 'jq_l1', 'jq_l2'
    :param start_date:
    :param end_date:
    :return:
    """

    @retry(TimeoutError, tries=_retry, delay=_delay)
    def _get_industry_daily_moneyflow(code, start_date='1970-01-01', end_date=None):
        symbol_list = get_industry_stocks(code, date=end_date)
        # 特殊处理
        if len(symbol_list) == 0:
            return pd.DataFrame(
                columns=['date', 'code', 'net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m',
                         'net_amount_s'])

        try:
            money_flow = get_money_flow(symbol_list, start_date=start_date, end_date=end_date)
        except Exception as e:
            err_msg = ';'.join(e.args)
            if '找不到标的' in err_msg:
                all_listed_symbols = get_all_securities('stock', date=end_date).index
                symbol_arr = np.array(symbol_list)
                symbol_list = list(symbol_arr[np.isin(symbol_arr, all_listed_symbols)])
                money_flow = get_money_flow(symbol_list, start_date=start_date, end_date=end_date)
                # symbols_not_exist = re.findall("\d+\.\w+", err_msg)
                # if len(symbols_not_exist) > 0:
                #     warnings.warn('行业编码{}中找不到标的:{}'.format(code, ','.join(symbols_not_exist)))
                #     for sym in symbols_not_exist:
                #         symbol_list.remove(sym)
                # money_flow = get_money_flow(symbol_list, start_date=start_date, end_date=end_date)
            else:
                raise Exception(err_msg)
        money_flow['code'] = code
        money_flow_types = ['net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s']
        money_flow = money_flow.groupby(['date', 'code'])[money_flow_types].sum().reset_index()
        return money_flow

    if end_date is not None:
        if pd.to_datetime(end_date) < pd.to_datetime(start_date):
            raise ValueError("end_date must be later than start_date.")

    industry_code_info = get_industries(industry)
    industry_code = industry_code_info.index.tolist()
    industry_mf_list = [_get_industry_daily_moneyflow(co, start_date=start_date, end_date=end_date) for co in
                        industry_code]
    industry_mf_df = pd.concat(industry_mf_list)
    industry_mf_df['date'] = pd.to_datetime(industry_mf_df['date']).dt.strftime('%Y%m%d').astype(int)
    return industry_mf_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_stock_index_weights(index_code, date=None, raw=False):
    if not re.match(r'\d{4}-\d{2}-\d{2}', str(date)):
        raise ValueError('date must be formatted as "%Y-%m-%d"')

    if isinstance(index_code, str):
        index_code = [index_code]

    if raw:
        index_weight_list = []
        for _index_code in index_code:
            _index_weights = get_index_weights(_index_code, date=date)
            if len(_index_weights) > 0:
                index_weight_list.append(_index_weights)
        index_weight_df = pd.concat(index_weight_list, keys=index_code, names=['index_code', 'code'])
        index_weight_df = index_weight_df.reset_index()[['date', 'index_code', 'code', 'weight']]
    else:
        index_weight_string_list = []
        for _index_code in index_code:
            _index_weights = get_index_weights(_index_code, date=date)
            if len(_index_weights) > 0:
                _jq_date = _index_weights['date'].iloc[0]
                _weight_string_list = ';'.join(
                    '{},{}'.format(c, w) for c, w in _index_weights['weight'].sort_index().items())
                index_weight_string_list.append([_jq_date, _index_code, _weight_string_list])
        index_weight_df = pd.DataFrame(index_weight_string_list, columns=['date', 'index_code', 'weight'])

    index_weight_df['date'] = pd.to_datetime(index_weight_df['date']).dt.strftime('%Y%m%d').astype(int)
    index_weight_df['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    return index_weight_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def getValuation(date, code=None):
    query_args = (valuation,)
    filters = (valuation.id > -1,)
    if code is not None:
        if not isinstance(code, list):
            code = [code]

        # filters = [valuation.code.in_([code])]
        filters += (valuation.code.in_(code),)

    # all
    f_query = lambda x: get_fundamentals(x, date=date)
    valuation_df = get_unlimited_data(f_query, query_args, filters, verbose=0)
    valuation_df.rename(columns={'day': 'date'}, inplace=True)
    valuation_df.drop('id', axis=1, inplace=True)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        # valuation_df[c] = pd.to_datetime(valuation_df[c]).dt.strftime('%Y%m%d').astype(int)
        valuation_df[c] = pd.to_datetime(valuation_df[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)
    return valuation_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_stock_money_flow(code, start_date=None, end_date=None, fields=None, count=None):
    """Summary
            
        字段名 含义  备注
        date    日期  
        sec_code    股票代码    
        change_pct  涨跌幅(%)  
        net_amount_main 主力净额(万) 主力净额 = 超大单净额 + 大单净额
        net_pct_main    主力净占比(%)    主力净占比 = 主力净额 / 成交额
        net_amount_xl   超大单净额(万)    超大单：大于等于50万股或者100万元的成交单
        net_pct_xl  超大单净占比(%)   超大单净占比 = 超大单净额 / 成交额
        net_amount_l    大单净额(万) 大单：大于等于10万股或者20万元且小于50万股或者100万元的成交单
        net_pct_l   大单净占比(%)    大单净占比 = 大单净额 / 成交额
        net_amount_m    中单净额(万) 中单：大于等于2万股或者4万元且小于10万股或者20万元的成交单
        net_pct_m   中单净占比(%)    中单净占比 = 中单净额 / 成交额
        net_amount_s    小单净额(万) 小单：小于2万股或者4万元的成交单
        net_pct_s   小单净占比(%)    小单净占比 = 小单净额 / 成交额

    Args:
        code (TYPE): Description
        start_date (None, optional): Description
        end_date (None, optional): Description
        fields (None, optional): Description
        count (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    money_flow = get_money_flow(code, start_date=start_date, end_date=end_date, fields=fields, count=count)
    money_flow.rename(columns={'sec_code': 'code'}, inplace=True)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        money_flow[c] = pd.to_datetime(money_flow[c]).dt.strftime('%Y%m%d').astype(int)
    return money_flow
