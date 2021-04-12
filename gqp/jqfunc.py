# -*- coding: utf-8 -*-
# @Author: gabrielfeng
# @Date:   2020-05-01 17:56:56
# @Last Modified by:   Gabriel
# @Last Modified time: 2021-04-12 11:53:45

import pandas as pd
import math
import numpy as np
import datetime
import re

from retry import retry
from sqlalchemy.sql import func
# from gqp.HDFStore_Dynamic import compress_price_data, db_NaN, db_NaT_floor, db_NaT_ceil
from jqdatasdk import *    # run at local
from sqlalchemy import or_, true
# from jqdata import *    # run at cloud
auth('*', '*')

_retry = 1
_delay = 1

db_NaN = -1
db_NaT_floor = pd.to_datetime('1970-01-01').date()
db_NaT_ceil = pd.to_datetime('2200-01-01').date()

bond_type_map = {
	703001:	'短期融资券',
	703002:	'质押式回购',
	703003:	'私募债',
	703004:	'企业债',
	703005:	'次级债',
	703006:	'一般金融债',
	703007:	'中期票据',
	703008:	'资产支持证券',
	703009:	'小微企业扶持债',
	703010:	'地方政府债',
	703011:	'公司债',
	703012:	'可交换私募债',
	703013:	'可转债',
	703014:	'集合债券',
	703015:	'国际机构债券',
	703016:	'政府支持机构债券',
	703017:	'集合票据',
	703018:	'外国主权政府人民币债券',
	703019:	'央行票据',
	703020:	'政策性金融债',
	703021:	'国债',
	703022:	'非银行金融债',
	703023:	'可分离可转债',
	703024:	'国库定期存款',
	703025:	'可交换债',
	703026:	'特种金融债',
}

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
        'open_interest': 'int64',    # future and option
        'deal_number': 'int64',    # cbond_price
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
def getAllTradingDate():
    """Get all trading dates

    Returns:
        TYPE: Description
    """
    all_trade_days = get_all_trade_days()
    all_trade_days = list(map(lambda x: int(datetime.datetime.strftime(x, '%Y%m%d')), all_trade_days))
    return np.array(all_trade_days)


@retry(TimeoutError, tries=_retry, delay=_delay)
def getAllSecurities(categories=None, date=None):
    """Summary

        display_name # 中文名称
        name # 缩写简称
        start_date # 开始日期, [datetime.date] 类型
        end_date # 结束日期，[datetime.date] 类型
        type # 类型，futures(期货)

    Args:
        categories (None, optional): Description
        date (None, optional): Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    if categories is None:
        categories = ['stock', 'index', 'fund', 'futures', 'etf', 'lof', 'fja', 'fjb', 'open_fund', 'bond_fund', 'stock_fund', 'QDII_fund', 'money_market_fund', 'mixture_fund', 'options']
    elif isinstance(categories, list):
        pass
    elif isinstance(categories, str):
        categories = [categories]
    else:
        raise ValueError('categories must be None or list or str')

    df = pd.concat([get_all_securities(c, date=date).assign(category=c) for c in categories])
    df.index.name = 'code'
    df = df.reset_index()
    df['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    df['start_date'] = pd.to_datetime(df['start_date'].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)
    df['end_date'] = pd.to_datetime(df['end_date'].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)
    return df


@retry(TimeoutError, tries=_retry, delay=_delay)
def getBar(code, date, freq, fields=None, dropna=False, fillna=None):
    d0 = datetime.datetime.strptime(str(date), '%Y%m%d')
    if freq[-1] == 'm' or freq == 'minute':
        d1 = d0 + datetime.timedelta(days=1)
    else:
        d1 = d0

    if fields is None:
        fields = ['open', 'high', 'low', 'close', 'volume', 'money', 'high_limit', 'low_limit', 'pre_close', 'avg',
                  'paused', 'factor', 'open_interest']

    price_f = ['open', 'high', 'low', 'close', 'high_limit', 'low_limit', 'pre_close', 'avg']
    price_f = [f for f in fields if f in price_f]

    if isinstance(code, str):
        code = [code]

    # jq_df = get_price(code, start_date=d0, end_date=d1, frequency=freq, fields=fields, panel=False)
    try:
        df = get_price(code, start_date=d0, end_date=d1, frequency=freq, fields=fields, fq='post', fill_paused=True, panel=False)
    except AttributeError as e:
        raise AttributeError(';'.join(e.argv))
        exit(222)

    if df.shape[0] == 0:
        return pd.DataFrame(columns=['date', 'stamp', 'code', 'open', 'high', 'low', 'close', 'volume', 'money',
                                     'high_limit', 'low_limit', 'pre_close', 'avg', 'paused', 'factor', 'open_interest'])

    df[price_f] = np.round(df[price_f].values / df[['factor']].values, 6)
    df['volume'] = np.round(df['volume'].values * df['factor'].values, 0)
    # df.rename(columns={'time':'timestamp'}, inplace=True)
    df['date'] = df['time'].dt.strftime('%Y%m%d')
    df['stamp'] = df['time'].dt.strftime('%H%M%S%f').str[:-3]

    if dropna:
        jq_df = df.dropna(subset=['close', 'volume'], how='all').copy()
    else:
        jq_df = df.copy()

    save_columns = ['date', 'stamp', 'code'] + fields
    if jq_df.shape[0] == 0:
        out_df = pd.DataFrame(columns=save_columns)
    else:
        out_df = jq_df[save_columns].copy()

    if fillna is not None:
        out_df = compress_price_data(out_df.fillna(fillna))

    return out_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_cbond_info(start_date=None, end_date=None, **kwargs):
    """
    Gets the cbond basic info.
        {
            703012: '可交换私募债',
            703013: '可转债',
            703023: '可分离可转债',
            703025: '可交换债',
        }

    :param      date:  The date
    :type       date:  start with list date

    :returns:   The cbond basic info.
    :rtype:     { return_type_description }
    """
    queries = (bond.CONBOND_BASIC_INFO, )

    filters = (
        # bond.CONBOND_BASIC_INFO.bond_type_id.in_([703013, 703025]),    # 可转债类型ID
        bond.CONBOND_BASIC_INFO.bond_type_id > 0,
    )
    if start_date is not None:
        filters += (bond.CONBOND_BASIC_INFO.list_date >= start_date, )

    if end_date is not None:
        filters += (bond.CONBOND_BASIC_INFO.list_date <= end_date, )

    cbond_info = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    cbond_info.columns = cbond_info.columns.str.lower()

    date_ceil_coln = ['convert_end_date', 'delist_date', 'maturity_date', 'issue_end_date', 'list_declare_date']
    for c in date_ceil_coln:
        cbond_info[c] = pd.to_datetime(cbond_info[c].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)

    # date_floor_coln = ['convert_start_date', 'list_date']
    date_floor_coln = ['convert_start_date', 'list_date', 'issue_start_date', 'interest_begin_date', 'last_cash_date']
    for c in date_floor_coln:
        cbond_info[c] = pd.to_datetime(cbond_info[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    cbond_info['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    cbond_info = cbond_info.drop('id', axis=1)
    return cbond_info


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_cbond_convert(codes, start_date=None, end_date=None, **kwargs):
    """
    Gets the cbond code.

    :param      date:  The date
    :type       date:  start with list date

    :returns:   The cbond code.
    :rtype:     { return_type_description }
    """
    queries = (
        bond.CONBOND_DAILY_CONVERT,
    )

    filters = (
        # bond.CONBOND_DAILY_CONVERT.bond_type_id == 703013,    # 可转债类型ID
        # bond.CONBOND_DAILY_CONVERT.code.in_(code),
        bond.CONBOND_DAILY_CONVERT.id > -1,    # 可转债类型ID
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (bond.CONBOND_DAILY_CONVERT.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (bond.CONBOND_DAILY_CONVERT.code.in_([codes]),)
    else:
        raise ValueError('codes must be None or list or str')

    if start_date is not None:
        filters += (bond.CONBOND_DAILY_CONVERT.date >= start_date, )

    if end_date is not None:
        filters += (bond.CONBOND_DAILY_CONVERT.date <= end_date, )

    cbond_info = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    cbond_info.columns = cbond_info.columns.str.lower()
    cbond_info = cbond_info.drop('id', axis=1)
    cbond_info['date'] = pd.to_datetime(cbond_info['date'].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)
    cbond_info = compress_price_data(cbond_info).fillna(db_NaN)
    return cbond_info


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_cbond_price(codes=None, start_date='1970-01-01', end_date='2200-01-01', **kwargs):
    queries = (
        bond.CONBOND_DAILY_PRICE,
    )

    filters = (
        bond.CONBOND_DAILY_PRICE.date >= start_date,
        bond.CONBOND_DAILY_PRICE.date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (bond.CONBOND_DAILY_PRICE.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (bond.CONBOND_DAILY_PRICE.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    cbond_price = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    cbond_price['stamp'] = int(15e7)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        # cbond_price[c] = pd.to_datetime(cbond_price[c]).dt.strftime('%Y%m%d').astype(int)
        cbond_price[c] = pd.to_datetime(cbond_price[c].fillna(0)).dt.strftime('%Y%m%d').astype(int)

    # dtypes_map = {
    #     int: ['date', 'volume'],
    #     str: ['code', 'name', 'exchange_code'],
    #     float: ['open', 'high', 'low', 'close', 'pre_close', 'money', 'deal_number', 'change_pct']
    # }
    # for t0, ks in dtypes_map.items():
    #     for c, t1 in cbond_price.dtypes.items():
    #         if c in ks:
    #             if t1 != t0:
    #                 cbond_price[c] = cbond_price[c].astype(t0)
    cbond_price = cbond_price.drop('id', axis=1)
    cbond_price = compress_price_data(cbond_price).fillna(db_NaN)
    return cbond_price


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_option_info(date=None, underlying_code=None, underlying_type=None, list_date=None, **kwargs):
    queries = (opt.OPT_CONTRACT_INFO, )
    filters = (opt.OPT_CONTRACT_INFO.id >= -1,)

    if date is not None:
        if isinstance(date, int):
            date = datetime.datetime.strptime(str(date), '%Y%m%d').date()
        filters += (opt.OPT_CONTRACT_INFO.list_date <= date, opt.OPT_CONTRACT_INFO.last_trade_date >= date)

    if underlying_code is not None:
        if isinstance(underlying_code, str):
            underlying_code = [underlying_code]
        filters += (opt.OPT_CONTRACT_INFO.underlying_symbol.in_(underlying_code),)

    if underlying_type is not None:
        if isinstance(underlying_type, str):
            underlying_type = [underlying_type]
        filters += (opt.OPT_CONTRACT_INFO.underlying_type.in_(underlying_type),)

    if list_date is not None:
        if not isinstance(list_date, list):
            list_date = [list_date]
        filters += (opt.OPT_CONTRACT_INFO.list_date.in_(list_date),)

    option_info = get_unlimited_data(opt.run_query, queries, filters, **kwargs)
    option_info['contract_type'] = option_info['contract_type'].map({'CO': 1, 'PO': 2})
    option_info.rename(columns={'underlying_symbol': 'underlying_code'}, inplace=True)
    option_info.drop(['id', 'trading_code'], axis=1, inplace=True)
    jq_date_columns = ['list_date', 'expire_date', 'last_trade_date', 'exercise_date', 'delivery_date', 'delist_date']
    for c in jq_date_columns:
        option_info[c] = pd.to_datetime(option_info[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    option_info['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    return option_info


@retry(TimeoutError, tries=_retry, delay=_delay)
def getFundamental(stat_dates, sections, pubDate=None):
    if pubDate is not None:
        if not isinstance(pubDate, list):
            pubDate = [pubDate]
    else:
        pubDate = []

    for statDate in stat_dates:
        for sec in sections:
            # # may be incomplete
            # rawdf = get_fundamentals(query(eval(sec)), statDate=statDate)

            # all
            f_query = lambda x: get_fundamentals(x, statDate=statDate)
            query_args = (eval(sec), )
            filters = (eval('{sec}.id >= -1'.format(sec=sec)),)
            if len(pubDate) > 0:
                filters += (eval('{sec}.pubDate.in_({pDate})'.format(sec=sec, pDate=pubDate)), )

            rawdf = get_unlimited_data(f_query, query_args, filters, verbose=0)

            if 'statDate.1' in rawdf.columns:
                rawdf.drop('statDate.1', axis=1, inplace=True)
            df = rawdf.rename(columns={'id': 'jqid'}).assign(table=sec, query_date=statDate)
            df = df.melt(id_vars=['code', 'statDate', 'pubDate', 'table', 'query_date'], var_name='field', value_name='value')
            # df['statDate'] = pd.to_datetime(df['statDate']).dt.strftime('%Y%m%d').astype(int)
            # df['pubDate'] = pd.to_datetime(df['pubDate']).dt.strftime('%Y%m%d').astype(float)
            jq_date_columns = ['statDate', 'pubDate']
            for c in jq_date_columns:
                df[c] = pd.to_datetime(df[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)
            yield df


@retry(TimeoutError, tries=_retry, delay=_delay)
def getFundamental_quarterly(stat_dates, sections=None, pubDate=None):
    """
    get quarter report

    stat_dates(list of strings): '2014q1', '2014q2', '2015q1', '2016q1', etc.
    sections(list of strings): 'balance', 'income', 'cash_flow', 'indicator'
    """
    if isinstance(stat_dates, str):
        stat_dates = [stat_dates]

    if sections is None:
        sections = ['balance', 'income', 'cash_flow', 'indicator']
    elif isinstance(sections, list):
        pass
    elif isinstance(sections, str):
        sections = [sections]
    else:
        raise ValueError('sections must be None or list or str')

    if pubDate is not None:
        if not isinstance(pubDate, list):
            pubDate = [pubDate]

    dfs = getFundamental(stat_dates, sections, pubDate=pubDate)
    return pd.concat(dfs)


@retry(TimeoutError, tries=_retry, delay=_delay)
def getFundamental_annually(stat_dates, sections=None, pubDate=None):
    """
    get annual report

    stat_dates(list of strings): '2014', '2015', '2016', etc.
    sections(list of strings): 'balance', 'income', 'cash_flow', 'indicator', 'bank_indicator', 'security_indicator', 'insurance_indicator'
    """

    if isinstance(stat_dates, str):
        stat_dates = [stat_dates]

    if sections is None:
        sections = ['balance', 'income', 'cash_flow', 'indicator', 'bank_indicator', 'security_indicator', 'insurance_indicator']
    elif isinstance(sections, list):
        pass
    elif isinstance(sections, str):
        sections = [sections]
    else:
        raise ValueError('sections must be None or list or str')


    if pubDate is not None:
        if not isinstance(pubDate, list):
            pubDate = [pubDate]

    dfs = getFundamental(stat_dates, sections, pubDate=pubDate)

    return pd.concat(dfs)


@retry(TimeoutError, tries=_retry, delay=_delay)
def getValuation(date, code=None):
    query_args = (valuation, )
    filters = (valuation.id > -1, )
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
def get_fund_main_info(codes=None, start_date='2200-01-01', end_date='1970-01-01'):
    """Summary

    Args:
        codes (None, optional): Description
        start_date (str, optional): Description
        end_date (str, optional): Description

    Returns:
        TYPE: pandas.DataFrame
            字段  名称  类型
            main_code   基金主体代码  varchar(12)
            name    基金名称    varchar(100)
            advisor 基金管理人   varchar(100)
            trustee 基金托管人   varchar(100)
            operate_mode_id 基金运作方式编码    int
            operate_mode    基金运作方式  varchar(32)
            underlying_asset_type_id    投资标的类型编码    int
            underlying_asset_type   投资标的类型  varchar(32)
            start_date  成立日期    date
            end_date    结束日期    date
            基金运作方式编码

            编码  基金运作方式
            401001  开放式基金
            401002  封闭式基金
            401003  QDII
            401004  FOF
            401005  ETF
            401006  LOF
            基金类别编码

            编码  基金类别
            402001  股票型
            402002  货币型
            402003  债券型
            402004  混合型
            402005  基金型
            402006  贵金属
            402007  封闭式

    Raises:
        ValueError: Description
    """
    query_args = (finance.FUND_MAIN_INFO,)
    filters = (
        finance.FUND_MAIN_INFO.start_date <= start_date,
        # finance.FUND_MAIN_INFO.end_date >= end_date,
        or_(
            finance.FUND_MAIN_INFO.end_date >= end_date,
            finance.FUND_MAIN_INFO.end_date.is_(None)
        )
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_MAIN_INFO.main_code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_MAIN_INFO.main_code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    fund_main_info_df = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)
    fund_main_info_df = fund_main_info_df.drop(['id'], axis=1).reset_index(drop=True)
    fund_main_info_df['start_date'] = pd.to_datetime(fund_main_info_df['start_date'].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)
    fund_main_info_df['end_date'] = pd.to_datetime(fund_main_info_df['end_date'].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)
    fund_main_info_df['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    fund_main_info_df.rename(columns={'main_code': 'code'}, inplace=True)
    return fund_main_info_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_fund_net_value(codes=None, start_date='1970-01-01', end_date='2200-01-01'):
    """Summary

    Args:
        codes (None, optional): Description
        start_date (str, optional): Description
        end_date (str, optional): Description

    Returns:
        TYPE: pandas.DataFrame

            字段  名称  类型  注释
            code    基金代码    varchar(12)
            day 交易日 date
            net_value   单位净值    decimal(20,6)   基金单位净值=（基金资产总值－基金负债）÷ 基金总份额
            sum_value   累计净值    decimal(20,6)   累计单位净值＝单位净值＋成立以来每份累计分红派息的金额
            factor  复权因子    decimal(20,6)   交易日最近一次分红拆分送股的复权因子
            acc_factor  累计复权因子  decimal(20,6)   复权因子的累乘
            refactor_net_value  累计复权净值  decimal(20,6)   单位净值*累计复权因子

    Raises:
        ValueError: Description
    """
    query_args = (finance.FUND_NET_VALUE,)
    filters = (
        finance.FUND_NET_VALUE.day >= start_date,
        finance.FUND_NET_VALUE.day <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_NET_VALUE.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_NET_VALUE.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    fund_net_value_df = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)
    fund_net_value_df = fund_net_value_df.drop(['id'], axis=1).reset_index(drop=True)
    fund_net_value_df.rename(columns={'day': 'date'}, inplace=True)
    fund_net_value_df['date'] = pd.to_datetime(fund_net_value_df['date']).dt.strftime('%Y%m%d').astype(int)
    return fund_net_value_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_fund_holdings(codes, start_date='1970-01-01', end_date='2200-01-01', report_type_id=None):
    """Summary

    Args:
        codes (TYPE): Description
        start_date (str, optional): Description
        end_date (str, optional): Description
        report_type_id (None, optional): Description

    Returns:
        TYPE: Description

            字段名称    中文名称    字段类型
            category  资产种类    varchar
            code    基金代码    varchar(12)
            period_start    开始日期    date
            period_end  报告期 date
            pub_date    公告日期    date
            report_type_id  报告类型编码  int
            report_type 报告类型    varchar(32)
            rank    持仓排名    int
            symbol  股票代码    varchar(32)
            name    股票名称    varchar(100)
            shares  持有股票    decimal(20,4)
            market_cap  持有股票的市值 decimal(20,4)
            proportion  占净值比例   decimal(10,4)
    Raises:
        ValueError: Description
    """
    # fetch stock detail
    query_args = (finance.FUND_PORTFOLIO_STOCK,)
    filters = (
        finance.FUND_PORTFOLIO_STOCK.pub_date >= start_date,
        finance.FUND_PORTFOLIO_STOCK.pub_date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_PORTFOLIO_STOCK.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_PORTFOLIO_STOCK.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    if report_type_id is None:
        pass
    elif isinstance(report_type_id, list):
        filters += (finance.FUND_PORTFOLIO_STOCK.report_type_id.in_(report_type_id),)
    elif isinstance(report_type_id, int):
        filters += (finance.FUND_PORTFOLIO_STOCK.report_type_id.in_([report_type_id]),)
    else:
        raise ValueError('codes must be None, list or int')

    fund_portfolio_stock_df = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)

    # fetch bond detail
    query_args = (finance.FUND_PORTFOLIO_BOND,)
    filters = (
        finance.FUND_PORTFOLIO_BOND.pub_date >= start_date,
        finance.FUND_PORTFOLIO_BOND.pub_date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_PORTFOLIO_BOND.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_PORTFOLIO_BOND.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    fund_portfolio_bond_df = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)

    fund_portfolio_detail = pd.concat([fund_portfolio_stock_df, fund_portfolio_bond_df], keys=['stock', 'bond'], names=['category', 'id2'])
    fund_portfolio_detail = fund_portfolio_detail.reset_index().drop(['id', 'id2'], axis=1)

    date_ceil_coln = ['period_end']
    for c in date_ceil_coln:
        fund_portfolio_detail[c] = pd.to_datetime(fund_portfolio_detail[c].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)

    # date_floor_coln = ['convert_start_date', 'list_date']
    date_floor_coln = ['period_start', 'pub_date']
    for c in date_floor_coln:
        fund_portfolio_detail[c] = pd.to_datetime(fund_portfolio_detail[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    return fund_portfolio_detail


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_fund_financials(codes, start_date='1970-01-01', end_date='2200-01-01', report_type_id=None):
    query_args = (finance.FUND_FIN_INDICATOR,)
    filters = (
        finance.FUND_FIN_INDICATOR.pub_date >= start_date,
        finance.FUND_FIN_INDICATOR.pub_date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_FIN_INDICATOR.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_FIN_INDICATOR.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    fund_financials = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)
    fund_financials = fund_financials.reset_index(drop=True).drop(['id'], axis=1)

    date_ceil_coln = ['period_end']
    for c in date_ceil_coln:
        fund_financials[c] = pd.to_datetime(fund_financials[c].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)

    # date_floor_coln = ['convert_start_date', 'list_date']
    date_floor_coln = ['period_start', 'pub_date']
    for c in date_floor_coln:
        fund_financials[c] = pd.to_datetime(fund_financials[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    return fund_financials


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_fund_portfolio_summary(codes, start_date='1970-01-01', end_date='2200-01-01'):
    query_args = (finance.FUND_PORTFOLIO,)
    filters = (
        finance.FUND_PORTFOLIO.pub_date >= start_date,
        finance.FUND_PORTFOLIO.pub_date <= end_date,
    )

    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_PORTFOLIO.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_PORTFOLIO.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    fund_portfolio_summary = get_unlimited_data(finance.run_query, query_args, filters, verbose=0)
    fund_portfolio_summary = fund_portfolio_summary.drop(['id'], axis=1).reset_index(drop=True)
    return fund_portfolio_summary


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_money_fund_daily_report(codes, start_date='1970-01-01', end_date='2200-01-01'):
    queires = (finance.FUND_MF_DAILY_PROFIT,)
    filters = (
        finance.FUND_MF_DAILY_PROFIT.end_date >= start_date,
        finance.FUND_MF_DAILY_PROFIT.end_date <= end_date,
    )
    if codes is None:
        pass
    elif isinstance(codes, list):
        filters += (finance.FUND_MF_DAILY_PROFIT.code.in_(codes),)
    elif isinstance(codes, str):
        filters += (finance.FUND_MF_DAILY_PROFIT.code.in_([codes]),)
    else:
        raise ValueError('codes must be None, list or str')

    mf_daily_profit = get_unlimited_data(finance.run_query, queires, filters, verbose=0)
    mf_daily_profit = mf_daily_profit.drop(['id'], axis=1).reset_index(drop=True)
    mf_daily_profit['end_date'] = pd.to_datetime(mf_daily_profit['end_date']).dt.strftime('%Y%m%d').astype(int)
    return mf_daily_profit


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_bond_info(start_date=None, end_date=None, bond_type_id=None, **kwargs):
    """
    Gets the bond basic info.
        {
            703001: '短期融资券', 703002: '质押式回购', 703003: '私募债',
            703004: '企业债', 703005: '次级债', 703006: '一般金融债',
            703007: '中期票据', 703008: '资产支持证券', 703009: '小微企业扶持债',
            703010: '地方政府债', 703011: '公司债', 703012: '可交换私募债',
            703013: '可转债', 703014: '集合债券', 703015: '国际机构债券',
            703016: '政府支持机构债券', 703017: '集合票据',
            703018: '外国主权政府人民币债券', 703019: '央行票据',
            703020: '政策性金融债', 703021: '国债', 703022: '非银行金融债',
            703023: '可分离可转债', 703024: '国库定期存款', 703025: '可交换债',
            703026: '特种金融债',
        }

    :param      date:  The date
    :type       date:  start with list date

    :returns:   The bond basic info.
    :rtype:     { return_type_description }
    """
    queries = (bond.BOND_BASIC_INFO, )

    filters = (
        # bond.CONBOND_BASIC_INFO.bond_type_id.in_([703013, 703025]),    # 可转债类型ID
        # bond.BOND_BASIC_INFO.bond_type_id > 0,
        bond.BOND_BASIC_INFO.bond_form_id > 0,    # dummy
    )

    if start_date is not None:
        filters += (bond.BOND_BASIC_INFO.list_date >= start_date, )

    if end_date is not None:
        filters += (bond.BOND_BASIC_INFO.list_date <= end_date, )

    if bond_type_id is not None:
        filters += (bond.BOND_BASIC_INFO.bond_type_id.in_(bond_type_id), )

    bond_info = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    bond_info.columns = bond_info.columns.str.lower()

    date_ceil_coln = ['delist_date', 'maturity_date']
    for c in date_ceil_coln:
        bond_info[c] = pd.to_datetime(bond_info[c].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)

    # date_floor_coln = ['convert_start_date', 'list_date']
    date_floor_coln = ['list_date', 'interest_begin_date', 'last_cash_date']
    for c in date_floor_coln:
        bond_info[c] = pd.to_datetime(bond_info[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    bond_info['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    bond_info = bond_info.drop('id', axis=1)
    return bond_info


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_bond_coupon(start_date=None, end_date=None, coupon_type_id=None, **kwargs):
    """
    Gets the bond basic info.
        {
            703001: '短期融资券', 703002: '质押式回购', 703003: '私募债',
            703004: '企业债', 703005: '次级债', 703006: '一般金融债',
            703007: '中期票据', 703008: '资产支持证券', 703009: '小微企业扶持债',
            703010: '地方政府债', 703011: '公司债', 703012: '可交换私募债',
            703013: '可转债', 703014: '集合债券', 703015: '国际机构债券',
            703016: '政府支持机构债券', 703017: '集合票据',
            703018: '外国主权政府人民币债券', 703019: '央行票据',
            703020: '政策性金融债', 703021: '国债', 703022: '非银行金融债',
            703023: '可分离可转债', 703024: '国库定期存款', 703025: '可交换债',
            703026: '特种金融债',
        }

    :param      date:  The date
    :type       date:  start with list date

    :returns:   The bond basic info.
    :rtype:     { return_type_description }
    """
    queries = (bond.BOND_COUPON, )

    filters = (
        # bond.CONBOND_BASIC_INFO.bond_type_id.in_([703013, 703025]),    # 可转债类型ID
        # bond.BOND_BASIC_INFO.bond_type_id > 0,
        bond.BOND_COUPON.id > 0,    # dummy
    )

    if start_date is not None:
        filters += (bond.BOND_COUPON.pub_date >= start_date, )

    if end_date is not None:
        filters += (bond.BOND_COUPON.pub_date <= end_date, )

    if coupon_type_id is not None:
        filters += (bond.BOND_COUPON.coupon_type_id.in_(coupon_type_id), )

    bond_coupon = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    bond_coupon.columns = bond_coupon.columns.str.lower()

    date_ceil_coln = ['coupon_end_date']
    for c in date_ceil_coln:
        bond_coupon[c] = pd.to_datetime(bond_coupon[c].fillna(db_NaT_ceil)).dt.strftime('%Y%m%d').astype(int)

    # date_floor_coln = ['convert_start_date', 'list_date']
    date_floor_coln = ['pub_date', 'coupon_start_date']
    for c in date_floor_coln:
        bond_coupon[c] = pd.to_datetime(bond_coupon[c].fillna(db_NaT_floor)).dt.strftime('%Y%m%d').astype(int)

    bond_coupon.rename(columns={'pub_date': 'pubdate'}, inplace=True)
    bond_coupon['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    bond_coupon = bond_coupon.drop('id', axis=1)
    return bond_coupon


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
            return pd.DataFrame(columns=['date', 'code', 'net_amount_main', 'net_amount_xl', 'net_amount_l', 'net_amount_m', 'net_amount_s'])

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
    industry_mf_list = [_get_industry_daily_moneyflow(co, start_date=start_date, end_date=end_date) for co in industry_code]
    industry_mf_df = pd.concat(industry_mf_list)
    industry_mf_df['date'] = pd.to_datetime(industry_mf_df['date']).dt.strftime('%Y%m%d').astype(int)
    return industry_mf_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_interbank_lend_rate(start_date=None, end_date=None, currency_id=None, market_id=None, **kwargs):
    """Summary
    Instruction
        currency_id - 货币编码
            1-人民币；2-港币；3-美元USD；4-日本元JPY；5-英镑GBP；6-欧元；7-瑞士法郎；16-新加坡元；20-德国马克；
            21-法国法郎；22-ECU；23-澳大利亚元；24-AUS 25-加拿大元；26-西班牙比塞塔；27-意大利里拉；28-荷兰盾；
            29-PTE；30-XEU；31-丹麦克朗；32-新西兰元；33-瑞典克朗
        market_id - 拆借市场编码
            1 - 香港银行同业拆借利率 HIBOR；
            2 - 伦敦银行同业拆借利率 LIBOR；
            3 - 中国银行同业拆借利率 CHIBOR；
            4 - 新加坡银行同业拆借利率 SIBOR；
            5 - 上海银行间同业拆放利率 SHIBOR；
        term_id - 拆借期限编码
            隔夜=20，一周=7，两周=14，三周=9，一月=1，两月=2，三月=3，四月=4,五月=5，
            六月=6，七月=21，八月=22，九月=23,十月=24，十一月=25，一年=12。注意不是每个拆借市场都支持所有的拆解周期

    Args:
        start_date (None, optional): Description
        end_date (None, optional): Description
        currency_id (None, optional): Description
        market_id (None, optional): Description
        **kwargs: Description

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    queries = (macro.MAC_LEND_RATE,)

    filters = (
        macro.MAC_LEND_RATE.id > -1,  # dummy
    )

    if start_date is not None:
        filters += (macro.MAC_LEND_RATE.day >= start_date,)

    if end_date is not None:
        filters += (macro.MAC_LEND_RATE.day <= end_date,)

    if currency_id is None:
        pass
    elif isinstance(currency_id, list):
        filters += (macro.MAC_LEND_RATE.currency_id.in_(currency_id),)
    elif isinstance(currency_id, int):
        filters += (macro.MAC_LEND_RATE.currency_id.in_([currency_id]),)
    else:
        raise ValueError('codes must be None, list or int')

    if market_id is None:
        pass
    elif isinstance(market_id, list):
        filters += (macro.MAC_LEND_RATE.market_id.in_(market_id),)
    elif isinstance(market_id, int):
        filters += (macro.MAC_LEND_RATE.market_id.in_([market_id]),)
    else:
        raise ValueError('codes must be None, list or int')

    lend_rate = get_unlimited_data(macro.run_query, queries, filters, **kwargs)
    lend_rate.columns = lend_rate.columns.str.lower()

    lend_rate['date'] = pd.to_datetime(lend_rate['day']).dt.strftime('%Y%m%d').astype(int)
    lend_rate['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    lend_rate = lend_rate.drop(['id', 'day'], axis=1)
    return lend_rate


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_macro(table_name, **kwargs):
    """Summary

    Args:
        table_name (TYPE): Description
        **kwargs: Description

    Returns:
        TYPE: pandas.DataFrame

            MAC_MANUFACTURING_PMI - 制造业采购经理指数（月度）
            MAC_AREA_GDP_QUARTER - 分地区国内生产总值表(季度)
            MAC_AREA_GDP_YEAR - 分地区国内生产总值表(年度)
            MAC_NONMANUFACTURING_PMI - 非制造业采购经理指数（月度）
            MAC_AREA_CPI_MONTH - 分地区居民消费价格指数（月度）
            MAC_CPI_MONTH - MAC_CPI_MONTH
            MAC_MONEY_SUPPLY_MONTH - 货币供应量(月度)
            MAC_MONEY_SUPPLY_YEAR - 货币供应量(年度)
            MAC_SOCIAL_SCALE_FINANCE - 社会融资规模及构成（年度）
            MAC_GOLD_FOREIGN_RESERVE - 黄金和外汇储备（月度）
    """
    jqdatasdk_table = eval('macro.{}'.format(table_name))
    queries = (jqdatasdk_table,)
    filters = (jqdatasdk_table.id > -1,)
    result = get_unlimited_data(macro.run_query, queries, filters, **kwargs)
    return result


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_ah_money_flow(start_date=None, end_date=None, **kwargs):
    """Summary

    Args:
        start_date (None, optional): Description
        end_date (None, optional): Description
        **kwargs: Description

    Returns:
        TYPE: pandas.DataFrame

            字段             名称               类型          备注/示例
            day                 交易日期        date
            link_id             市场通编码      int
            link_name           市场通名称      varchar(32) 包括以下四个名称： 沪股通，深股通，港股通(沪）,港股通(深）;其中沪股通和深股通属于北向资金，港股通（沪）和港股通（深）属于南向资金。
            currency_id         货币编码        int
            currency            货币名称        varchar(16)
            buy_amount          买入成交额      decimal(20,4)   亿
            buy_volume          买入成交数      decimal(20,4)   笔
            sell_amount         卖出成交额      decimal(20,4)   亿
            sell_volume         卖出成交数      decimal(20,4)   笔
            sum_amount          累计成交额      decimal(20,4)   买入成交额+卖出成交额
            sum_volume          累计成交数目    decimal(20,4)   买入成交量+卖出成交量
            quota               总额度          decimal(20, 4)  亿（2016-08-16号起，沪港通和深港通不再设总额度限制）
            quota_balance       总额度余额      decimal(20, 4)  亿
            quota_daily         每日额度        decimal(20, 4)  亿
            quota_daily_balance 每日额度余额    decimal(20, 4)  亿
    """
    queries = (finance.STK_ML_QUOTA, )

    filters = (
        finance.STK_ML_QUOTA.id > 0,    # dummy
    )

    if start_date is not None:
        filters += (finance.STK_ML_QUOTA.day >= start_date, )

    if end_date is not None:
        filters += (finance.STK_ML_QUOTA.day <= end_date, )

    money_flow = get_unlimited_data(finance.run_query, queries, filters, **kwargs)
    money_flow.columns = money_flow.columns.str.lower()
    money_flow.rename(columns={'day': 'date'}, inplace=True)
    money_flow.drop('id', axis=1, inplace=True)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        money_flow[c] = pd.to_datetime(money_flow[c]).dt.strftime('%Y%m%d').astype(int)
    return money_flow


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_stock_money_flow(code, start_date=None, end_date=None, fields=None, count=None):
    """Summary

    Args:
        code (TYPE): Description
        start_date (None, optional): Description
        end_date (None, optional): Description
        fields (None, optional): Description
        count (None, optional): Description

    Returns:
        TYPE: pandas.DataFrame

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
    """
    money_flow = get_money_flow(code, start_date=start_date, end_date=end_date, fields=fields, count=count)
    money_flow.rename(columns={'sec_code': 'code'}, inplace=True)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        money_flow[c] = pd.to_datetime(money_flow[c]).dt.strftime('%Y%m%d').astype(int)
    return money_flow


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
                _weight_string_list = ';'.join('{},{}'.format(c, w) for c, w in _index_weights['weight'].sort_index().items())
                index_weight_string_list.append([_jq_date, _index_code, _weight_string_list])
        index_weight_df = pd.DataFrame(index_weight_string_list, columns=['date', 'index_code', 'weight'])

    index_weight_df['date'] = pd.to_datetime(index_weight_df['date']).dt.strftime('%Y%m%d').astype(int)
    index_weight_df['update_time'] = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    return index_weight_df


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_repo_daily_price(code=None, name=None, start_date='1970-01-01', end_date='2200-01-01', **kwargs):
    """REPO_DAILY_PRICE
    # https://www.joinquant.com/help/api/help#bond:%E5%9B%BD%E5%80%BA%E9%80%86%E5%9B%9E%E8%B4%AD%E6%97%A5%E8%A1%8C%E6%83%85%EF%BC%88REPO_DAILY_PRICE%EF%BC%89
    from jqdata import bond
    df=bond.run_query(query(bond.REPO_DAILY_PRICE))


    Args:
        code (None, optional): Description
        name (None, optional): Description
        start_date (str, optional): Description
        end_date (str, optional): Description
        **kwargs: Description

    Returns:
        TYPE: pandas.DataFrame

            名称  类型  描述
            date    date    交易日期
            code    varchar(12) 回购代码，如 '204001.XSHG'
            name    varchar(20) 回购简称，如 'GC001'
            exchange_code   varchar(12) 证券市场编码。XSHG-上海证券交易所；XSHE-深圳证券交易所
            pre_close   decimal(10,4)   前收盘利率(%)
            open    decimal(10,4)   开盘利率(%)
            high    decimal(10,4)   最高利率(%)
            low decimal(10,4)   最低利率(%)
            close   decimal(10,4)   收盘利率(%)
            volume  bigint  成交量（手）
            money   decimal（20,2）   成交额（元）
            deal_number int 成交笔数（笔）

    Raises:
        ValueError:

    """
    queries = (
        bond.REPO_DAILY_PRICE,
    )

    filters = (
        bond.REPO_DAILY_PRICE.date >= start_date,
        bond.REPO_DAILY_PRICE.date <= end_date,
    )

    if code is None:
        pass
    elif isinstance(code, list):
        filters += (bond.REPO_DAILY_PRICE.code.in_(code),)
    elif isinstance(code, str):
        filters += (bond.REPO_DAILY_PRICE.code.in_([code]),)
    else:
        raise ValueError('code must be None, list or str')

    if name is None:
        pass
    elif isinstance(name, list):
        filters += (bond.REPO_DAILY_PRICE.name.in_(name),)
    elif isinstance(name, str):
        filters += (bond.REPO_DAILY_PRICE.name.in_([name]),)
    else:
        raise ValueError('name must be None, list or str')

    repo_price = get_unlimited_data(bond.run_query, queries, filters, **kwargs)
    repo_price['stamp'] = int(15e7)
    jq_date_columns = ['date']
    for c in jq_date_columns:
        repo_price[c] = pd.to_datetime(repo_price[c].fillna(0)).dt.strftime('%Y%m%d').astype(int)

    repo_price = repo_price.drop('id', axis=1)
    repo_price = compress_price_data(repo_price).fillna(db_NaN)
    return repo_price


@retry(TimeoutError, tries=_retry, delay=_delay)
def get_future_global_daily(code=None, name=None, start_date='1970-01-01', end_date='2200-01-01', **kwargs):
    """REPO_DAILY_PRICE
    # https://www.joinquant.com/help/api/help#Future:%E8%8E%B7%E5%8F%96%E5%A4%96%E7%9B%98%E6%9C%9F%E8%B4%A7%E6%97%A5%E8%A1%8C%E6%83%85%E6%95%B0%E6%8D%AE

    Args:
        code (None, optional): Description
        name (None, optional): Description
        start_date (str, optional): Description
        end_date (str, optional): Description
        **kwargs: Description

    Returns:
        TYPE: pandas.DataFrame

        字段	名称	类型	非空	含义
        code	期货代码	varchar(64)	Y	代码列表详见下方期货代码名称对照表
        name	期货名称	varchar(64)
        day	日期	date	Y
        open	开盘价	decimal(20,6)
        close	收盘价	decimal(20,6)
        low	最低价	decimal(20,6)
        high	最高价	decimal(20,6)
        volume	成交量	decimal(20,6)
        change_pct	涨跌幅（%）	decimal(20,4)		（当日收盘价-前收价）/前收价
        amplitude	振幅（%）	decimal(20,6)		（当日最高点的价格－当日最低点的价格）/前收价
        pre_close	前收价	decimal(20,6)

    Raises:
        ValueError:

    """
    queries = (
        finance.FUT_GLOBAL_DAILY,
    )

    filters = (
        finance.FUT_GLOBAL_DAILY.day >= start_date,
        finance.FUT_GLOBAL_DAILY.day <= end_date,
    )

    if code is None:
        pass
    elif isinstance(code, list):
        filters += (finance.FUT_GLOBAL_DAILY.code.in_(code),)
    elif isinstance(code, str):
        filters += (finance.FUT_GLOBAL_DAILY.code.in_([code]),)
    else:
        raise ValueError('code must be None, list or str')

    fut_price = get_unlimited_data(finance.run_query, queries, filters, **kwargs)
    fut_price = fut_price.rename(columns={'day': 'date'})
    jq_date_columns = ['date']
    for c in jq_date_columns:
        fut_price[c] = pd.to_datetime(fut_price[c].fillna(0)).dt.strftime('%Y%m%d').astype(int)

    fut_price = fut_price.drop('id', axis=1)
    fut_price = compress_price_data(fut_price).fillna(db_NaN)
    return fut_price


if __name__ == '__main__':
    df = get_future_global_daily('OIL', '2015-01-01', '2021-01-01')
