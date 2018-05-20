"""查询企业基本面和宏观
get_security_info
查询单个标的的信息
get_fundamentals
查询财务数据，包含估值表、利润表、现金流量表、资产负债表、银行专项指标、证券专项指标、保险专项指标, 具体见https://www.joinquant.com/data/dict/fundamentals
get_fundamentals_continuously
查询多日的财务数据
get_extras
查询股票是否是ST
查询基金的累计净值、单位净值
查询期货的结算价、持仓量

macro.run_query
查询宏观经济数据，具体数据见官网API https://www.joinquant.com/data/dict/macroData
"""

"""现金流相关
get_billbord_list
查询股票龙虎榜数据
get_locked_shares
查询股票限售解禁股数据
get_margincash_stocks
获取融资标的列表
get_marginsec_stocks
获取融券标的列表
get_mtss
查询股票、基金的融资融券数据
get_money_flow
查询某只股票的资金流向数据
"""

import pandas as pd
import numpy as np
import scipy.stats as scs

import jqdatasdk as jqd
# from jqdatasdk import query, macro, valuation, income, cash_flow

# login
jqd.auth('13510687238','freestyle')

# Exploration on Industries
indus = jqd.get_industries('jq_l1')
stkcodes_by_indus = {indcode: jqd.get_industry_stocks(indcode) for indcode in indus.index}
indus['#stocks'] = pd.Series({k: len(v) for k, v in stkcodes_by_indus.items()})
indus.sort_values('#stocks')

# Basic Infos
begD = '2013-05-01'
endD = '2018-05-01'
trade_days = jqd.get_trade_days(start_date=begD, end_date=endD)
stkcodes = jqd.get_industry_stocks('HY005')

# Prices
def f_get_price(codes, start_date, end_date, fields, verbose=True):
    for co in codes:
        data = jqd.get_price(co, start_date=start_date, end_date=end_date,
                             fields=fields, skip_paused=True, fq='pre')
        data['code'] = co
        if verbose:
            print co, data.head()
        yield data

def f_get_daily_return(codes, start_date, end_date, verbose=True):
    for co in codes:
        data = jqd.get_price(co, start_date=start_date, end_date=end_date,
                             fields='close', skip_paused=True, fq='pre')
        rtn = np.log(data['close']).diff().rename(co)
        if verbose:
            print co, rtn.head()
        yield rtn


rtn_df = pd.concat(f_get_daily_return(stkcodes, begD, endD), 1)



# Fundamentals
# def f_get_fundamental(codes, dates, sections):
#     for sec in sections:
#         expression_str = 'jqd.query({}).filter({}.code == '.format(sec, sec)
#         for co in codes:
#             q = eval(expression_str + "'{}')".format(co))
#             for td in dates:
#                 yield jqd.get_fundamentals(q, date=td.strftime('%Y-%m-%d'))
#         print sec, 'loaded'


def f_get_fundamental(codes, dates, sections, verbose=True):
    codes = [codes] if isinstance(codes, str) else codes
    dates = [dates] if isinstance(dates, str) else dates
    sections = [sections] if isinstance(sections, str) else sections

    for sec in sections:    # sections写循环是为避开提取全市场时的提取上限
        recorded_info_index = np.full((1, 3), np.nan)
        for td in dates:
            td_str = td.strftime('%Y-%m-%d')
            q = eval('jqd.query(' + sec + ').filter(jqd.valuation.code.in_(codes))')
            df = jqd.get_fundamentals(q, date=td_str)

            if 'valuation' in sec:
                df['pubDate'] = df['statDate'] = df['day']

            if df.shape > 0:
                new_info_index = np.array(df[['code', 'pubDate', 'statDate']])
                is_existed = np.isin(new_info_index, recorded_info_index).all(1)

                if sum(~is_existed) > 0:
                    out_df = df[~is_existed]
                    out_df.columns.name = 'field'
                    out_df_info_index = np.array(out_df[['code', 'pubDate', 'statDate']])
                    recorded_info_index = np.vstack([recorded_info_index, out_df_info_index])
                    out_df_reshaped = out_df.set_index('code').stack().rename('value').reset_index()
                    out_df_reshaped['table'] = sec.split('.')[1]

                    if verbose:
                        print '\n', sec, td_str
                        print 'total recorded: ', recorded_info_index.shape[0]
                        print '# of lastest requested: ', df.shape
                        print '# of lastest requested in recorded: ', is_existed.sum()
                        print '# of lastest appended (requested): ', out_df.shape[0]
                        print '# of lastest appended (reshaped): ', out_df_reshaped.shape[0]

                    yield out_df_reshaped


field_inputs = ['valuation', 'balance', 'cash_flow', 'income', 'indicator']
# special_fields = ['bank_indicator', 'security_indicator', 'insurance_indicator']
# field_inputs = list(special_fields[0])
field_inputs = ['jqd.' + fi for fi in field_inputs]

fundamentals_dfs = []
for code in stkcodes:
    fundamentals = f_get_fundamental(stkcodes, trade_days, field_inputs, verbose=True)
    fundamentals_df = pd.concat(fundamentals)
    fundamentals_dfs.append(fundamentals_df)


# Macros
macros_methods = [eval('jqd.query(jqd.macro.{})'.format(method)) for method in dir(jqd.macro) if method.isupper()]



# def categorize_macros_table():
#     time_fields = ['day', 'stat_year', 'stat_month']
#     for method in dir(jqd.macro):
#         if method.isupper():
#             meval = eval('dir(jqd.macro.{})'.format(method))
#             has_time_fields = np.isin(time_fields, meval)
#             if has_time_fields.any():
#                 yield [has_time_fields]

# def get_all_macros_table_names(stat_time='2013'):
#     for method in dir(jqd.macro):
#         if method.isupper():
#             try:
#                 meval_res = eval('jqd.macro.{}'.format(method))
#                 time_fields = np.array(['day', 'stat_year', 'stat_month', 'stat_quarter'])
#                 time_colns = time_fields[np.isin(time_fields, dir(meval_res))]
#                 if len(time_colns) != 0:
#                     print 'Alert |', method, 'meval_res', '|',  'none or multiple time columns'
#                 else:
#                     stat_time_name = time_colns[0]

#                 query_eval = eval('jqd.query(meval_res).filter(jqd.macro.{}.{}=={})'.format(method, stat_time_name, stat_time))
#                 yield query_eval
#             except Exception as e:
#                 if 'SQL expression' in e.__str__():
#                     print 'OK |', method, ': ', meval_res, '|', e
#                 else:
#                     print e
#                     print 'Alert |', method, ': ', meval_res, '|', e

def query_to_get_all_macros():
    for method in dir(jqd.macro):
        if method.isupper():
            try:
                query_eval = eval('jqd.query(jqd.macro.{})'.format(method))
                yield query_eval
            except Exception as e:
                if 'SQL expression' in e.__str__():
                    print 'OK |', method, ': ', meval_res, '|', e
                else:
                    print e
                    print 'Alert |', method, ': ', meval_res, '|', e

table_names = query_to_get_all_macros()
queries = list(table_names)
macros_dfs = []
for q in queries:
    df = jqd.macro.run_query(q)
    macros_dfs.append(df)



# Cash Relatives
jqd.get_billbord_list(start_date=begD)
jqd.get_locked_shares(stock_list=stkcodes[0], start_date=begD, forward_count=1000) # 只有这种组合可运行
jqd.get_mtss(security_list=stkcodes, start_date=begD)
jqd.get_money_flow(security_list=stkcodes, start_date=begD)



# Drafts
# MAC_SALE_RETAIL_MONTH

# q = jqd.query(jqd.macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_YEAR
#         ).filter(jqd.macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_YEAR.stat_year=='2014')
# df = jqd.macro.run_query(q)


# q = jqd.query(jqd.macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_QUARTER)
#         ).filter(jqd.macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_QUARTER.stat_quarter=='2015-06')
# df = jqd.macro.run_query(q)


# q = jqd.query(jqd.macro.MAC_RMB_EXCHANGE_RATE).filter(jqd.macro.MAC_RMB_EXCHANGE_RATE.day=='2013')
# df = jqd.macro.run_query(q)

# q = jqd.query(jqd.macro.MAC_LEND_RATE)
# df = jqd.macro.run_query(q)


# q = jqd.query(jqd.macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_QUARTER.stat)
# df = jqd.macro.run_query(q)

# df = jqd.macro.run_query(q)
