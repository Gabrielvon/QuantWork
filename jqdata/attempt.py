# -*- coding: utf-8 -*-
#



import jqdatasdk as jqd
from jqdatasdk import query, macro, valuation, income, cash_flow, jy

jqd.auth('13510687238','freestyle')


# In[] Run Query
df = jy.run_query(query(jy.LC_StockArchives).limit(10))

# In[] Macro
# Query doc: https://www.joinquant.com/data/dict/macroData
macro_q = macro.MAC_CURRENCY_STATE_YEAR
macro_q = macro.MAC_INDUSTRY_AREA_AGR_OUTPUT_VALUE_QUARTER
macro_q = macro.MAC_STK_MARKET
q = query(macro_q)
df = macro.run_query(q)


# In[] Fundamentals
# Query doc: https://www.joinquant.com/data/dict/fundamentals
q0 = query(
           valuation.markercap
    ).filter(
        valuation.code == '000001.XSHE',
    )

q1 = query(income
    ).filter(
        income.code == '000001.XSHE',
    )

q2 = query(cash_flow)

ret = jqd.get_fundamentals(q0, statDate='2014')
df = jqd.get_fundamentals(q, '2015-10-15')
df = jqd.get_fundamentals_continuously(q0, count=1000)

#


# Get Codes
codes = jqd.get_index_stocks('000001.XSHG', date=None)

# Get all underlying
"""
type: 类型，stock(股票)，index(指数)，etf(ETF基金)，fja（分级A），fjb（分级B），
open_fund（开放式基金）, bond_fund（债券基金）, stock_fund（股票型基金）,
QDII_fund（QDII 基金）, money_market_fund（货币基金）, mixture_fund（混合型基金）
"""
stock_infos = jqd.get_all_securities(types='stock', date=None)
index_infos = jqd.get_all_securities(types='index', date=None)
future_infos = jqd.get_all_securities(types='future', date=None)
future_infos = jqd.get_all_securities(types='etf', date=None)


# Cash Flow
cf = jqd.get_money_flow(codes[:5], start_date=None, end_date=None, fields=None, count=None)