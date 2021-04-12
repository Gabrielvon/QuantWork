from jqdatasdk import *
import pandas as pd
import numpy as np
import seaborn as sns
import datetime


stock_bmc = ['510050.XSHG', '510300.XSHG', '000905.XSHG']
gold_bmc = ['518880.XSHG']
bond_bmc = ["511010.XSHG"]  # "511030.XSHG", "511260.XSHG"
codes = stock_bmc + gold_bmc + bond_bmc
beginDate = "2015-01-01"
endDate = datetime.datetime.today().strftime('%Y-%m-%d')
# endDate = "2020-01-01"
rawdf = get_price(codes, start_date=beginDate, end_date=endDate, frequency='daily', fields=['close'], skip_paused=False,
                  fq='pre', count=None, panel=False, fill_paused=True)
cn_market_df = rawdf.pivot(index='time', columns='code', values='close')

ret_df = np.log(cn_market_df).diff()
cret_df = ret_df.cumsum()
_ = cret_df.plot.line(figsize=(15, 8), grid=True, secondary_y='BTC-USD')

_ = sns.heatmap(ret_df.corr(method='spearman'))

fund_codes = get_all_securities(types=['QDII_fund'], date=None)
# fund_codes[fund_codes['display_name'].str.contains('富国')].sort_values('type')
fund_codes[fund_codes['display_name'].str.contains('富国')].sort_index()

from pandas_datareader import data as web
import matplotlib.pyplot as plt

# python3 才让读yahoo!数据
index1 = '^IXIC'  # Nasdaq
index2 = '^DJI'  # 道琼斯
index3 = '^GSPC'  # 标普500
index4 = 'GSG'  # commodity
index5 = 'CMR.TO'  # money market
index6 = '^XAU'  # 黄金
index8 = 'BTC-USD'  # 比特币

listindex = [index1, index2, index3, index4, index5, index6, index8]

stockpool = ['AAPL', 'MSFT', 'BABA']

symbols = listindex + stockpool
results = []
for code in symbols:
    try:
        daydata = web.DataReader(code, data_source='yahoo', start='1/1/2015')
        dupindex = daydata.index.duplicated(keep='last')
        if dupindex.any():
            daydata = daydata.loc[~dupindex, :]
        results.append(daydata['Adj Close'].rename(code))
    except Exception as e:
        print(code, 'failed')

gl_market_df = pd.concat(results[:-3], axis=1)

ret_df = np.log(gl_market_df).diff()
cret_df = ret_df.cumsum()
_ = cret_df.plot.line(figsize=(15, 8), grid=True, secondary_y='BTC-USD')

_ = sns.heatmap(ret_df.corr(method='spearman'))

# Test

finance.run_query(query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.main_code == "001371").limit(10))
rs = finance.run_query(query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code == "001371").limit(3000))

rs = macro.run_query(query(macro.MAC_CPI_MONTH))
rs

sentiment = finance.run_query(query(finance.CCTV_NEWS).filter(finance.CCTV_NEWS.day == '2020-01-22'))

index_codes = get_all_securities(types=['index'], date=None)

fund_codes = get_all_securities(types=['fund'], date=None)
# fund_codes[fund_codes['display_name'].str.contains('富国')].sort_values('type')
fund_codes[fund_codes['display_name'].str.contains('富国')].sort_index()

fund_codes = get_all_securities(types=['QDII_fund'], date=None)
# fund_codes[fund_codes['display_name'].str.contains('富国')].sort_values('type')
fund_codes[fund_codes['display_name'].str.contains('富国')].sort_index()

fund_codes = get_all_securities(types=['etf'], date=None)
# fund_codes[fund_codes['display_name'].str.contains('富国')].sort_values('type')
fund_codes[fund_codes['display_name'].str.contains('债')].sort_values('start_date')

get_price()

