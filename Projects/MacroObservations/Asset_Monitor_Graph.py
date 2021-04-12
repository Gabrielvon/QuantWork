import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import gqp.utilities as gut
import gqp.jqfunc as jqf
import akshare as ak
import yfinance as yf

start_date = "2014-12-01"
end_date = "2021-04-01"

asset_name_mapping = {
    'CL': 'NYMEX原油',
    'ES': '标普500(期)',
    'GC': 'COMEX黄金',
    'OIL': 'BRENT原油',
    'XAU': '伦敦金',
    'cpi': 'CN-CPI',
    'house_price': '房价',
    '000300.XSHG': '沪深300',
    '000905.XSHG': '中证500',
    '159920.XSHE': '恒生ETF(深)',
    '399481.XSHE': '企债指数(深)',
    '511010.XSHG': '国债ETF',
    '518880.XSHG': '黄金ETF',
    '^HSI': '恒生指数',
    '^GSPC': '标普500'
}



# 恒生指数

# # 定期存款
# # --------------------------------------------------------------------------------
# repo = jqf.get_bond_info(start_date=start_date, end_date=end_date)
# repo['bond_type_id'].value_counts()
# rs = ak.macro_china_lpr()
# rs['LPR1Y'].sort_index()

# # BTC
# # --------------------------------------------------------------------------------
# import gqp.ccfunc as ccf
# count = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
# btc = ccf.get_cc_daily_price('XBTUSD', end_date, count, verbose=0)
# btc['date'] = pd.to_datetime(btc['timestamp']).dt.strftime('%Y%m%d').astype(int)
# btc_price = btc[['date', 'close']]
# btc.pivot('date', 'symbol', 'close')
# btc[btc['date'].duplicated(keep=False)]

# 标普, 黄金, 原油
# --------------------------------------------------------------------------------
asset_list = ['ES', 'GC', 'XAU', 'CL', 'OIL']
global_asset = jqf.get_future_global_daily(asset_list, start_date=start_date, end_date=end_date)
global_asset['date_ts'] = pd.to_datetime(global_asset['date'], format='%Y%m%d')
global_asset['month'] = global_asset['date_ts'].dt.strftime('%Y-%m')
global_asset = global_asset.sort_values(['code', 'date_ts']).groupby(['code', 'month']).last().reset_index()
global_asset_mom_lchg = np.log(global_asset.pivot('month', 'code', 'close')).diff()

# GDP, CPI，房价
# --------------------------------------------------------------------------------
cpi_month_raw = jqf.get_macro('MAC_CPI_MONTH')
cpi_month = cpi_month_raw.pivot('stat_month', 'area_name', 'cpi_month')['全国'].rename('cpi').astype(float)
cpi_month.index = pd.to_datetime(cpi_month.index, format='%Y-%m')
cpi_mom_lret = np.log(cpi_month).diff()
cpi_mom_lret.index = pd.to_datetime(cpi_mom_lret.index, format='%Y-%m-%d').strftime('%Y-%m')
cpi_mom_lret.index.name = 'month'

house_price_index_month = jqf.get_macro('MAC_INDUSTRY_ESTATE_70CITY_INDEX_MONTH')
idx1 = house_price_index_month['area_name'].isin(['北京', '上海', '深圳', '广州'])
idx2 = house_price_index_month['fixed_base_type'] == '上月=100'
house_price_mom = house_price_index_month[idx1 & idx2]
house_price_mom_lchg = np.log(house_price_mom.pivot('stat_month', 'area_name', 'commodity_house_idx') / 100)
hp_mom_avglret = house_price_mom_lchg.mean(axis=1).rename('house_price')
hp_mom_avglret.index.name = 'month'

# 沪深300，企债指数
# --------------------------------------------------------------------------------
codes = ['000300.XSHG', '518880.XSHG', '511010.XSHG', '399481.XSHE', '159920.XSHE', '000905.XSHG']
rawdf = jqf.get_price(codes, start_date=start_date, end_date=end_date, frequency='daily', fields=['close'], skip_paused=False, fq='pre', count=None, panel=False, fill_paused=True)

hsi = yf.Ticker("^HSI")
hsi_raw = hsi.history(interval="1d", start=start_date, end=end_date)
hsi_data = hsi_raw.reset_index()[['Date', 'Close']].assign(code='^HSI')
hsi_data.columns = ['time', 'close', 'code']

spx = yf.Ticker("^GSPC")
spx_raw = spx.history(interval="1d", start=start_date, end=end_date)
spx_data = spx_raw.reset_index()[['Date', 'Close']].assign(code='^GSPC')
spx_data.columns = ['time', 'close', 'code']

index_data = rawdf.append(hsi_data)
index_data = index_data.append(spx_data)
index_data['month'] = index_data['time'].dt.strftime('%Y-%m')
cn_market_df = index_data.sort_values(['code', 'time']).groupby(['code', 'month']).last()['close'].unstack('code')
cn_market_mom_lret = np.log(cn_market_df).diff()


# 保存清洗后的数据
# --------------------------------------------------------------------------------
dataset_filepath = 'dataset'
gut.init_data_directory([dataset_filepath])
global_asset.to_csv(f'{dataset_filepath}/global_future.csv')
cpi_month_raw.to_csv(f'{dataset_filepath}/cpi.csv')
house_price_index_month.to_csv(f'{dataset_filepath}/house_price.csv')
cn_market_df.to_csv(f'{dataset_filepath}/market_data.csv')


# 汇总及绘图
# --------------------------------------------------------------------------------
final_stats0 = pd.concat([global_asset_mom_lchg, cpi_mom_lret, hp_mom_avglret, cn_market_mom_lret], axis=1).dropna(how='any')

final_stats1 = final_stats0.copy()
final_stats1.index = pd.to_datetime(final_stats1.index, format='%Y-%m')

final_stats1_sum = final_stats1.resample('1y').sum()
final_stats1_sum = gut.convet_log_simple_return(final_stats1_sum, 0)
final_stats1_sum.index = pd.Index(final_stats1_sum.index.strftime('%Y'), name='Year')
# final_stats1_sum.columns = ['标普500', 'COMEX黄金', '伦敦金', 'NYMEX-OIL', 'BRENT-OIL', 'CN-CPI', '房价', '企债指数(深)', '沪深300', '国债ETF', '黄金ETF']
final_stats1_sum.columns = final_stats1_sum.columns.map(asset_name_mapping)

final_stats1_std = final_stats1.resample('1y').std()
final_stats1_std.index = pd.Index(final_stats1_std.index.strftime('%Y'), name='Year')
# final_stats1_std.columns = ['标普500', 'COMEX黄金', '伦敦金', 'NYMEX-OIL', 'BRENT-OIL', 'CN-CPI', '房价', '企债指数(深)', '沪深300', '国债ETF', '黄金ETF']
final_stats1_std.columns = final_stats1_std.columns.map(asset_name_mapping)

sns.set(font_scale=1.3)
#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

# fig, ax = plt.subplots(figsize=(16, 8))         # Sample figsize in inches
# fig.suptitle('历年资产收益(%)')
# sns.heatmap(final_stats1_sum, annot=True, linewidths=.5, ax=ax)
# plt.show()

# fig, ax = plt.subplots(figsize=(16, 8))         # Sample figsize in inches
# fig.suptitle('历年资产收益波动率(%)')
# sns.heatmap(final_stats1_std, annot=True, linewidths=.5, ax=ax)
# plt.show()


fig, ax = plt.subplots(figsize=(28, 10))         # Sample figsize in inches
fig.suptitle('历年资产收益和波动率情况(%)')
label_ravel = ['{:.4f}\n({:.4f})'.format(a, b) for a, b in zip(np.ravel(final_stats1_sum), np.ravel(final_stats1_std))]
labels = np.reshape(label_ravel, final_stats1_sum.shape)
sns.heatmap(final_stats1_sum, annot=labels, linewidths=.5, ax=ax, fmt="")
plt.savefig('Asset_Monitor.png')
plt.show()


