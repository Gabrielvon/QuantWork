import os
import sys
os.environ['NUMEXPR_MAX_THREADS'] = '8'
# sys.platform.lower() == 'darwin'
sys.path.append("E:\\Synology\\Drive\\PycharmProjects\\QuantStudy")
log_path = "E:\\Synology\\Drive\\PycharmProjects\\QuantStudy\\database\\PQL\\import_daily.log"

import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from gqp import jqfunc as jqf
import matplotlib.pyplot as plt


future_codes = jqf.get_all_securities(types=['futures'])
begindate = '2012-01-01'
enddate = '2021-04-09'
future_au_quote = jqf.get_price('AU9999.XSGE', begindate, enddate, fields=['close', 'money'])    # 合约乘数:1000克/手, 单位: 元/克
future_ag_quote = jqf.get_price('AG9999.XSGE', begindate, enddate, fields=['close', 'money'])    # 合约乘数:15千克/手, 单位: 元/千克
future_au_ag_ratio = 1000 * future_au_quote['close'] / (15 * future_ag_quote['close'])
future_au_ag_ratio.reset_index(drop=True).plot.line(figsize=(15, 8), grid=True)
plt.show()

etf_au_quote = jqf.get_price('518880.XSHG', begindate, enddate, fields=['close', 'money'])
etf_au_quote = jqf.get_price('518800.XSHG', begindate, enddate, fields=['close', 'money'])
lof_ag_quote = jqf.get_price('161226.XSHE', begindate, enddate, fields=['close', 'money'])

(etf_au_quote['close'] / lof_ag_quote['close']).dropna().plot.line(figsize=(15, 8), grid=True)
plt.show()
