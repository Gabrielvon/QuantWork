import pandas as pd
import numpy as np
import scipy.stats as scs
import pickle
import jqdatasdk as jqd


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


# Login
jqd.auth('13510687238','freestyle')

# Basic Infos
begD = '2013-05-01'
endD = '2018-05-01'
trade_days = jqd.get_trade_days(start_date=begD, end_date=endD)
stkcodes = jqd.get_industry_stocks('HY005')


# Fundamental
field_inputs = ['valuation', 'balance', 'cash_flow', 'income', 'indicator']
# special_fields = ['bank_indicator', 'security_indicator', 'insurance_indicator']
# field_inputs = list(special_fields[0])
field_inputs = ['jqd.' + fi for fi in field_inputs]

fundamentals_dfs = []
for code in stkcodes:
    fundamentals = f_get_fundamental(stkcodes, trade_days, field_inputs, verbose=True)
    fundamentals_df = pd.concat(fundamentals)
    fundamentals_dfs.append(fundamentals_df)

fundamentals_dfs.to_csv('fundamental.csv')
