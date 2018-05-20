import pandas as pd
import numpy as np
import scipy.stats as scs
import pickle
import jqdatasdk as jqd


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



# Login
jqd.auth('13510687238','freestyle')

# Basic Infos
begD = '2013-05-01'
endD = '2018-05-01'
trade_days = jqd.get_trade_days(start_date=begD, end_date=endD)
stkcodes = jqd.get_industry_stocks('HY005')


# Macros
macros_table_names = query_to_get_all_macros()
queries = list(table_names)
macros_dfs = []
for q in queries:
    df = jqd.macro.run_query(q)
    macros_dfs.append(df)

with open('macros_dfs.pkl', 'a+') as pkl:
    pickle.dump(macros_dfs)
