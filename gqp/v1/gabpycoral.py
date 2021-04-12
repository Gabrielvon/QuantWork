from __future__ import print_function

import pandas as pd
import pycoraldb
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime
import time

import sys
sys.path.append('/Users/gabrielfeng/Nutstore/MyNutshell/FanCapitial/RefPkg')
sys.path.append('E:\MyNutshell\FanCapitial\RefPkg')


class db():

    def __init__(self, address=None, username=None, password=None):
        if address.lower() == 'external':
            self.address = 'coraldb://180.168.106.62:59020'
        elif address.lower() == 'internal':
            self.address = 'coraldb://192.168.211.236:59020'
        elif address.split(':')[0] == 'coraldb':
            self.address = address
        else:
            raise ValueError('Check your address.')

        self.client = pycoraldb.CoralDBClient(self.address)
        if (username is None) | (password is None):
            self.__gab_login()
        else:
            self.username = username
            self.password = password
            self.client.login(self.username, self.password)

    def __gab_login(self):
        self.client.login("fengwl", '5e24d6484fedd7e3e52c28d3155b0ed7')

    def _getTick(self, date, n_attempt=5, timesleep=5):
        code, fields = self.query['code'], self.query['fields']
        cnt = 0
        while cnt <= n_attempt:
            try:
                df = self.client.getBar(code=code, beginDate=date, fields=fields)
                break
            except Exception as e:
                print(e)
                time.sleep(timesleep)
                cnt += 1
                continue
        return df.toDataFrame()

    def _getHolo(self, date, n_attempt=5, timesleep=5):
        code, fields = self.query['code'], self.query['fields']
        cnt = 0
        while cnt <= n_attempt:
            try:
                df = self.client.getBar(code=code, beginDate=date, fields=fields, holo=True)
                break
            except Exception as e:
                print(e)
                time.sleep(timesleep)
                cnt += 1
                continue
        return df.toDataFrame()

    def _getCycle(self, date, n_attempt=5, timesleep=5):
        code, fields = self.query['code'], self.query['fields']
        cycle = self.query['n_cycle'] * self.query['freq']
        cnt = 0
        while cnt <= n_attempt:
            try:
                df = self.client.getBar(code=code, beginDate=date, fields=fields, cycle=cycle)
                break
            except Exception as e:
                print(e)
                time.sleep(timesleep)
                cnt += 1
                continue
        return df.toDataFrame()

    def _getKnock(self, date, n_attempt=5, timesleep=5):
        code, fields = self.query['code'], self.query['fields']
        cnt = 0
        while cnt <= n_attempt:
            try:
                df = self.client.getKnock(code=code, beginDate=date, fields=fields)
                break
            except Exception as e:
                print(e)
                time.sleep(timesleep)
                cnt += 1
                continue
        return df.toDataFrame()

    def _getOrder(self, date, n_attempt=5, timesleep=5):
        code, fields = self.query['code'], self.query['fields']
        cnt = 0
        while cnt <= n_attempt:
            try:
                df = self.client.getKnock(code=code, beginDate=date, fields=fields)
                break
            except Exception as e:
                print(e)
                time.sleep(timesleep)
                cnt += 1
                continue
        return df.toDataFrame()

    def query_init(self, query):
        """[summary]

        Arguments:
            paras_dict {dict} -- dict in following format
            {'code': str, 'begin': int or str, 'end': int or str, 'fields': list, 'type':str,
            'dtype': 'tick', 'n_cycle': int, 'freq': pycoraldb object}

        """
        if isinstance(query, str):
            dt = (datetime.today() - pd.to_timedelta('1d')).strftime('%Y%m%d')
            self.query = {'code': query, 'begin': dt, 'end': dt, 'fields': '*', 'dtype': 'tick'}
        elif isinstance(query, dict):
            self.query = query
        else:
            raise TypeError('query should be dict.')

        try:
            beginD, endD = self.query['begin'], self.query['end']
        except Exception:
            beginD, endD = self.query['begin'], self.query['begin']
        valid_dates = self.client.getTradingDays(beginD, endD).values
        self.TD = [i[0] for i in valid_dates]

    # def getTick(self, n_jobs=2):
    #     pool = ThreadPool(processes=n_jobs)
    #     res = pool.map(self._getTick, self.TD)
    #     return pd.concat(res)

    # def getHolo(self, n_jobs=2):
    #     pool = ThreadPool(processes=n_jobs)
    #     res = pool.map(self._getHolo, self.TD)
    #     return pd.concat(res)

    # def getCycle(self, n_jobs=2):
    #     pool = ThreadPool(processes=n_jobs)
    #     res = pool.map(self._getCycle, self.TD)
    #     return pd.concat(res)

    def getBar(self, n_jobs=2):
        pool = ThreadPool(processes=n_jobs)
        if self.query['dtype'].lower() == 'tick':
            res = pool.map(self._getTick, self.TD)
        elif self.query['dtype'].lower() == 'holo':
            res = pool.map(self._getHolo, self.TD)
        elif self.query['dtype'].lower() == 'cycle':
            res = pool.map(self._getCycle, self.TD)
        elif self.query['dtype'].lower() == 'knock':
            res = pool.map(self._getKnock, self.TD)
        elif self.query['dtype'].lower() == 'order':
            res = pool.map(self._getOrder, self.TD)
        else:
            raise ValueError('datatype it not right.')
        try:
            self.rawdata = pd.concat(res)
            return self.rawdata
        except ValueError:
            print('Warning: dates got no valid trading days.')
            self.rawdata = res
            return self.rawdata

    def getBarMultiples(self, codes, n_jobs, verbose=False):
        if verbose:
            itit = tqdm(codes)
        else:
            itit = codes
        res = []
        for co in itit:
            res.append(self.getBar(updates={'code': co}))
        res = pd.concat(res, keys=codes)
        self.rawdata = res
        return self.rawdata

    def to_txt(self, codes, path='./', n_jobs=4):
        header = 'timestamp,' + self.query['fields'] + '\n'
        update_freq = 1
        for i, co in tqdm(enumerate(codes, 1), desc='codes', mininterval=10):
            print('\n{}: {}   {}'.format(i, datetime.now(), co))
            self.query.update({'code': co})
            myc, mye = co.split('.')
            if (i - 1) % update_freq == 0:
                filename = path + '{}_{}_{}_{}_{}.txt'.format(myc, mye,
                                                              self.query['dtype'], self.query['begin'], self.query['end'])
                # try:
                #     f = open(filenamefn, 'r+')
                #     saved_infos = f.readlines()
                #     if header not in saved_infos:
                #         f.writelines(header)
                # except IOError:
                #     f = open(filename, 'a+')
                #     f.writelines(header)
                f = open(filename, 'a+')
                saved_infos = f.readlines()
                if header not in saved_infos:
                    f.writelines(header)

            for dt in tqdm(self.TD, desc='dates', mininterval=1):
                print(dt)
                try:
                    temp_df = self.getBar(n_jobs)
                    temp_df.to_csv(f, header=False, index=False)
                except Exception as e:
                    err = [co, dt, str(e)]
                    # f_error.writelines(', '.join([str(s) for s in err]))
                    print('Code: %s;\nDate: %s;\nErorr: %s' % tuple(err))
            if i % update_freq == 0:
                print('{} were recorded.'.format(i))
                f.close()
                # f_error.close()
            if not f.closed:
                f.close()


def print_df(df):
    print(df.shape)
    print(df.head(10))
    print(df.tail(10))


def test_db(DB):
    DB.query.update({'dtype': 'tick'})
    df_tick = DB.getBar(4)
    print_df(df_tick)

    DB.query.update({'dtype': 'holo'})
    df_holo = DB.getBar(4)
    print_df(df_holo)

    DB.query.update({'dtype': 'cycle'})
    df_cycle = DB.getBar(4)
    print_df(df_cycle)

    DB.query.update({'dtype': 'knock'})
    df_knock = DB.getBar(4)
    print_df(df_knock)

    DB.query.update({'dtype': 'order'})
    df_order = DB.getBar(4)
    print_df(df_order)


def test_save(DB):
    codes = ['000001.SZ', '600601.SH']
    DB.to_txt(codes)


def main():
    DB = db('external')
    DB.query_init({'code': '000001.SZ', 'begin': 20180201, 'end': 20180205,
                   'fields': 'code,date', 'dtype': 'tick',
                   'n_cycle': 1, 'freq': pycoraldb.H})
    # test_db(DB)
    DB.query.update({'dtype': 'cycle'})
    test_save(DB)


if __name__ == '__main__':
    main()
