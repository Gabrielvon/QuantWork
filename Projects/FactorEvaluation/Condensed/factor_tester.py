# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:23:56 2016

@author: Gabriel F.
"""

import pandas as pd
import numpy as np
import utility as util
import matplotlib.pyplot as plt
import seaborn as sn
# import scipy.stats as scs
# from minepy import MINE


class factor_tester():

    def __segmentation(self, in_df, flag):
        if flag == 1:
            segm = util.split_data(in_df['avg_factor'].values, self.split_num + 1, flag=1)
        elif flag == 2:
            labels_2 = in_df.groupby('code', sort=False).apply(
                lambda x: util.split_data(x['avg_factor'], self.split_num + 1, flag=1))
            segm = np.hstack(labels_2)
        elif flag == 3:
            segm = util.split_data(in_df['avg_factor'].values, self.split_num + 1, flag=2)
        elif flag == 4:
            labels_4 = in_df.groupby('code', sort=False).apply(
                lambda x: util.split_data(x['avg_factor'], self.split_num + 1, flag=2))
            segm = np.hstack(labels_4)
        else:
            raise ValueError('flag is not defined correctly.')
        return segm

    def __revert_compressed_data(self, x, cyc):
        cnt = x['count'].values
        xrp = np.repeat(x['sum_factor'].values / cnt, cnt)
        mu = x['avg_' + str(cyc)].values
        sd = x['std_' + str(cyc)].values
        yrp = util.revert_compressed_val(zip(mu, sd, cnt))
        return np.stack([xrp, yrp], 1)

    def __calc_all_res_between_group(self, in_df, flag):
        df = in_df.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby('segment')
        mu_bt_grps = grp_by_segm[self.sum_column_names].mean()
        mu_bt_grps = grp_by_segm[[c for c in in_df.columns if 'avg' in str(c)]].mean()
        res = {tcyc: util.calc_ic(mu_bt_grps[['avg_factor', 'avg_' + tcyc]].values) for tcyc in self.ts_cycle_list}
        res_ss = pd.Series(res, name='ic')
        res_ss.index.name = 'cycle'
        return res_ss

    def __calc_all_res_within_group(self, in_df, flag):
        df = in_df.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby('segment')

        sum_values = grp_by_segm[['sum_' + s for s in self.ts_cycle_list]].sum().values
        avgrtn_by_segm = sum_values / grp_by_segm[['count']].sum().values
        avgrtn_by_segm = pd.DataFrame(avgrtn_by_segm, index=grp_by_segm.groups.keys(), columns=self.ts_cycle_list)
        avgrtn_by_segm.index.name = 'segment'

        def func1(x):
            res = [util.calc_ic(x[['avg_factor', 'avg_' + i]].values) for i in self.ts_cycle_list]
            return pd.Series(res, index=self.ts_cycle_list)

        def func2(x):
            cyc = self.ts_cycle_list
            res = [util.update_std(x['sumsq_' + i].sum(), x['sum_' + i].sum(), x['count'].sum()) for i in cyc]
            return pd.Series(res, index=cyc)

        ic_by_segm = grp_by_segm.apply(func1)
        std_by_segm = grp_by_segm.apply(func2)

        res = pd.concat([avgrtn_by_segm, ic_by_segm, std_by_segm], 1, keys=['avgrtn', 'ic', 'std(avgrtn)'])
        res.columns.names = ['type', 'cycle']
        return res

    def __init__(self, input_data, split_num=10):

        self.ts_cycle_list = np.unique([el for cn in input_data.columns for el in cn.split('_') if el.isdigit()])
        self.sum_column_names = ['sum_' + s for s in ['factor'] + list(self.ts_cycle_list)]
        self.split_num = split_num
        self.subsets = input_data['subset'].unique()
        self.codes = np.unique(input_data['code'])

        input_df = input_data.reset_index(drop=True).copy()

        sum_values = input_df[self.sum_column_names].values
        avg_vals = pd.DataFrame(sum_values / input_df[['count']].values,
                                columns=['avg_factor'] + ['avg_' + i for i in self.ts_cycle_list],
                                index=input_df.index)

        self.input_df = input_df.join(avg_vals).copy()
        self.input_df['ts'] = pd.to_datetime(self.input_df['ts'], format='%Y%m%d')
        self.fig_title_by_flags = {
            1: 'Processed As whole (Equal Gap per Group)',
            2: 'Processed by Code and Concatenate (Equal Gap per Group)',
            3: 'Processed As whole (Equal Numbers per Group)',
            4: 'Processed by Code and Concatenate (Equal Numbers per Group)'
        }

        self.res_between_group_dict = {}
        self.res_within_group_dict = {}
        for i in range(1, 5):
            self.input_df['segment{}'.format(i)] = self.__segmentation(self.input_df, i)
            self.res_between_group_dict[i] = self.__calc_all_res_between_group(self.input_df, 1)
            self.res_within_group_dict[i] = self.__calc_all_res_within_group(self.input_df, 1)

    def plot_res_between_groups(self, maxcol=1, flag=1):
        """Summary

            Plot in one plot with results between groups.

        Args:
            maxcol (int, optional): number of columns in the plots
            flag (int, optional): 1：基于样本全体处理后分组的情况；2：基于样本个体处理后合并分组的情况；

        """
        df = self.res_between_group_dict[flag]
        res_cycles = df.rename('value').reset_index()

        g = sn.factorplot(data=res_cycles, x='cycle', y='value', kind='bar', sharey=False)
        g.fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        g.fig.set_size_inches(5, 4)
        g.fig.tight_layout(rect=(0, 0, 1, 0.97))
        return g

    def plot_factor_among_segment(self, in_df, flag):
        df = in_df.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby('segment')
        comp_cnt = grp_by_segm.size()
        total_cnt = grp_by_segm['count'].sum()

        std_factor = grp_by_segm.apply(
            lambda x: util.update_std(
                x['sumsq_factor'].sum(), x['sum_factor'].sum(), x['count'].sum()
            )
        )

        res = pd.concat([comp_cnt, total_cnt, std_factor], 1, keys=[
                        'compressed_count', 'overall_count', 'std(sum_factor)'])
        res.columns.name = 'type'
        self.factor_among_segment = res.copy()
        res_graph = res.stack().rename('value').reset_index()
        sn.set_style("whitegrid")
        g = sn.factorplot(data=res_graph, x='segment', y='value', col='type',
                          kind='bar', sharey=False, alpha=0.8)
        return g

    def plot_res_within_group(self, flag=1):
        """Summary

            Plot in several plots with results within group.

        Args:
            cycles (None, optional): Description
            flag (int, optional): 1：对总体绝对大小分组；2：对个体分组后的组别合并

        """
        df = self.res_within_group_dict[flag]
        res = df.stack(['type', 'cycle']).reset_index().rename(columns={0: 'value'})
        g = sn.factorplot(data=res, x='segment', y='value',
                          row='cycle', col='type',
                          kind='bar', alpha=0.8, dodge=True,
                          sharex=False, size=2, sharey=False,
                          legend_out=True, margin_titles=True)
        g.fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        g.fig.set_size_inches(16, 9)
        g.fig.tight_layout(rect=(0, 0, 1, 0.97))
        g.add_legend()
        return g

    def plot_sectional_return(self, cycles, codes=None, tfreq='1d', flag=1,
                              tformat='%y%m%d', figsize=(15, 8), dropna=True):
        """Summary

        Args:
            cycles (TYPE): Description
            codes (None, optional): Description
            tfreq (str, optional): Description
            flag (int, optional): 1：对总体绝对大小分组；2：对个体分组后的组别合并
            tformat (str, optional): x轴坐标的显示格式
            dropna (bool, optional): 是否去除NA值

        Returns:
            TYPE: Description
        """
        df = self.input_df.rename(columns={'segment' + str(flag): 'segment'})

        if codes is not None:
            df = df[df['code'].isin(list(codes))]

        grp_by_ts_segm = df.groupby(['ts', 'segment'])
        avg_rtn_by_ts_segm = grp_by_ts_segm.apply(lambda x: x['sum_' + str(cycles)].sum() / x['count'].sum())

        avg_rtn_by_ts = avg_rtn_by_ts_segm.unstack('segment')
        self.avg_rtn_by_ts = avg_rtn_by_ts.copy()
        avg_cumrtn = avg_rtn_by_ts.fillna(0).cumsum()
        avg_cumrtn.index = pd.to_datetime(avg_cumrtn.index, format='%Y%m%d')

        fig, ax = plt.subplots(figsize=(15, 8))
        avg_cumrtn.plot.line(grid=True, ax=ax)
        ttl_2nd = '\nCummulative Segmented Return (daily)'
        ttl_3rd = '\nForward: ' + str(cycles) + 's'
        ax.set_title(ttl_2nd + ttl_3rd, y=0.99, fontsize=10)
        fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig, ax

    def plot_return_dist(self, cycles, flag=1, codes=None, kind='box'):

        df = self.input_df.rename(columns={'segment' + str(flag): 'segment'})

        if codes is not None:
            df = df[df['code'].isin(list(codes))]

        fig, ax = plt.subplots(figsize=(12, 8))
        dat = [g.values for i, g in df.groupby('segment')['avg_' + str(cycles)]]
        if kind == 'box':
            sn.boxplot(data=dat, orient='v', whis=3, showmeans=True, showfliers=False, ax=ax)
        elif kind == 'violin':
            sn.violinplot(data=dat, orient='v', whis=3, showmeans=True, showfliers=False, ax=ax)
        ax.set_title(r'Outliers: $\mu \pm 3\sigma$')
        ax.grid(True)

        fig_title = self.fig_title_by_flags[flag]
        plt.title(fig_title + '\nReturn Distribution within Each Segmentation\nForward: ' +
                  str(cycles) + 's', fontsize=12)
        fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.98))
        return fig, ax

    def plot_trades_activity(self, cycles, flag=1, freq='1d', codes=None):

        df = self.input_df.rename(columns={'segment' + str(flag): 'segment'})
        if codes is not None:
            df = df[df['code'].isin(list(codes))]

        grp_by_ts = df.groupby(pd.Grouper(key='ts', freq=freq))
        df_cnt_stack_df = grp_by_ts['segment'].apply(pd.value_counts).reset_index()
        df_cnt_stack_df.columns = ['ts', 'segment', 'count']

        factor_grps_labels = util.split_data(df_cnt_stack_df[['count']].values, self.split_num + 1)
        df_cnt_stack_df['section'] = factor_grps_labels
        self.df_cnt = df_cnt_stack_df

        col = np.sort(df_cnt_stack_df['segment'].unique())
        col_ord = zip(col[:self.split_num / 2], col[self.split_num / 2:][::-1])
        col_ord = [j for i in col_ord for j in i]

        sn.set_style('whitegrid')
        g = sn.FacetGrid(data=df_cnt_stack_df, hue='section',
                         col='segment', col_order=col_ord, col_wrap=2,
                         sharey='row', legend_out=True,
                         palette=sn.color_palette('Set1', n_colors=10),
                         xlim=df_cnt_stack_df['ts'].values[[0, -1]])
        g.map(plt.scatter, 'ts', 'count', alpha=0.2, marker='o')

        ttl_1st = self.fig_title_by_flags[flag]
        ttl_2nd = '\nTrades Activity'
        ttl_3rd = '\nForward: ' + str(cycles) + 's'
        g.fig.suptitle(ttl_1st + ttl_2nd + ttl_3rd, y=0.99, fontsize=12)
        g.fig.set_size_inches(18, 9)
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0, 0, 0.95, 0.97))
        g.add_legend()
        return g

    def get_abnormal_dates(self, sigma=3):
        mask = util.detect_outlier(self.df_cnt['count'], sigma)
        dates_wt_outliers = self.df_cnt[mask]
        return dates_wt_outliers

    def plot_seperated(self, regroup=None):

        df = self.input_df.copy()
        if regroup is not None:
            df['user_grp'] = df['code'].map(regroup)
        else:
            Warning('regroup is not defined properly, please assign')
            df['user_grp'] = df['code']
        grp_by_users = df.groupby('user_grp', sort=True)[['code', 'ts', 'avg_factor']]
        colns = ['segment' + str(i) for i in self.fig_title_by_flags.keys()]

        def __get_segm(grp):
            for i, x in grp:
                segments = np.stack([self.__segmentation(x, 1) for i in self.fig_title_by_flags.keys()], 1)
                yield pd.DataFrame(segments, columns=colns, index=x.index)

        df[colns] = pd.concat(__get_segm(grp_by_users))
        self.input_df2 = df.copy()

    def plot_res_between_groups_seperated(self, flag=1, figsize=(10, 5)):
        df = self.input_df2.copy()
        raw_df_grp = df.groupby('user_grp', sort=False)
        data_graph = raw_df_grp.apply(lambda x: self.__calc_all_res_between_group(x, flag))
        self.data_bt_grp_sep = data_graph.copy()
        data_graph_melted = data_graph.stack().reset_index().rename(columns={0: 'value'})
        g = sn.factorplot(data=data_graph_melted, x='user_grp', y='value', hue='cycle',
                          kind='bar', alpha=0.3, dodge=True,
                          sharex=True, sharey=False, margin_titles=True, legend=False)

        g.fig.suptitle(self.fig_title_by_flags[flag] +
                       '\nPerformance by User-defined Group', x=0.8, y=0.99, fontsize=12)
        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.92))
        g.add_legend()
        return g

    def plot_factor_among_segment_seperated(self, flag):
        df = self.input_df2.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby(['user_grp', 'segment'])
        comp_cnt = grp_by_segm.size()
        total_cnt = grp_by_segm['count'].sum()
        std_factor = grp_by_segm.apply(
            lambda x: util.update_std(
                x['sumsq_factor'].sum(), x['sum_factor'].sum(), x['count'].sum()
            )
        )
        res = pd.concat([comp_cnt, total_cnt, std_factor], 1, keys=[
                        'compressed_count', 'overall_count', 'std(sum_factor)'])
        res.columns.name = 'type'
        self.data_ft_segm_sep = res.copy()
        res_graph = res.stack().rename('value').reset_index()
        g = sn.factorplot(data=res_graph, x='segment', y='value', row='user_grp',
                          col='type', kind='bar', sharey=False, alpha=0.8)
        return g

    def plot_res_within_group_seperated(self, res_type='avgrtn', flag=1):
        df = self.input_df2.rename(columns={'segment' + str(flag): 'segment'})
        raw_df_grp = df.groupby('user_grp')
        data_graph = raw_df_grp.apply(lambda x: self.__calc_all_res_within_group(x, flag))
        self.data_res_wt_grp_sep = data_graph

        if res_type in data_graph.columns:
            data_graph = data_graph[res_type].stack().rename(res_type).reset_index()
            g = sn.factorplot(data=data_graph, x='segment', y=res_type,
                              row='user_grp', col='cycle', hue='segment',
                              kind='bar', alpha=0.8, dodge=False,
                              legend=False, sharex=True, sharey=False, margin_titles=True)
        else:
            data_graph_melted = data_graph.stack().reset_index().melt(id_vars=['cycle', 'segment', 'user_grp'])
            g = sn.FacetGrid(data=data_graph_melted, row='type', col='cycle', hue='user_grp',
                             size=2, aspect=3, sharex=False, sharey=False,
                             legend_out=True, margin_titles=True)
            g.map(plt.scatter, 'segment', 'value', alpha=0.8)

        ttl_2nd = '\nby Segmentation within User-defined Groups'
        g.fig.suptitle(self.fig_title_by_flags[flag] + ttl_2nd, y=0.99, fontsize=12)
        g.fig.set_size_inches(20, 15)
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.97))
        g.add_legend()
        return g

    def plot_sectional_return_seperated(self, flag=1, cycles=None, user_grp=None):
        # plot sectional
        df = self.input_df2.rename(columns={'segment' + str(flag): 'segment'})
        user_grp = user_grp if user_grp is not None else df['user_grp'].unique()
        df = df[df['user_grp'].isin(user_grp)]
        cycles = cycles if cycles is not None else self.ts_cycle_list

        data_graph2 = []
        for i in cycles:
            grp_by_ug_ts_segm = df.groupby(['user_grp', 'ts', 'segment'])
            avg_rtn_by_ug_ts_segm = grp_by_ug_ts_segm.apply(lambda x: x['sum_' + str(i)].sum() / x['count'].sum())
            cumsum_rtn = avg_rtn_by_ug_ts_segm.unstack('ts').cumsum(1)
            data_graph2.append(cumsum_rtn.stack())

        data_graph2 = pd.concat(data_graph2, keys=self.ts_cycle_list, names=['cycle', 'user_grp', 'segment', 'ts'])
        data_graph2 = data_graph2.rename('avgrtn').reset_index()
        self.data_segm_rtn_sep = data_graph2

        sn.set_style("darkgrid")
        g = sn.FacetGrid(data=data_graph2, row='cycle', col='user_grp', hue='segment',
                         size=3, sharex='col', sharey='col', legend_out=True, margin_titles=True)
        g = g.map(plt.plot, 'ts', 'avgrtn', lw=1, alpha=0.8, marker='.', ms=3)
        ttl_2nd = '\n by Segmentation'
        ttl_3rd = ' within User-defined Groups'
        g.fig.suptitle(self.fig_title_by_flags[flag] + ttl_2nd + ttl_3rd, y=0.99, fontsize=12)
        g.fig.set_size_inches(20, 15)
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.97))
        g.add_legend()
        return g
