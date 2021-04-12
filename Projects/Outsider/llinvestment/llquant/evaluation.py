# -*- coding: utf-8 -*-
# @Author: Gabriel Feng
# @Date:   2021-03-15 15:36:43
# @Last Modified by:   Gabriel Feng
# @Last Modified time: 2021-03-15 16:00:28

# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
# import scipy.stats as scs
# from datetime import datetime
# from minepy import MINE
import llquant.utilities as ut


class factor_tester():

    def __init__(self, input_data, split_num=10, verbose=True):
        self.input_data = input_data.copy()
        ts_cycle_list = [cn for cn in input_data.columns if cn.isdigit()]
        self.ts_cycle_list = self._sort_str_by_digits(ts_cycle_list)
        self.split_num = split_num
        self.pcts = [0., 1., 3., 5., 10., 90., 95., 97., 99., 100.]
        self.flags = range(1, 7)
        self.fig_title_by_flags = {
            1: 'Processed as Whole (Equal Gap per Group)',
            2: 'Processed by Codes  (Equal Gap per Group)',
            3: 'Processed as Whole (Equal Numbers per Group)',
            4: 'Processed by Codes (Equal Numbers per Group)',
            5: 'Stacked Tail by Codes (Equal Numbers per Group)',
            6: 'Stacked Tail by Codes (Equal Numbers per Group)',
        }
        self._reset()

        for i in self.flags:
            segment_i = self._segmentation(self.input_data, i)
            self.input_data['segment{}'.format(i)] = segment_i
        # plot settings
        try:
            sn.set(style="darkgrid", font="SimHei", rc={'axes.unicode_minus': False})
        except Exception as e:
            print(e)
            sn.set(style="darkgrid")
        self.palette0 = sn.color_palette(n_colors=len(self.ts_cycle_list))
        self.palette1 = sn.color_palette("coolwarm", n_colors=self.split_num + 1)
        self.cm_line0 = plt.cm.coolwarm
        self.cm_line1 = sn.diverging_palette(220, 20, n=self.split_num + 1)

    def _segmentation(self, in_df, flag):
        """Summary
            一般观察组合如下:
                - 整体与个体分组的对比: [1, 2], [3, 4], [5, 6];
                - 观察单个因子的分散程度，极大极小值: [1, 3, 5], [2, 4, 6];
                - 比较多个因子，假设有两个因子a和b: [3a, 3b], [4a, 4b], [5a, 5b], [6a, 6b];
        Args:
            in_df (TYPE): Description
            flag (TYPE):
                1: 整体等距分段; 2: 个体等距分段;
                3: 整体等量分段; 4: 个体等量分段;
                5: 整体自定义分位点分段; 6: 个体自定义分位点分段;
        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        in_df_dropna = in_df.dropna(subset=['code', 'factor'])

        if flag == 1:
            segm = ut.split_data(
                in_df_dropna['factor'].values, ncut=self.split_num, flag=1)
        elif flag == 2:
            def __segm_f2(g):
                return ut.split_data(g['factor'].values, ncut=self.split_num, flag=1)
            segm = np.hstack(in_df_dropna.groupby('code', sort=False, observed=True).apply(__segm_f2))
        elif flag == 3:
            segm = ut.split_data(in_df_dropna['factor'].values, ncut=self.split_num, flag=2)
        elif flag == 4:
            def __segm_f4(g):
                return ut.split_data(g['factor'].values, ncut=self.split_num, flag=2)
            segm = np.hstack(in_df_dropna.groupby('code', sort=False, observed=True).apply(__segm_f4))
        elif flag == 5:
            segm = ut.split_data(in_df_dropna['factor'].values, ncut=self.pcts, flag=3)
        elif flag == 6:
            def __segm_f6(g):
                return ut.split_data(g['factor'].values, ncut=self.pcts, flag=3)
            segm = np.hstack(in_df_dropna.groupby('code', sort=False, observed=True).apply(__segm_f6))
        # elif flag == 7:
        #     def __segm_f7(grps):
        #         for k, g in grps:
        #             if g.shape[0] > 3:
        #                 rs = [np.nan] * g.shape[0]
        #             else:
        #                 rs = ut.split_data(g['factor'].values, ncut=self.pcts, flag=3)
        #             yield pd.Series(segm, index=g.index, name=k)
        #     segm = pd.concat(__segm_f7(in_df.groupby(pd.Grouper(key='ts', freq='1d'))))
        else:
            raise ValueError('flag is not defined correctly.')

        return pd.Series(segm, index=in_df_dropna.index, name=flag)

    # def __revert_compressed_data(self, x, cyc):
    #     cnt = x['count'].values
    #     xrp = np.repeat(x['factor'].values / cnt, cnt)
    #     mu = x['avg_' + str(cyc)].values
    #     sd = x['std_' + str(cyc)].values
    #     yrp = ut.revert_compressed_val(zip(mu, sd, cnt))
    #     return np.stack([xrp, yrp], 1)

    def _calc_all_res_between_group(self, in_df, flag):
        df = in_df.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby('segment')
        mu_bt_grps = grp_by_segm[['factor'] + self.ts_cycle_list].mean()
        res = {tcyc: ut.calc_ic(*mu_bt_grps[['factor', tcyc]].values.T) for tcyc in self.ts_cycle_list}
        res_ss = pd.Series(res, name='ic')
        res_ss.index.name = 'cycle'
        return res_ss

    def _calc_all_res_within_group(self, in_df, flag):
        df = in_df.rename(columns={'segment' + str(flag): 'segment'})
        grp_by_segm = df.groupby('segment')

        def __calc_ic(grps):
            for _, gp in grps:
                for tcyc in self.ts_cycle_list:
                    yield ut.calc_ic(*gp[['factor', tcyc]].values.T)

        mu_by_segm = grp_by_segm[self.ts_cycle_list].mean()
        cnt_by_segm = grp_by_segm[self.ts_cycle_list].count()
        ic_by_segm = pd.DataFrame(np.reshape(list(__calc_ic(grp_by_segm)), mu_by_segm.shape),
                                  index=grp_by_segm.groups.keys(),
                                  columns=self.ts_cycle_list)
        std_by_segm = grp_by_segm[self.ts_cycle_list].std()

        res = pd.concat([mu_by_segm, cnt_by_segm, ic_by_segm, std_by_segm],
                        1, keys=['avgrtn', 'count', 'ic', 'std(avgrtn)'])
        res.columns.names = ['type', 'cycle']
        return res

    def _reset(self):
        self.res_between_group_dict = {}
        self.res_within_group_dict = {}
        self.avg_rtn_by_ts = {}
        self.avg_rtn_by_ts_in_tail = {}
        self.factor_among_segment = {}
        self.trades_activity = {}
        self.data_bt_grp_sep = {}
        self.data_segm_sep = {}
        self.data_wt_grp_sep = {}
        self.data_segm_rtn_sep = {}

    def _sort_str_by_digits(self, str_list):
        return sorted(str_list, key=lambda x: int(x.split('_')[-1]))

    def plot_res_between_groups(self, flag=None, figsize=None):
        """Summary

            Plot in one plot with results between groups.

        """
        if flag is None:
            flag = self.flags
        elif isinstance(flag, (int, str)):
            flag = [flag]

        for i in flag:
            if i not in self.res_between_group_dict:
                self.res_between_group_dict[i] = self._calc_all_res_between_group(self.input_data, i)

        df = pd.concat(self.res_between_group_dict, names=['flag', 'cycle'])
        df = df[df.index.get_level_values('flag').isin(self.flags)]

        if isinstance(df, pd.core.series.Series):
            df = df.to_frame()

        if figsize is None:
            figsize = (10 * df.shape[1], 5)

        res_cycles = df.reset_index().melt(id_vars=['flag', 'cycle'])
        res_cycles['cycle'] = res_cycles['cycle'].astype(int)

        g = sn.catplot(data=res_cycles, x='flag', y='value',
                       col='variable', hue='cycle', kind='bar', ci=None,
                       alpha=0.3, dodge=True, sharex=True, sharey=False,
                       margin_titles=True, legend=True, palette=self.palette0)

        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.tight_layout(rect=(0, 0, 1, .97))
        return g

    def plot_res_within_group(self, flag, figsize=None, calc_all=True):
        """Summary

            Plot in several plots with results within group.

        Args:
            cycles (None, optional): Description

        """
        if calc_all:
            for i in self.flags:
                self.res_within_group_dict[i] = self._calc_all_res_within_group(self.input_data, i)
        else:
            if flag not in self.res_between_group_dict:
                self.res_within_group_dict[flag] = self._calc_all_res_within_group(self.input_data, flag)

        if figsize is None:
            figsize = (16, 2 * len(self.ts_cycle_list))

        df = self.res_within_group_dict[flag]
        df.index.name = 'segment'
        res = df.stack(['type', 'cycle']).reset_index().rename(
            columns={0: 'value'})
        res['cycle'] = res['cycle'].astype(int)
        g = sn.catplot(data=res, x='segment', y='value', row='cycle', col='type',
                       kind='bar', alpha=0.8, dodge=True, sharex=False, sharey=False,
                       height=2, legend_out=True, margin_titles=True,
                       palette=self.palette1)
        # g.fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.tight_layout(rect=(0, 0, 1, 0.97))
        g.add_legend()
        return g

    def __calc_avg_rtn_by_ts(self, x):
        avg_df = x.groupby(pd.Grouper(key='ts', freq=self.temp_freq)).mean()
        return avg_df.dropna(how='all')

    def plot_sectional_return(self, flag, cycle, tfreq='1d',
                              tformat='%y%m%d', figsize=(15, 8), dropna=True):
        """Summary

        Args:
            cycle (TYPE): Description
            codes (None, optional): Description
            tfreq (str, optional): Description
            flag (int, optional): 1：对总体绝对大小分组；2：对个体分组后的组别合并
            tformat (str, optional): x轴坐标的显示格式
            dropna (bool, optional): 是否去除NA值

        Returns:
            TYPE: Description
        """

        data_key = '_'.join(map(str, [flag, cycle, tfreq]))
        self.temp_freq = tfreq
        if data_key in self.avg_rtn_by_ts.keys():
            avg_rtn_by_ts = self.avg_rtn_by_ts[data_key]
        else:
            df = self.input_data.rename(columns={'segment' + str(flag): 'segment'})
            grp_by_ts_segm = df.groupby('segment')[['ts', str(cycle)]]
            avg_rtn_by_ts_segm = grp_by_ts_segm.apply(self.__calc_avg_rtn_by_ts)[str(cycle)]
            avg_rtn_by_ts = avg_rtn_by_ts_segm.unstack('segment')
            self.avg_rtn_by_ts[data_key] = avg_rtn_by_ts.copy()

        avg_cumrtn = avg_rtn_by_ts.fillna(0).cumsum()
        avg_cumrtn.index = pd.to_datetime(avg_cumrtn.index, format='%Y%m%d')

        fig, ax = plt.subplots(figsize=figsize)
        avg_cumrtn.plot.line(grid=True, ax=ax, cmap=self.cm_line0)
        ttl_2nd = '\nCummulative Segmented Return'
        ttl_3rd = '\nForward: ' + str(cycle) + 's | Freq: ' + self.temp_freq
        fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        ax.set_title(ttl_2nd + ttl_3rd, y=0.99, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig, ax

    def plot_sectional_return_in_stacked_tails(self, flag, cycle, tfreq='1d',
                                               tformat='%y%m%d', figsize=(15, 8), dropna=True):
        """Summary

        Args:
            cycle (TYPE): Description
            codes (None, optional): Description
            tfreq (str, optional): Description
            tformat (str, optional): x轴坐标的显示格式
            dropna (bool, optional): 是否去除NA值

        Returns:
            TYPE: Description
        """

        def __calc_avg_cum_rtn(tf):
            tmp_df = df.loc[tf, ['ts', str(cycle)]]
            grp_by_ts_segm = tmp_df.groupby(pd.Grouper(key='ts', freq=tfreq))
            return grp_by_ts_segm.mean()

        data_key = '_'.join(map(str, [flag, cycle, tfreq]))
        if data_key in self.avg_rtn_by_ts_in_tail.keys():
            avg_rtn_by_ts_in_tail = self.avg_rtn_by_ts_in_tail[data_key]
        else:
            df = self.input_data.rename(columns={'segment' + str(flag): 'segment'})
            n_segm = len(self.pcts) - 1
            tail_grp_arr = np.full((len(df), n_segm), 0, dtype=bool)
            for i in range(0, n_segm // 2, 1):
                tail_grp_arr[:, i] = df['segment'].isin(range(i + 1))
                tail_grp_arr[:, n_segm - i - 1] = df['segment'].isin(range(n_segm - i - 1, n_segm))
                avg_rtn_by_ts_in_tail = pd.concat(map(__calc_avg_cum_rtn, tail_grp_arr.T), 1)
                self.avg_rtn_by_ts_in_tail[data_key] = avg_rtn_by_ts_in_tail.copy()

        avg_cumrtn = avg_rtn_by_ts_in_tail.fillna(0).cumsum()
        avg_cumrtn.index = pd.to_datetime(avg_cumrtn.index, format='%Y%m%d')

        fig, ax = plt.subplots(figsize=figsize)
        avg_cumrtn.plot.line(grid=True, ax=ax, cmap=self.cm_line0)
        ttl_2nd = '\nCummulative Segmented Return'
        ttl_3rd = '\nForward: ' + str(cycle) + 's | Freq: ' + tfreq
        ax.set_title(ttl_2nd + ttl_3rd, y=0.99, fontsize=10)
        fig.suptitle(self.fig_title_by_flags[flag], y=0.99, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig, ax

    def plot_return_dist(self, flag, cycle, codes=None, kind='box', figsize=(12, 8)):

        df = self.input_data.rename(columns={'segment' + str(flag): 'segment'})
        df = df[df['code'].isin(list(codes))] if codes is not None else df

        fig, ax = plt.subplots(figsize=figsize)
        dat = [g.values for i, g in df.groupby('segment')[str(cycle)]]
        if kind == 'box':
            sn.boxplot(data=dat, orient='v', whis=3, showmeans=True, showfliers=False,
                       ax=ax, palette=self.palette1)
        elif kind == 'violin':
            sn.violinplot(data=dat, orient='v', whis=3, showmeans=True, showfliers=False,
                          ax=ax, palette=self.palette1)
        ax.set_title(r'Outliers: $\mu \pm 3\sigma$')
        ax.grid(True)

        fig_title = self.fig_title_by_flags[flag]
        plt.title(fig_title + '\nReturn Distribution within Each Segmentation\nForward: ' +
                  str(cycle) + 's', fontsize=12)
        fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.98))
        return fig, ax

    def plot_trades_activity(self, flag, cycle, tfreq='1d', codes=None):
        data_key = '_'.join(map(str, [flag, cycle, tfreq]))
        if data_key in self.trades_activity.keys():
            df_cnt_stack_df = self.trades_activity[data_key]
        else:
            df = self.input_data.rename(columns={'segment' + str(flag): 'segment'})
            df = df[df['code'].isin(list(codes))] if codes is not None else df

            grp_by_ts = df.groupby(pd.Grouper(key='ts', freq=tfreq))
            df_cnt_stack_df = grp_by_ts['segment'].apply(pd.value_counts).reset_index()
            df_cnt_stack_df.columns = ['ts', 'segment', 'count']

            factor_grps_labels = ut.split_data(df_cnt_stack_df[['count']].values, self.split_num + 1)
            df_cnt_stack_df['section'] = factor_grps_labels
            self.trades_activity[data_key] = df_cnt_stack_df

        col = np.sort(df_cnt_stack_df['segment'].unique())
        col_ord = zip(col[:self.split_num // 2], col[self.split_num // 2:][::-1])
        col_ord = [j for i in col_ord for j in i]

        g = sn.FacetGrid(data=df_cnt_stack_df, hue='section',
                         col='segment', col_order=col_ord, col_wrap=2,
                         sharey='row', legend_out=True,
                         palette=sn.color_palette('Set1', n_colors=10),
                         xlim=df_cnt_stack_df['ts'].values[[0, -1]])
        g.map(plt.scatter, 'ts', 'count', alpha=0.2, marker='o')

        ttl_1st = self.fig_title_by_flags[flag]
        ttl_2nd = '\nTrades Activity'
        ttl_3rd = '\nForward: ' + str(cycle) + 's | Freq: ' + tfreq
        g.fig.suptitle(ttl_1st + ttl_2nd + ttl_3rd, y=0.99, fontsize=12)
        g.fig.set_size_inches(18, 9)
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0, 0, 0.95, 0.97))
        g.add_legend()
        return g

    def get_abnormal_dates(self, sigma=3):
        mask = self.df_cnt.apply(lambda x: ut.detect_outlier(x, sigma)).stack()
        dates_wt_outliers = self.df_cnt.stack()[mask].reset_index()
        dates_wt_outliers.columns = ['ts', 'sec', 'value']
        return dates_wt_outliers

    def plot_seperated(self, regroup=None):

        df = self.input_data.copy()
        if regroup is not None:
            df['user_grp'] = df['code'].map(regroup)
        else:
            Warning('regroup is not defined properly, please assign')
            df['user_grp'] = df['code']

        grp_by_users = df.groupby('user_grp', sort=True)[['code', 'ts', 'factor']]
        colns = ['segment' + str(i) for i in self.fig_title_by_flags.keys()]

        def __get_segm(grp):
            for i, x in grp:
                # segments = np.stack([self.__segmentation(x, fi) for fi in self.fig_title_by_flags.keys()], 1)
                # yield pd.DataFrame(segments, columns=colns, index=x.index)
                segments = pd.concat([self.__segmentation(x, fi) for fi in self.fig_title_by_flags.keys()], 1)
                yield segments

        df[colns] = pd.concat(__get_segm(grp_by_users))
        self.input_data2 = df.copy()

    def plot_res_between_groups_seperated(self, flag=1, figsize=(10, 5)):
        if flag in self.data_bt_grp_sep.keys():
            data_graph = self.data_bt_grp_sep[flag]
        else:
            df = self.input_data2.copy()
            raw_df_grp = df.groupby('user_grp', sort=False)
            data_graph = raw_df_grp.apply(lambda x: self.__calc_all_res_between_group(x, flag))
            self.data_bt_grp_sep[flag] = data_graph.copy()

        data_graph_melted = data_graph.stack().reset_index().rename(columns={0: 'value'})
        g = sn.factorplot(data=data_graph_melted, x='user_grp', y='value', hue='cycle',
                          kind='bar', alpha=0.3, dodge=True, sharex=True, sharey=False,
                          margin_titles=True, legend=False, palette=self.palette0)

        fig_title = self.fig_title_by_flags[flag]
        g.fig.suptitle(fig_title + '\nPerformance by User-defined Group', x=0.8, y=0.99, fontsize=12)
        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.92))
        g.add_legend()
        return g

    def plot_res_within_group_seperated(self, flag=1, res_type='avgrtn', figsize=None):
        data_key = '_'.join(map(str, [flag, res_type]))
        if data_key in self.data_wt_grp_sep.keys():
            data_graph = self.data_wt_grp_sep[data_key]
        else:
            df = self.input_data2.rename(columns={'segment' + str(flag): 'segment'})
            raw_df_grp = df.groupby('user_grp')
            data_graph = raw_df_grp.apply(lambda x: self._calc_all_res_within_group(x, flag))
            self.data_wt_grp_sep[data_key] = data_graph.copy()

        if figsize is None:
            figsize = (4 * len(self.ts_cycle_list), 4 * len(np.unique(list(self.regroup.values()))))

        if res_type in data_graph.columns:
            data_graph_melted = data_graph[res_type].stack().rename(res_type).reset_index()
            data_graph_melted['cycle'] = data_graph_melted['cycle'].astype(int)
            g = sn.factorplot(data=data_graph_melted, x='segment', y=res_type,
                              row='user_grp', col='cycle', hue='segment',
                              kind='bar', alpha=0.8, dodge=False,
                              legend=False, sharex=True, sharey=False,
                              margin_titles=True, palette=self.palette1)
        else:
            data_graph_melted = data_graph.stack().reset_index().melt(id_vars=['cycle', 'segment', 'user_grp'])
            data_graph_melted['cycle'] = data_graph_melted['cycle'].astype(int)
            g = sn.FacetGrid(data=data_graph_melted, row='type', col='cycle', hue='user_grp',
                             size=2, aspect=3, sharex=False, sharey=False,
                             legend_out=True, margin_titles=True, palette=self.palette1)
            g.map(plt.scatter, 'segment', 'value', alpha=0.8)

        ttl_2nd = '\nby Segmentation within User-defined Groups'
        g.fig.suptitle(self.fig_title_by_flags[flag] + ttl_2nd, y=0.99, fontsize=12)
        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.97))
        g.add_legend()
        return g

    def plot_sectional_return_seperated(self, flag, cycle=None, tfreq='1d', user_grp=None, figsize=None):
        data_key = '_'.join(map(str, [flag, cycle, tfreq]))
        if data_key in self.data_segm_rtn_sep.keys():
            data_graph2 = self.data_segm_rtn_sep[data_key]
        else:
            df = self.input_data2.rename(columns={'segment' + str(flag): 'segment'})
            user_grp = user_grp if user_grp is not None else df['user_grp'].unique()
            cycle = [str(cycle)] if not isinstance(cycle, list) else str(cycle)
            df = df[df['user_grp'].isin(user_grp)]

            grp_by_ug_ts_segm = df.groupby(['user_grp', 'segment'])
            self.temp_freq = tfreq
            avg_rtn_by_ug_ts_segm = grp_by_ug_ts_segm.apply(self.__calc_avg_rtn_by_ts)[str(cycle)]
            avg_rtn_by_ug_ts_segm.columns.name = 'cycle'
            cumsum_rtn = avg_rtn_by_ug_ts_segm.stack().unstack('ts').cumsum(1)
            data_graph2 = cumsum_rtn.stack().rename('avgrtn').reset_index()
            self.data_segm_rtn_sep[data_key] = data_graph2.copy()

        if figsize is None:
            figsize = (4 * len(self.ts_cycle_list), 6 * len(np.unique(list(self.regroup.values()))))

        g = sn.FacetGrid(data=data_graph2, row='cycle', col='user_grp', hue='segment',
                         size=3, sharex='col', sharey='col', legend_out=True,
                         margin_titles=True, palette=self.cm_line1)
        g = g.map(plt.plot, 'ts', 'avgrtn', lw=1, alpha=0.8, marker='.', ms=3)
        ttl_2nd = '\nCummulative Segmented Return | User-defined Groups'
        ttl_3rd = '\nForward: ' + str(cycle) + 's | Freq: ' + self.temp_freq
        g.fig.suptitle(self.fig_title_by_flags[flag] + ttl_2nd + ttl_3rd, y=0.99, fontsize=12)
        g.fig.set_size_inches(figsize[0], figsize[1])
        g.fig.autofmt_xdate()
        g.fig.tight_layout(rect=(0.01, 0.01, 0.95, 0.97))
        g.add_legend()
        return g
