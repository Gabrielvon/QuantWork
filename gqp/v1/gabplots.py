import pandas as pd
import gabFunc as gfc
import gabTS as gts
import gabStat as gstat
import gabBT as gbt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class plot3D:
    def __init__(self, df, zlabel=''):
        if isinstance(df, pd.core.series.Series):
            self.z_label = df.name
            df = df.unstack()
        else:
            self.z_label = zlabel

        self.x = df.columns
        self.x_label = df.columns.name
        self.y = df.index
        self.y_label = df.index.name
        self.x, self.y = np.meshgrid(self.x, self.y)
        self.z = df.values

    def formatting(self, ax):
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.04f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.04f'))
        try:
            ax.set_zlabel(self.z_label)
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
        except AttributeError:
            ax.set_title(self.z_label)
        return ax

    def surface(self, colorbar=True):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.x, self.y, self.z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax = self.formatting(ax)
        if colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig, ax

    def wireframe(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(self.x, self.y, self.z, rstride=10, cstride=10)
        ax = self.formatting(ax)
        return fig, ax

    def contour(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        cset = ax.contour(self.x, self.y, self.z,
                          extend3d=True, cmap=cm.coolwarm)
        ax = self.formatting(ax)
        ax.clabel(cset, fontsize=9, inline=1)
        return fig, ax

    def heatmap(self, colorbar=True):
        fig, ax = plt.subplots()
        hm = ax.pcolor(self.x, self.y, self.z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        ax = self.formatting(ax)
        if colorbar:
            fig.colorbar(hm, shrink=0.5, aspect=5)
        return fig, ax, hm

    # def barplot(self):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
    #         xs = np.arange(20)
    #         ys = np.random.rand(20)

    #         # You can provide either a single color or an array. To demonstrate this,
    #         # the first bar of each set will be colored cyan.
    #         cs = [c] * len(xs)
    #         cs[0] = 'c'
    #         ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    #     ax = self.formatting(ax)


def plt_format(fig=None, ax=None, flag=0):
    if fig:
        if flag == 0:
            fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        if flag == 1:
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                                top=0.95, wspace=0.3, hspace=0.2)
    if ax:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.04f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.04f'))


def heatmap_in_one_figure(vals, pars, cmap=None):
    """
    vals: lists of dataframes
    pars: nrow and ncol of plots in the figure
    """

    from mpl_toolkits.axes_grid1 import AxesGrid
    fig = plt.figure()
    grid = AxesGrid(fig, 111,
                    nrows_ncols=pars,
                    axes_pad=0.05,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    for val, ax in zip(vals, grid):
        myplot3d = plot3D(val)
        hm = ax.pcolor(myplot3d.x, myplot3d.y, myplot3d.z,
                       vmin=0, vmax=1, cmap=cmap)

    grid.cbar_axes[0].colorbar(hm)

    return fig, ax


def plot_acf(data, lags=50, title=None):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, lags=lags, ax=axs[0])
    plot_pacf(data, lags=lags, ax=axs[1])
    fig.suptitle(title)
    return fig, axs


def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    import statsmodels.tsa.api as smta
    import statsmodels.api as sma
    import scipy.stats as scs
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smta.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smta.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sma.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


def mycmap(x, n=None):
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    return colors


def scale_index(index, n, form='%Y-%m-%d %H:%M:%S', dtype='str'):
    xaxis_idx = map(int, np.linspace(0, len(index) - 1, n))
    if isinstance(index, pd.core.indexes.datetimes.DatetimeIndex):
        xaxis_val = index.strftime(form).astype(dtype)
    else:
        xaxis_val = index.astype(dtype)
    return xaxis_idx, xaxis_val[xaxis_idx]


# In[]
class plot_bt_res():
    def __init__(self, trading_rec, param_cols):
        self.trarec_grp = trading_rec.groupby(param_cols)
        self.trading_rec = trading_rec
        self.bt_res = self.trarec_grp.apply(lambda x: gbt.BT_wCost(x)[0]).unstack(level=-1).fillna(np.nan)
        self.trec = self.trarec_grp.apply(lambda x: gbt.BT_wCost(x)[1])
        self.trec_unstack = self.trec.unstack(param_cols)
        self.underly = self.trec['Underlying'].unique()
        self.indiv_tr = {ud: self.trec[self.trec['Underlying'] == ud] for ud in self.underly}

    def heatmap_basic(self, xaxis=0, bt_res=None):
        # Overview: Basic (Heatmap)
        if bt_res is None:
            bt_res = self.bt_res
            underly = self.underly
        else:
            underly = bt_res['Underlyings'].apply(lambda x: ','.join(x)).unique()

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        axs = axs.flatten()
        for i, coln in enumerate(['count|PnL', 'WinLoss', 'AvgReturn', 'TotalReturn', 'sum|PnL', 'MaxDD']):
            df = bt_res[coln].unstack(xaxis)
            df.sort_index(axis=0, level=0, inplace=True, ascending=False)
            df.sort_index(axis=1, level=0, inplace=True)
            sn.heatmap(df, annot=True, annot_kws={"size": 8}, ax=axs[i])
            axs[i].set_title(coln)

        fig.suptitle(underly)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig, axs

    def barplot_basic(self):
        df = self.bt_res[['count|PnL', 'WinLoss', 'AvgReturn', 'TotalReturn', 'sum|PnL', 'MaxDD']]
        df.columns.name = 'res_type'
        df = df.stack().reset_index().rename(columns={0: 'res_value'})
        df.name = 'res_value'
        df = df
        g = sn.factorplot(x='sec_le', y='res_value', col='dur', row='res_type', hue='freq',
                          data=df, kind='bar', sharey='row', size=2, aspect=1.5,
                          margin_titles=False)
        return g

    def heatmap_parts(self, xaxis=0, bt_res=None):
        # Overview: Long & Short (Heatmap)
        if bt_res is None:
            bt_res = self.bt_res
            underly = self.underly
        else:
            underly = bt_res['Underlyings'].apply(lambda x: ','.join(x)).unique()

        for k in ['Long', 'Short']:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            i = 0
            for coln, ss in bt_res.filter(like=k).iteritems():
                if 'pct|Loss|Enter' in coln:
                    continue
                df = ss.unstack(xaxis)
                df.sort_index(axis=0, level=0, inplace=True, ascending=False)
                df.sort_index(axis=1, level=0, inplace=True)
                sn.heatmap(df, annot=True, annot_kws={"size": 8}, ax=axs[i])
                axs[i].set_title(coln)
                i += 1

            fig.suptitle(underly)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig, axs

    def cummulative_returns(self):
        # Cummulative Return
        cret0 = self.trec_unstack.loc[:, 'CumRet(RReturn)'].ffill()

        # Integration by refine_tradingRecord
        self.trarec_refined = gbt.refine_tradingRecord(self.trading_rec.sort_values('dt_Enter'))
        emat_int, trec_int = gbt.BT_wCost(self.trarec_refined)
        cret1 = pd.Series(trec_int['CumRet(RReturn)'], name='integrated')

        fig, ax = plt.subplots(figsize=(9, 10))
        cret0.plot.line(grid=True, title=', '.join(self.underly), cmap=cm.rainbow, ax=ax)
        cret1.plot.line(ls='-.', color='k', grid=True, ax=ax, legend=True)
        fig.tight_layout(rect=(0, 0, 1, 1))
        return fig, ax

    def holding_period(self, grp_name=None):
        if grp_name:
            hp = self.trec.get_group(grp_name)['HoldingPeriod']
        else:
            hp = self.trec['HoldingPeriod']
        hp_dist = np.floor(hp.dt.seconds / 3600).value_counts().sort_index().reset_index()
        ax = sn.barplot(x='index', y='HoldingPeriod', data=hp_dist)
        plt.show()
        return hp_dist, ax

    def count_by_direc(self, coln):
        """
        """
        df_time = pd.concat([np.round(self.trec['HoldingPeriod'].dt.seconds / 3600),
                             self.trec['dt_Enter'].dt.hour,
                             self.trec['dt_Enter'].dt.minute,
                             self.trec['dt_Exit'].dt.hour,
                             self.trec['dt_Exit'].dt.minute],
                            1, keys=['hp', 'en_h', 'en_m', 'ex_h', 'ex_m'])

        time_dist = pd.concat([df_time[(self.trec['Type_Enter'] == 1) & (self.trec['RReturn'] > 0)][coln].value_counts(),
                               df_time[(self.trec['Type_Enter'] == -1) &
                                       (self.trec['RReturn'] > 0)][coln].value_counts(),
                               df_time[(self.trec['Type_Enter'] == 1) & (
                                   self.trec['RReturn'] < 0)][coln].value_counts(),
                               df_time[(self.trec['Type_Enter'] == -1) & (self.trec['RReturn'] < 0)][coln].value_counts()],
                              1, keys=['lo&pos', 'sh&pos', 'lo&neg', 'sh&neg'])
        return time_dist.fillna(0)

    def profit_period(self):

        time_dist = pd.concat([self.count_by_direc('en_h'), self.count_by_direc('ex_h')], 1)
        time_dist.plot.bar(subplots=True, layout=(2, 4), figsize=(15, 8))
        plt.show()
        hp_dist = self.count_by_direc('hp')
        hp_dist.plot.bar(subplots=True, layout=(2, 4), figsize=(15, 8))
        plt.show()

    def trades_occurence(self):
        trades_freq = self.trarec_grp.apply(lambda x: x.groupby('Underlying').size())
        trades_freq.name = 'Occurence'
        print('Number of trades for each underlying:\n\n')
        print(trades_freq)
        trades_freq = trades_freq.reset_index()
        sn.factorplot(x='sec_le', y='Occurence', col='dur', row='freq', hue='Underlying',
                      data=trades_freq, kind='bar')

    def daily_trades_freq(self):
        trarec_grp = plt_bt_res.trec.groupby(plt_bt_res.trec.index.names[:-1] + ['Underlying'])
        res = trarec_grp.size().unstack('Underlying')
        print '\n\nFrequencies of the number of trades everyday:\n', res.T

    def print_basic(self, coerext='145500'):
        idx_coer = self.trec['dt_Exit'].dt.strftime('%H:%M:%S') == coerext
        coerext_pct = gfc.true_pct(idx_coer)
        print '%Coerce Exit: ', coerext_pct, '\n\n'
        if coerext_pct > 0:
            emat_coerce, trec_coerce = gbt.BT_wCost(self.trec[idx_coer])
            emat_complete, trec_complete = gbt.BT_wCost(self.trec[~idx_coer])
            coer_vs_compl = pd.concat([emat_coerce, emat_complete], 1, keys=['coerce', 'complete']).T
            print coer_vs_compl[['TotalReturn', 'WinLoss']]
            print coer_vs_compl.filter(like='count')
            print coer_vs_compl.filter(like='Long')
            print coer_vs_compl.filter(like='Short')
        else:
            print 'No coerce exit.'

    def bar_parts(self, paras):
        """Summary
        Eligible for only given paramater.
        """
        given_bt_res = self.bt_res.loc[pd.IndexSlice[paras], :]
        res_long = given_bt_res.filter(like='Long')
        res_long = pd.Series(res_long.values, name='long', index=res_long.index.str.split(':').str[0])
        res_short = given_bt_res.filter(like='Short')
        res_short = pd.Series(res_short.values, name='short', index=res_short.index.str.split(':').str[0])
        ls_res = pd.concat([res_long, res_short], 1)

        ls_status = ls_res.stack().reset_index()
        ls_status.columns = ['Type', 'Direc', 'Results']
        sn.factorplot(x='Direc', y='Results', data=ls_status, col='Type', kind='bar', sharey=False)
        plt.tight_layout()
        plt.show()

    def seperated_heatmap_basics(self, underly=None):
        if underly is None:
            underly = self.underly
        for ud in underly:
            tr = self.indiv_tr[ud]
            btr = tr.groupby(tr.index.names[:-1]).apply(lambda x: gbt.BT_wCost(x)[0]).unstack(-1).fillna(np.nan)
            self.heatmap_basic(bt_res=btr)

    def seperated_heatmap_parts(self, underly=None):
        if underly is None:
            underly = self.underly
        for ud in underly:
            tr = self.indiv_tr[ud]
            btr = tr.groupby(tr.index.names[:-1]).apply(lambda x: gbt.BT_wCost(x)[0]).unstack(-1).fillna(np.nan)
            self.heatmap_parts(bt_res=btr)
