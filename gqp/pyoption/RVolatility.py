import numpy as np
import pandas as pd

"""
Reference:
    1. <Bennett_2012_measuring_historic_volatility>: Close-to-Close, Exponentially Weighted, Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang Volatility
    2. https://breakingdownfinance.com/finance-topics/risk-management/parkinson-volatility/
    3. https://breakingdownfinance.com/finance-topics/risk-management/garman-klass-volatility/
"""


# def close_to_close(cl, F=252, N=1):
#     if len(cl) < 10:
#         return np.nan
#     change = (cl - np.mean(cl)) ** 2
#     vol = np.sqrt(np.sum(change))
#     return vol * np.sqrt(F / (N - 1))


# def parkinson(hi, lo, F=252, N=1):
#     if len(hi) < 10:
#         return np.nan
#     sum_of_change = np.sum(np.log(hi / lo) ** 2) / (4 * np.log(2))
#     vol = np.sqrt(sum_of_change)
#     return vol * np.sqrt(F / N)


# def garman_klass(op, hi, lo, cl, F=252, N=1):
#     if len(cl) < 10:
#         return np.nan
#     change = 0.5 * np.log(hi / lo) ** 2 - (2 * np.log(2) - 1) * np.log(cl / op) ** 2
#     vol = np.sqrt(np.sum(change))
#     return vol * np.sqrt(F / N)


# def rogers_satchell(op, hi, lo, cl, F=252, N=1):
#     if len(cl) < 10:
#         return np.nan
#     change = np.log(hi / cl) * np.log(hi / op) + np.log(lo / cl) * np.log(lo / op)
#     vol = np.sqrt(np.sum(change))
#     return vol * np.sqrt(F / N)


# def gklass_yzhang(op, hi, lo, cl, F=252, N=1):
#     """Garman Klass and Yang Zhang extension"""
#     if len(cl) < 10:
#         return np.nan
#     op1, hi1, lo1, cl1 = op[1:], hi[1:], lo[1:], cl[1:]
#     cl0 = cl[:-1]
#     change = np.log(op1 / cl0) ** 2 + 0.5 * np.log(hi1 / lo1) ** 2 - (2 * np.log(2) - 1) * np.log(cl1 / op1) ** 2
#     vol = np.sqrt(np.sum(change))
#     return vol * np.sqrt(F / N)


# def yang_zhang(op, hi, lo, cl, F=252, N=1):
#     if len(cl) < 10:
#         return np.nan
#     op1, hi1, lo1, cl1 = op[1:], hi[1:], lo[1:], cl[1:]
#     cl0 = cl[:-1]
#     k = 0.34 / (1.34 + (N + 1) / (N - 1))
#     var_overnight = np.sum((np.log(op1 / cl0) - np.mean(np.log(op1 / cl0))) ** 2) / (N - 1)
#     var_oc = np.sum((np.log(cl1 / op1) - np.mean(np.log(cl1 / op1))) ** 2) / (N - 1)
#     var_rs = rogers_satchell(op1, hi1, lo1, cl1, F, N) ** 2
#     vol = np.sqrt(F) * np.sqrt(var_overnight + k * var_oc + (1 - k) * var_rs)
#     return vol


# # Serialize
# def numpy_shift(a, shift):
#     shifted_a = np.roll(a, shift)
#     shifted_a[:shift] = np.nan
#     return shifted_a


# def close_to_close(cl, F=252, N=1):
#     logret = np.log(cl / numpy_shift(cl, 1))
#     return np.sqrt(F / N) * pd.Series(logret).rolling(N, min_periods=N).std().values


# def parkinson(hi, lo, F=252, N=1):
#     # someone use logarithm base on 10 instead of natural logarithm.
#     hilo = np.log(hi / lo)
#     variance = (F / N) * (1 / (4 * np.log(2))) * pd.Series(hilo ** 2).rolling(N, min_periods=N).sum().values
#     return np.sqrt(variance)


# def garman_klass(op, hi, lo, cl, F=252, N=1):
#     hilo = np.log(hi / lo)
#     clop = np.log(cl / op)
#     change = 0.5 * hilo ** 2 - (2 * np.log(2) - 1) * clop ** 2
#     variance = (F / N) * pd.Series(change).rolling(N, min_periods=N).sum().values
#     return np.sqrt(variance)


# def rogers_satchell(op, hi, lo, cl, F=252, N=1):
#     hicl = np.log(hi / cl)
#     hiop = np.log(hi / op)
#     locl = np.log(lo / cl)
#     loop = np.log(lo / op)

#     change = hicl * hiop + locl * loop
#     variance = (F / N) * pd.Series(change).rolling(N, min_periods=N).sum().values
#     return np.sqrt(variance)


# def gklass_yzhang(op, hi, lo, cl, F=252, N=1):
#     """Garman Klass and Yang Zhang extension"""
#     opcl_1 = np.log(op / numpy_shift(cl, 1))
#     hilo = np.log(hi / lo)
#     clop = np.log(cl / op)
#     change = opcl_1 ** 2 + 0.5 * hilo ** 2 - (2 * np.log(2) - 1) * clop ** 2
#     variance = (F / N) * pd.Series(change).rolling(N, min_periods=N).sum().values
#     return np.sqrt(variance)


# def yang_zhang(op, hi, lo, cl, k0=0.34, F=252, N=1):
#     opcl_1 = np.log(op / numpy_shift(cl, 1))
#     clop = np.log(cl / op)
#     k = k0 / ((1 + k0) + (N + 1) / (N - 1))

#     chg_overnight = (opcl_1 - pd.Series(opcl_1).rolling(N, min_periods=N).mean().values) ** 2
#     var_overnight = (1 / (N - 1)) * pd.Series(chg_overnight).rolling(N, min_periods=N).sum().values

#     chg_clop = (clop - pd.Series(clop).rolling(N, min_periods=N).mean().values) ** 2
#     var_clop = (1 / (N - 1)) * pd.Series(chg_clop).rolling(N, min_periods=N).sum().values

#     var_rs = rogers_satchell(op, hi, lo, cl, F, N) ** 2

#     variance = F * (var_overnight + k * var_clop + (1 - k) * var_rs)
#     return np.sqrt(variance)


# Applicable for intraday calculation
def numpy_shift(a, shift):
    shifted_a = np.roll(a, shift)
    shifted_a[:shift] = np.nan
    return shifted_a


def close_to_close(cl, N, B):
    """Summary

    Args:
        cl (TYPE): close price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description

    Deleted Parameters:
        k (TYPE): the number of calendar days within the time series
        F (int, optional): the number of calendar days per year
    """
    logret = np.log(cl / numpy_shift(cl, 1))
    variance = pd.Series(logret).rolling(N, min_periods=N).var().values
    return np.sqrt(B * variance)


def parkinson(hi, lo, N, B):
    """Summary

    Args:
        hi (TYPE): highest price
        lo (TYPE): lowest price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description

    Deleted Parameters:
        k (TYPE): the number of calendar days within the time series
        F (int, optional): the number of calendar days per year
    """
    # someone use logarithm base on 10 instead of natural logarithm.
    hilo = np.log(hi / lo)
    variance = (1 / (N - 1)) * (1 / (4 * np.log(2))) * pd.Series(hilo ** 2).rolling(N, min_periods=N).sum().values
    return np.sqrt(B * variance)


def garman_klass(op, hi, lo, cl, N, B):
    """Summary

    Args:
        op (TYPE): open price
        hi (TYPE): highest price
        lo (TYPE): lowest price
        cl (TYPE): close price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description

    Deleted Parameters:
        k (TYPE): the number of calendar days within the time series
        F (int, optional): the number of calendar days per year
    """
    hilo = np.log(hi / lo)
    clop = np.log(cl / op)
    change = 0.5 * hilo ** 2 - (2 * np.log(2) - 1) * clop ** 2
    variance = (1 / (N - 1)) * pd.Series(change).rolling(N, min_periods=N).sum().values
    return np.sqrt(B * variance)


def rogers_satchell(op, hi, lo, cl, N, B):
    """Summary

    Args:
        op (TYPE): open price
        hi (TYPE): highest price
        lo (TYPE): lowest price
        cl (TYPE): close price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description

    Deleted Parameters:
        k (TYPE): the number of calendar days within the time series
        F (int, optional): the number of calendar days per year
    """
    hicl = np.log(hi / cl)
    hiop = np.log(hi / op)
    locl = np.log(lo / cl)
    loop = np.log(lo / op)

    change = hicl * hiop + locl * loop
    variance = (1 / (N - 1)) * pd.Series(change).rolling(N, min_periods=N).sum().values
    return np.sqrt(B * variance)


def gklass_yzhang(op, hi, lo, cl, N, B):
    """Garman Klass and Yang Zhang extension

    Args:
        op (TYPE): open price
        hi (TYPE): highest price
        lo (TYPE): lowest price
        cl (TYPE): close price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description

    Deleted Parameters:
        k (TYPE): the number of calendar days within the time series
        F (int, optional): the number of calendar days per year
    """
    opcl_1 = np.log(op / numpy_shift(cl, 1))
    hilo = np.log(hi / lo)
    clop = np.log(cl / op)
    change = opcl_1 ** 2 + 0.5 * hilo ** 2 - (2 * np.log(2) - 1) * clop ** 2
    variance = (1 / (N - 1)) * pd.Series(change).rolling(N, min_periods=N).sum().values
    return np.sqrt(B * variance)


def yang_zhang(op, hi, lo, cl, N, B):
    """Summary

    Args:
        op (TYPE): open price
        hi (TYPE): highest price
        lo (TYPE): lowest price
        cl (TYPE): close price
        N (TYPE): the number of samples, here is the rolling window
        B (TYPE): annualized factor. B = F * b, where F is calendar days in one year, b is the number of sample per day

    Returns:
        TYPE: Description
    """
    if N == 1:
        return np.full_like(cl, np.nan)

    q0 = 0.34
    opcl_1 = np.log(op / numpy_shift(cl, 1))
    clop = np.log(cl / op)
    k = q0 / ((1 + q0) + (N + 1) / (N - 1))

    chg_overnight = (opcl_1 - pd.Series(opcl_1).rolling(N, min_periods=N).mean().values) ** 2
    var_overnight = (1 / (N - 1)) * pd.Series(chg_overnight).rolling(N, min_periods=N).sum().values

    chg_clop = (clop - pd.Series(clop).rolling(N, min_periods=N).mean().values) ** 2
    var_clop = (1 / (N - 1)) * pd.Series(chg_clop).rolling(N, min_periods=N).sum().values

    var_rs = rogers_satchell(op, hi, lo, cl, N, B=1) ** 2

    variance = var_overnight + k * var_clop + (1 - k) * var_rs
    return np.sqrt(B * variance)


def calc_vix2(options, R=0.03, Nmid=30 * 60 * 24, Nbase=365 * 60 * 24):
    """
    Calculating volatility index

    Args:
        options (dataframe):
            formatted dataframe includes the following columns in corresponding data type:
                Data columns (total 7 columns):
                timestamp          datetime64[ns]
                cp_flag            int64
                expire_date        datetime64[ns]
                exercise_price     float64
                ap1                float64
                bp1                float64
                volume_multiple    int64
                dtypes: datetime64[ns](2), float64(3), int64(2)

            plus, expire_date must be unique.

        R (float, optional): risk free rate
        Nmid (TYPE, optional): time to maturity of one month in minutes
        Nbase (TYPE, optional): periods in unit of minutes per year

    Returns:
        float64: volatility index value given option data

    """
    def calc_time_to_maturity(expire_date, current_date, Nbase):
        """Summary

        Args:
            expire_date (timestamp): Description
            current_date (timestamp): Description
            Nbase (float): periods in unit of minutes per year

        Returns:
            TYPE: Description
        """

        NT = (expire_date.replace(hour=9, minute=30, second=0) - current_date) / np.timedelta64(1, 'm')
        return NT, NT / Nbase

    def calc_forward_price(opt, R, T):
        """Summary

        Args:
            opt (dataframe): formatted dataframe of option data with unique expire date
            R (float): risk-free rate
            T (float): time to maturity

        Returns:
            TYPE: Description

        Deleted Parameters:
            options (dataframe): formatted dataframe of option data with unique expire date

        Raises:
            ValueError: Description
        """
        if opt['expire_date'].nunique() > 1:
            raise ValueError('got multiple expire dates.')
        opt['premium'] = opt[['ap1', 'bp1']].mean(1)
        premium = opt.pivot(index='exercise_price', columns='cp_flag', values='premium')
        premium_spread = premium[1] - premium[2]
        S = premium_spread.abs().idxmin()
        F = S + np.exp(R * T) * premium_spread[S]
        return F

    def calc_sigma_sq(opt, R, T, F):
        """Summary

        Args:
            opt (dataframe): formatted dataframe of option data
            R (float): risk-free rate
            T (float): time to maturity
            F (float): forward price

        Returns:
            float: variance

        Raises:
            ValueError: Description

        Deleted Parameters:
            options (float): formatted dataframe of option data
        """
        def _select_Ki(ss):
            # kbp: series with k as index and bid as values
            cnt = 0
            selected_Ks = []
            for k, bid in ss.items():
                if bid == 0:
                    cnt += 1
                    if cnt == 2:
                        break
                else:
                    selected_Ks.append(k)
                    cnt = 0
            return selected_Ks

        if opt['expire_date'].nunique() > 1:
            raise ValueError('got multiple expire dates.')

        df = opt.set_index('exercise_price')
        sorted_call = df.loc[df['cp_flag'] == 1, 'bp1'].sort_index()
        sorted_put = df.loc[df['cp_flag'] == 2, 'bp1'].sort_index()
        # K0 = sorted_call.index[sorted_call.index < F][-1]    # using put returns the same result
        idx_downside = sorted_call.index < F
        if sum(idx_downside) > 0:
            K0 = sorted_call.index[idx_downside][-1]    # using put returns the same result
        else:
            K0 = sorted_call.index[0]

        part_call = sorted_call[sorted_call.index > K0]
        part_put = sorted_put[sorted_put.index < K0]

        Ks_from_call = _select_Ki(part_call)
        Ks_from_put = _select_Ki(part_put.iloc[::-1])[::-1]
        Ki = np.hstack([Ks_from_put, K0, Ks_from_call])
        if np.any(np.diff(Ki) < 0):
            raise ValueError('sorting problemss in Ks while processing')
        if len(Ki) != len(set(Ki)):
            raise ValueError('not identical K in Ks while processing')
        dKi = np.hstack([np.diff(Ki[:2]), (Ki[2:] - Ki[:-2]) / 2, np.diff(Ki[-2:])])

        df['QK'] = df[['bp1', 'ap1']].mean(1)
        QKi = pd.concat([
            df[df['cp_flag'] == 2].loc[Ks_from_put, 'QK'],
            pd.Series(np.sum(df.loc[K0, ['bp1', 'ap1']].values) / 4, index=[str(K0)]),
            df[df['cp_flag'] == 1].loc[Ks_from_call, 'QK']
        ]).values

        sigma_sq = (2 / T) * np.sum((dKi / Ki ** 2) * np.exp(R * T) * QKi) - (F / K0 - 1) ** 2 / T
        return sigma_sq

    date1, date2 = np.sort(options['expire_date'].unique())[:2]
    # 近月
    opt = options[options['expire_date'] == date1].copy()
    expire_date, current_date = opt[['expire_date', 'timestamp']].iloc[0, :]
    NT1, T1 = calc_time_to_maturity(expire_date, current_date, Nbase)
    F1 = calc_forward_price(opt, R, T1)
    sigma_sq1 = calc_sigma_sq(opt, R, T1, F1)
    if NT1 <= Nmid:
        # 次近月
        opt = options[options['expire_date'] == date2].copy()
        expire_date, current_date = opt[['expire_date', 'timestamp']].iloc[0, :]
        NT2, T2 = calc_time_to_maturity(expire_date, current_date, Nbase)
        F2 = calc_forward_price(opt, R, T2)
        sigma_sq2 = calc_sigma_sq(opt, R, T2, F2)

        # Calc iVX
        vol1 = T1 * sigma_sq1 * (NT2 - Nmid) / (NT2 - NT1)
        vol2 = T2 * sigma_sq2 * (Nmid - NT1) / (NT2 - NT1)
        sigma_sq1 = (vol1 + vol2) * Nbase / Nmid

    if sigma_sq1 >= 0:
        iVX = 100 * np.sqrt(sigma_sq1)
    else:
        iVX = np.nan
    return iVX



def calc_vix(options, R=0.03, Nmid=30 * 60 * 24, Nbase=365 * 60 * 24):
    """
    Calculating volatility index

    Args:
        options (dataframe):
            formatted dataframe includes the following columns in corresponding data type:
                Data columns (total 7 columns):
                timestamp          datetime64[ns]
                cp_flag            int64
                expire_date        datetime64[ns]
                exercise_price     float64
                close                float64
                volume_multiple    int64
                dtypes: datetime64[ns](2), float64(3), int64(2)

            plus, expire_date must be unique.

        R (float, optional): risk free rate
        Nmid (TYPE, optional): time to maturity of one month in minutes
        Nbase (TYPE, optional): periods in unit of minutes per year

    Returns:
        float64: volatility index value given option data

    """
    def calc_time_to_maturity(expire_date, current_date, Nbase):
        """Summary

        Args:
            expire_date (timestamp): Description
            current_date (timestamp): Description
            Nbase (float): periods in unit of minutes per year

        Returns:
            TYPE: Description
        """

        NT = (expire_date.replace(hour=9, minute=30, second=0) - current_date) / np.timedelta64(1, 'm')
        return NT, NT / Nbase

    def calc_forward_price(opt, R, T):
        """Summary

        Args:
            opt (dataframe): formatted dataframe of option data with unique expire date
            R (float): risk-free rate
            T (float): time to maturity

        Returns:
            TYPE: Description

        Deleted Parameters:
            options (dataframe): formatted dataframe of option data with unique expire date

        Raises:
            ValueError: Description
        """
        if opt['expire_date'].nunique() > 1:
            raise ValueError('got multiple expire dates.')
        opt['premium'] = opt['close'].copy()
        premium = opt.pivot(index='exercise_price', columns='cp_flag', values='premium')
        premium_spread = premium[1] - premium[2]
        S = premium_spread.abs().idxmin()
        F = S + np.exp(R * T) * premium_spread[S]
        return F

    def calc_sigma_sq(opt, R, T, F):
        """Summary

        Args:
            opt (dataframe): formatted dataframe of option data
            R (float): risk-free rate
            T (float): time to maturity
            F (float): forward price

        Returns:
            float: variance

        Raises:
            ValueError: Description

        Deleted Parameters:
            options (float): formatted dataframe of option data
        """
        def _select_Ki(ss):
            # kbp: series with k as index and bid as values
            cnt = 0
            selected_Ks = []
            for k, bid in ss.items():
                if bid == 0:
                    cnt += 1
                    if cnt == 2:
                        break
                else:
                    selected_Ks.append(k)
                    cnt = 0
            return selected_Ks

        if opt['expire_date'].nunique() > 1:
            raise ValueError('got multiple expire dates.')

        df = opt.set_index('exercise_price')
        sorted_call = df.loc[df['cp_flag'] == 1, 'close'].sort_index()
        sorted_put = df.loc[df['cp_flag'] == 2, 'close'].sort_index()
        # K0 = sorted_call.index[sorted_call.index < F][-1]    # using put returns the same result
        idx_downside = sorted_call.index < F
        if sum(idx_downside) > 0:
            K0 = sorted_call.index[idx_downside][-1]    # using put returns the same result
        else:
            K0 = sorted_call.index[0]

        part_call = sorted_call[sorted_call.index > K0]
        part_put = sorted_put[sorted_put.index < K0]

        Ks_from_call = _select_Ki(part_call)
        Ks_from_put = _select_Ki(part_put.iloc[::-1])[::-1]
        Ki = np.hstack([Ks_from_put, K0, Ks_from_call])
        if np.any(np.diff(Ki) < 0):
            raise ValueError('sorting problemss in Ks while processing')
        if len(Ki) != len(set(Ki)):
            raise ValueError('not identical K in Ks while processing')
        dKi = np.hstack([np.diff(Ki[:2]), (Ki[2:] - Ki[:-2]) / 2, np.diff(Ki[-2:])])

        df['QK'] = df['close'].copy()
        QKi = pd.concat([
            df[df['cp_flag'] == 2].loc[Ks_from_put, 'QK'],
            pd.Series(np.sum(df.loc[K0, 'close'].values) / 4, index=[str(K0)]),
            df[df['cp_flag'] == 1].loc[Ks_from_call, 'QK']
        ]).values

        sigma_sq = (2 / T) * np.sum((dKi / Ki ** 2) * np.exp(R * T) * QKi) - (F / K0 - 1) ** 2 / T
        return sigma_sq

    date1, date2 = np.sort(options['expire_date'].unique())[:2]
    # 近月
    opt = options[options['expire_date'] == date1].copy()
    expire_date, current_date = opt[['expire_date', 'timestamp']].iloc[0, :]
    NT1, T1 = calc_time_to_maturity(expire_date, current_date, Nbase)
    F1 = calc_forward_price(opt, R, T1)
    sigma_sq1 = calc_sigma_sq(opt, R, T1, F1)
    if NT1 <= Nmid:
        # 次近月
        opt = options[options['expire_date'] == date2].copy()
        expire_date, current_date = opt[['expire_date', 'timestamp']].iloc[0, :]
        NT2, T2 = calc_time_to_maturity(expire_date, current_date, Nbase)
        F2 = calc_forward_price(opt, R, T2)
        sigma_sq2 = calc_sigma_sq(opt, R, T2, F2)

        # Calc iVX
        vol1 = T1 * sigma_sq1 * (NT2 - Nmid) / (NT2 - NT1)
        vol2 = T2 * sigma_sq2 * (Nmid - NT1) / (NT2 - NT1)
        sigma_sq1 = (vol1 + vol2) * Nbase / Nmid

    if sigma_sq1 >= 0:
        iVX = 100 * np.sqrt(sigma_sq1)
    else:
        iVX = np.nan
    return iVX