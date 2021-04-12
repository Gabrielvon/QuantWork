import numpy as np
# import pandas as pd
import math
import scipy.stats as scs
from gqp.pyoption import option_numba as optnb
from copy import copy


def chi(k, kPut, kATM, kCall, flag):
    if flag == 1:
        out = (math.log(kATM / k) * math.log(kCall / k)) / (math.log(kATM / kPut) * math.log(kCall / kPut))
    elif flag == 2:
        out = (math.log(k / kPut) * math.log(kCall / k)) / (math.log(kATM / kPut) * math.log(kCall / kATM))
    elif flag == 3:
        out = (math.log(k / kPut) * math.log(k / kATM)) / (math.log(kCall / kPut) * math.log(kCall / kATM))
    else:
        raise ValueError('no such flag: ' + str(flag))
    return out


def x_weights(k, kPut, kATM, kCall, vegaK, vegaPut, vegaATM, vegaCall):
    if k == kPut:
        x1, x2, x3 = 1, 0, 0
    elif k == kATM:
        x1, x2, x3 = 0, 1, 0
    elif k == kCall:
        x1, x2, x3 = 0, 0, 1
    else:
        x1 = (vegaK / vegaPut) * chi(k, kPut, kATM, kCall, flag=1)
        x2 = (vegaK / vegaATM) * chi(k, kPut, kATM, kCall, flag=2)
        x3 = (vegaK / vegaCall) * chi(k, kPut, kATM, kCall, flag=3)
    return x1, x2, x3


def d1(F, K, sigma, tau):
    return (math.log(F / K) + (0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))


def d2(F, K, sigma, tau):
    return d1(F, K, sigma, tau) - sigma * math.sqrt(tau)


def firstOrderVol(k, kPut, kATM, kCall, volPut, volATM, volCall):
    y1 = chi(k, kPut, kATM, kCall, flag=1)
    y2 = chi(k, kPut, kATM, kCall, flag=2)
    y3 = chi(k, kPut, kATM, kCall, flag=3)
    return np.float64(y1 * volPut + y2 * volATM + y3 * volCall)  # eventuellement entourer par np.float64


def VannaVolgaImpliedVol(k, kPut, kATM, kCall, volPut, volATM, volCall, F, tau):
    """secondOrderVol"""
    y1 = chi(k, kPut, kATM, kCall, flag=1)
    y2 = chi(k, kPut, kATM, kCall, flag=2)
    y3 = chi(k, kPut, kATM, kCall, flag=3)

    v_1st = y1 * volPut + y2 * volATM + y3 * volCall
    D1 = v_1st - volATM

    d1d2_k = d1(F, k, volATM, tau) * d2(F, k, volATM, tau)
    d1d2_Put = d1(F, kPut, volPut, tau) * d2(F, kPut, volPut, tau)
    d1d2_Call = d1(F, kCall, volCall, tau) * d2(F, kCall, volCall, tau)

    D2 = y1 * d1d2_Put * ((volPut - volATM) ** 2) + y3 * d1d2_Call * ((volCall - volATM) ** 2)

    v_2nd = volATM + (-volATM + math.sqrt(volATM ** 2 + d1d2_k * (2 * volATM * D1 + D2))) / d1d2_k
    return v_2nd


def get_anchor_points(data, delta_lv=0.25, maxiter=100, verbose=0):
    """without calibration"""

    grp = data.copy()
    assert grp[['spot_price', 'expire_date', 'rf', 'tau', 'q']].nunique().max(
    ) == 1, "'spot_price', 'rf', 'tau', 'q' can only have unique value"
    assert grp['exercise_price'].nunique() > 3, 'exercise_price must have more than three unique values'
    assert grp['cp_flag'].nunique() == 2, 'data must include call(1) and put(2)'

    cp, S0, r0, tau0, q0 = grp[['cp_flag', 'spot_price', 'rf', 'tau', 'q']].values[0]
    F_t = S0 * np.exp(r0 * tau0)    # Forward price
    grp['abs_XmF'] = (grp['exercise_price'].abs() - F_t).abs()
    grp['abs_delta_dist'] = (grp['delta'].abs() - delta_lv).abs()
    kATM, volATM = grp.loc[grp['abs_XmF'].idxmin(), ['exercise_price', 'bsiv']].values
    grp_wo_ATM = grp[grp['exercise_price'] != kATM].copy()
    kPut, volPut = grp_wo_ATM.loc[grp['cp_flag'] == 2, :].sort_values('abs_delta_dist')[['exercise_price', 'bsiv']].values[0]
    kCall, volCall = grp_wo_ATM.loc[grp['cp_flag'] == 1, :].sort_values('abs_delta_dist')[['exercise_price', 'bsiv']].values[0]
    return kPut, kATM, kCall, volPut, volATM, volCall, F_t, tau0


def get_calib_anchor_points(data, delta_lv=0.25, maxiter=100, calibrate=False, verbose=0):
    """with calibration"""

    grp = data.copy()
    assert grp[['spot_price', 'expire_date', 'rf', 'tau', 'q']].nunique().max(
    ) == 1, "'spot_price', 'rf', 'tau', 'q' can only have unique value"
    assert grp['exercise_price'].nunique() > 3, 'exercise_price must have more than three unique values'
    assert grp['cp_flag'].nunique() == 2, 'data must include call(1) and put(2)'

    grp['flat_vol'] = np.sum((grp['vega'] / grp['vega'].sum()) * grp['bsiv'])
    grp['bs_price'] = optnb.get_option_value(
        *grp[['cp_flag', 'spot_price', 'exercise_price', 'rf', 'tau', 'flat_vol', 'q']].values.T)
    cp, S0, r0, tau0, q0, fvol0 = grp[['cp_flag', 'spot_price', 'rf', 'tau', 'q', 'flat_vol']].values[0]
    F_t = S0 * np.exp(r0 * tau0)    # Forward price
    grp['abs_XmF'] = (grp['exercise_price'].abs() - F_t).abs()
    grp['abs_delta_dist'] = (grp['delta'].abs() - delta_lv).abs()
    _colns = ['exercise_price', 'bsiv', 'vega', 'market_price', 'bs_price']
    cp_ATM, kATM, volATM, vegaPut, valPut_MKT, valPut_BS = grp.loc[grp['abs_XmF'].idxmin(), ['cp_flag'] + _colns]
    grp_wo_ATM = grp[grp['exercise_price'] != kATM].copy()
    kPut, volPut, vegaATM, valATM_MKT, valATM_BS = grp_wo_ATM.loc[grp['cp_flag'] == 2, :].sort_values('abs_delta_dist')[_colns].iloc[0]
    kCall, volCall, vegaCall, valCall_MKT, valCall_BS = grp_wo_ATM.loc[grp['cp_flag'] == 1, :].sort_values('abs_delta_dist')[_colns].iloc[0]
    alpha = -scs.norm.ppf(delta_lv * np.exp(r0 * tau0))

    # calibrate atm
    for i in np.arange(maxiter):
        if i == 0:
            vol = volATM
            k = kATM
        else:
            pre_k = copy(k)
            k = S0 * np.exp((r0 + 0.5 * vol ** 2) * tau0)    # k_(i), vol_(i-1)
            vega = optnb.vega(*np.array([[S0, k, r0, tau0, vol]]).T)[0]
            xPut, xATM, xCall = x_weights(k, kPut, kATM, kCall, vega, vegaPut, vegaATM, vegaCall)
            valATM = valATM_BS + \
                xPut * (valPut_MKT - valPut_BS) + \
                xATM * (valATM_MKT - valATM_BS) + \
                xCall * (valCall_MKT - valCall_BS)
            vol = optnb.implied_volatility_series(
                *np.array([[cp_ATM, S0, k, r0, tau0, q0, valATM]]).T, maxiter=100)[0]    # vol_(i)
            if np.abs(k - pre_k) < 1e-8:
                break

    calib_kATM, calib_volATM = k, vol

    # calibrate 25-delta call
    for i in np.arange(maxiter):
        if i == 0:
            vol = volCall
            k = kCall
        else:
            pre_k = copy(k)
            k = S0 * np.exp(alpha * vol * math.sqrt(tau0) + (r0 + 0.5 * vol ** 2) * tau0)    # k_(i), vol_(i-1)
            vega = optnb.vega(*np.array([[S0, k, r0, tau0, vol]]).T)[0]
            xPut, xATM, xCall = x_weights(k, kPut, kATM, kCall, vega, vegaPut, vegaATM, vegaCall)
            valATM = valATM_BS + \
                xPut * (valPut_MKT - valPut_BS) + \
                xATM * (valATM_MKT - valATM_BS) + \
                xCall * (valCall_MKT - valCall_BS)
            vol = optnb.implied_volatility_series(
                *np.array([[1, S0, k, r0, tau0, q0, valATM]]).T, maxiter=100)[0]    # vol_(i)
            if np.abs(k - pre_k) < 1e-8:
                break

    calib_kCall, calib_volCall = k, vol

    # calibrate 25-delta put
    for i in np.arange(maxiter):
        if i == 0:
            vol = volPut
            k = kPut
        else:
            pre_k = copy(k)
            k = S0 * np.exp(-alpha * vol * math.sqrt(tau0) + (r0 + 0.5 * vol ** 2) * tau0)    # k_(i), vol_(i-1)
            vega = optnb.vega(*np.array([[S0, k, r0, tau0, vol]]).T)[0]
            xPut, xATM, xCall = x_weights(k, kPut, kATM, kCall, vega, vegaPut, vegaATM, vegaCall)
            valATM = valATM_BS + \
                xPut * (valPut_MKT - valPut_BS) + \
                xATM * (valATM_MKT - valATM_BS) + \
                xCall * (valCall_MKT - valCall_BS)
            vol = optnb.implied_volatility_series(
                *np.array([[2, S0, k, r0, tau0, q0, valATM]]).T, maxiter=100)[0]    # vol_(i)
            if np.abs(k - pre_k) < 1e-8:
                break

    calib_kPut, calib_volPut = k, vol

    if verbose > 0:
        print('Last iteration: ')
        print('\tPut: \t Iter(', i, ')\t', k - pre_k, calib_kPut, calib_volPut)
        print('\tATM: \t Iter(', i, ')\t', k - pre_k, calib_kATM, calib_volATM)
        print('\tCall: \t Iter(', i, ')\t', k - pre_k, calib_kCall, calib_volCall)

    if verbose > 1:
        print('Compare: ')
        print('\tBefore Calibration: \t', kPut, kATM, kCall)
        print('\tBefore Calibration: \t', volPut, volATM, volCall)
        print('\tAfter Calibration: \t', calib_kPut, calib_kATM, calib_kCall)
        print('\tAfter Calibration: \t', calib_volPut, calib_volATM, calib_volCall)
    return calib_kPut, calib_kATM, calib_kCall, calib_volPut, calib_volATM, calib_volCall, F_t, tau0
