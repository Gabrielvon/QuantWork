import scipy.optimize as sco
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import option_numba as optnb


lb = np.array([-20, 0, -5, -0.05, 0])
ub = np.array([0.001, 0.2, 0.01, 0.02, 0.01])
chi0_guess = (ub + lb) / 2
# lb = -np.inf
# ub = np.inf


def log(*ss):
    """Do logging to stdout"""
    for s in ss:
        print(s)


def numpy_fillna_triang(arr, depth=1):
    def _fillna_triang(arr):
        triang_mean = pd.Series(arr).rolling(3, min_periods=1, center=True).mean()
        return pd.Series(arr).fillna(triang_mean)

    for i in range(depth):
        arr = _fillna_triang(arr)
        if not np.any(np.isnan(arr)):
            break

    return arr


def rawSVI(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def logstrike(S0, K, T, r, q):
    return np.log(K / S0 * np.exp(-(r - q) * T))


def straightSVI(x, m1, m2, q1, q2, c):
    return ((m1 + m2) * x + q1 + q2 + np.sqrt(((m1 + m2) * x + q1 + q2)**2 - 4 * (m1 * m2 * x**2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))) / 2


def residSVI(chi, logstrikes, tau, iv):
    """Residuals function for fitting implied volatility"""
    w = straightSVI(logstrikes, chi[0], chi[1], chi[2], chi[3], chi[4])
    return iv - np.sqrt(np.array(w) / tau)


def std2straight(a):
    """Obtain asymptotic parameters from alternative parametrization"""
    m1 = -a[1] / a[0] / 2. - np.sqrt((a[1] / a[0])**2 / 4. - a[2] / a[0])
    m2 = -a[1] / a[0] / 2. + np.sqrt((a[1] / a[0])**2 / 4. - a[2] / a[0])
    q1 = (1 + m1 * a[3]) / a[0] / (m2 - m1)
    q2 = (1 + m2 * a[3]) / a[0] / (m1 - m2)
    c = q1 * q2 - a[4] / a[0]
    return np.array([m1, m2, q1, q2, c])


def chi_guess(T, logstrikes, iv, maxfev=1200):
    """Function to obtain initial parameter vector for fit"""
    # Split data in five intervals and calculate mean x and mean y
    kmin = np.min(logstrikes)
    kmax = np.max(logstrikes)
    klo = [kmin + i * (kmax - kmin) / 5. for i in range(5)]
    kup = [kmin + (i + 1) * (kmax - kmin) / 5. for i in range(5)]
    xm = np.array([np.mean(logstrikes[(lo <= logstrikes) & (logstrikes <= u)]) for lo, u in zip(klo, kup)])    # Xm = E[logStrikes]
    ym = np.array([np.mean(T * iv[(lo <= logstrikes) & (logstrikes <= u)]**2) for lo, u in zip(klo, kup)])    # Ym = E[t * iv**2]
    xm = numpy_fillna_triang(xm)
    ym = numpy_fillna_triang(ym)

    # Determine quadratic form through these five average points
    un = np.array([1 for l in klo])
    A = np.matrix([ym * ym, ym * xm, xm * xm, ym, un]).T
    a = np.linalg.solve(A, -xm)

    # If it's already a hyperbola, we have our initial guess
    if 4 * a[0] * a[2] < a[1]**2:
        ap = std2straight(a)
        return np.minimum(np.maximum(ap, lb), ub, out=chi0_guess, where=~np.isnan(ap))

    # Otherwise, flip to approximating hyperbola and do a least squares fit to the five points
    a[0] = -a[0]

    def residHyp(chi):
        return straightSVI(xm, chi[0], chi[1], chi[2], chi[3], chi[4]) - ym

    ap = sco.leastsq(residHyp, std2straight(a), maxfev=maxfev)[0]
    # ap = std2straight(a)
    # ap = np.minimum(np.maximum(ap, lb), ub, out=chi0_guess, where=~np.isnan(ap))
    # ap = sco.least_squares(residHyp, x0=ap, bounds=[lb, ub], verbose=0, xtol=1e-6)['x']
    # ap = np.minimum(np.maximum(ap, lb), ub, out=chi0_guess, where=~np.isnan(ap))
    return ap


def straightSVIp(x, m1, m2, q1, q2, c):
    H = np.sqrt(((m1 + m2) * x + q1 + q2)**2 - 4 * (m1 * m2 * x**2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))
    return ((m1 + m2) + ((m1 + m2) * ((m1 + m2) * x + q1 + q2) - 4 * m1 * m2 * x - 2 * (m1 * q2 + m2 * q1)) / H) / 2


def straightSVIpp(x, m1, m2, q1, q2, c):
    H = np.sqrt(((m1 + m2) * x + q1 + q2)**2 - 4 * (m1 * m2 * x**2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))
    A = (2 * (m1 + m2)**2 - 8 * m1 * m2) / H
    B = (2 * (m1 + m2) * ((m1 + m2) * x + q1 + q2) - 8 * m1 * m2 * x - 4 * (m1 * q2 + m2 * q1))**2 / H**3 / 2
    return (A - B) / 4


def calendar(chi1, logstrike1, chi2, logstrike2):
    """Function to quantify calendar arbitrage between two slices T1 > T2 on grid"""
    # all logstrike are from organized grid with sorted tau
    w1 = straightSVI(logstrike1, chi1[0], chi1[1], chi1[2], chi1[3], chi1[4])
    w2 = straightSVI(logstrike2, chi2[0], chi2[1], chi2[2], chi2[3], chi2[4])
    return np.sum(np.maximum(0, w2 - w1))


def butterfly(chi, logstrike):
    """Function to quantify butterfly arbitrage in a slice on grid"""
    # all logstrike are from organized grid with sorted tau
    w = straightSVI(logstrike, chi[0], chi[1], chi[2], chi[3], chi[4])
    wp = straightSVIp(logstrike, chi[0], chi[1], chi[2], chi[3], chi[4])
    wpp = straightSVIpp(logstrike, chi[0], chi[1], chi[2], chi[3], chi[4])
    g = (1. - (logstrike * wp) / (2. * w))**2 - wp**2 / 4. * (1. / w + 1. / 4.) + wpp / 2.
    return np.sum(np.maximum(0, -g))


def residuals(chiT, logstrikeT, chiTp, logstrikeTp, optpri, bs_args, penalty):
    """Residuals function for fitting option prices with penalties on arbitrage"""
    # all logstrike are from organized grid with sorted tau
    cp_flag, S, K, R, T, logstrike, Q = bs_args
    cpen, bpen = penalty
    w = straightSVI(logstrike, chiT[0], chiT[1], chiT[2], chiT[3], chiT[4])

    sig = np.sqrt(np.array(w) / T)
    bs = optnb.get_option_value(cp_flag, S, K, R, T, sig, Q)    # bs value

    zero_calarbT = (chiTp is None) or (logstrikeTp is None)
    calarbT = calendar(chiT, logstrikeT, chiTp, logstrikeTp) if not zero_calarbT else 0    # calarb
    butarbT = butterfly(chiT, logstrikeT)    # butarb
    e = optpri - bs    # price error
    return e + (np.sqrt(sum(e)**2 + (cpen * calarbT + bpen * butarbT)**2 * len(e)) - sum(e)) / len(e)


def straight2raw(chi):
    """Obtain rawSVI parameters from asymptotic parametrization"""
    a = (chi[0] * chi[3] - chi[1] * chi[2]) / (chi[0] - chi[1])
    b = abs(chi[0] - chi[1]) / 2.
    rho = (chi[0] + chi[1]) / abs(chi[0] - chi[1])
    m = -(chi[2] - chi[3]) / (chi[0] - chi[1])
    sigma = np.sqrt(4 * chi[4]) / abs(chi[0] - chi[1])
    return [a, b, rho, m, sigma]


def RND(k, m1, m2, q1, q2, c):
    """Calculate risk neutral density wrt logstrike"""
    w = straightSVI(k, m1, m2, q1, q2, c)
    wp = straightSVIp(k, m1, m2, q1, q2, c)
    wpp = straightSVIpp(k, m1, m2, q1, q2, c)
    g = (1. - k * wp / (2. * w))**2 - wp**2 / 4. * (1. / w + 1. / 4.) + wpp / 2.
    return g / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * ((-k - w / 2.)**2 / w))


def get_svi_iv(S, K, R, T, Q, IV, maxfev=1200, verbose=0):
    # Variable to store parameter vectors chi
    expirs = sorted(set(T))
    chi = pd.DataFrame(index=expirs, columns=['m1', 'm2', 'q1', 'q2', 'c'])

    # Variables required
    logK = logstrike(S, K, T, R, Q)

    # Fit implied volatilities directly to obtain first guess on parameter vectors
    if verbose > 1:
        log('Calculating first guess on parameters ...')
    if verbose > 1:
        iteration = tqdm(enumerate(expirs), total=len(expirs))
    else:
        iteration = enumerate(expirs)
    for i, t in iteration:
        if verbose > 2:
            log('Fitting implied volatility on slice ' + str(i) + ', T=' + str(T) + ' ...')
        logk, tau, iv = [v[T == t] for v in [logK, T, IV]]
        chi0 = chi_guess(t, logk, iv, maxfev=maxfev)

        chi.loc[t, :] = sco.leastsq(residSVI, chi0, args=(logk, tau, iv), maxfev=maxfev)[0]
        # chi.loc[t, :] = sco.least_squares(residSVI, x0=chi0, bounds=[lb, ub], verbose=0, xtol=1e-6, args=(logk, tau, iv))['x']

        if verbose > 2:
            log('Got parameters:', chi.loc[t, :].to_frame().T)
    if verbose > 1:
        log('Summary of initial guess for parameters:', chi)
    variables = chi.loc[T, :].assign(k=logK).astype(float)
    w = straightSVI(*variables[['k', 'm1', 'm2', 'q1', 'q2', 'c']].values.T)
    calibrated_iv = np.sqrt(w / T)
    if verbose > 0:
        log('Calibrated implied volatilities from SVI: ', calibrated_iv)
    return chi, calibrated_iv


def get_svi_iv_with_penalty(cp_flag, S, K, R, T, Q, P, chi, bpen=128, cpen=128, blim=0.001, clim=0.001, maxfev=1200, verbose=0):
    """
        bpen = 128                # initial butterfly penalty factor
        cpen = 128                # initial calendar penalty factor
        blim = 0.001              # target butterfly arbitrage bound
        clim = 0.001              # target calendar arbitrage bound
    """
    # Variable to store parameter vectors chi
    expirs = sorted(set(T))
    chi = chi.copy()
    assert expirs == chi.index.tolist()

    # Reduce arbitrage by fitting option prices with penalties on calendar and butterfly arbitrage
    maxbutarb = float("Inf")
    maxcalarb = float("Inf")
    while maxbutarb > blim or maxcalarb > clim:
        if verbose > 1:
            log('\nButterfly penalty factor: ' + str(bpen))
        if verbose > 1:
            log('Calendar penalty factor: ' + str(cpen))
        tp = 0
        maxbutarb = 0
        maxcalarb = 0
        for j, t in enumerate(expirs):
            if verbose > 2:
                log('Fitting mid prices on slice ' + str(j) + ', T=' + str(t) + ' ...')

            cp_flag_, S_, K_, R_, Q_, P_ = [v[T == t] for v in [cp_flag, S, K, R, Q, P]]
            tauT = np.full_like(K_, t)
            tauTp = np.full_like(K_, tp)

            chiT = chi.loc[t, :].tolist()
            logstrikeT = logstrike(S_, K_, tauT, R_, Q_)
            chiTp = chi.loc[tp, :].tolist() if tp else None
            logstrikeTp = logstrike(S_, K_, tauTp, R_, Q_) if tp else None
            bs_args = [cp_flag_, S_, K_, R_, tauT, logstrikeT, Q_]
            penalty = [cpen, bpen]
            chi.loc[t, :] = sco.leastsq(residuals, chiT, args=(logstrikeT, chiTp, logstrikeTp, P_, bs_args, penalty), maxfev=maxfev)[0]
            # chi.loc[t, :] = sco.least_squares(residuals, x0=chiT, bounds=[lb, ub], verbose=0, xtol=1e-6, args=(logstrikeT, chiTp, logstrikeTp, P_, bs_args, penalty))['x']
            if verbose > 2:
                log('Got parameters:', chi.loc[t, :].to_frame().T)

            butarb = butterfly(chi.loc[t, :], logstrikeT)
            if verbose > 2:
                log('Butterfly penalty for slice is ' + str(bpen * butarb))
            zero_calarb = (chiTp is None) or (logstrikeTp is None)
            calarb = calendar(chiT, logstrikeT, chiTp, logstrikeTp) if not zero_calarb else 0    # calarb
            if verbose > 2:
                log('Calendar penalty for slice is ' + str(cpen * calarb))
            maxbutarb = np.maximum(maxbutarb, butarb)
            maxcalarb = np.maximum(maxcalarb, calarb)
            tp = t
        if maxbutarb > clim:
            bpen *= 2
        if maxcalarb > clim:
            cpen *= 2

    if verbose > 0:
        log('Maximum remaining butterfly arbitrage is ' + str(maxbutarb))
    if verbose > 0:
        log('Maximum remaining calendar arbitrage is ' + str(maxcalarb))
    if verbose > 0:
        log('Summary of final parameters:', chi)

    variables = chi.loc[T, :].assign(k=logstrike(S, K, T, R, Q)).astype(float)
    w = straightSVI(*variables[['k', 'm1', 'm2', 'q1', 'q2', 'c']].values.T)
    calibrated_iv = np.sqrt(w / T)

    return chi, calibrated_iv


def calibrate_iv_with_chi(chi, n, T, logStrike):
    chi = chi.copy()
    chi.index = np.round(chi.index * 365) - n
    variables = chi.loc[np.round(T * 365), :].assign(k=logStrike).astype(float)
    w = straightSVI(*variables[['k', 'm1', 'm2', 'q1', 'q2', 'c']].values.T)
    calibrated_iv = np.sqrt(w / T)
    return calibrated_iv
