import scipy.optimize as sco
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import option_numba as optnb


def logstrike(S, K, tau, r, q):
    """ log-moneyness

    Args:
        S (TYPE): underlying(spot) price
        K (TYPE): strike price
        tau (TYPE): time to expire
        r (TYPE): risk free rate
        q (TYPE): dividend rate

    Returns:
        TYPE: log strike
    """
    return np.log(K / S * np.exp(-(r - q) * tau))


def svi_parameter_bounds(parameterization):
    """ pre defined bounds
        svi_parameter_bounds returns the parameter bounds of an SVI parameterization. The parameters are
        assumed to be in the following order:
        raw = [a b m rho sigma]
        natural = [delta mu rho omega zeta]
        jumpwing = [v psi p c vt]

    Args:
        parameterization (TYPE): Description

    Returns:
        TYPE: Description
    """

    large = 1e5
    if parameterization == 'raw':
        lb = [-large, 0, -large, -1, 0]
        ub = [large, large, large, 1, large]
    elif parameterization == 'natural':
        lb = [-large, -large, -1, 0, 0]
        ub = [large, large, 1, large, large]
    elif parameterization == 'jumpwing':
        lb = [0, -large, 0, 0, 0]
        ub = [large, large, large, large, large]
    else:
        raise ValueError('Unknown parameterization')
    return np.array(lb), np.array(ub)


def svi_convert_parameters(param_old, type_old, type_new, tau=None):
    """Summary
        svi_convertparameters converts the parameter set of one type of SVI
        formulation to another. The parameterizations are assumed to be:
            - raw =(a,b,m,rho, sigma)
            - natural = (delta, mu, rho, omega, zeta)
            - jumpwing = (v, psi, p, c, vt)

    Args:
        param_old (TYPE): (5x1) = original parameters
        type_old (str): formulation of original parameters (raw, natural, jumpwing)
        type_new (str): formulation of new parameters (raw, natural, jumpwings)
        tau (None, optional): Description

    Returns:
        TYPE: new parameters

    Raises:
        ValueError: Description
    """
    if type_old not in ['raw', 'natural', 'jumpwing']:
        raise ValueError('type_old has to be one of: raw, natural, jumpwing')

    if type_new not in ['raw', 'natural', 'jumpwing']:
        raise ValueError('type_new has to be one of: raw, natural, jumpwing')

    if ((type_old == 'jumpwing') or (type_new == 'jumpwing')) and tau is None:
        raise ValueError('tau is required for tailwings formulation')

    if type_old == 'raw':
        a, b, m, rho, sigma = param_old
        if type_new == 'raw':
            param_new = param_old
        elif type_new == 'natural':
            a, b, m, rho, sigma = param_old
            omega = 2 * b * sigma / np.sqrt(1 - rho ** 2)
            delta = a - omega / 2 * (1 - rho ** 2)
            mu = m + rho * sigma / np.sqrt(1 - rho ** 2)
            zeta = np.sqrt(1 - rho ** 2) / sigma
            param_new = [delta, mu, rho, omega, zeta]
        elif type_new == 'jumpwing':
            w = a + b * (-rho * m + np.sqrt(m ** 2 + sigma ** 2))
            v = w / tau
            psi = 1 / np.sqrt(w) * b / 2 * (-m / np.sqrt(m ** 2 + sigma ** 2) + rho)
            p = 1 / np.sqrt(w) * b * (1 - rho)
            c = 1 / np.sqrt(w) * b * (1 + rho)
            vt = 1 / tau * (a + b * sigma * np.sqrt(1 - rho ** 2))
            param_new = [v, psi, p, c, vt]
        else:
            param_new = []

    elif type_old == 'natural':
        if type_new == 'raw':
            delta, mu, rho, omega, zeta = param_old
            a = delta + omega / 2 * (1 - rho ** 2)
            b = omega * zeta / 2
            m = mu - rho / zeta
            sigma = np.sqrt(1 - rho ** 2) / zeta
            param_new = [a, b, m, rho, sigma]
        elif type_new == 'natural':
            param_new = param_old
        elif type_new == 'jumpwing':
            param_temp = svi_convert_parameters(param_old, 'natural', 'raw', tau)
            param_new = svi_convert_parameters(param_temp, 'raw', 'jumpwing', tau)
        else:
            param_new = []

    elif type_old == 'jumpwing':
        if type_new == 'raw':
            v, psi, p, c, vt = param_old
            w = v * tau

            b = np.sqrt(w) / 2 * (c + p)
            rho = 1 - p * np.sqrt(w) / b
            beta = rho - 2 * psi * np.sqrt(w) / b
            alpha = np.sign(beta) * np.sqrt(1 / beta ** 2 - 1)
            m = (v - vt) * tau / (b * (-rho + np.sign(alpha) * np.sqrt(1 + alpha ** 2) - alpha * np.sqrt(1 - rho ** 2)))
            if m == 0:
                sigma = (vt * tau - w) / b / (np.sqrt(1 - rho ** 2) - 1)
            else:
                sigma = alpha * m

            a = vt * tau - b * sigma * np.sqrt(1 - rho ** 2)

            if sigma < 0:
                sigma = 0

            param_new = [a, b, m, rho, sigma]

        elif type_new == 'natural':
            param_temp = svi_convert_parameters(param_old, 'jumpwing', 'raw', tau)
            param_new = svi_convert_parameters(param_temp, 'raw', 'natural', tau)
        elif type_new == 'jumpwing':
            param_new = param_old
        else:
            param_new = []

    else:
        param_new = []
    return param_new


def svi_raw(k, param):
    """Raw SVI

    Args:
        k (TYPE): log strike price (log-moneyness)
        param (TYPE): svi parameters in fix order: a, b, m, rho, sigma
            - a: level of variance
            - b: slope of wings
            - m: translates smile to right
            - rho: counter-clockwise rotation of smile
            - sigma: reduces ATM curvature of the smile
    Returns:
        TYPE: total implied variance (w)

    Note:
        volatility = np.sqrt(w / tau)
    """
    # check that input is consistent
    assert len(param) == 5, 'There have to be five parameters: a, b, m, rho, sigma'
    a, b, m, rho, sigma = param

    # calculate total variance
    totalvariance = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return np.array(totalvariance)


def validate_svi_raw(param):
    # assuming all variables are real.
    a, b, m, rho, sigma = param
    a1 = b >= 0
    a2 = np.abs(rho) < 0
    a3 = sigma > 0
    a4 = a + b * sigma * np.sqrt(1 - rho ** 2)
    return a1 and a2 and a3 and a4


def svi_natural(k, param):
    """Natural SVI

    Args:
        k (TYPE): log strike price (log-moneyness)
        param (TYPE): svi parameters in fix order: delta, mu, rho, omega, zeta
            - delta: level of variance
            - mu: slope of wings
            - rho: translates smile to right
            - omega: counter-clockwise rotation of smile
            - zeta: reduces ATM curvature of the smile
    Returns:
        TYPE: total implied variance (w)

    Note:
        volatility = np.sqrt(w / tau)
    """
    # check that input is consistent
    assert len(param) == 5, 'There have to be five parameters: delta, mu, rho, omega, zeta'
    delta, mu, rho, omega, zeta = param

    # make sure that parameter restrictions are satisfied
    # validate_svi_natural(param)

    # calculate total variance
    totalvariance = delta + omega / 2 * (1 + zeta * rho * (k - mu) +
                                         np.sqrt((zeta * (k - mu) + rho) ** 2 + (1 - rho ** 2)))
    return totalvariance


def validate_svi_natural(param):
    delta, mu, rho, omega, zeta = param
    assert omega >= 0, 'omega has to be non-negative'
    assert abs(rho) < 1, '|rho| has to be smaller than 1'
    assert zeta > 0, 'zeta has to be positive'
    assert delta + omega * (1 - rho ** 2) >= 0, 'delta + omega (1-rho**2) has to be non-negative'
    return True


def svi_jumpwing(k, param, tau):
    """Summary
    SVI - Stochastic Volatility Inspired parameterization of the implied
    volatility smile. This function implements the jump-wings formulation.

    Args:
        k (TYPE): log-moneyness at which to evaluate the total implied
        param (TYPE): Description
           - v: ATM variance
           - psi: ATM skew
           - p: slope of left/put wing
           - c: slope of right/call wing
           - vt: minimum implied variance
        tau (TYPE): time to expire

    Returns:
        TYPE: total implied variance (w)

    Note:
        volatility = np.sqrt(w / tau)
    """
    assert len(param) == 5, 'There have to be five parameters: v, psi, p, c, vt'
    v, _, p, c, vt = param

    # # make sure that parameter restrictions are satisfied
    # validate_svi_jumpwing(param)

    # convert parameters to raw formulation
    param_raw = svi_convert_parameters(param, 'jumpwing', 'raw', tau)

    # calculate total variance
    totalvariance = svi_raw(k, param_raw)

    return totalvariance


def validate_svi_jumpwing(param):
    v, _, p, c, vt = param
    assert v >= 0, 'v has to be non-negative'
    assert p >= 0, 'p has to be non-negative'
    assert c >= 0, 'c has to be non-negative'
    assert vt >= 0, 'vt has to be non-negative'
    return True


def heston_like(theta, param):
    # Heston-like parameterization
    if isinstance(param, (float, int)):
        lambd = param
    else:
        lambd = param[0]
    value = 1 / (lambd * theta) * (1 - (1 - np.exp(-lambd * theta)) / (lambd * theta))
    return value


def power_law(theta, param):
    # Power-law parameterization
    eta, gamma = param
    value = eta / (theta ** (gamma) * (1 + theta) ** (1 - gamma))
    return value


def svi_surface(k, theta, rho, phifun, phi_param, tau):
    """Summary
    svi_surface calcualtes the surface SVI free of statis arbitrage.

    Args:
        k (TYPE): Description
        theta (TYPE): ATM variance time at which to evaluate the surface. If
            theta and k have the same dimensions, then the surface is evaluated
            for each given (k,theta) pair. If the dimensions are different, the
            function evaluates the surface on the grid given by k and theta
        rho (TYPE): rho has to be a scalar
        phifun (TYPE): Description
        phi_param (TYPE): Description
        tau (TYPE): Description

    Returns:
        TYPE: total implied variance

    Raises:
        ValueError: Description
    """
    # evaluate phi
    if phifun == 'heston_like':
        phi = heston_like(theta, phi_param)
    elif phifun == 'power_law':
        phi = power_law(theta, phi_param)
    else:
        raise ValueError('Incorrect function for phi')

    if isinstance(theta, (float, int)):
        totalvariance = theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))
    elif len(k) == len(theta):
        totalvariance = theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))
    else:
        T = len(theta)
        K = len(k)
        totalvariance = np.zeros((K, T))
        for t in np.arange(T):
            totalvariance[:, t] = theta[t] / 2 * \
                (1 + rho * phi[t] * k + np.sqrt((phi[t] * k + rho) ** 2 + (1 - rho ** 2)))

    return totalvariance


# def alpha_func(alpha, tau, eta_alpha, p):
#     # the infinitesimal & function ATM bid-ask position size adjustment
#     T0 = np.min(alpha.index)
#     alpha0 = alpha.loc[T0]
#     alphaT = alpha.loc[tau]
#     eta_alpha_T = eta_alpha.loc[tau]
#     return alpha0 + (alphaT - alpha0) * (1 - np.exp(-eta_alpha_T * p))


# def psi_func(psi, tau, eta_psi, p):
#     # the infinitesimal & function wing curvature bid-ask position size adjustment
#     T0 = np.min(psi.index)
#     psi0 = psi.loc[T0]
#     eta_psi_T = eta_psi.loc[tau]
#     return psi0 + (1 - psi0) * (1 - np.exp(-eta_psi_T * p))


# def svi_ivp(k, a, b, rho, m, sigma, beta, alpha, psi, eta_alpha, eta_phi, p=1, mu=1, eta=4):
#     """Summary

#     Args:
#         k (TYPE): Description
#         a (TYPE): the vertical displacement of the smile
#         b (TYPE): the angle between left and right asymptotes
#         rho (TYPE): the orientation of the graph
#         m (TYPE): the horizontal displacement of the smile
#         sigma (TYPE): the smoothness of the vertex
#         beta (TYPE): the downside transform for making the wings sub-linear
#         alpha (TYPE): the infinitesimal & function ATM bid-ask position size adjustment
#         psi (TYPE): the infinitesimal & function wing curvature bid-ask position size adjustment
#         eta_alpha (TYPE) and eta_phi (TYPE): the ATM & wing curvature bid-ask market impact elasticity (or liquidity horizon) respectively
#         p (int, optional): the position size
#         mu (int, optional) and eta (int, optional): the change of variable from strike space to modified strike space

#     Returns:
#         TYPE: Description
#     """

#     return


def ivp_bid_ask_adjustment_alpha(a, alpha, eta_alpha, p):
    """Summary

    Inputs must be satisfying the following:
        - 1-d array
        - have same length
        - ascendingly sorted by tau

    Args:
        a (TYPE): the parameter a from raw SVI
        alpha (TYPE): ATM spread, alpha series fro the bid-ask adjustment
        eta_alpha (TYPE): the liquidity elasticity of the ATM
        p (TYPE): Description

    Returns:
        TYPE: Description
    """

    return alpha + (a - alpha) * (1 - np.exp(-eta_alpha * p))


def ivp_bid_ask_adjustment_psi(psi, eta_psi, p):
    """Summary

    Inputs must be satisfying the following:
        - 1-d array
        - have same length
        - ascendingly sorted by tau

    Args:
        psi (TYPE): ATM spread, the symmetry of this bid-ask adjustment
        eta_psi (TYPE): the liquidity elasticity of the wings
        p (TYPE): Description

    Returns:
        TYPE: Description
    """

    return psi + (1 - psi) * (1 - np.exp(-eta_psi * p))


def ivp_mid(k, param):
    """Summary

    Inputs must be satisfying the following:
        - 1-d array
        - have same length
        - ascendingly sorted by tau

    Args:
        k (TYPE): Description
        a (TYPE): the vertical displacement of the smile
        b (TYPE): the angle between left and right asymptotes
        rho (TYPE): the orientation of the graph
        m (TYPE): the horizontal displacement of the smile
        sigma (TYPE): the smoothness of the vertex
        beta (TYPE): the downside transform for making the wings sub-linear
        mu (int, optional): default as 1, the change of variable from strike space to modified strike space
        eta (TYPE): default as 4

    Returns:
        TYPE: Description
    """
    a, b, rho, m, sigma, beta, mu, eta = param
    z = k / beta ** (mu + eta * np.abs(k - m))
    w = a + b * (rho * (z - m) + np.sqrt((z - m) ** 2 + sigma ** 2))
    return w


def ivp_ask(k, param, alpha, eta_alpha, psi, eta_psi, p):
    """Summary

    Inputs must be satisfying the following:
        - 1-d array
        - have same length
        - ascendingly sorted by tau

    Args:
        k (TYPE): Description
        a (TYPE): the vertical displacement of the smile
        b (TYPE): the angle between left and right asymptotes
        rho (TYPE): the orientation of the graph
        m (TYPE): the horizontal displacement of the smile
        sigma (TYPE): the smoothness of the vertex
        beta (TYPE): the downside transform for making the wings sub-linear
        mu (int, optional): default as 1, the change of variable from strike space to modified strike space
        eta (TYPE): default as 4
        alpha (TYPE): the infinitesimal & function ATM bid-ask position size adjustment
        eta_alpha (TYPE) and eta_phi (TYPE): the ATM & wing curvature bid-ask market impact elasticity (or liquidity horizon) respectively
        psi (TYPE): the infinitesimal & function wing curvature bid-ask position size adjustment
        eta_psi (TYPE): Description
        p (int, optional): the position size

    Returns:
        TYPE: Description
    """
    a, b, rho, m, sigma, beta, mu, eta = param
    z0 = k / beta ** (mu + eta * np.abs(k - m))
    psi = ivp_bid_ask_adjustment_psi(psi, eta_psi, p)
    z = z0 * (1 + psi)
    alpha = ivp_bid_ask_adjustment_alpha(a, alpha, eta_alpha, p)
    w = a + b * (rho * (z - m) + np.sqrt((z - m) ** 2 + sigma ** 2)) + alpha

    # assert np.all(alpha > 0), 'alpha must be greater than 1'
    # assert np.all((psi < 1) & (psi > 0)), 'psi must be within (0, 1)'
    return w


def ivp_bid(k, param, alpha, eta_alpha, psi, eta_psi, p):
    """Summary

    Inputs must be satisfying the following:
        - 1-d array
        - have same length
        - ascendingly sorted by tau

    Args:
        k (TYPE): Description
        a (TYPE): the vertical displacement of the smile
        b (TYPE): the angle between left and right asymptotes
        rho (TYPE): the orientation of the graph
        m (TYPE): the horizontal displacement of the smile
        sigma (TYPE): the smoothness of the vertex
        beta (TYPE): the downside transform for making the wings sub-linear
        mu (int, optional): default as 1, the change of variable from strike space to modified strike space
        eta (TYPE): default as 4
        alpha (TYPE): the infinitesimal & function ATM bid-ask position size adjustment
        eta_alpha (TYPE) and eta_phi (TYPE): the ATM & wing curvature bid-ask market impact elasticity (or liquidity horizon) respectively
        psi (TYPE): the infinitesimal & function wing curvature bid-ask position size adjustment
        eta_psi (TYPE): Description
        p (int, optional): the position size

    Returns:
        TYPE: Description
    """
    a, b, rho, m, sigma, beta, mu, eta = param
    z0 = k / beta ** (mu + eta * np.abs(k - m))
    psi = ivp_bid_ask_adjustment_psi(psi, eta_psi, p)
    z = z0 * (1 - psi)
    alpha = ivp_bid_ask_adjustment_alpha(a, alpha, eta_alpha, p)
    w = a + b * (rho * (z - m) + np.sqrt((z - m) ** 2 + sigma ** 2)) - alpha

    # assert np.all(alpha > 0), 'alpha must be greater than 1'
    # assert np.all((psi < 1) & (psi > 0)), 'psi must be within (0, 1)'
    return w


def ivp_constraints(k, b, rho, m, sigma, beta, tau):
    # value should be less or equal than 4
    z = (k - m) / beta ** np.abs(k - m)
    part1 = (1 + np.abs(k - m) * np.log(beta)) / beta ** np.abs(k - m)
    part2 = rho + z / np.sqrt((z - m) ** 2 + sigma ** 2)
    value = np.abs(tau * part1 * b * part2)
    return value


def butterfly(P_minus, P, P_plus):
    return P_minus - 2 * P + P_plus


def calendar(P_minus, P_plus):
    return P_plus - P_minus


def svi_raw_calibrate(tmpdf, param_init=None):

    def resid(param, logstrike, actual, tau):
        w = svi_raw(logstrike, param)
        pred = np.sqrt(w / tau)
        return np.sum((actual - pred)**2)

    tmpdf = tmpdf.copy()
    taus = np.sort(tmpdf['tau'].unique())
    if param_init is None:
        param_init = [0, 0, 0, 0, 0]
    param_calib = {}
    for t in taus:
        idx_kgrp = tmpdf['tau'] == t
        logstrikes, actual = tmpdf.loc[idx_kgrp, ['logstrike', 'bsiv']].values.T
        # optim_res = sco.minimize(resid, param_init, args=(logstrikes, actual,t), method='Nelder-Mead', tol=1e-10)
        optim_res = sco.minimize(resid, param_init, args=(logstrikes, actual, t), method='L-BFGS-B', tol=1e-10)
        param_calib[t] = optim_res.x
        w_svi_raw = svi_raw(logstrikes, param_calib[t])
        tmpdf.loc[idx_kgrp, 'calibiv'] = np.sqrt(w_svi_raw / t)
    return tmpdf, param_calib


def svi_raw_calibrate_epoch(tmpdf, epoch=10):

    def resid(param, logstrike, actual, tau):
        w = svi_raw(logstrike, param)
        pred = np.sqrt(w / tau)
        return np.sum((actual - pred)**2)

    tmpdf = tmpdf.copy()
    taus = np.sort(tmpdf['tau'].unique())
    param_calib = {}
    for t in taus:
        idx_kgrp = tmpdf['tau'] == t
        logstrikes, actual = tmpdf.loc[idx_kgrp, ['logstrike', 'bsiv']].values.T

        res = np.empty((epoch, 6))
        for i in range(epoch):
            rand = np.random.randn(5)
            param_init = [max(0, rand[0]) * 1e-2, rand[1] * 1e-2, (rand[2] - 0.5)
                          * 2e-2, (rand[3] - 0.5) * 2, rand[4] * 1e-2]
            optim_res = sco.minimize(resid, param_init, args=(logstrikes, actual, t), method='Nelder-Mead', tol=1e-10)
            # optim_res = sco.minimize(resid, param_init, args=(logstrikes, actual,t), method='L-BFGS-B', tol=1e-10)
            res[i, 0] = optim_res.fun
            res[i, 1:] = optim_res.x

        param_calib[t] = res[np.nanargmin(res[:, 0]), 1:]
        w_svi_raw = svi_raw(logstrikes, param_calib[t])
        tmpdf.loc[idx_kgrp, 'calibiv'] = np.sqrt(w_svi_raw / t)
    return tmpdf, param_calib


def svi_raw_predict(tmpdf, params):
    tmpdf = tmpdf.copy()
    taus = np.sort(tmpdf['tau'].unique())
    for t in taus:
        idx_kgrp = tmpdf['tau'] == t
        logstrikes, actual = tmpdf.loc[idx_kgrp, ['logstrike', 'bsiv']].values.T
        w_svi_raw = svi_raw(logstrikes, params[t])
        tmpdf.loc[idx_kgrp, 'calibiv'] = np.sqrt(w_svi_raw / t)
    return tmpdf


def resid(param, logstrike, actual, tau):
    w = svi_raw(logstrike, param)
    # idx_valid = w >= 0
    # pred = np.sqrt(w[idx_valid] / tau[idx_valid])
    # diff = actual[idx_valid] - pred
    actual_w = actual ** 2 * tau
    diff = actual_w - w
    return np.sum(diff**2)


def svi_raw_fit_epoch(logstrike, iv, tau, epoch=20, min_valid=10, max_iter=100, tol=1e-06, verbose=False):
    is_valid_params = False
    n = 0
    all_valid_res = []
    np.random.seed(64)
    while not is_valid_params:
        tmpresarr = np.empty((epoch, 7))
        trainxarr = np.array(logstrike)
        trainyarr = np.array(iv)
        tauarr = np.array(tau)
        for i in range(epoch):
            rand = np.random.rand(5)
            param_init = [max(0, rand[0]) * 1e-2, rand[1] * 1e-2, (rand[2] - 0.5) * 2e-2, (rand[3] - 0.5) * 2, rand[4] * 1e-2]
            optim_res = sco.minimize(
                resid,
                param_init,
                args=(trainxarr, trainyarr, tauarr),
                method='L-BFGS-B',
                bounds=[(-np.inf, np.inf), (0, np.inf), (-np.inf, np.inf), (-1, 1), (0, np.inf)],
                tol=tol,
            )

            tmpresarr[i, 0] = optim_res.fun
            tmpresarr[i, 1] = optim_res.status
            tmpresarr[i, 2:] = optim_res.x

        a, b, m, rho, sigma = tmpresarr[:, 2:].T
        tmpidx = (tmpresarr[:, 0] != 0) & (tmpresarr[:, 1] == 0) & (a + b * sigma * np.sqrt(1 - rho ** 2) >= 0)
        valid_res = tmpresarr[tmpidx]
        if valid_res.shape[0] > 0:
            all_valid_res.append(valid_res)

        n += 1
        if verbose:
            print('iteration: ', n, '\n', valid_res)

        if (len(all_valid_res) > min_valid) or (n >= max_iter):
            is_valid_params = True

    if len(all_valid_res) > 0:
        allresarr = np.vstack(all_valid_res)
        best_param = allresarr[np.nanargmin(allresarr[:, 0]), 2:]
    else:
        best_param = np.full(5, np.nan)
    return best_param
