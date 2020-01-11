#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.stats as scs
import scipy.integrate as sci
import scipy.optimize as sco


def HestonPiIntegrand(u, x, tau, typo, *parms):
    """
    parms have the parameters in this order: v, vbar, lambd, eta, rho
    """
    v, vbar, lambd, eta, rho = parms

    alpha = -u ** 2 / 2 - 1j * u / 2 + 1j * typo * u
    beta = lambd - rho * eta * typo - rho * eta * 1j * u
    gamma = eta ** 2 / 2
    d = np.sqrt(beta ** 2 - 4 * alpha * gamma)
    r_plus = (beta + d) / eta ** 2
    r_minus = (beta - d) / eta ** 2
    g = r_minus / r_plus

    D = r_minus * ((1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    C = lambd * (r_minus * tau - (2 / eta ** 2) * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))

    return np.real(np.exp(C * vbar + D * v + 1j * u * x) / (1j * u))


def HestonPi(x, tau, typo, *parms):
    def integrand(u):
        return HestonPiIntegrand(u, x, tau, typo, *parms)
    return 1 / 2 + (1 / np.pi) * sci.quad(integrand, 0, 100)[0]


def HestonCall(F, K, tau, *parms):
    '''Heston call'''
    x = np.log(F / K)
    P1 = HestonPi(x, tau, 1, *parms)
    P0 = HestonPi(x, tau, 0, *parms)
    return K * (np.exp(x) * P1 - P0)


def evaluate_HestonCall(chromo, estopts, spots, strikes, taus, rfr, dr):
    """docstring for heston_evaluate"""
    diffs = [p - HestonCall(s * np.exp((r - q) * t), k, t, *chromo)
             for i, (p, s, k, t, r, q) in enumerate(zip(estopts, spots, strikes, taus, rfr, dr))]
    return np.array(diffs)


def check_real_root(lambd, vbar, eta):
    """check if it is guaranteed to have a real root in heston model"""
    return 2 * lambd * vbar / eta ** 2 > 1


def calibrate_heston_parameter(chromo, estopts, spots, strikes, taus, rfr, dr, verbose=0):
    bounds = ([0, 0, 0, 0, -1], [1, 1, 20, 5, 1])
    p1 = sco.least_squares(
        evaluate_HestonCall,
        x0=chromo,
        bounds=bounds,
        verbose=verbose,
        xtol=1e-5,
        args=(estopts, spots, strikes, taus, rfr, dr)
    )['x']
    return p1
