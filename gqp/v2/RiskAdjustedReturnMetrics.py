# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy.stats as scs

"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.

Reference:
Main: http://www.turingfinance.com/computational-investing-with-python-week-one/
Supplement: https://www.portfolioeffect.com/docs/glossary/measures/relative-return-measures/up-capture-ratio

"""


def vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return np.cov(m)[0][1] / np.std(market)


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)


def var(returns, alpha):
    """Value at Risk
    This method calculates the historical simulation var of the returns

    Args:
        returns (TYPE): Description
        alpha (TYPE): Description

    Returns:
        TYPE: Description
    """
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def modified_var(returns, alpha):
    # cornish fisher expansion
    _, _, mu, variance, S, K = scs.describe(returns)
    z = scs.norm.ppf(1 - alpha)
    z_cf_alpha = z + (z ** 2 - 1) * S / 6 + (z ** 3 - 3 * z) * (K - 3) / 24 - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    return mu - z_cf_alpha * np.sqrt(variance)


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = np.sum(sorted_returns[:index])
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)


def prices(returns, base):
    # Converts returns into prices
    return np.hstack((100, base * np.cumproduct(1 + returns)))


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = sorted((dd(returns, i) for i in range(len(returns))))
    total_dd = np.sum(np.abs(drawdowns[:periods + 1]))
    return total_dd / periods


def average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = sorted((math.pow(dd(returns, i), 2.0) for i in range(len(returns))))
    total_dd = np.sum(np.abs(drawdowns[:periods + 1]))
    return total_dd / periods


def excess_alpha(er, rf, market):
    return er - rf - np.mean(market - rf)


def jensen_alpha(er, returns, market, rf):
    return (er - rf) - beta(returns, market) * np.mean(market - rf)


def treynor_ratio(er, returns, market, rf):
    """Treynor ratio

    Args:
        er (TYPE): expected return
        returns (TYPE): realized return
        market (TYPE): market reurn
        rf (TYPE): risk-free rate

    Returns:
        TYPE: Description
    """
    return (er - rf) / beta(returns, market)


def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)


def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.mean(diff) / vol(diff)


def modigliani_ratio(er, returns, benchmark, rf):
    """Summary
    the Modigliani ratio a.k.a the M2 ratio, is a combination the Sharpe and
    information ratio in that it adjusts the expected excess returns of the
    portfolio above the risk free rate by the expected excess returns of a
    benchmark portfolio, E(rb), or the market E(rM), above the risk
    free rate,

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        benchmark (TYPE): Description
        rf (TYPE): Description

    Returns:
        TYPE: Description
    """
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    return (er - rf) / var(returns, alpha)


def modified_excess_var(er, returns, rf, alpha):
    """
    a.k.a modified sharpe ratio. Modified Sharpe Ratio is a performance
    measure which amends traditional Sharpe Ratio to account for
    non-Gaussianity of return distribution.

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        rf (TYPE): Description
        alpha (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (er - rf) / modified_var(returns, alpha)


def conditional_sharpe_ratio(er, returns, rf, alpha):
    """Summary
    a.k.a STARR ratio (Stable Tail Adjusted Return Ratio)

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        rf (TYPE): Description
        alpha (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (er - rf) / cvar(returns, alpha)


def rachev_ratio(returns, rf, alpha_a, alpha_b):
    """Summary
    Rachev Ratio Reward-to-risk measure and defined as the ratio between the
    CVaR of the opposite of the excess return at a given confidence level 1−α
    and the CVaR of the excess return at another confidence level 1−β.

    Args:
        returns (TYPE): Description
        rf (TYPE): Description
        alpha_a (TYPE): Description
        alpha_b (TYPE): Description

    Returns:
        TYPE: Description
    """
    return cvar(rf - returns, alpha_b) / cvar(returns - rf, alpha_a)


def omega_ratio(er, returns, rf, target=0):
    return (er - rf) / lpm(returns, target, 1)


def sortino_ratio(er, returns, rf, target=0):
    return (er - rf) / math.sqrt(lpm(returns, target, 2))


def kappa_three_ratio(er, returns, rf, target=0):
    return (er - rf) / math.pow(lpm(returns, target, 3), float(1 / 3))


def gain_loss_ratio(returns, target=0):
    return hpm(returns, target, 1) / lpm(returns, target, 1)


def upside_potential_ratio(returns, target=0):
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))


def calmar_ratio(er, returns, rf):
    """
    The Calmar ratio discounts the expected excess return of a portfolio by
    the worst expected maximum draw down for that portfolio

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        rf (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (er - rf) / max_dd(returns)


def sterling_ration(er, returns, rf, periods):
    """Summary
    The Sterling ratio discounts the expected excess return of a portfolio by
    the average of the NN worst expected maximum drawdowns for that portfolio

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        rf (TYPE): Description
        periods (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (er - rf) / average_dd(returns, periods)


def burke_ratio(er, returns, rf, periods):
    """Summary
    The Burke ratio is similar to the Sterling ratio except that it is less
    sensitive to outliers. It discounts the expected excess return of a
    portfolio by the square root of the average of the NN worst expected
    maximum drawdowns squared for that portfolio

    Args:
        er (TYPE): Description
        returns (TYPE): Description
        rf (TYPE): Description
        periods (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (er - rf) / math.sqrt(average_dd_squared(returns, periods))


def up_capture_ratio(returns, market, target=0):
    up_loc = market > target
    return np.mean(returns[up_loc]) / np.mean(market[up_loc])


def down_capture_ratio(returns, market, target=0):
    down_loc = market < target
    return np.mean(returns[down_loc]) / np.mean(market[down_loc])


def up_number_ratio(returns, market, target=0):
    m_loc = market > target
    r_loc = returns > target
    return float(sum(m_loc & r_loc)) / sum(m_loc)


def down_number_ratio(returns, market, target=0):
    m_loc = market < target
    r_loc = returns < target
    return float(sum(m_loc & r_loc)) / sum(m_loc)


def up_percentage_ratio(returns, market, target=0):
    m_loc = market > target
    excess_r_loc = returns[m_loc] > market[m_loc]
    return float(sum(excess_r_loc)) / sum(m_loc)


def down_percentage_ratio(returns, market, target=0):
    m_loc = market < target
    worse_r_loc = returns[m_loc] < market[m_loc]
    return float(sum(worse_r_loc)) / sum(m_loc)


def test_risk_metrics(r, m):
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_risk_adjusted_metrics(r, m, e, f):
    # Risk-adjusted return based on Volatility
    print("Excess Alpha =", excess_alpha(e, r, m))
    print("Jensen Alpha =", jensen_alpha(e, r, m, f))
    print("Treynor Ratio =", treynor_ratio(e, r, m, f))
    print("Sharpe Ratio =", sharpe_ratio(e, r, f))
    print("Information Ratio =", information_ratio(r, m))
    # Risk-adjusted return based on Value at Risk
    print("Excess VaR =", excess_var(e, r, f, 0.05))
    print("Modified Sharpe Ratio =", modified_excess_var(e, r, f, 0.05))
    print("Conditional Sharpe Ratio =", conditional_sharpe_ratio(e, r, f, 0.05))
    print("Rachev Ratio Reward-to-risk =", rachev_ratio(r, f, 0.05, 0.20))
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r, f))
    print("Sortino Ratio =", sortino_ratio(e, r, f))
    print("Kappa 3 Ratio =", kappa_three_ratio(e, r, f))
    print("Gain Loss Ratio =", gain_loss_ratio(r))
    print("Upside Potential Ratio =", upside_potential_ratio(r))
    # Risk-adjusted return based on Drawdown risk
    print("Calmar Ratio =", calmar_ratio(e, r, f))
    print("Sterling Ratio =", sterling_ration(e, r, f, 5))
    print("Burke Ratio =", burke_ratio(e, r, f, 5))
    # Relative Return Measures
    print("Up Capture Ratio =", up_capture_ratio(r, m, 0))
    print("Down Capture Ratio =", down_capture_ratio(r, m, 0))
    print("Up Number Ratio =", up_number_ratio(r, m, 0))
    print("Down Number Ratio =", down_number_ratio(r, m, 0))
    print("Up Percentage Ratio =", up_percentage_ratio(r, m, 0))
    print("Down Percentage Ratio =", down_percentage_ratio(r, m, 0))


if __name__ == "__main__":
    # Returns from the portfolio (r) and market (m)
    r = np.random.uniform(-1, 1, 50)
    m = np.random.uniform(-1, 1, 50)
    # Expected return
    e = np.mean(r)
    # Risk free rate
    f = 0.06
    # Alpha
    alpha_a = 0.05
    alpha_b = 0.20  # prepared for rachev_ratio
    # Periods
    peri = 5  # prepared for sterling ratio and burke ratio

    test_risk_metrics(r, m)
    test_risk_adjusted_metrics(r, m, e, f)
