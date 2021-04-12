import math
import numpy as np
from numba import jit
from numba.typed import List

@jit(nopython=True)
def anynan(array):
    for a in array:
        if math.isnan(a):
            return True
    return False


@jit(nopython=True)
def nnan(array):
    i = 0
    for a in array:
        if math.isnan(a):
            i += 1
    return i


@jit(nopython=True)
def check_equal_length(args):
    m = len(args[0])
    for n in List(args):
        if len(n) != m:
            return False
    return True

# @jit(nopython=True)
# def check_equal_length(*args):
#     m = len(args[0])
#     # print(len(args), numba.typeof(args))
#     for i in range(len(args)):
#         print(args[i])
#         # print(len(np.array(args[i])))
#         pass
#         # if len(n) != m:
#         #     return False
#     return True

@jit(nopython=True)
def N(x):
    """ N helper function
    Cumulative distribution function for the standard normal distribution

    Args:
        x (float): Description

    Returns:
        TYPE: Description
    """

    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@jit(nopython=True)
def phi(x):
    """ Phi helper function
    Density distribution function for the standard normal distribution

    Args:
        x (float): Description

    Returns:
        TYPE: Description
    """
    return math.exp(-0.5 * x * x) / (math.sqrt(2.0 * math.pi))


# @jit('f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])', nopython=True)
@jit(nopython=True)
def _black_scholes_call_value(S, K, r, t, v, q):
    if anynan(List([S, K, r, t, v, q])):
        return np.nan

    if v == 0:
        c = S * math.exp(-q * t) - K * math.exp(- r * t)
    else:
        sqrtT = math.sqrt(t)
        d1 = (math.log(S / K) + (r - q + 0.5 * v**2) * t) / (v * sqrtT)
        d2 = d1 - v * sqrtT
        c = S * math.exp(-q * t) * N(d1) - K * math.exp(- r * t) * N(d2)
    return max(0, c)


# @jit('f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])', nopython=True)
@jit(nopython=True)
def _black_scholes_put_value(S, K, r, t, v, q):
    if anynan(List([S, K, r, t, v, q])):
        return np.nan

    if v == 0:
        p = K * math.exp(- r * t) - S * math.exp(-q * t)
    else:
        sqrtT = math.sqrt(t)
        d1 = (math.log(S / K) + (r - q + 0.5 * v**2) * t) / (v * sqrtT)
        d2 = d1 - v * sqrtT
        p = K * math.exp(- r * t) * N(-d2) - S * math.exp(-q * t) * N(-d1)

    return max(0, p)


@jit(nopython=True)
def black_scholes_call_value(S, K, R, T, V, Q):
    assert check_equal_length(List([S, K, R, T, V, Q])), "Key values have different lengths."
    callResult = List()
    for args in zip(S, K, R, T, V, Q):
        callResult.append(_black_scholes_call_value(*args))
    return callResult

# @jit(nopython=True)
# def black_scholes_call_value(S, K, R, T, V, Q):
#     if check_equal_length(S, K, R, T, V, Q):
#         pass
#     else:
#         raise ValueError("Key values have different lengths.")
#     callResult = []
#     for args in zip(S, K, R, T, V, Q):
#         callResult.append(_black_scholes_call_value(*args))
#     return callResult


@jit(nopython=True)
def black_scholes_put_value(S, K, R, T, V, Q):
    assert check_equal_length(List([S, K, R, T, V, Q])), "Key values have different lengths."
    putResult = List()
    for args in zip(S, K, R, T, V, Q):
        putResult.append(_black_scholes_put_value(*args))
    return putResult


@jit(nopython=True)
def get_option_value(cp_flag, S, K, R, T, V, Q):
    assert check_equal_length(List([cp_flag, S, K, R, T, V, Q])), "Key values have different lengths."
    option_values = List()
    for args in zip(cp_flag, S, K, R, T, V, Q):
        if args[0] == 1:
            option_values.append(_black_scholes_call_value(*args[1:]))
        elif args[0] == 2:
            option_values.append(_black_scholes_put_value(*args[1:]))
    return option_values


@jit(nopython=True)
def call_implied_volatility(S, K, r, t, q, mktprice, upper=5.0, lower=0.000001, tol=1e-10, maxiter=np.inf):
    """Black-scholes call implied volatility

    Args:
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        q (float): dividend for underlying
        mktprice (float): option price from market
        upper (float, optional): upper bound for implied volatility
        lower (float, optional): lower bound for implied volatility
        tol (float, optional): error tolerance
        maxiter (int, optional): max iteration

    Returns:
        float: call implied volatility
    """
    # init
    if anynan(List([S, K, r, t, q, mktprice])):
        return np.nan

    if abs(mktprice - _black_scholes_call_value(S, K, r, t, 0, q)) < tol:
        return lower

    C = 0
    sigma = 0.3
    iternum = 0

    # 先判断波动率是否在给定初值区间内能够算出
    C_lower = _black_scholes_call_value(S, K, r, t, lower, q)
    C_upper = _black_scholes_call_value(S, K, r, t, upper, q)

    if (C_upper - mktprice) * (C_lower - mktprice) < 0:    # 在这个区间中有解
        while abs(C - mktprice) > tol:
            # print('iter:', iternum, 'price:', C, 'sigma: ', sigma, 'err:', C - mktprice)
            C = _black_scholes_call_value(S, K, r, t, sigma, q)

            if abs(C - mktprice) < tol:
                break

            if C - mktprice > 0:
                upper = sigma
                sigma = (sigma + lower) / 2
            else:
                lower = sigma
                sigma = (sigma + upper) / 2

            iternum += 1
            if iternum >= maxiter:
                break

    else:
        # print 'Can not calculate implied vol!'
        sigma = np.nan

    return sigma


@jit(nopython=True)
def put_implied_volatility(S, K, r, t, q, mktprice, upper=5.0, lower=0.000001, tol=1e-10, maxiter=np.inf):
    """Black-scholes put implied volatility

    Args:
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        q (float): dividend for underlying
        mktprice (float): option price from market
        upper (float, optional): upper bound for implied volatility
        lower (float, optional): lower bound for implied volatility
        tol (float, optional): error tolerance
        maxiter (int, optional): max iteration

    Returns:
        float: put implied volatility
    """
    # init
    if anynan(List([S, K, r, t, q, mktprice])):
        return np.nan

    if abs(mktprice - _black_scholes_put_value(S, K, r, t, 0, q)) < tol:
        return lower

    P = 0
    sigma = 0.3
    iternum = 0

    # 先判断波动率是否在给定初值区间内能够算出
    P_lower = _black_scholes_put_value(S, K, r, t, lower, q)
    P_upper = _black_scholes_put_value(S, K, r, t, upper, q)
    if (P_upper - mktprice) * (P_lower - mktprice) < 0:    # 在这个区间中有解
        while abs(P - mktprice) > tol:
            # print('iter:', iternum, 'price:', P, 'sigma: ', sigma, 'err:', P - mktprice)
            P = _black_scholes_put_value(S, K, r, t, sigma, q)

            if abs(P - mktprice) < tol:
                break

            if P - mktprice > 0:
                upper = sigma
                sigma = (sigma + lower) / 2
            else:
                lower = sigma
                sigma = (sigma + upper) / 2

            iternum += 1
            if iternum >= maxiter:
                break

    else:
        # print 'Can not calculate implied vol!'
        sigma = np.nan
    return sigma


# @jit(nopython=True)
def implied_volatility_series(cp_flag, S, K, r, t, q, option_price, upper=5, lower=0.000001, tol=1e-10, maxiter=np.inf):
    """ Serialize implied volatility calculation

    Args:
        cp_flag (int): option type: 1 for call, 2 for put
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        q (float): dividend for underlying
        option_price (float): option price from market
        upper (float, optional): upper bound for implied volatility
        lower (float, optional): lower bound for implied volatility
        tol (float, optional): error tolerance
        maxiter (int, optional): max iteration

    Returns:
        array[float]: implied volatility
    """
    assert check_equal_length(List([cp_flag, S, K, r, t, q, option_price])), "Key values have different lengths."
    ivs = List()
    for cp, S0, X0, r0, t0, q0, p0 in zip(cp_flag, S, K, r, t, q, option_price):
        if anynan(List([cp, S0, X0, r0, t0, q0, p0])) or (t0 == 0):
            iv = np.nan
        elif cp == 1:
            iv = call_implied_volatility(S0, X0, r0, t0, q0, p0, upper=upper, lower=lower, tol=tol, maxiter=maxiter)
        elif cp == 2:
            iv = put_implied_volatility(S0, X0, r0, t0, q0, p0, upper=upper, lower=lower, tol=tol, maxiter=maxiter)
        else:
            iv = np.nan
        ivs.append(iv)
    return ivs


@jit(nopython=True)
def _call_delta(S, K, r, t, vol):
    """Black-Scholes call delta

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: call delta
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)

    return N(d1)


@jit(nopython=True)
def _put_delta(S, K, r, t, vol):
    """Black-Scholes put delta

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: put delta
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)

    return N(d1) - 1.0


@jit(nopython=True)
def _gamma(S, K, r, t, vol):
    """Black-Scholes gamma

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: gamma
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)

    return phi(d1) / (S * vol * math.sqrt(t))


@jit(nopython=True)
def _vega(S, K, r, t, vol):
    """Black-Scholes vega
    From <Options, Futures and Other Deriative>, vega is defined as absolute
    values instead of percentage. However, it is usually described as
    percentage by dividing 100: a 1%(0.01) increase in the implied volatility
    from increases the value of the option by approximately vega.

    Ex. if vega is 1 (unit of vol, not in percentage), the option price will increase certain unit (the result from this function).

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: vega
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)

    return (S * phi(d1) * math.sqrt(t)) / 100.0


@jit(nopython=True)
def _call_theta(S, K, r, t, vol):
    """Black-Scholes call theta

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: call theta
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))

    theta = -((S * phi(d1) * vol) / (2.0 * math.sqrt(t))) - (r * K * math.exp(-r * t) * N(d2))
    return theta / 365.0


@jit(nopython=True)
def _put_theta(S, K, r, t, vol):
    """Black-Scholes put theta

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: put theta
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))

    theta = -((S * phi(d1) * vol) / (2.0 * math.sqrt(t))) + (r * K * math.exp(-r * t) * N(-d2))
    return theta / 365.0


@jit(nopython=True)
def _call_rho(S, K, r, t, vol):
    """Black-Scholes call rho

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: call rho
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))

    rho = K * t * math.exp(-r * t) * N(d2)
    return rho / 100.0


@jit(nopython=True)
def _put_rho(S, K, r, t, vol):
    """Black-Scholes put rho

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: put rho
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))

    rho = -K * t * math.exp(-r * t) * N(-d2)
    return rho / 100.0


@jit(nopython=True)
def _volga(S, K, r, t, vol):
    """ Black-Scholes volga
    https://financetrainingcourse.com/education/2014/06/vega-volga-and-vanna-the-volatility-greeks/

    Args:
        S (float): option price
        K (float): strike price
        r (float): annual risk free rate
        t (float): number of days over 365
        vol (float): annual volatility

    Returns:
        float: volga
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))
    return (phi(d1) * math.sqrt(t) * d1 * d2 / vol) / 100.0


@jit(nopython=True)
def _vanna(S, K, r, t, vol):
    """Black-Scholes vanna
    https://financetrainingcourse.com/education/2014/06/vega-volga-and-vanna-the-volatility-greeks/

    Args:
        S (float): option price
        K (float): strike price
        r (float): annual risk free rate
        t (float): number of days over 365
        vol (float): annual volatility

    Returns:
        float: vanna
    """
    d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * math.sqrt(t))
    return (phi(d1) * math.sqrt(t) * d2 / vol) / 100.0


# @jit(nopython=True)
# def _veta(S, K, r, t, vol):
#     """Black-Scholes veta: vega w.r.t tau

#     Args:
#         S (float): underlying
#         K (float): strike price
#         r (float): rate
#         t (float): time to expiration
#         vol (float): volatility

#     Returns:
#         float: veta
#     """
#     d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)

#     return (S * phi(d1) * (1 / (2 * math.sqrt(t)) - math.log(S / K) * t / (2 * vol) + (r + 0.5 * vol ** 2) / math.sqrt(t))) / 100.0


# @jit(nopython=True)
# def _veta(S, K, r, t, vol):
#     """Black-Scholes veta: vega w.r.t tau

#     Args:
#         S (float): underlying
#         K (float): strike price
#         r (float): rate
#         t (float): time to expiration
#         vol (float): volatility

#     Returns:
#         float: veta
#     """
#     # d1 = (1.0 / (vol * math.sqrt(t))) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
#     k1 = math.log(S / K) / vol
#     k2 = (r + 0.5 * vol**2)
#     pi = math.pi
#     sqrtT = math.sqrt(t)
#     d1 = k1 / sqrtT + k2 * sqrtT

#     return 0.5 * S * (k1 * d1 * phi(d1) + t * k2 / math.sqrt(2 * pi) + sqrtT * phi(d1))


@jit(nopython=True)
def _veta(S, K, r, t, vol, q=0):
    """Black-Scholes veta: vega w.r.t tau

    Args:
        S (float): underlying
        K (float): strike price
        r (float): rate
        t (float): time to expiration
        vol (float): volatility

    Returns:
        float: veta
    """
    sqrtT = math.sqrt(t)
    d1 = (1.0 / (vol * sqrtT)) * (math.log(S / K) + (r + 0.5 * vol**2.0) * t)
    d2 = d1 - (vol * sqrtT)

    return -S * math.exp(-q * t) * phi(d1) * sqrtT * (q + (r - q) * d1 / (vol * sqrtT) - (1 + d1 * d2) / (2 * t)) / 100


@jit(nopython=True)
def call_delta(S, K, R, T, V):
    """ Serialize call delta calculation

    Returns:
        array[float]: call_delta
    """
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    call_delta_results = List()
    for args in zip(S, K, R, T, V):
        call_delta_results.append(_call_delta(*args))
    return call_delta_results


@jit(nopython=True)
def put_delta(S, K, R, T, V):
    """ Serialize call delta calculation

    Returns:
        array[float]: put_delta
    """
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    put_delta_results = List()
    for args in zip(S, K, R, T, V):
        put_delta_results.append(_put_delta(*args))
    return put_delta_results


@jit(nopython=True)
def call_theta(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    call_theta_results = List()
    for args in zip(S, K, R, T, V):
        call_theta_results.append(_call_theta(*args))
    return call_theta_results


@jit(nopython=True)
def put_theta(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    put_theta_results = List()
    for args in zip(S, K, R, T, V):
        put_theta_results.append(_put_theta(*args))
    return put_theta_results


@jit(nopython=True)
def call_rho(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    call_rho_results = List()
    for args in zip(S, K, R, T, V):
        call_rho_results.append(_call_rho(*args))
    return call_rho_results


@jit(nopython=True)
def put_rho(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    put_rho_results = List()
    for args in zip(S, K, R, T, V):
        put_rho_results.append(_put_rho(*args))
    return put_rho_results


@jit(nopython=True)
def delta(cp_flag, S, K, R, T, V):
    """ Serialize implied volatility calculation

    Args:
        cp_flag (int): option type: 1 for call, 2 for put
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        v (float): volatility
    Returns:
        array[float]: delta no matter call or put
    """
    assert check_equal_length(List([cp_flag, S, K, R, T, V])), "Key values have different lengths."
    deltas = List()
    for cp, S0, X0, r0, t0, v0 in zip(cp_flag, S, K, R, T, V):
        if anynan(List([cp, S0, X0, r0, t0, v0])) or (t0 == 0):
            delta = np.nan
        elif cp == 1:
            delta = _call_delta(S0, X0, r0, t0, v0)
        elif cp == 2:
            delta = _put_delta(S0, X0, r0, t0, v0)
        else:
            delta = np.nan
        deltas.append(delta)
    return deltas


@jit(nopython=True)
def theta(cp_flag, S, K, R, T, V):
    """ Serialize implied volatility calculation

    Args:
        cp_flag (int): option type: 1 for call, 2 for put
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        v (float): volatility
    Returns:
        array[float]: theta no matter call or put
    """
    assert check_equal_length(List([cp_flag, S, K, R, T, V])), "Key values have different lengths."
    thetas = List()
    for cp, S0, X0, r0, t0, v0 in zip(cp_flag, S, K, R, T, V):
        if anynan(List([cp, S0, X0, r0, t0, v0])) or (t0 == 0):
            theta = np.nan
        elif cp == 1:
            theta = _call_theta(S0, X0, r0, t0, v0)
        elif cp == 2:
            theta = _put_theta(S0, X0, r0, t0, v0)
        else:
            theta = np.nan
        thetas.append(theta)
    return thetas


@jit(nopython=True)
def rho(cp_flag, S, K, R, T, V):
    """ Serialize implied volatility calculation

    Args:
        cp_flag (int): option type: 1 for call, 2 for put
        S (float): option price from black-schole pricing
        K (float): strike price
        r (float): annual risk-free rate
        t (float): time to expiration
        v (float): volatility
    Returns:
        array[float]: theta no matter call or put
    """
    assert check_equal_length(List([cp_flag, S, K, R, T, V])), "Key values have different lengths."
    rhos = List()
    for cp, S0, X0, r0, t0, v0 in zip(cp_flag, S, K, R, T, V):
        if anynan(List([cp, S0, X0, r0, t0, v0])) or (t0 == 0):
            rho = np.nan
        elif cp == 1:
            rho = _call_rho(S0, X0, r0, t0, v0)
        elif cp == 2:
            rho = _put_rho(S0, X0, r0, t0, v0)
        else:
            rho = np.nan
        rhos.append(rho)
    return rhos


@jit(nopython=True)
def gamma(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    gamma_results = List()
    for args in zip(S, K, R, T, V):
        gamma_results.append(_gamma(*args))
    return gamma_results


@jit(nopython=True)
def vega(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    vega_results = List()
    for args in zip(S, K, R, T, V):
        vega_results.append(_vega(*args))
    return vega_results


@jit(nopython=True)
def volga(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    volga_results = List()
    for args in zip(S, K, R, T, V):
        volga_results.append(_volga(*args))
    return volga_results


@jit(nopython=True)
def vanna(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    vanna_results = List()
    for args in zip(S, K, R, T, V):
        vanna_results.append(_vanna(*args))
    return vanna_results


@jit(nopython=True)
def veta(S, K, R, T, V):
    assert check_equal_length(List([S, K, R, T, V])), "Key values have different lengths."
    veta_results = List()
    for args in zip(S, K, R, T, V):
        veta_results.append(_veta(*args))
    return veta_results


def max_increment(cp_flag, strike, pre_close):
    """call max increment
    rule based on sse: http://www.sse.com.cn/assortment/options/contract/c/c_20151016_3999892.shtml

    Args:
        cp_flag (TYPE): contract type, 1 for call, 2 for put
        strike (TYPE): contract strike price
        pre_close (TYPE): pre close price for underlying asset

    Returns:
        TYPE: Description

    Raises:
        ValueError: cp_flag must be either 1 (call) or 2 (put).
    """
    if np.all(cp_flag == 1):
        max_incre = np.maximum(0.005 * pre_close, 0.1 * np.minimum(2 * pre_close - strike, pre_close))
    elif np.all(cp_flag == 2):
        max_incre = np.maximum(0.005 * strike, 0.1 * np.minimum(2 * strike - pre_close, pre_close))
    else:
        raise ValueError('cp_flag must be either 1 (call) or 2 (put).')
    return max_incre


def max_decrement(pre_close):
    """
    rule based on sse: http://www.sse.com.cn/assortment/options/contract/c/c_20151016_3999892.shtml
    - rules are the same for both call and put

    Args:
        pre_close (TYPE): pre close price for underlying asset

    Returns:
        TYPE: Description
    """
    return 0.1 * pre_close


def is_trading_curb(new_price, reference_price, unit=0.0001, verbose=False):
    """
    rule based on sse: http://www.sse.com.cn/assortment/options/contract/c/c_20151016_3999892.shtml

    Args:
        new_price (TYPE): current option price from market
        reference_price (TYPE): reference price from lastest auction
        unit (float, optional): minimum price tick
        verbose (bool, optional): show condition information

    Returns:
        TYPE: Description
    """
    abs_change = abs(new_price - reference_price)
    cond1 = abs_change / reference_price >= 0.5
    cond2 = abs_change >= 5 * unit
    if verbose:
        print('abs_change / reference_price >= 0.5:', cond1)
        print('abs_change >= 5 * unit:', cond2)
    return cond1 & cond2


def check_moneyness(cp_flag, S, K):
    """Check moneyness

    Args:
        cp_flag (TYPE): call put flag
        S (TYPE): underlying price
        K (TYPE): strike price

    Returns:
        TYPE: Description

    Raises:
        ValueError: Description
    """
    is_S_gr_K = (S - K) > 0
    if S == K:
        return 'ATM'
    elif cp_flag == 1:
        if is_S_gr_K:
            return 'ITM'
        else:
            return 'OTM'
    elif cp_flag == 2:
        if is_S_gr_K:
            return 'OTM'
        else:
            return 'ITM'
    else:
        raise ValueError('Check cp_flag.')


def calc_logstrike(S, K, tau, r, q):
    F = S * np.exp((r - q) * tau)
    return np.log(K / F)


if __name__ == '__main__':
    import numba
    input = np.array([
        [1., 2.731312, 3.50, 5.00, 6.358598, 0.00, 1.507480],
        [2., 2.731312, 3.30, 0.05, 0.063585, 0.00, 0.593907],
        [1., 2.731312, 2.80, 0.05, 0.063585, 0.00, 0.059605],
        [1., 2.731312, 2.95, 0.05, 0.063585, 0.00, 0.015647],
        [1., 2.731312, 3.40, 5.00, 6.358598, 0.00, 1.974015],
    ])

    cp, S0, X0, r0, t0, q0, p0 = input[0].T
    call_implied_volatility(S0, X0, r0, t0, q0, p0)
    cp, S0, X0, r0, t0, q0, p0 = input[1].T
    put_implied_volatility(S0, X0, r0, t0, q0, p0)
    cp_flag, S, K, r, t, q, option_price = input.T
    v = implied_volatility_series(cp_flag, S, K, r, t, q, option_price)
    impvol = np.array(v)
    delta(cp_flag, S, K, r, t, impvol)
    theta(cp_flag, S, K, r, t, impvol)
    rho(cp_flag, S, K, r, t, impvol)
    gamma(S, K, r, t, impvol)
    vega(S, K, r, t, impvol)
    volga(S, K, r, t, impvol)
    vanna(S, K, r, t, impvol)
    veta(S, K, r, t, impvol)
