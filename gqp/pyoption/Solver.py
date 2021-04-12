import numpy as np
# import pandas as pd
from scipy.optimize import linprog


def solve_weight1(_ivsp, _price, _delta, _gamma, _theta, _rho, _vega, full=False):
    """
    Maxmize sum of iv spread
        Subject to
            - other greeks < threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """
    # Objective
    c = -np.abs(_ivsp).copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta)])
    b_eq = np.array([1.0])
    A_ub = np.array([_delta, _gamma, _theta, _rho, -_delta, -_gamma, -_theta, -_rho])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    b_ub = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    # b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    # bnd = (0, 1)
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def solve_weight2(_ivsp, _price, _delta, _gamma, _theta, _rho, _vega, full=False):
    """
    Maxmize sum of vega
        Subject to
            - other greeks <= threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """

    # Objective
    c = _vega.copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta)])
    b_eq = np.array([1.0])
    A_ub = np.array([_delta, _gamma, _theta, _rho, -_delta, -_gamma, -_theta, -_rho])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    b_ub = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    # b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    # bnd = (0, 1)
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def solve_weight3(_ivsp, _price, _delta, _gamma, _theta, _rho, _vega, full=False):
    """ Self Financing
    Maxmize sum of vega
        Subject to
            - sum of cashflow = 0
            - other greeks < threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """
    # Objective
    c = _vega.copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta), _price])
    b_eq = np.array([1.0, 0.0])
    A_ub = np.array([_delta, _gamma, _theta, _rho, -_delta, -_gamma, -_theta, -_rho])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    b_ub = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    # b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def solve_weight4(_ivsp, _price, _delta, _gamma, _theta, _rho, _vega, full=False):
    """
    Maxmize sum of vega
        Subject to
            - sum of cashflow = 0
            - other greeks < threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """
    # Objective
    c = -_vega.copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta)])
    b_eq = np.array([1.0])
    A_ub = np.array([_delta, _gamma, -_delta, -_gamma])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01])
    # b_ub = np.array([0.001, 0.001, 0.001, 0.001])
    b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def solve_weight5(_ivsp, _price, _delta, _gamma, _theta, _rho, _vega, full=False):
    """
    Maxmize sum of iv spread
        Subject to
            - other greeks < threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """
    # Objective
    c = -np.abs(_ivsp).copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta)])
    b_eq = np.array([1.0])
    A_ub = np.array([_delta, _gamma, -_delta, -_gamma])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01])
    # b_ub = np.array([0.001, 0.001, 0.001, 0.001])
    b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    # bnd = (0, 1)
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def solve_weight6(_ivsp, _weight, _delta, _gamma, _theta, _rho, _vega, full=False):
    """
    Maxmize sum of iv spread
        Subject to
            - other greeks < threshold (approximately zero)
        Parameter bounds:
            - (0, 1) for iv_spread > 0;
            - (-1, 0) for iv_spread < 0;
    """
    # Objective
    c = -np.abs(_weight).copy()

    # Constrains
    A_eq = np.array([np.ones_like(_delta)])
    b_eq = np.array([1.0])
    A_ub = np.array([_delta, _gamma, -_delta, -_gamma])
    # b_ub = np.array([0.01, 0.01, 0.01, 0.01])
    # b_ub = np.array([0.001, 0.001, 0.001, 0.001])
    b_ub = np.array([0.0001, 0.0001, 0.0001, 0.0001])

    # Parameter bounds
    # bnd = (0, 1)
    bnd = [np.sort([0, s]) for s in np.sign(_ivsp)]

    # Optimize
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bnd)
    if not full:
        if res.status == 0:
            out = res.x
        else:
            out = None
    else:
        out = res
    return out


def optimize_weight(prigap, delta, gamma, constraint=3, category='Continuous'):
    # category: Continuous or Integer
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum
    prob = LpProblem("Optimize Weight", LpMaximize)

    # weight_type = 'Integer'
    codes = list(prigap.keys())
    weight_var = {c: LpVariable('w_' + c, *np.sort([0, np.sign(prigap[c])]), cat=category) for c in codes}

    prob += lpSum([prigap[i] * weight_var[i] for i in codes]), 'Objective'
    prob += lpSum([1 * weight_var[i] for i in codes]) == 1, 'sum of weights'

    if constraint > 2:
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) <= 0.0001, 'sum of upper delta'
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) >= -0.0001, 'sum of lower delta'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) <= 0.0001, 'sum of upper gamma'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) >= -0.0001, 'sum of lower gamma'
    elif constraint > 1:
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) <= 0.0001, 'sum of upper delta'
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) >= -0.0001, 'sum of lower delta'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) <= 0.01, 'sum of upper gamma'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) >= -0.01, 'sum of lower gamma'
    elif constraint > 0:
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) <= 0.01, 'sum of upper delta'
        prob += lpSum([delta[i] * weight_var[i] for i in codes]) >= -0.01, 'sum of lower delta'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) <= 0.01, 'sum of upper gamma'
        prob += lpSum([gamma[i] * weight_var[i] for i in codes]) >= -0.01, 'sum of lower gamma'

    status = prob.solve()
    if status == 1:
        optim_weights = {v.name.split('_')[1]: v.varValue for v in prob.variables()}
    else:
        optim_weights = {v.name.split('_')[1]: np.nan for v in prob.variables()}
    return optim_weights


# def optimize_allocation(prigap, delta, gamma, theta, _size, constraint=3, init_capital=1e6, category='Continuous'):
#     # category: Continuous or Integer
#     from pulp import LpProblem, LpMaximize, LpVariable, lpSum
#     prob = LpProblem("Optimize Weight", LpMaximize)

#     codes = list(prigap.keys())
#     size_var = {c: LpVariable('pos_' + c, *(0.5 * init_capital * np.sort([0, np.sign(prigap[c])])), cat=category) for c in codes}
#     prob += lpSum([prigap[c] * size_var[c] - 0.00021 if size_var[c] != _size[c] else prigap[c] * size_var[c] for c in codes]), 'Objective'
#     prob += lpSum([size_var[c] for c in codes]) <= init_capital, 'sum of allocation'

#     if constraint > 2:
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper delta'
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower delta'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper gamma'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower gamma'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper theta'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower theta'
#     elif constraint > 1:
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper delta'
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower delta'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper gamma'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower gamma'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper theta'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower theta'
#     elif constraint > 0:
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper delta'
#         prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower delta'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper gamma'
#         prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower gamma'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper theta'
#         prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower theta'

#     status = prob.solve()
#     if status == 1:
#         optim_weights = {v.name.split('_')[1]: v.varValue for v in prob.variables()}
#     else:
#         optim_weights = {v.name.split('_')[1]: np.nan for v in prob.variables()}

#     return optim_weights


# def optimize_allocation(prigap, _size, init_capital=1e6, constraint=3, category='Continuous', verbose=0, delta=None, gamma=None, theta=None, vega=None, vanna=None, volga=None):
#     """Summary

#     Args:
#         prigap (TYPE): Description
#         _size (TYPE): Description
#         delta (None, optional): Description
#         gamma (None, optional): Description
#         theta (None, optional): Description
#         vega (None, optional): Description
#         vanna (None, optional): Description
#         volga (None, optional): Description
#         constraint (int, optional): Description
#         init_capital (float, optional): Description
#         category (str, optional): Continuous or Integer
#         verbose (int, optional): Description

#     Returns:
#         TYPE: Description
#     """
#     from pulp import LpProblem, LpMaximize, LpVariable, lpSum, solvers
#     prob = LpProblem("Optimize Weight", LpMaximize)

#     codes = list(prigap.keys())
#     size_var = {c: LpVariable('pos_' + c, *(0.5 * init_capital *
#                                             np.sort([0, np.sign(prigap[c])])), cat=category) for c in codes}
#     prob += lpSum([prigap[c] * size_var[c] - 0.00021 if size_var[c] != _size[c]
#                    else prigap[c] * size_var[c] for c in codes]), 'Objective'
#     prob += lpSum([size_var[c] for c in codes]) <= init_capital, 'sum of allocation'

#     if constraint > 2:
#         if delta is not None:
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper delta'
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower delta'
#         if gamma is not None:
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper gamma'
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower gamma'
#         if theta is not None:
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper theta'
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower theta'
#         if vega is not None:
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper vega'
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower vega'
#         if vanna is not None:
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper vanna'
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower vanna'
#         if volga is not None:
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) <= 0.0001, 'sum of upper volga'
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) >= -0.0001, 'sum of lower volga'

#     elif constraint > 1:
#         if delta is not None:
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper delta'
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower delta'
#         if gamma is not None:
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper gamma'
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower gamma'
#         if theta is not None:
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper theta'
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower theta'
#         if vega is not None:
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper vega'
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower vega'
#         if vanna is not None:
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper vanna'
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower vanna'
#         if volga is not None:
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) <= 0.001, 'sum of upper volga'
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) >= -0.001, 'sum of lower volga'
#     elif constraint > 0:
#         if delta is not None:
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper delta'
#             prob += lpSum([delta[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower delta'
#         if gamma is not None:
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper gamma'
#             prob += lpSum([gamma[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower gamma'
#         if theta is not None:
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper theta'
#             prob += lpSum([theta[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower theta'
#         if vega is not None:
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper vega'
#             prob += lpSum([vega[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower vega'
#         if vanna is not None:
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper vanna'
#             prob += lpSum([vanna[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower vanna'
#         if volga is not None:
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) <= 0.01, 'sum of upper volga'
#             prob += lpSum([volga[i] * size_var[i] for i in codes]) >= -0.01, 'sum of lower volga'

#     status = prob.solve()
#     # status = prob.solve(solvers.PULP_CBC_CMD(maxSeconds=120))
#     if status == 1:
#         optim_weights = {v.name.split('_')[1]: v.varValue for v in prob.variables()}
#     else:
#         optim_weights = {v.name.split('_')[1]: np.nan for v in prob.variables()}

#     if verbose > 3:
#         print('status: ', status, 'constraint: ', constraint)

#     return optim_weights


# def optimize_allocation(
#         prigap, _size, multiple, max_total_size=10, transaction_cost=0.00021, constraint=3, category='Continuous', verbose=0,
#         delta=None, gamma=None, theta=None, vega=None, vanna=None, volga=None,
# ):
#     """Summary

#     Args:
#         prigap (TYPE): target for objective
#         _size (TYPE): current holding position
#         delta (None, optional: delta in dollar amount
#         gamma (None, optional): gamma in dollar amount
#         theta (None, optional): theta in dollar amount
#         vega (None, optional): vega in dollar amount
#         vanna (None, optional): vanna in dollar amount
#         volga (None, optional): volga in dollar amount
#         constraint (int, optional): level of constraint
#         max_total_size (float, optional): max total size
#         category (str, optional): Continuous or Integer
#         verbose (int, optional): print information

#     Returns:
#         TYPE: Description
#     """
#     from pulp import LpProblem, LpMaximize, LpVariable, lpSum
#     # from pulp import solvers
#     prob = LpProblem("Optimize Weight", LpMaximize)

#     codes = list(prigap.keys())
#     size_var = {c: LpVariable('pos_' + c, *(max_total_size * np.sort([0, np.sign(prigap[c])])), cat=category) for c in codes}
#     prob += lpSum([(prigap[c] - transaction_cost) * size_var[c] if size_var[c] != 0 else prigap[c] * size_var[c] for c in codes]), 'Objective'
#     prob += lpSum([size_var[c] for c in codes]) <= max_total_size, 'sum of allocation'

#     if delta is not None:
#         total_delta = lpSum([delta[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
#     if gamma is not None:
#         total_gamma = lpSum([gamma[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
#     if theta is not None:
#         total_theta = lpSum([theta[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
#     if vega is not None:
#         total_vega = lpSum([vega[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
#     if vanna is not None:
#         total_vanna = lpSum([vanna[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
#     if volga is not None:
#         total_volga = lpSum([volga[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])

#     if constraint > 2:
#         if delta is not None:
#             prob += total_delta <= 0.0001, 'sum of upper delta'
#             prob += total_delta >= -0.0001, 'sum of lower delta'
#         if gamma is not None:
#             prob += total_gamma <= 0.0001, 'sum of upper gamma'
#             prob += total_gamma >= -0.0001, 'sum of lower gamma'
#         if theta is not None:
#             prob += total_theta <= 0.0001, 'sum of upper theta'
#             prob += total_theta >= -0.0001, 'sum of lower theta'
#         if vega is not None:
#             prob += total_vega <= 0.0001, 'sum of upper vega'
#             prob += total_vega >= -0.0001, 'sum of lower vega'
#         if vanna is not None:
#             prob += total_vanna <= 0.0001, 'sum of upper vanna'
#             prob += total_vanna >= -0.0001, 'sum of lower vanna'
#         if volga is not None:
#             prob += total_volga <= 0.0001, 'sum of upper volga'
#             prob += total_volga >= -0.0001, 'sum of lower volga'
#     elif constraint > 1:
#         if delta is not None:
#             prob += total_delta <= 0.001, 'sum of upper delta'
#             prob += total_delta >= -0.001, 'sum of lower delta'
#         if gamma is not None:
#             prob += total_gamma <= 0.001, 'sum of upper gamma'
#             prob += total_gamma >= -0.001, 'sum of lower gamma'
#         if theta is not None:
#             prob += total_theta <= 0.001, 'sum of upper theta'
#             prob += total_theta >= -0.001, 'sum of lower theta'
#         if vega is not None:
#             prob += total_vega <= 0.001, 'sum of upper vega'
#             prob += total_vega >= -0.001, 'sum of lower vega'
#         if vanna is not None:
#             prob += total_vanna <= 0.001, 'sum of upper vanna'
#             prob += total_vanna >= -0.001, 'sum of lower vanna'
#         if volga is not None:
#             prob += total_volga <= 0.001, 'sum of upper volga'
#             prob += total_volga >= -0.001, 'sum of lower volga'
#     elif constraint > 0:
#         if delta is not None:
#             prob += total_delta <= 0.01, 'sum of upper delta'
#             prob += total_delta >= -0.01, 'sum of lower delta'
#         if gamma is not None:
#             prob += total_gamma <= 0.01, 'sum of upper gamma'
#             prob += total_gamma >= -0.01, 'sum of lower gamma'
#         if theta is not None:
#             prob += total_theta <= 0.01, 'sum of upper theta'
#             prob += total_theta >= -0.01, 'sum of lower theta'
#         if vega is not None:
#             prob += total_vega <= 0.01, 'sum of upper vega'
#             prob += total_vega >= -0.01, 'sum of lower vega'
#         if vanna is not None:
#             prob += total_vanna <= 0.01, 'sum of upper vanna'
#             prob += total_vanna >= -0.01, 'sum of lower vanna'
#         if volga is not None:
#             prob += total_volga <= 0.01, 'sum of upper volga'
#             prob += total_volga >= -0.01, 'sum of lower volga'

#     status = prob.solve()
#     # status = prob.solve(solvers.PULP_CBC_CMD(maxSeconds=120))
#     if status == 1:
#         optim_weights = {v.name.split('_')[1]: v.varValue for v in prob.variables()}
#     else:
#         optim_weights = {v.name.split('_')[1]: np.nan for v in prob.variables()}

#     if verbose > 3:
#         print('status: ', status, 'constraint: ', constraint)

#     return optim_weights, status


def optimize_allocation(
        prigap, _size, multiple, max_total_size=10, transaction_cost=0.00021, category='Continuous', verbose=0,
        greek_bounds=None, delta=None, gamma=None, theta=None, vega=None, vanna=None, volga=None, more_data=False
):
    """Summary

    Args:
        prigap (TYPE): target for objective
        _size (TYPE): current holding position
        delta (None, optional: delta in dollar amount
        gamma (None, optional): gamma in dollar amount
        theta (None, optional): theta in dollar amount
        vega (None, optional): vega in dollar amount
        vanna (None, optional): vanna in dollar amount
        volga (None, optional): volga in dollar amount
        constraint (int, optional): level of constraint
        max_total_size (float, optional): max total size
        category (str, optional): Continuous or Integer
        verbose (int, optional): print information

    Returns:
        TYPE: Description
    """
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum
    # from pulp import solvers
    prob = LpProblem("Optimize_Weight", LpMaximize)

    codes = list(prigap.keys())
    size_var = {c: LpVariable('pos_' + c, *(1e10 * np.sort([0, np.sign(prigap[c])])), cat=category) for c in codes}
    # size_var = {c: LpVariable('pos_' + c, -max_total_size, max_total_size, cat=category) for c in codes}

    # # ---------------------------------------- Objective ---------------------------------------- #
    # # prob += lpSum([prigap[c] * size_var[c] - transaction_cost * abs(size_var[c]) if size_var[c] != 0 else prigap[c] * size_var[c] for c in codes]), 'Objective'
    # # prob += lpSum([(prigap[c] - transaction_cost) * size_var[c] if prigap[c] >= 1e-10 else (prigap[c] + transaction_cost) * size_var[c] if prigap[c] <= -1e-10 else prigap[c] * size_var[c] for c in codes]), 'Objective'
    # expected_profit = []
    # for c in codes:
    #     if prigap[c] >= 1e-10:
    #         _expp = (prigap[c] - transaction_cost) * size_var[c]
    #     elif prigap[c] <= -1e-10:
    #         _expp = (prigap[c] + transaction_cost) * size_var[c]
    #     else:
    #         _expp = 0 * size_var[c]
    #     expected_profit.append(_expp)

    # prob += lpSum(expected_profit), 'Objective'

    # # ---------------------------------------- Constraints ---------------------------------------- #
    # # Constraints: open position
    # open_position_var = []
    # for c in codes:
    #     if _size[c] > 0:
    #         if prigap[c] > 0:
    #             open_position_var.append(_size[c] + size_var[c])
    #         else:
    #             open_position_var.append(_size[c] - size_var[c])
    #     else:
    #         if prigap[c] > 0:
    #             open_position_var.append(-_size[c] + size_var[c])
    #         else:
    #             open_position_var.append(-_size[c] - size_var[c])

    # prob += lpSum(open_position_var) <= max_total_size, 'sum of open position'

    # ---------------------------------------- Objective ---------------------------------------- #
    # ---------------------------------------- Constraints ---------------------------------------- #
    # Constraints: open position
    expected_profit = []
    open_position_var = []
    for c in codes:
        if _size[c] > 0:
            if prigap[c] > 0:
                size = _size[c] + size_var[c]
                expected_profit.append(prigap[c] * size - transaction_cost * size_var[c])
                open_position_var.append(size)
            else:
                size = _size[c] - size_var[c]
                expected_profit.append(-prigap[c] * size + transaction_cost * size_var[c])
                open_position_var.append(size)
        else:
            if prigap[c] > 0:
                size = -_size[c] + size_var[c]
                expected_profit.append(prigap[c] * size - transaction_cost * size_var[c])
                open_position_var.append(size)
            else:
                size = -_size[c] - size_var[c]
                expected_profit.append(-prigap[c] * size + transaction_cost * size_var[c])
                open_position_var.append(size)

    prob += lpSum(expected_profit), 'Objective'
    prob += lpSum(open_position_var) <= max_total_size, 'sum of open position'

    # Constraints: greeks
    if delta is not None:
        total_delta = lpSum([delta[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_delta <= greek_bounds['delta'][1], 'sum of upper delta'
        prob += total_delta >= greek_bounds['delta'][0], 'sum of lower delta'
    if gamma is not None:
        total_gamma = lpSum([gamma[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_gamma <= greek_bounds['gamma'][1], 'sum of upper gamma'
        prob += total_gamma >= greek_bounds['gamma'][0], 'sum of lower gamma'
    if vega is not None:
        total_vega = lpSum([vega[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_vega <= greek_bounds['vega'][1], 'sum of upper vega'
        prob += total_vega >= greek_bounds['vega'][0], 'sum of lower vega'
    if theta is not None:
        total_theta = lpSum([theta[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_theta <= greek_bounds['theta'][1], 'sum of upper theta'
        prob += total_theta >= greek_bounds['theta'][0], 'sum of lower theta'
    if vanna is not None:
        total_vanna = lpSum([vanna[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_vanna <= greek_bounds['vanna'][1], 'sum of upper vanna'
        prob += total_vanna >= greek_bounds['vanna'][0], 'sum of lower vanna'
    if volga is not None:
        total_volga = lpSum([volga[c] * (size_var[c] + _size[c]) / multiple[c] for c in codes])
        prob += total_volga <= greek_bounds['volga'][1], 'sum of upper volga'
        prob += total_volga >= greek_bounds['volga'][0], 'sum of lower volga'

    # ---------------------------------------- Solver ---------------------------------------- #
    status = prob.solve()
    # status = prob.solve(solvers.PULP_CBC_CMD(maxSeconds=120))
    if status == 1:
        optim_weights = {v.name.split('_')[1]: v.varValue for v in prob.variables()}
    else:
        optim_weights = {v.name.split('_')[1]: np.nan for v in prob.variables()}
        # optim_weights = {v.name.split('_')[1]: -_size[v.name.split('_')[1]] for v in prob.variables()}

    # if verbose > 3:
    #     print('status: ', status, 'constraint: ', constraint)

    if more_data:
        return optim_weights, status, prob
    else:
        return optim_weights, status


def cvx_allocation(base_df, greek_df, max_total_size=10, transaction_cost=0, greek_bounds=None, verbose=0, solver='CVXOPT', **kwargs):
    import cvxpy as cp
    _size, expret = base_df.copy().T.values
    size_adj = cp.Variable(len(expret))
    size = _size + size_adj
    # objective = cp.Maximize(size @ expret - total_tc)
    objective = cp.Maximize(cp.sum(size * expret))

    conditions = [cp.sum(cp.abs(size)) <= max_total_size]
    conditions.extend([size[i] >= 0 if pg > 0 else size[i] <= 0 for i, pg in enumerate(expret)])
    if 'delta' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['delta'].values)) <= greek_bounds['delta'])
    if 'gamma' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['gamma'].values)) <= greek_bounds['gamma'])
    if 'vega' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['vega'].values)) <= greek_bounds['vega'])
    if 'theta' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['theta'].values)) <= greek_bounds['theta'])
    if 'vanna' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['vanna'].values)) <= greek_bounds['vanna'])
    if 'volga' in greek_df:
        conditions.append(cp.sum(cp.abs(size * greek_df['volga'].values)) <= greek_bounds['volga'])

    prob = cp.Problem(objective, conditions)
    if verbose > 0:
        verbose_solve = True
    else:
        verbose_solve = False

    # https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
    if solver not in dir(cp):
        raise ValueError('installed_solvers: ', cp.installed_solvers())

    try:
        prob.solve(solver=eval('cp.' + solver), verbose=verbose_solve, **kwargs)
        optimal_vars = size_adj.value
    except cp.SolverError as e:
        if verbose > 0:
            print(e)
        optimal_vars = np.full_like(expret, np.nan)

    return dict(zip(base_df.index, optimal_vars)), prob.status


def linprog_allocation(base_df, greek_df, max_total_size=10, transaction_cost=2.1, greek_bounds={}, verbose=0, **kwargs):
    # -------------------------------- Initialize -------------------------------- #
    import scipy.optimize as sco
    _size, expret, penalty = base_df.values.T
    greek_arr = greek_df.values
    tc = transaction_cost

    grkbound = [greek_bounds[k] for k in greek_df.columns if k in greek_bounds]
    ngreeks = len(grkbound)
    ulen = base_df.shape[0]
    greek_arrT = greek_arr.T

    # ---------------------------------- Prepare A ub ---------------------------------- #
    # greek
    greek_ext = np.vstack([
        -np.hstack([greek_arrT, greek_arrT, greek_arrT, greek_arrT]),
        np.hstack([greek_arrT, greek_arrT, greek_arrT, greek_arrT])
    ])
    # total position for all contracts
    ones_vec_1x = np.ones((1, ulen))    # maximum total position in vector by row
    totpos_ext = np.vstack([
        np.hstack([ones_vec_1x, ones_vec_1x, -ones_vec_1x, -ones_vec_1x]),
    ])
    # variables bounds
    ones_mat_1x = np.diag(np.ones((ulen)))
    zeros_mat_1x = np.diag(np.zeros((ulen)))
    variables_ext = np.vstack([
        np.hstack([-ones_mat_1x, zeros_mat_1x, zeros_mat_1x, zeros_mat_1x]),
        np.hstack([zeros_mat_1x, ones_mat_1x, zeros_mat_1x, zeros_mat_1x]),
        np.hstack([zeros_mat_1x, -ones_mat_1x, zeros_mat_1x, zeros_mat_1x]),
        np.hstack([zeros_mat_1x, zeros_mat_1x, -ones_mat_1x, zeros_mat_1x]),
        np.hstack([zeros_mat_1x, zeros_mat_1x, ones_mat_1x, zeros_mat_1x]),
        np.hstack([zeros_mat_1x, zeros_mat_1x, zeros_mat_1x, ones_mat_1x]),
    ])

    # ---------------------------------- Prepare b ub ---------------------------------- #
    # greek
    grk_chg_bound = np.ravel(np.array(grkbound).T - np.dot(_size, greek_arr))
    grk_chg_bound[:ngreeks] = -grk_chg_bound[:ngreeks]
    bnd_greek_chg = grk_chg_bound[:, None]
    # totpos bounds
    totpos_chg_bound = max_total_size - np.sum(np.abs(_size))
    bnd_totpos_chg = np.array([[totpos_chg_bound]])
    # variables bounds
    zeros_vec_1x = np.zeros(ulen)
    bnd_variables_chg = np.hstack([
        np.hstack([zeros_vec_1x, zeros_vec_1x, np.maximum(_size, 0)]),
        np.hstack([zeros_vec_1x, -np.minimum(_size, 0), zeros_vec_1x]),
    ])[:, None]

    # ------------------------ Make A ub matrix and b ub vector ------------------------ #
    # A ub matrix
    A_ub_mat = np.vstack([greek_ext, totpos_ext, variables_ext])

    # b ub vector
    b_ub_vec = np.vstack([bnd_greek_chg, bnd_totpos_chg, bnd_variables_chg])

    # ------------------------------ Make objective c vec ------------------------------ #
    expret = np.hstack([expret - (tc + penalty), expret + (tc + penalty),
                        expret - (tc + penalty), expret + (tc + penalty)])
    objective_vars = -expret
    # objective_vars = np.hstack([tc - expret, - tc - expret, tc - expret, - tc - expret])
    if verbose > 1:
        print('A_ub_mat: ', A_ub_mat.dtype, A_ub_mat.shape)
        print('b_ub_vec: ', b_ub_vec.dtype, b_ub_vec.shape)
        print('Objective: ', objective_vars.dtype, objective_vars.shape)

    # --------------------------- Solve the program ----------------------------------- #
    res = sco.linprog(c=objective_vars, A_ub=A_ub_mat, b_ub=b_ub_vec, bounds=(None, None), **kwargs)
    if verbose > 0:
        print('Iter[{0}], {1}[{2}]: {3}'.format(res.nit, res.success, res.status, res.message))

    optimal_values = -res.fun
    status = res.status

    if status == 0:
        size_chg_var = np.reshape(res.x, (4, ulen)).T
    else:
        size_chg_var = np.full((ulen, 4), np.nan)

    size_dict = dict(zip(base_df.index, _size + size_chg_var.sum(1)))
    return size_dict, status, [size_chg_var, optimal_values, res]
