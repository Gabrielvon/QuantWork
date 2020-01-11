import numpy as np
import scipy.stats as scs


def update_std(sumsq_x, sum_x, cnt):
    """online update standard deviation

    Args:
        sumsq_x (float): cummulative sum squre of x
        sum_x (float): cummulative sum of x
        cnt (float): cummulativef number of x, i.e. count

    Returns:
        float: global standard deviation
    """
    var = (sumsq_x - sum_x**2 / cnt) / cnt
    return np.sqrt(var)


def revert_compressed_val(comp_val):
    """Expand compressed value

    Expand compressed value, which originally from normal distribution.

    Args:
        comp_val (tuple or list): mean, std and count of compressed group

    Returns:
        TYPE: Description
    """
    def __revert_compressed_val(comp_val, seed=False):
        # comp_arr = np.array(comp_val, dtype=[('mu', float), ('std', float), ('cnt', int)])
        for mu, sd, ct in comp_val:
            if ct > 1:
                if seed:
                    np.random.seed(seed)
                yield scs.norm.rvs(loc=mu, scale=sd, size=int(ct))
            elif ct == 1:
                yield mu
    return np.hstack(__revert_compressed_val(comp_val))


def update_maxminscale(stats_on_target, lastest_minmax):
    """
    online maxminscale normalization

    Args:
        stats_on_target (TYPE): recorded stats in the past which is going to be updated. i.e. [sum of square, sum, count] of last
        lastest_minmax (TYPE): current status i.e. current [min, max]

    Returns:
        tuple: sumsq, sum and count
    """
    target_xss, target_xs, target_xct = stats_on_target
    xmn, xmx = lastest_minmax

    zss = (target_xss - 2 * xmn * target_xs + target_xct * xmn**2) / (xmx - xmn)**2
    zs = (target_xs - target_xct * xmn) / (xmx - xmn)
    zct = target_xct

    return zss, zs, zct


def update_zscore(stats_on_target, lastest_mustd):
    """online zscore computation

    Args:
        stats_on_target (TYPE): recorded stats in the past which is going to be updated. i.e. [sum of square, sum, count] of last
        lastest_mustd (TYPE): current status i.e. current [mean, std]

    Returns:
        tuple: sumsq, sum and count
    """
    target_xss, target_xs, target_xct = stats_on_target
    xmu, xstd = lastest_mustd

    zss = (target_xss - 2 * target_xs * xmu + target_xct * xmu ** 2) / xstd ** 2
    zs = (target_xs - target_xct * xmu) / xstd
    zct = target_xct

    return zss, zs, zct
