import numpy as np
import scipy.integrate


def _integrate_one_way(dt, start_point, rss, bTrace, rtol, atol):
    direction = np.sign(dt)
    dt = np.abs(dt)
    t = 0.0
    xout = np.atleast_2d(start_point.copy())

    def finish_integration(t, coord):
        r = np.linalg.norm(coord)
        ret = (r - 1) * (r - rss)
        return ret

    finish_integration.terminal = True
    # The integration domain is deliberately huge, because the
    # the interation automatically stops when an out of bounds error
    # is thrown
    t_span = (0, 1e4)

    def fun(t, y):
        return bTrace(t, y, direction)

    res = scipy.integrate.solve_ivp(
        fun, t_span, start_point, method='LSODA',
        rtol=rtol, atol=atol, events=finish_integration)

    xout = res.y
    return xout


def _integrate_seeds(seeds, rss=None, bTrace=None,
                     rtol=None, atol=None):
    out = []
    for start_point in seeds:
        args = (start_point, rss, bTrace, rtol, atol, )
        xforw = _integrate_one_way(1, *args)
        xback = _integrate_one_way(-1, *args)
        xback = np.flip(xback, axis=1)
        xout = np.row_stack((xback.T, xforw.T))
        out.append(xout)
    return out
