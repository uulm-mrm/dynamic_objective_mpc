import itertools
import math
import time
from contextlib import contextmanager
from functools import wraps
from typing import Iterable, Union

import numpy as np
from numba import jit
from numpy.typing import NDArray


@contextmanager
def timeblock(name: str):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print("\nTime block:", name + " " + str(t1 - t0))


def set_and_measure(timestamp: float) -> tuple[float, float]:
    new_timestamp: float = time.perf_counter()
    delta_time: float = new_timestamp - timestamp
    return delta_time, new_timestamp


def timeme(original_function=None, N: int = 1, logger=None, t_min: float = 500e-7):
    def _decorate(function):
        @wraps(function)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = None
            for _ in range(N):
                result = function(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = (end_time - start_time) / N

            if total_time < t_min:
                return result

            if logger is not None:
                logger.info(f'Function {function.__name__}: {total_time:.6f} s ({N} runs)')
            else:
                print(f'Function {function.__name__}: {total_time:.6f} s ({N} runs)')
            return result

        return timeit_wrapper

    if original_function is not None:
        return _decorate(original_function)

    return _decorate


@jit(nopython=True, cache=True, nogil=True)
def get_frenet_differences(x, y, phi, xr, yr, phir) -> tuple[float, float, float]:
    """
    Return differences in Frenet coordinates
    :param x:
    :param y:
    :param phi:
    :param xr:
    :param yr:
    :param phir:
    :return:
    """
    e_lag = (x - xr) * math.cos(phir) + (y - yr) * math.sin(phir)
    e_cont = -(x - xr) * math.sin(phir) + (y - yr) * math.cos(phir)
    e_phi = angle_diff(phi, phir)
    return e_lag, e_cont, e_phi

@jit(nopython=True, cache=True, nogil=True)
def interpolate(a, b, t):
    return (1 - t) * a + t * b

@jit(nopython=True, cache=True, nogil=True)
def interpolate_at_array(float_idx: float, values):
    lower_idx: int = int(np.floor(float_idx))
    upper_idx: int = int(np.ceil(float_idx))

    lower_idx = max(0, lower_idx)
    upper_idx = min(len(values) - 1, upper_idx)

    a = values[lower_idx]
    b = values[upper_idx]
    t = float_idx - lower_idx
    return interpolate(a, b, t)


@jit(nopython=True, cache=True, nogil=True)
def get_ellipsis_coords(x_obs, y_obs, phi_obs, l: float, w: float, eps: float, disk_r: float, n_se: int):
    t = np.linspace(0, 2 * math.pi, 50)
    a = (l / 2 + eps + disk_r)
    b = (w / 2 + eps + disk_r)
    c, s = np.cos(t), np.sin(t)
    x = a * np.abs(c) ** (2 / n_se) * np.sign(c)
    y = b * np.abs(s) ** (2 / n_se) * np.sign(s)

    coords = np.vstack((x, y))
    origin = np.array([x_obs, y_obs])[:, np.newaxis]
    P = get_rot_matrix(phi_obs)
    return P @ coords + origin


@jit(nopython=True, cache=True, nogil=True)
def get_rot_matrix(phi: float) -> NDArray:
    return np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


@jit(nopython=True, cache=True, nogil=True)
def get_rect_vertices(x0: float, y0: float, width: float, l: float, phi: float) -> NDArray:
    """
    Around geometric center
    :param x0:
    :param y0:
    :param width:
    :param l:
    :param phi:
    :return:
    """
    origin = np.array([x0, y0])[:, np.newaxis]
    l_h: float = l / 2
    w_h: float = width / 2
    coords = np.array(
        [[- l_h, l_h, l_h, - l_h, - l_h],
         [- w_h, - w_h, w_h, w_h, - w_h]])
    P = get_rot_matrix(phi)

    return (P @ coords) + origin


@jit(nopython=True, cache=True, nogil=True)
def get_wheel_vertices(x: float, y: float, yaw: float, delta: float, wb: float, track_width: float):
    wheel_bounds = np.asarray([
        (-0.4, -0.1, 1),
        (0.4, -0.1, 1),
        (0.4, 0.1, 1),
        (-0.4, 0.1, 1),
        (-0.4, -0.1, 1)
    ])

    trans = np.asarray([
        [np.cos(yaw), -
        np.sin(yaw), x],
        [np.sin(yaw), np.cos(
            yaw), y],
        [0.0, 0.0, 1.0]
    ])

    rl_wheel_trans = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, track_width / 2.0],
        [0.0, 0.0, 1.0]
    ])

    rr_wheel_trans = rl_wheel_trans.copy()
    rr_wheel_trans[1, 2] -= track_width
    fl_wheel_trans = np.asarray([
        [np.cos(delta), -np.sin(delta), wb],
        [np.sin(delta), np.cos(delta), track_width / 2.0],
        [0.0, 0.0, 1.0]
    ])
    fr_wheel_trans = fl_wheel_trans.copy()
    fr_wheel_trans[1, 2] -= track_width

    fl_wheel_trans = trans @ fl_wheel_trans
    fr_wheel_trans = trans @ fr_wheel_trans
    rl_wheel_trans = trans @ rl_wheel_trans
    rr_wheel_trans = trans @ rr_wheel_trans

    rl_wheel_bounds = wheel_bounds @ rl_wheel_trans.T
    rr_wheel_bounds = wheel_bounds @ rr_wheel_trans.T
    fl_wheel_bounds = wheel_bounds @ fl_wheel_trans.T
    fr_wheel_bounds = wheel_bounds @ fr_wheel_trans.T

    return rl_wheel_bounds, rr_wheel_bounds, fl_wheel_bounds, fr_wheel_bounds

@jit(nopython=True, cache=True, nogil=True)
def get_ego_vertices(x0: float, y0: float, width: float, lf: float, lb: float, phi: float) -> NDArray:
    """
    Plotted with correct rear axis
    :param x0:
    :param y0:
    :param width:
    :param lf:
    :param lb:
    :param phi:
    :return:
    """
    origin = np.array([x0, y0])[:, np.newaxis]
    coords = np.array(
        [[lf, lf, -lb, -lb, lf],
         [width / 2, -width / 2, -width / 2, width / 2, width / 2]
         ])
    P = get_rot_matrix(phi)
    return (P @ coords) + origin


@jit(nopython=True, cache=True, nogil=True)
def get_ref_s_idx(s0: float, ref: NDArray, res: float) -> int:
    max_index: int = len(ref) - 1
    return int(min(max(s0 / res, 0), max_index))


@jit(nopython=True, cache=True, nogil=True)
def v_integrate(v_lim, res: float, a_max: float, a_min: float, factor: float) -> NDArray:
    a_eff = max(a_max, abs(a_min)) * factor

    # integrate forward and backward
    v_forward = v_lim[0] * np.ones_like(v_lim)
    v_backward = v_lim[-1] * np.ones_like(v_lim)

    vi: float = v_forward[0]  # start with current
    for i in range(len(v_lim)):
        v_forward[i] = vi
        vi = math.sqrt(vi * vi + 2 * res * a_eff)
        vi = min(vi, v_lim[i])

    start_idx: int = len(v_lim) - 1
    vi: float = v_backward[-1]
    for i in range(start_idx, -1, -1):
        v_backward[i] = vi
        vi = math.sqrt(vi * vi + 2 * res * a_eff)
        vi = min(vi, v_lim[i])

    return np.minimum(v_forward, v_backward)


@jit(nopython=True, cache=True, nogil=True)
def n_integrate(n_max, kappa, dn: float, is_pos: bool) -> NDArray:
    # integrate forward and backward
    n_forward = np.zeros_like(n_max)
    n_backward = np.zeros_like(n_max)

    ni: float = n_max[0]
    for i in range(len(n_max)):
        n_forward[i] = ni
        # dn_eff: float = dn
        dn_eff: float = dn * (1 - kappa[i] * ni)
        # print("dn -> dn_eff", dn, "->", dn_eff)
        # print("at n =", ni, "with kappa =", kappa[i])
        if is_pos:
            ni = min(ni + dn_eff, n_max[i])
        else:
            ni = max(ni - dn_eff, n_max[i])

    ni: float = n_max[-1]
    for i in range(len(n_max) - 1, -1, -1):
        n_backward[i] = ni
        # dn_eff: float = dn
        dn_eff: float = dn * (1 - kappa[i] * ni)
        if is_pos:
            ni = min(ni + dn_eff, n_max[i])
        else:
            ni = max(ni - dn_eff, n_max[i])

    if is_pos:
        return np.minimum(n_forward, n_backward)
    else:
        return np.maximum(n_forward, n_backward)


@jit(nopython=True, cache=True, nogil=True)
def get_vlim(kappa_ref: list, v_lim: NDArray, alat_max: float) -> NDArray:
    """
    add curvature dependant limit
    :param alat_max:
    :param v_lim:
    :param kappa_ref:
    :return:
    """
    for i, kappa in enumerate(kappa_ref):
        kappa_abs: float = np.abs(kappa)
        if kappa_abs > 0.0:
            v_lim[i] = np.minimum(np.sqrt(alat_max / kappa_abs), v_lim[i])

    return v_lim


def pairwise(iterable: Iterable):
    """
    s -> (s0, s1), (s1, s2), (s2, s3), ...
    Args:
        iterable:

    Returns:

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


@jit(nopython=True, cache=True, nogil=True)
def clip_angle_minpi_pluspi(angle: float) -> float:
    while angle < 0:
        angle += 2 * np.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


@jit(nopython=True, cache=True, nogil=True)
def angle_diff(a1: float, a2: float) -> float:
    """
    :param a1:
    :param a2:
    :return:
    """

    # ensure -pi to +pi
    a2: float = clip_angle_minpi_pluspi(a2)
    a1: float = clip_angle_minpi_pluspi(a1)

    diff: float = a2 - a1

    return clip_angle_minpi_pluspi(diff)

def get_angle_diffs(yaws: list) -> list:
    return [angle_diff(yaw1, yaw2) for (yaw1, yaw2) in pairwise(yaws)]


def get_curvatures(yaws: Union[NDArray, list], ds: Union[NDArray, float]) -> Union[NDArray, list]:
    if isinstance(ds, float):
        return [angle_diff(yaw1, yaw2) / ds for (yaw1, yaw2) in pairwise(yaws)]
    return np.array([angle_diff(yaw1, yaw2) for (yaw1, yaw2) in pairwise(yaws)]) / np.asarray(ds)


def calc_sref_kapparef(x: Union[NDArray, list], y: Union[NDArray, list], yaw: Union[NDArray, list]) \
        -> tuple[NDArray, NDArray]:
    if len(x) > 1:
        delta_s = np.linalg.norm([np.diff(x), np.diff(y)], axis=0)
        kappa_ref = get_curvatures(yaw, delta_s)
        s_ref = np.cumsum(delta_s)
        kappa_ref = np.hstack([kappa_ref[0], kappa_ref])
        s_ref = np.hstack([0, s_ref])
    else:
        kappa_ref = np.array([0])
        s_ref = np.array([0])
    return s_ref, kappa_ref


def test():
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4, 5]
    yaw = [np.deg2rad(45) for _ in x]

    x_arr = np.array(x)
    y_arr = np.array(y)
    yaw_arr = np.array(yaw)

    s_ref, kappa_ref = calc_sref_kapparef(x, y, yaw)
    s_ref_arr, kappa_ref_arr = calc_sref_kapparef(x_arr, y_arr, yaw_arr)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(s_ref, kappa_ref)
    plt.plot(s_ref_arr, kappa_ref_arr)
    plt.show()


if __name__ == "__main__":
    test()
