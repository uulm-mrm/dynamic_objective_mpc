import math
from typing import Any

import numpy as np
from numba import jit
from numpy.typing import NDArray


########################################################################################################################
# Interpolations #######################################################################################################
########################################################################################################################
@jit(nopython=True, cache=True, nogil=True)
def interpolate(t: float, val0: float, val1: float):
    return (1 - t) * val0 + t * val1


@jit(nopython=True, cache=True, nogil=True)
def interpolate_xy(t: float, xs: tuple, ys: tuple):
    x = interpolate(t, *xs)
    y = interpolate(t, *ys)
    return x, y


@jit(nopython=True, cache=True, nogil=True)
def interpolate_xyphi(t: float, xs: tuple, ys: tuple, phis: tuple):
    x = interpolate(t, *xs)
    y = interpolate(t, *ys)
    phi = interpolate(t, *phis)
    return x, y, phi


@jit(nopython=True, cache=True, nogil=True)
def interpolate_sxyphi(t: float, ss: tuple, xs: tuple, ys: tuple, phis: tuple):
    s = interpolate(t, *ss)
    x = interpolate(t, *xs)
    y = interpolate(t, *ys)
    phi = interpolate(t, *phis)
    return s, x, y, phi


########################################################################################################################
# Helper ###############################################################################################################
########################################################################################################################
@jit(nopython=True, cache=True, nogil=True)
def dist_2d(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


@jit(nopython=True, cache=True, nogil=True)
def dist_2d_squared(x1, x2, y1, y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)


@jit(nopython=True, cache=True, nogil=True)
def angle_diff(a: float, b: NDArray) -> NDArray:
    diff: NDArray = a - b
    diff = np.where(diff > math.pi, diff - 2 * math.pi, diff)
    diff = np.where(diff < -math.pi, diff + 2 * math.pi, diff)
    return diff


@jit(nopython=True, cache=True, nogil=True)
def find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2):
    if len(xref) == 1:
        return 0
    vabs = abs(sref[idxmindist] - sref[idxmindist2])
    if vabs == 0:
        return 0
    vl = np.empty(2)
    u = np.empty(2)
    vl[0] = xref[idxmindist2] - xref[idxmindist]
    vl[1] = yref[idxmindist2] - yref[idxmindist]
    u[0] = x - xref[idxmindist]
    u[1] = y - yref[idxmindist]
    t = (vl[0] * u[0] + vl[1] * u[1]) / vabs / vabs
    return t


########################################################################################################################
# Transforms ###########################################################################################################
########################################################################################################################

# Misc. ################################################################################################################
@jit(nopython=True, cache=True, nogil=True)
def get_phi(x, y, sref, xref, yref, phiref):
    idxmindist: int = find_closest_point(x, y, xref, yref)
    idxmindist2: int = find_closest_neighbour(x, y, xref, yref, idxmindist)
    t = find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2)

    phi = (1 - t) * phiref[idxmindist] + t * phiref[idxmindist2]

    return phi


# Frenet -> Cart #######################################################################################################

@jit(nopython=True, cache=True, nogil=True)
def find_closest_s(si, sref):
    idxmindist = np.argmin(np.abs(si - sref))
    return idxmindist


@jit(nopython=True, cache=True, nogil=True)
def find_second_closest_s(si, sref, idxmindist):
    d1 = np.abs(si - sref[idxmindist - 1])  # distance to node before
    d2 = np.abs(si - sref[(idxmindist + 1) % sref.size])  # distance to node after
    idxmindist2 = idxmindist + 1 if d1 > d2 else idxmindist - 1
    return idxmindist2 % sref.size


@jit(nopython=True, cache=True, nogil=True)
def transform_pose_frenet2cart_scalar(si, ni, alpha, sref, xref, yref, phiref):
    idxmindist = find_closest_s(si, sref)
    idxmindist2 = find_second_closest_s(si, sref, idxmindist)
    t = (si - sref[idxmindist]) / (sref[idxmindist2] - sref[idxmindist])
    xs = (xref[idxmindist], xref[idxmindist2])
    ys = (yref[idxmindist], yref[idxmindist2])
    phis = (phiref[idxmindist], phiref[idxmindist2])
    x_int, y_int, psi_int = interpolate_xyphi(t, xs, ys, phis)

    x = x_int - ni * np.sin(psi_int)
    y = y_int + ni * np.cos(psi_int)
    phi = psi_int + alpha
    return x, y, phi


# Cart -> Frenet #######################################################################################################
@jit(nopython=True, cache=True, nogil=True)
def transform_point_cart2frenet(x: float, y: float, sref, xref, yref, phiref) -> tuple[float, float]:
    idxmindist: int = find_closest_point(x, y, xref, yref)
    idxmindist2: int = find_closest_neighbour(x, y, xref, yref, idxmindist)
    t = find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2)
    ss = (sref[idxmindist], sref[idxmindist2])
    xs = (xref[idxmindist], xref[idxmindist2])
    ys = (yref[idxmindist], yref[idxmindist2])
    phis = (phiref[idxmindist], phiref[idxmindist2])
    s_int, x_int, y_int, psi_int = interpolate_sxyphi(t, ss, xs, ys, phis)

    s = s_int
    n = np.cos(psi_int) * (y - y_int) - np.sin(psi_int) * (x - x_int)
    return s, n


@jit(nopython=True, cache=True, nogil=True)
def transform_pose_cart2frenet(x: float, y: float, phi: float, sref, xref, yref, phiref) -> tuple[float, float, float]:
    idxmindist: int = find_closest_pose(x, y, phi, xref, yref, phiref)
    idxmindist2: int = find_closest_neighbour(x, y, xref, yref, idxmindist)
    t = find_projection(x, y, xref, yref, sref, idxmindist, idxmindist2)
    ss = (sref[idxmindist], sref[idxmindist2])
    xs = (xref[idxmindist], xref[idxmindist2])
    ys = (yref[idxmindist], yref[idxmindist2])
    phis = (phiref[idxmindist], phiref[idxmindist2])
    s_int, x_int, y_int, psi_int = interpolate_sxyphi(t, ss, xs, ys, phis)

    s = s_int
    n = np.cos(psi_int) * (y - y_int) - np.sin(psi_int) * (x - x_int)
    alpha = phi - psi_int
    return s, n, alpha


# Cart -> Frenet (Index precise) #######################################################################################

@jit(nopython=True, cache=True, nogil=True)
def find_closest_point(x: float, y: float, xref: NDArray, yref: NDArray) -> int:
    xdiff: NDArray = x - xref
    ydiff: NDArray = y - yref
    dists2: NDArray = xdiff * xdiff + ydiff * ydiff
    return np.argmin(dists2)


@jit(nopython=True, cache=True, nogil=True)
def find_closest_pose(x: float, y: float, phi: float, xref: NDArray, yref: NDArray, phiref: NDArray,
                      angle_lim: float = math.pi / 2, angle_weight: float = 0.0) -> int:
    xdiff: NDArray = x - xref
    ydiff: NDArray = y - yref
    psidiff = angle_diff(phi, phiref)
    dists2: NDArray = xdiff * xdiff + ydiff * ydiff + (psidiff * psidiff) * angle_weight
    dists2[psidiff > angle_lim] = np.inf
    return np.argmin(dists2)


@jit(nopython=True, cache=True, nogil=True)
def find_closest_neighbour(x, y, xref, yref, idxmindist):
    dist_before = dist_2d_squared(x, xref[idxmindist - 1], y, yref[idxmindist - 1])
    idxmindist_after = (idxmindist + 1) % xref.size
    dist_after = dist_2d_squared(x, xref[idxmindist_after], y, yref[idxmindist_after])

    idxmindist2 = idxmindist - 1 if dist_before < dist_after else idxmindist + 1

    return idxmindist2 % xref.size


@jit(nopython=True, cache=True, nogil=True)
def get_closest_point(x: float, y: float, xref: NDArray, yref: NDArray) -> tuple[
    int, Any, Any]:
    s_idx = find_closest_point(x, y, xref, yref)
    x_proj = xref[s_idx]
    y_proj = yref[s_idx]
    return s_idx, x_proj, y_proj


@jit(nopython=True, cache=True, nogil=True)
def get_closest_pose(x: float, y: float, phi: float, xref: NDArray, yref: NDArray, phiref: NDArray,
                     angle_lim: float, angle_weight: float) -> tuple[int, Any, Any, Any]:
    s_idx = find_closest_pose(x, y, phi, xref, yref, phiref, angle_lim, angle_weight)
    x_proj = xref[s_idx]
    y_proj = yref[s_idx]
    phi_proj = phiref[s_idx]
    return s_idx, x_proj, y_proj, phi_proj


def test():
    x_vec = np.arange(0, 10, 0.1)
    y_vec = np.arange(0, 10, 0.1)
    y_vec = np.zeros_like(x_vec)  #0, 10, 0.1)
    phi_ref = np.ones_like(x_vec) * np.deg2rad(45)

    s_vec = np.cumsum(np.sqrt(np.diff(x_vec) ** 2 + np.diff(y_vec) ** 2))
    s_vec = np.hstack([0, s_vec])

    print("x_vec", x_vec)
    print("y_vec", y_vec)
    print("s_vec", s_vec)
    x = 15.0
    y = 0.0
    s_idx, x_proj, y_proj = get_closest_point(x, y, x_vec, y_vec)
    s, n = transform_point_cart2frenet(x, y, s_vec, x_vec, y_vec, phi_ref)
    print("s_idx", s_idx)

    print("s", s)
    print("n", n)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x_vec, y_vec, color="blue")
    plt.scatter(x, y, color="red")
    plt.scatter(y_proj, y_proj, color="green")
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == '__main__':
    test()
