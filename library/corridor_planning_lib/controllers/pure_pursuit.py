# Adapted from https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/pure_pursuit/pure_pursuit.py

import numpy as np
import math
from numba import jit
from numpy.typing import NDArray
from typing import Final
from corridor_planning_lib import util

class PurePursuit:
    def __init__(self, wb: float, res: float):
        # Parameters
        self.K: Final[float] = 1.0  # look forward gain
        self.MAX_LD: Final[float] = 10.0  # [m] look-ahead distance
        self.MIN_LD: Final[float] = wb + 1.0  # [m] look-ahead distance
        self.WB: Final[float] = wb  # [m] wheelbase of vehicle
        self.RES: Final[float] = res

    def get_look_ahead_distance(self, v: float) -> float:
        return np.clip(self.K * v, self.MIN_LD, self.MAX_LD)

    @staticmethod
    def get_dist(x: float, y: float, px: float, py: float) -> float:
        return math.hypot(x - px, y - py)

    def get_target_point(self, x: float, y: float, l_dist: float, x_vec, y_vec, phi_vec, is_forward_segment: bool) -> [float, float]:
        """
        Get target point that is l_dist (Look ahead distance) away
        :param is_forward_segment:
        :param phi_vec:
        :param x:
        :param y:
        :param l_dist:
        :param x_vec:
        :param y_vec:
        :return:
        """

        # extrapolation
        phi: float = phi_vec[-1] if is_forward_segment else phi_vec[-1] + math.pi
        dx_last: float = self.RES * math.cos(phi)
        dy_last: float = self.RES * math.sin(phi)

        i: int = 0
        while i < 1000:
            if i < len(x_vec):
                # take value
                px, py = x_vec[i], y_vec[i]
            else:
                # extrapolate
                di: int = i - (len(x_vec)-1)
                dx: float = di * dx_last
                dy: float = di * dy_last
                px = x_vec[-1] + dx
                py = y_vec[-1] + dy

            dist: float = self.get_dist(x, y, px, py)
            i += 1
            if dist > l_dist:
                return px, py

        # print("MAX_ITERS reached, target point could not be found!")
        return px, py


    def get_delta(self, x: float, y: float, yaw: float, v: float, x_vec, y_vec, phi_vec, is_forward_segment) -> float:
        l_dist: float = self.get_look_ahead_distance(v)
        tx, ty = self.get_target_point(x, y, l_dist, x_vec, y_vec, phi_vec, is_forward_segment)
        alpha: float = util.angle_diff(yaw, math.atan2(ty - y, tx - x))
        return math.atan(2.0 * self.WB * math.sin(alpha) / l_dist)


@jit(nopython=True, cache=True, nogil=True)
def find_closest_point(x: float, y: float, xref: NDArray, yref: NDArray) -> int:
    xdiff: NDArray = x - xref
    ydiff: NDArray = y - yref
    dists2: NDArray = xdiff * xdiff + ydiff * ydiff
    return np.argmin(dists2)

def test():
    import imviz as viz
    main_title = "Pure Pursuit Test"
    viz.set_main_window_title(main_title)
    viz.style_colors_light()
    viz.set_main_window_size((10000, 10000))
    viz.show_main_window()

    # data
    res: float = 0.1
    s_vec = np.arange(0, 20, res)
    d_vec = np.sin(0.5*s_vec)
    path_phi: float = np.deg2rad(45)
    x_vec = math.cos(path_phi) * s_vec + math.sin(path_phi) * d_vec
    y_vec = math.sin(path_phi) * s_vec - math.sin(path_phi) * d_vec

    x: float = 0.0
    y: float = 7.0
    v: float = 3.0
    phi: float = np.deg2rad(45)

    pp = PurePursuit(wb=2.5, res=0.1)

    while True:
        idx = find_closest_point(x, y, x_vec, y_vec)
        x_forward, y_forward = x_vec[idx:], y_vec[idx:]
        # x_forward, y_forward = np.flip(x_vec[:idx]), np.flip(y_vec[:idx])

        delta: float = pp.get_delta(x, y, phi, v, x_forward, y_forward)
        dx, dy = math.cos(phi + delta)*2, math.sin(phi + delta)*2

        if not viz.wait(vsync=False):
            continue

        window_title = "map"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            if viz.begin_plot(window_title, flags=viz.PlotFlags.EQUAL):
                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                viz.plot(x_vec, y_vec, color="gray", fmt="-o")
                viz.plot(x_forward, y_forward, color="blue", fmt="-o")
                viz.plot([pp.tx], [pp.ty], color="green", fmt="o")
                ego = np.array([x, y])
                ego = viz.drag_point("drag_ego", ego, radius=5.0, color="blue")
                x, y = ego[0], ego[1]
                viz.plot_circle((x, y), radius=pp.l_dist, color="green")

                # plot ego
                dx_ego, dy_ego = math.cos(phi)*2, math.sin(phi)*2
                viz.plot([x, x+dx_ego], [y, y+dy_ego], color="blue")

                # plot steer
                viz.plot([x, x+dx], [y, y+dy], color="red")

                viz.end_plot()
            viz.end_window()

if __name__ == "__main__":
    test()

