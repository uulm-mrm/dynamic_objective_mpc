import math
from typing import Optional

import corridor_planning_lib.util as util
import numpy as np
from corridor_planning_lib.transforms import (transform_pose_cart2frenet, transform_pose_frenet2cart_scalar,
                                              transform_point_cart2frenet, get_phi)
from numba import jit
from numpy.typing import NDArray


class Obstacle:
    def __init__(self, x: float, y: float, phi: float, v: float = 0.0, w: float = 1.0, l: float = 2.0, N: int = 50):
        self.x: float = x
        self.y: float = y
        self.phi: float = phi
        self.v = v
        self.w: float = w
        self.l: float = l

        self.s: float = 0
        self.n: float = 0
        self.alpha: float = 0
        self.omega: float = 0

        # predicted trajectories
        self.s_pred = self.s * np.ones(N + 1)
        self.n_pred = self.n * np.ones(N + 1)
        self.alpha_pred = self.alpha * np.ones(N + 1)
        self.x_pred = N * self.x * np.ones(N + 1)
        self.y_pred = N * self.y * np.ones(N + 1)
        self.phi_pred = N * self.phi * np.ones(N + 1)

        self.vertices: NDArray = util.get_rect_vertices(x, y, w, l, phi)

    def project(self, sref, xref, yref, phiref):
        # transform static obs to frame
        [self.s, self.n, self.alpha] = transform_pose_cart2frenet(self.x, self.y, self.phi, sref, xref, yref, phiref)

    def move_along_path(self, dt: float, sref, xref, yref, phiref):
        """
        move along path, n=const, alpha=const
        :param dt:
        :param sref:
        :param xref:
        :param yref:
        :param phiref:
        :return:
        """


        vx: float = self.v * math.cos(self.phi)
        vy: float = self.v * math.sin(self.phi)

        self.x += vx * dt
        self.y += vy * dt

        s, _, _ = transform_pose_cart2frenet(self.x, self.y, self.phi, sref, xref, yref, phiref)
        _, _, phi = transform_pose_frenet2cart_scalar(self.s, 0, 0, sref, xref, yref, phiref)
        self.omega = util.angle_diff(self.phi, phi) / dt

        # assert abs(self.omega) < 0.2 + 0.01, (self.omega, self.phi, phi, util.angle_diff(self.phi, phi))
        self.phi = phi

        self.omega = np.clip(self.omega, -0.2, 0.2)

        # stop at end
        if self.s > sref[-1]-10:
            self.v = max(self.v - 0.001, 0.0)

        self.vertices = util.get_rect_vertices(self.x, self.y, self.w, self.l, self.phi)

    def move_cvtr(self, dt: float) -> [float, float, float]:
        ds: float = self.v * dt
        self.x = self.x + math.cos(self.phi) * ds
        self.y = self.y + math.sin(self.phi) * ds
        self.phi = util.clip_angle_minpi_pluspi(self.phi + self.omega * dt)
        self.vertices = util.get_rect_vertices(self.x, self.y, self.w, self.l, self.phi)

    def predict_cvtr(self, dt: float, x0, y0, phi0) -> [float, float, float]:
        ds: float = self.v * dt
        x = x0 + math.cos(phi0) * ds
        y = y0 + math.sin(phi0) * ds
        phi = util.clip_angle_minpi_pluspi(phi0 + self.omega * dt)
        return x, y, phi

    @staticmethod
    @jit(nopython=True, cache=True, nogil=True)
    def get_closest_sn(x, y, phi, w, l, sref, xref, yref, phiref) -> [float, float]:
        # take smallest s and smallest n
        vertices = util.get_rect_vertices(x, y, w, l, phi)
        s_min = np.inf
        n_min = np.inf
        for x_vert, y_vert in zip(vertices[0], vertices[1]):
            s, n = transform_point_cart2frenet(x_vert, y_vert, sref, xref, yref, phiref)
            s_min = min(s, s_min)
            n_min = min(abs(n), abs(n_min))
        return s, n

    def predict(self, dt: float, N: int, sref, xref, yref, phiref):
        # predict with CVCTR
        if abs(self.v) > 0:
            x = self.x
            y = self.y
            phi = self.phi
            for i in range(N+1):
                x, y, phi = self.predict_cvtr(i * dt, x, y, phi)
                self.x_pred[i] = x
                self.y_pred[i] = y
                self.phi_pred[i] = phi

                self.s_pred[i], self.n_pred[i] = self.get_closest_sn(x, y, phi, self.w, self.l, sref, xref, yref, phiref)

            # # do not predict in loops
            # # get main prediction direction and clip
            # if self.s_pred[1] - self.s_pred[0] >= 0:
            #     self.s_pred_filt = np.clip(self.s_pred, self.s, 1e3)
            #     assert (self.s_pred_filt >= self.s).all(), self.s_pred
            # else:
            #     self.s_pred_filt = np.clip(self.s_pred, -1e3, self.s)
            #     assert (self.s_pred_filt <= self.s).all(), self.s_pred

        else:
            self.s_pred[:], self.n_pred[:] = self.get_closest_sn(self.x, self.y, self.phi, self.w, self.l, sref, xref, yref, phiref)
            self.n_pred[:] = self.n
            self.x_pred[:] = self.x
            self.y_pred[:] = self.y
            self.phi_pred[:] = self.phi


class EgoData:
    def __init__(self, s_list = None):
        # last index of ego
        self.ego_idx: int = 0
        self.max_ego_idx: int = 0
        self.segment_start_idx: int = 0
        self.segment_end_idx: int = len(s_list) if s_list is not None else 0


class GoalTolerance:
    def __init__(self, s: float, n: float, yaw: float):
        self.s: float = s
        self.n: float = n
        self.yaw: float = yaw

    def is_within_tol(self, x, y, phi, xr, yr, phir) -> tuple[bool, tuple[float, float, float]]:
        """
        Get differences and check if they are within the given tolerance
        :param x:
        :param y:
        :param phi:
        :param xr:
        :param yr:
        :param phir:
        :return:
        """
        # Differences
        e_lag, e_cont, e_phi = util.get_frenet_differences(x, y, phi, xr, yr, phir)

        if abs(e_lag) < self.s and abs(e_cont) < self.n and abs(e_phi) < np.deg2rad(self.yaw):
            return True, (e_lag, e_cont, e_phi)
        return False, (e_lag, e_cont, e_phi)


class Path:
    def __init__(self, s = None, x = None, y= None, phi = None, kappa = None, directions: NDArray[bool] = None, v_max = None,
                 n_min = None, n_max = None, path_id: int = 42, goal_index: int = None):
        # default empty
        if s is None:
            self.s = []
            self.x = []
            self.y = []
            self.phi = []
            self.kappa = []
            self.directions = []
            self.v_max = []
            self.border_min = []
            self.border_max = []
            self.goal_index = 0
            self.ego_data = EgoData()

        else:
            self.s = s
            self.x = x
            self.y = y
            self.phi = phi
            self.kappa = kappa
            self.directions: NDArray[bool] = directions

            self.v_max = v_max * np.ones_like(s) if np.isscalar(n_min) else v_max
            self.border_min = n_min * np.ones_like(s) if np.isscalar(n_min) else n_min
            self.border_max = n_max * np.ones_like(s) if np.isscalar(n_max) else n_max

            self.goal_index = goal_index if goal_index is not None else len(x) - 1
            self.ego_data = EgoData(self.s)

        self.path_id: int = path_id

        # always not set at start
        self.smooth_border_min = None
        self.smooth_border_max = None
        self.n_corridor_center = None
        self.dir_change_indices = []
        self.idx_params_set: int = -1


    def values(self, s_i: int = None, g_i: int = None):
        if s_i is None and g_i is None:
            return self.s, self.x, self.y, self.phi, self.kappa, self.directions, self.v_max

        if s_i is not None and g_i is None:
            return self.s[s_i:], self.x[s_i:], self.y[s_i:], self.phi[s_i:], self.kappa[s_i:], self.directions[
                                                                                               s_i:], self.v_max[s_i:]

        if s_i is None and g_i is not None:
            return self.s[:g_i], self.x[:g_i], self.y[:g_i], self.phi[:g_i], self.kappa[:g_i], self.directions[
                                                                                               :g_i], self.v_max[:g_i]

        # if s_i is not None and g_i is not None:
        return self.s[s_i:g_i], self.x[s_i:g_i], self.y[s_i:g_i], self.phi[s_i:g_i], self.kappa[s_i:g_i], \
            self.directions[s_i:g_i], self.v_max[s_i:g_i]

    def are_new_params_needed(self):
        return self.ego_data.ego_idx != self.idx_params_set

    def set_params_used(self):
        self.idx_params_set = self.ego_data.ego_idx

    def set(self, attr_str: str, val: float, s_idx: int, e_idx: int = None):
        attr_values = getattr(self, attr_str)

        if e_idx is None:
            attr_values[s_idx:] = val
            return

        e_idx = min(e_idx, len(attr_values) - 1)
        attr_values[s_idx:e_idx] = val

    def set_dir_changes(self):
        # x_diff = np.gradient(self.x)
        # y_diff = np.gradient(self.y)
        # phis = np.arctan2(y_diff, x_diff)
        # angle_diffs = [abs(util.angle_diff(phi, self.phi[i])) for i, phi in enumerate(phis)]

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(self.directions, label="directions")
        # plt.plot(phis, label="phis")
        # plt.plot(self.phi, label="path phi")
        # plt.plot(angle_diffs, label="diffs")

        # for i, diff in enumerate(angle_diffs):
        #     if diff > math.pi/2:
        #         print("higher")
        #         self.directions[i] = False
        #     else:
        #         print("lower")
        #         self.directions[i] = True

        # plt.plot(self.directions, label="filt directions")
        # plt.legend()
        # plt.show()

        self.dir_change_indices = np.array([0])
        change_indices = np.where(np.abs(np.diff(self.directions)) > 0)[0]
        self.dir_change_indices = np.hstack((self.dir_change_indices, change_indices, len(self.directions)))

    def unwrap_angles(self, ego_angle: float):
        angles = np.hstack((ego_angle, self.phi))
        angles = np.unwrap(angles, period=2 * np.pi)
        self.phi = angles[1:]

    def get_interpolated_val(self, which: str, float_idx: float):
        return util.interpolate_at_array(float_idx, getattr(self, which))

    def smooth_boundaries(self):
        self.smooth_border_min = util.n_integrate(self.border_min, self.kappa, dn=0.1, is_pos=False)
        self.smooth_border_max = util.n_integrate(self.border_max, self.kappa, dn=0.1, is_pos=True)

    def is_ego_at_end(self) -> bool:
        end_idx: int = len(self.x) - 1
        return self.ego_data.ego_idx == end_idx

    def set_v_max_on_reverse(self, v_max_reverse: float):
        rev_indices: NDArray = self.directions == False
        self.v_max[rev_indices] = np.clip(self.v_max[rev_indices], 0.0, v_max_reverse)

    def prepare_path(self, params) -> None:
        self.set_dir_changes()
        self.set_v_max_on_reverse(params.general.v_max_reverse)

    def prepare_bounds(self):
        self.smooth_boundaries()

    def is_idx_relevant(self, idx: int) -> bool:
        if idx is None:
            return False
        return self.ego_data.segment_start_idx < idx <= self.ego_data.segment_end_idx

    def set_relevant_idx(self) -> bool:
        # standard case and maximum case also
        segment_start_idx: int = -1
        segment_end_idx: int = -1

        # search for dir change
        for (prior_idx, next_idx) in util.pairwise(self.dir_change_indices):
            if next_idx > self.ego_data.ego_idx:
                segment_end_idx = next_idx
                segment_start_idx = prior_idx
                break

        # at end of path
        if not segment_start_idx > -1:
            segment_end_idx = self.dir_change_indices[-1]
            segment_start_idx = self.dir_change_indices[-2]

        # paste values
        on_new_segment = False
        if self.ego_data.segment_start_idx != segment_start_idx:
            on_new_segment: bool = True
        self.ego_data.segment_start_idx = segment_start_idx
        self.ego_data.segment_end_idx = segment_end_idx
        return on_new_segment

    def shift_ego_near_switch(self, check_idx: int) -> bool:
        """
        Check if ego vehicle is near direction change. If yes shift param arr start to switching point
        :return:
        """

        diff: int = abs(self.ego_data.ego_idx - self.ego_data.segment_end_idx)
        at_last_segment: bool = self.ego_data.segment_end_idx == len(self.x)

        # if near direction change, shift ego index to switching point to
        if diff <= check_idx and not at_last_segment:
            self.ego_data.ego_idx = self.ego_data.segment_end_idx
            return True

        return False

    def get_possible_start_offset(self, des_offset_idx: int):
        des_start_idx: int = self.ego_data.ego_idx - des_offset_idx
        possible_start_idx: int = max(des_start_idx, self.ego_data.segment_start_idx)
        return self.ego_data.ego_idx - possible_start_idx

    def get_segment_forward(self, which: str, start_offset: int = None, last_idx: int = None) -> NDArray:
        if last_idx is None:
            last_idx = self.ego_data.segment_end_idx + 1
        else:
            last_idx = min(self.ego_data.segment_end_idx + 1, last_idx + 1)

        # do not index with smaller index at second slice
        last_idx = max(last_idx, self.ego_data.ego_idx + 1)

        # slice values out
        offset_idx = start_offset if start_offset is not None else 0
        values = getattr(self, which)[self.ego_data.ego_idx-offset_idx:last_idx]

        if values.size == 0:
            return np.asarray([getattr(self, which)[-1]])
        return values

    def get_forward(self, which: str) -> NDArray:
        values = getattr(self, which)[self.ego_data.ego_idx:]
        if values.size == 0:
            return np.array([getattr(self, which)[-1]])
        return values.copy()

    def get_segment_forward_copy(self, which: str) -> NDArray:
        values = getattr(self, which)[self.ego_data.ego_idx:self.ego_data.segment_end_idx + 1]
        if values.size == 0:
            return np.array([getattr(self, which)[-1]])
        return values.copy()

    def get_at_end(self, which: str) -> float:
        return getattr(self, which)[-1]

    def get_at_goal(self, which: str) -> float:
        return self.get_at_idx(which, self.goal_index)

    def get_at_ego(self, which: str) -> float:
        return self.get_at_idx(which, self.ego_data.ego_idx)

    def get_at_idx(self, which: str, idx: int) -> float:
        attr = getattr(self, which)
        max_idx: int = len(attr) - 1
        return attr[np.clip(idx, 0, max_idx)]

    def get_from_idx(self, which: str, idx: int) -> float:
        attr = getattr(self, which)
        max_idx: int = len(attr) - 1
        return attr[np.clip(idx, 0, max_idx):]

    def get_at_segment_middle(self, which: str):
        values = getattr(self, which)[self.ego_data.segment_start_idx:self.ego_data.segment_end_idx + 1]
        if values.size == 0:
            return np.array([getattr(self, which)[-1]])

        return values[int(len(values) / 2)]

    def is_segment_forward(self) -> bool:
        return self.get_at_segment_middle("directions")

    def get_segment(self, which: str) -> NDArray:
        values = getattr(self, which)[self.ego_data.segment_start_idx:self.ego_data.segment_end_idx + 1]
        if values.size == 0:
            return np.array([getattr(self, which)[-1]])
        return values

    def get_at_segment_end(self, which: str) -> float:
        attr = getattr(self, which)
        return attr[min(self.ego_data.segment_end_idx, max(len(attr) - 1, 0))]

    def get_forward_from_theta(self, which: str, theta: float, res: float):
        assert not np.isnan(theta)
        values = self.get_segment_forward(which)
        theta_0 = self.get_at_ego("s")
        idx: int = int((theta - theta_0) / res)
        sliced_values = values[idx:]
        if sliced_values.size == 0:
            return np.array([values[-1]])
        return sliced_values


    # def padd(self, nb_elements: int, v_min: float):
    #     """
    #     Add last state at end
    #     :param v_min:
    #     :param nb_elements:
    #     :return:
    #     """
    #     self.s = np.hstack([self.s, np.ones(nb_elements) * self.s[-1]])
    #     self.x = np.hstack([self.x, np.ones(nb_elements) * self.x[-1]])
    #     self.y = np.hstack([self.y, np.ones(nb_elements) * self.y[-1]])
    #     self.phi = np.hstack([self.phi, np.ones(nb_elements) * self.phi[-1]])
    #     self.kappa = np.hstack([self.kappa, np.ones(nb_elements) * self.kappa[-1]])
    #     self.directions = np.hstack([self.directions, np.ones(nb_elements) * self.directions[-1]])
    #     self.border_min = np.hstack([self.border_min, np.ones(nb_elements) * self.border_min[-1]])
    #     self.border_max = np.hstack([self.border_max, np.ones(nb_elements) * self.border_max[-1]])
    #     self.smooth_border_min = np.hstack([self.smooth_border_min, np.ones(nb_elements) * self.smooth_border_min[-1]])
    #     self.smooth_border_max = np.hstack([self.smooth_border_max, np.ones(nb_elements) * self.smooth_border_max[-1]])
    #     self.v_max = np.hstack([self.v_max, v_min * np.ones(nb_elements) * self.v_max])  # all zeros at end

    def concatenate(self, add_path):
        self.s = np.hstack([self.s, add_path.s])
        self.x = np.hstack([self.x, add_path.x])
        self.y = np.hstack([self.y, add_path.y])
        self.phi = np.hstack([self.phi, add_path.phi])
        self.kappa = np.hstack([self.kappa, add_path.kappa])
        self.directions = np.hstack([self.directions, add_path.directions])
        self.border_min = np.hstack([self.border_min, add_path.border_min])
        self.border_max = np.hstack([self.border_max, add_path.border_max])
        self.smooth_border_min = np.hstack([self.smooth_border_min, add_path.smooth_border_min])
        self.smooth_border_max = np.hstack([self.smooth_border_max, add_path.smooth_border_max])
        self.v_max = np.hstack([self.v_max, add_path.v_max])

    # def clip_and_pad(self, idx: int):
    #     self.s[idx:] = self.s[idx]
    #     self.x[idx:] = self.x[idx]
    #     self.y[idx:] = self.y[idx]
    #     self.phi[idx:] = self.phi[idx]
    #     self.kappa[idx:] = 0.0
    #     self.directions[idx:] = self.directions[idx]
    #     self.border_min[idx:] = self.border_min[idx]
    #     self.border_max[idx:] = self.border_max[idx]
    #     self.smooth_border_min[idx:] = self.smooth_border_min[idx]
    #     self.smooth_border_max[idx:] = self.smooth_border_max[idx]
    #     self.v_max[idx:] = 1.0
