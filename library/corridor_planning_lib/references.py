import numpy as np
from numpy.typing import NDArray
from typing import Final
import math

from corridor_planning_lib import util
from corridor_planning_lib.params import Params
from numba import jit

class References:
    def __init__(self, nb_param: int, res: float):
        self.nb_param: int = nb_param
        self.theta_ego: float = 0.0
        self.theta_0: float = 0.0
        self.theta_f: float = 0.0
        self.param_offset: float = 0.0

        self.theta: NDArray = np.zeros(nb_param)
        self.x: NDArray = np.zeros(nb_param)
        self.y: NDArray = np.zeros(nb_param)
        self.phi: NDArray = np.zeros(nb_param)
        self.kappa: NDArray = np.zeros(nb_param)
        self.v_max_raw: NDArray = np.zeros(nb_param)
        self.v_max: NDArray = np.zeros(nb_param)
        self.v_init: NDArray = np.zeros(nb_param)
        self.dtheta_max: NDArray = np.zeros(nb_param)
        self.n_min: NDArray = np.zeros(nb_param)
        self.n_max: NDArray = np.zeros(nb_param)

        self.res: float = res

        self.v_min: Final[float] = 1.0
        self.desired_offset: Final[float] = 5.0

    def get_at_theta(self, which: str, theta: float):
        attr = getattr(self, which)
        max_idx: int = len(attr) - 1
        idx: int = int(min((theta - self.theta_0) / self.res, max_idx))
        idx = max(idx, 0)
        return attr[idx]

    @staticmethod
    @jit(cache=True, nopython=True)
    def get_vmax_kappa(v_max: NDArray, kappa: NDArray, a_lat_max: float) -> NDArray:
        # init with kappa for init
        assert len(v_max) == len(kappa)

        N: int = min(10, len(kappa))
        kappa_filt = np.abs(np.convolve(kappa, np.ones(N)/N, mode='same'))
        assert len(kappa) == len(kappa_filt)
        # avoid division by zero
        eps: float = 1e-8
        v_max_kappa = np.sqrt(a_lat_max / (kappa_filt+eps))

        assert len(v_max) == len(v_max_kappa)
        return np.minimum(v_max, v_max_kappa)

    def set(self, path, stop_idx: int, theta_goal_behind: float, v: float, theta: float, params: Params):
        self.theta_ego = theta

        # Get Start index 5m behind vehicle
        des_offset_idx = int(self.desired_offset / self.res)
        offset_idx = path.get_possible_start_offset(des_offset_idx=des_offset_idx)
        offset: float = offset_idx * self.res

        self.theta_0 = self.theta_ego - offset + self.param_offset
        self.theta_f = path.get_at_idx("s", stop_idx)
        self.x = path.get_segment_forward("x", start_offset=offset_idx, last_idx=stop_idx)
        self.y = path.get_segment_forward("y", start_offset=offset_idx, last_idx=stop_idx)
        self.phi = path.get_segment_forward("phi", start_offset=offset_idx, last_idx=stop_idx)
        self.kappa = path.get_segment_forward("kappa", start_offset=offset_idx, last_idx=stop_idx)
        self.n_min = path.get_segment_forward("smooth_border_min", start_offset=offset_idx)
        self.n_max = path.get_segment_forward("smooth_border_max", start_offset=offset_idx)

        # dtheta at end must be limitted at end to avoid generation of reward through virtual overshooting
        self.v_max_raw = path.get_segment_forward("v_max", start_offset=offset_idx, last_idx=stop_idx).copy()

        idx_stop: int = (stop_idx - path.ego_data.ego_idx + offset_idx)

        # stop or desired speed at end
        v_end = self.v_min
        if theta_goal_behind > 0:
            acc_red_factor: float = 0.5  # must be smaller because v_max does not decrease further
            t_to_stop: float = math.sqrt(2*theta_goal_behind/params.solver.along_max)
            v_end = t_to_stop * params.solver.along_max * acc_red_factor
            v_end = max(v_end, self.v_min)
        self.v_max_raw[idx_stop-1:] = v_end

        # start near ego velocity
        # v_ego_idx: int = min(len(self.v_max_raw) - 1, offset_idx)
        # v_min_ego = max(abs(v), self.v_min)
        # self.v_max_raw[v_ego_idx] = v_min_ego
        # self.v_max_raw[max(v_ego_idx-1, 0)] = v_min_ego

        # set vmax
        self.v_max = util.v_integrate(self.v_max_raw, self.res, params.solver.along_max, -params.solver.along_max,
                                      factor=1.0)

        self.dtheta_max = self.v_max

        # set init
        v_max_kappa = self.get_vmax_kappa(self.v_max_raw, self.kappa, params.solver.alat_max)
        self.v_init = util.v_integrate(v_max_kappa,
                                       self.res, params.general.init_acc, -params.general.init_acc,
                                       factor=1.0)

        self.theta = np.arange(self.theta_0, self.theta_0 + len(self.x) * self.res, self.res)