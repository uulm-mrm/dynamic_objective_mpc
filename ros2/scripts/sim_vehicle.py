from collections import deque

import numpy as np
from corridor_planning_lib.mpc.real_vehicle_settings import get_real_integrator, get_real_bicycle_model
from corridor_planning_lib.param_setter import ParamSetter, ParamType
from corridor_planning_lib.params import VehicleParams
from numpy.typing import NDArray


class RealSimDiff:
    def __init__(self):
        self.a_offset: float = 0.0  # positive too much gas/ less brake
        self.a_mult: float = 1.0
        self.wb_offset: float = 0.0
        self.delta_offset: float = 0.0


class SimVehicle:
    def __init__(self, x0: NDArray, nx: int, nu: int, delta_t: float):

        self.delta_t: float = delta_t

        self.x0_prev: NDArray = np.zeros(nx)

        assert len(x0) == nx

        self.nx = nx
        self.nu = nu

        # start values
        self.x0 = x0.copy()
        self.u0 = np.array([0.0, 0.0])
        self.nx = len(self.x0)

        self.x_idx, self.y_idx, self.phi_idx, self.v_idx = list(
            [idx for idx in range(self.nx)])
        self.acc_idx, self.delta_idx = list(
            [idx for idx in range(self.nu)])

        self.diff_params: RealSimDiff = RealSimDiff()
        self.vehicle_params = VehicleParams()

        # predictions
        self.model = get_real_bicycle_model()
        self.integrator = get_real_integrator(self.model)
        self.param_setter = ParamSetter(self.model, ParamType.p, N=50)

        # simulated dead times
        dead_time_acc = 0.180
        dead_time_steer = 0.180

        # dead time structs
        a_length = int(dead_time_acc / delta_t)
        delta_length = int(dead_time_steer / delta_t)
        self.a_shift: deque = deque()
        self.delta_shift: deque = deque()

        # fill with values
        for i in range(a_length):
            self.a_shift.append(self.u0[self.acc_idx])
        for i in range(delta_length):
            self.delta_shift.append(self.u0[self.delta_idx])

    def integrate(self, a_ctrl: float, delta_ctrl: float, t_ros: float, errors: bool = False,
                  runtime_errors: bool = False) -> None:

        self._set_params(t_ros, errors, runtime_errors)

        if len(self.a_shift) == 0:
            a_now = a_ctrl
        else:
            # get new controls
            a_now: float = self.a_shift.popleft()
            self.a_shift.append(a_ctrl)

        if len(self.delta_shift) == 0:
            delta_now = delta_ctrl
        else:
            delta_now: float = self.delta_shift.popleft()
            self.delta_shift.append(delta_ctrl)

        self.x0_prev = self.x0.copy()

        # set current input
        self.u0 = np.array([a_now, delta_now])

        # sim with controlled values
        self.integrator.set("T", self.delta_t)
        self.x0 = self.integrator.simulate(self.x0, self.u0)

    def meas_x0(self) -> NDArray:
        return np.hstack([self.x0, self.u0[self.delta_idx], 0])

    def meas_a(self) -> float:
        return (self.x0[self.v_idx] - self.x0_prev[self.v_idx]) / self.delta_t

    def _set_params(self, t_ros: float, errors: bool, runtime_errors: bool) -> None:

        if not errors:
            self.diff_params.a_offset = 0.0
            self.diff_params.a_mult = 1.0
            self.diff_params.wb_offset = 0.0
            self.diff_params.delta_offset = 0.0

        # with errors
        if runtime_errors:
            if t_ros > 10:
                self.diff_params.a_offset = 0.0
            if t_ros > 20:
                self.diff_params.a_offset = -0.5

        # set in solver array
        self.param_setter.set("a_offset", self.diff_params.a_offset)
        self.param_setter.set("a_mult", self.diff_params.a_mult)
        # set params of simulated vehicle
        self.param_setter.set("wb", self.vehicle_params.wb)
        self.param_setter.set(
            "wb_offset", self.diff_params.wb_offset)
        self.param_setter.set(
            "delta_offset", self.diff_params.delta_offset)
        self.integrator.set("p", self.param_setter.param_arr)
