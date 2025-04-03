#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node

from aduulm_logger_python import create_logger, logger

from corridor_planning_lib.corridor_planner import CorridorPlanner
from corridor_planning_lib.data_structures import Path, Obstacle
from corridor_planning_lib.visualization import VisModule

from sim_vehicle import SimVehicle
from sim_params import SimParams
from path_modifications import add_forward_curve, add_backward_curve, add_forward_straight, add_backward_straight


class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')
        create_logger(self)

        self.params: SimParams = SimParams()

        self.declare_parameter('track_switch', self.params.track_switch)
        self.declare_parameter('ignore_goal', self.params.ignore_goal)
        self.declare_parameter('baseline', self.params.baseline)
        self.params.track_switch = self.get_parameter(
            'track_switch').get_parameter_value().integer_value
        self.params.ignore_goal = self.get_parameter(
            'ignore_goal').get_parameter_value().bool_value
        self.params.baseline = self.get_parameter(
            'baseline').get_parameter_value().integer_value

        self.cp = CorridorPlanner(logger, self.params.sim_delta_t)

        match self.params.baseline:
            case 0:
                # separated baseline
                self.cp.do_separate_planning = True
            case 1:
                # switch baseline
                self.cp.do_hard_switch = True
            case 0:
                # dynamic objective MPC
                pass

        self.vis = VisModule(self)

        # simulation states
        self.path: Path = Path()
        self.goal_pose: Optional[NDArray] = None
        self.reinit: bool = True

        # defaults
        self.obstacles: list = []

        self.set_track()

        # create sim vehicle
        nx_sim = self.cp.nx - 2
        nu_sim = self.cp.nu - 1
        x0_meas: Optional[NDArray] = np.zeros(nx_sim)
        x0_meas[0:3] = np.array(
            [self.path.x[0], self.path.y[0], self.path.phi[0]])

        self.vehicle = SimVehicle(
            x0_meas, nx_sim, nu_sim, self.params.sim_delta_t)

        # Run
        self.reinit = True
        self.simulate()

    def set_track(self):
        """
        Set the used track/path
        :return:
        """
        match self.params.track_switch:
            case 0:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(self.params.phi_rot_deg)]
                add_forward_straight(self.path, self.params,
                                     length=10, start_pose=start_pose)
                add_forward_curve(self.path, self.params,
                                  kappa=0.12, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.12, length=10)
                add_forward_straight(self.path, self.params, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.2, length=10)
                add_forward_straight(self.path, self.params, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_straight(self.path, self.params, length=10)
                add_forward_straight(self.path, self.params, length=10)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)

                self.path.border_min[50:100] += 2.0
                self.path.border_min[200:250] += 1.5
                self.path.border_min[300:400] += 3.5
                self.path.border_min[500:600] += 1.5
                self.path.border_min[600:700] += 3.5
                self.path.border_max[750:800] -= 3.5
                self.path.border_min[1200::] += 3.0
            case 1:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(90)]
                add_forward_straight(self.path, self.params,
                                     length=10, start_pose=start_pose)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params, kappa=0.3, length=2)
                add_forward_curve(self.path, self.params, kappa=-0.3, length=2)
                add_forward_straight(self.path, self.params, length=10)
                add_forward_curve(self.path, self.params, kappa=0.2, length=5)
                add_forward_curve(self.path, self.params, kappa=-0.2, length=5)
                add_forward_curve(self.path, self.params,
                                  kappa=0.20, length=10)
                add_forward_curve(self.path, self.params, kappa=0.20, length=5)
                add_forward_straight(self.path, self.params, length=10)
                add_backward_straight(self.path, self.params, length=2)
                add_backward_curve(self.path, self.params,
                                   kappa=-0.1, length=15)
                add_backward_straight(self.path, self.params, length=2)

                if not self.params.ignore_goal:
                    self.goal_pose = self.get_goal_pose(d_long=-3.0)

                self.path.border_min[220:240] += 3.8
                self.path.border_max[320:340] -= 3.5

            case 2:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(self.params.phi_rot_deg)]
                add_forward_straight(self.path, self.params,
                                     length=10, start_pose=start_pose)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_straight(self.path, self.params, length=5)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.2, length=10)
                add_forward_curve(self.path, self.params, kappa=0.2, length=10)
                add_backward_curve(self.path, self.params,
                                   kappa=-0.1, length=10)
                add_backward_curve(self.path, self.params,
                                   kappa=0.1, length=10)
                add_backward_straight(self.path, self.params, length=2)

                self.path.border_max[100:110] -= 3.5
                self.path.border_min[180:200] += 3.5
                self.path.border_max[300:350] -= 4.0
                self.path.border_min[300:350] += 3.0
                self.path.border_min[610:630] += 3.0
                self.path.border_max[700:750] -= 2.5

                if not self.params.ignore_goal:
                    self.goal_pose = self.get_goal_pose(d_long=-2.0)

            case 3:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(self.params.phi_rot_deg)]
                add_forward_straight(self.path, self.params,
                                     length=20, start_pose=start_pose)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_straight(self.path, self.params, length=20)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_straight(self.path, self.params, length=20)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_straight(self.path, self.params, length=20)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_straight(self.path, self.params, length=20)

            case 4:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(self.params.phi_rot_deg)]
                add_forward_straight(self.path, self.params,
                                     length=70, start_pose=start_pose)
                add_forward_curve(self.path, self.params,
                                  kappa=0.05, length=35)
                add_forward_straight(self.path, self.params, length=70)
                self.path.border_min[:] = -10
                self.path.border_max[:] = 10

            case 5:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(self.params.phi_rot_deg)]
                add_forward_straight(self.path, self.params,
                                     length=5, start_pose=start_pose)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.1, length=10)
                add_forward_curve(self.path, self.params, kappa=0.1, length=10)
                add_forward_straight(self.path, self.params, length=5)

                half_idx: int = int(len(self.path.border_max) / 2)
                self.path.border_max[half_idx-25:half_idx+25] -= 3.5
                self.path.border_min[half_idx-25:half_idx+25] += 3.5

                if not self.params.ignore_goal:
                    self.goal_pose = self.get_goal_pose()

            case 6:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(90)]

                full_course: bool = True
                if full_course:
                    add_forward_straight(self.path, self.params,
                                         length=5, start_pose=start_pose)
                    add_forward_curve(self.path, self.params,
                                      kappa=0.2, length=5)
                    add_forward_curve(self.path, self.params,
                                      kappa=-0.2, length=10)
                    add_forward_curve(self.path, self.params,
                                      kappa=0.2, length=5)
                    add_forward_straight(self.path, self.params, length=5)
                    start_pose = None

                add_backward_curve(self.path, self.params,
                                   kappa=-0.2, length=10, start_pose=start_pose)
                add_backward_curve(self.path, self.params,
                                   kappa=0.2, length=10)
                add_backward_straight(self.path, self.params, length=5)
                add_backward_curve(self.path, self.params, kappa=0.2, length=5)

                self.path.border_max[:] = 5.0
                self.path.border_min[:] = -5.0

                # comment in for outside goal
                self.path.border_max[-80:-70] -= 3.7
                self.path.border_min[-25:-16] += 4.0

                if not self.params.ignore_goal:
                    # outside goal
                    self.goal_pose = self.get_goal_pose(
                        d_lat=-3.0, d_long=-4.1, psi_diff=0.1)

                    # inside goal
                    # self.goal_pose = self.get_goal_pose(d_lat=3, d_long=1, psi_diff=0.6)

            case 7:
                start_pose = [self.params.offset, self.params.offset,
                              np.deg2rad(90)]
                add_forward_straight(self.path, self.params,
                                     length=2, start_pose=start_pose)
                add_forward_curve(self.path, self.params,
                                  kappa=0.2, length=5)
                add_forward_curve(self.path, self.params,
                                  kappa=-0.2, length=5)
                add_forward_curve(self.path, self.params,
                                  kappa=0.1, length=10)
                add_backward_curve(self.path, self.params,
                                   kappa=-0.1, length=5)
                add_backward_curve(self.path, self.params,
                                   kappa=-0.2, length=10)
                add_backward_straight(self.path, self.params,
                                      length=5)
                self.path.border_max[:] = 5.0
                self.path.border_min[:] = -5.0

                self.path.border_max[-50:-40] -= 4.5

                if not self.params.ignore_goal:
                    self.goal_pose = self.get_goal_pose(
                        d_lat=3.0, d_long=-6.0, psi_diff=-0.0)

            case _:
                raise Exception(
                    f"Track with number {self.params.track_switch} does not exist")

    def get_goal_pose(self, d_lat: float = -1.0, d_long: float = 2.0, psi_diff: float = -0.2):
        # set goal pose
        psi_end = self.path.phi[-1]
        x_diff = math.cos(psi_end) * d_long - math.sin(psi_end) * d_lat
        y_diff = math.sin(psi_end) * d_long + math.cos(psi_end) * d_lat
        return np.hstack([self.path.x[-1] + x_diff, self.path.y[-1] + y_diff, self.path.phi[-1] + psi_diff])

    def simulate(self):
        """
        Start simulation loop
        :return:
        """
        current_time_ros: float = 0.0

        self.path.goal_index = len(self.path.x) - 1
        self.path.prepare_path(self.cp.params)
        self.path.prepare_bounds()

        # init to prevent huge t_int values
        self.vis.setup()

        # initial values
        a_out: float = self.vehicle.u0[self.vehicle.acc_idx]
        delta_out: float = self.vehicle.u0[self.vehicle.delta_idx]
        while True:
            # stop sim on pause
            pause = self.vis.vis_step()
            if pause:
                continue

            # fixed time
            delta_ros = self.params.sim_delta_t
            current_time_ros += delta_ros

            # integrate with real vehicle model
            self.vehicle.integrate(a_out, delta_out, current_time_ros,
                                   self.params.model_errors, self.params.runtime_model_errors)
            x0_meas = self.vehicle.meas_x0()

            # trajectory planning
            trans_state: int = - \
                1 if self.vehicle.x0[self.vehicle.v_idx] < 0 else 1
            goal_pose_only: bool = False
            self.cp.plan(self.path, self.goal_pose, goal_pose_only, self.params.pose_goal_tol,
                         x0_meas.copy(), self.obstacles, current_time_ros,
                         delta_ros, self.reinit, is_automation_on=True, trans_state=trans_state)
            self.reinit = False

            v_target, a_target, delta_target, ddelta_target = self.cp.get_controls()

            # Default
            a_ctrl = a_target
            delta_ctrl = delta_target

            # No intermediate controller

            # Base case
            a_out = a_ctrl
            delta_out = delta_ctrl


if __name__ == "__main__":
    rclpy.init()

    sim_node = SimNode()
    rclpy.spin(sim_node)

    sim_node.destroy_node()
    rclpy.shutdown()
