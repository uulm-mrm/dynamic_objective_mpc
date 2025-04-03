import math
import time
from collections import deque
from typing import Optional, Final

import corridor_planning_lib.util as util
import numba
import numpy as np
from numpy.typing import NDArray

from corridor_planning_lib.data_structures import Path, GoalTolerance
from corridor_planning_lib.dead_time_comp import DeadTimeComp
from corridor_planning_lib.mpc.integrator_settings import get_integrator, get_integrator_bicycle_model
from corridor_planning_lib.mpc.ocp_settings import get_solver, get_bicycle_model, blend_weights
from corridor_planning_lib.param_setter import ParamSetter, ParamType
from corridor_planning_lib.params import Params
from corridor_planning_lib.references import References
from corridor_planning_lib.transforms import get_closest_pose, transform_pose_cart2frenet, transform_point_cart2frenet
from corridor_planning_lib.violations import ViolationsChecker
from corridor_planning_lib.controllers.pure_pursuit import PurePursuit
from corridor_planning_lib.history import History

class CorridorPlanner:
    # @util.timeme
    def __init__(self, logger, plan_period: float):
        self.logger = logger

        # vehicle info
        self.trans_state: int = 0
        self.platform_id: str = ""

        # planning
        self.res: float = 0.1
        self.plan_period: float = plan_period
        self.do_set_params: bool = False
        self.first_traj_planned: bool = False
        self.obstacles: list = []

        # maneuver to goal behind path
        self.segment_just_switched: bool = True

        # inputs
        self.path: Optional[Path] = None
        self.goal_state: Optional[NDArray] = None
        self.goal_pose_only: bool = False
        self.goal_frenet_error = np.inf * np.ones(3)
        self.goal_reached: bool = False
        self.relevant_path: Optional[Path] = None

        # timing
        self.t_plan: float = 0.0
        self.t_cycle: float = 0.0
        self.tstamp_rec: float = 0.0
        self.t_ros4vis: float = 0.0
        self.tstamp_viol_check: float = 0.0
        self.last_iter: int = 50

        # ocp and integrator
        self.model, self.constraint, self.extra_vals = get_bicycle_model()
        self.solver, self.N, self.Tf = get_solver(self.model, self.constraint)
        self.integrator_model = get_integrator_bicycle_model()
        self.integrator = get_integrator(self.integrator_model)
        self.delta_t: float = self.Tf / self.N
        self.time_tot: float = 0
        self.residuals: float = 0
        self.optimizer_error: bool = False
        self.do_reinit_by_error: bool = False

        # dimensions
        self.nx: int = self.model.x.size()[0]
        self.nx_int: int = self.integrator_model.x.size()[0]
        self.nu: int = self.model.u.size()[0]
        self.nu_int: int = self.integrator_model.u.size()[0]
        self.nx_frenet: int = self.model.x_frenet.size()[0]

        self.x0: NDArray = np.zeros(self.nx)
        self.u0: NDArray = np.zeros(self.nu)
        self.x0_meas: NDArray = np.zeros(self.nx)  # for difference between planned and actual

        # struct to save results
        self.trajX: NDArray = np.zeros((self.nx, self.N + 1))
        self.init_trajX: NDArray = np.zeros((self.nx, self.N + 1))
        self.trajU: NDArray = np.zeros((self.nu, self.N))
        self.init_trajU: NDArray = np.zeros((self.nu, self.N))
        self.t_traj_calc: float = 0.0

        # automatic indices for easier access
        self.x_idx, self.y_idx, self.phi_idx, self.v_idx, self.delta_idx, self.theta_idx = list(
            [idx for idx in range(self.nx)])
        self.acc_idx, self.ddelta_idx, self.dtheta_idx = list(
            [idx for idx in range(self.nu)])

        # dead time compensation method
        self.x0_dead_time: NDArray = np.zeros(self.nx)
        self.dead_time_comp: DeadTimeComp = DeadTimeComp(self.nu, self.plan_period)

        # parameters and references
        self.runtimes: deque = deque(maxlen=int(100))
        self.cycle_times: deque = deque(maxlen=int(100))

        self.params_set: bool = False
        self.param_setter = ParamSetter(self.model, ParamType.p, self.N)
        self.int_param_setter = ParamSetter(self.integrator_model, ParamType.p, self.N)
        self.params = Params()
        assert (len(self.params.solver.q) == self.nx)
        assert (len(self.params.solver.r) == self.nu)
        assert (len(self.params.solver.q_frenet) == self.nx_frenet)
        self.references = References(self.model.params.N_refpath, self.res)

        # (re)initialization
        self.do_reinit: bool = False
        self.t_last_reinit: float = -1e3
        self.t_reinit_period: float = 0.250
        self.pure_pursuit = PurePursuit(wb=self.params.vehicle.wb, res=self.res)

        # Debug and visualization
        self.history = History()
        self.runtimes: deque = deque(maxlen=int(100))
        self.cycle_times: deque = deque(maxlen=int(100))
        self.violation_checker = ViolationsChecker(logger)

        # will be set from node
        self.do_separate_planning: bool = False
        self.do_hard_switch: bool = False


    # @util.timeme
    def set_logger(self, logger) -> None:
        self.logger = logger

    # @util.timeme
    def get_next_target_values(self) -> tuple[float, float, float, float]:
        dead_time: float = self.params.vehicle.get_min_dead_time()
        if not self.params.mode.is_trajectory_planner and self.params.mode.mpc_dead_time_comp and dead_time > 0.0:
            idx: int = 0
            v_target: float = self.trajX[self.v_idx, idx]
            delta_target: float = self.trajX[self.delta_idx, idx]

            a_target: float = self.trajU[self.acc_idx, idx]
            delta_dot_target: float = self.trajU[self.ddelta_idx, idx]
        else:
            idx: int = 0
            v_target: float = self.trajX[self.v_idx, idx]
            delta_target: float = self.trajX[self.delta_idx, idx]

            a_target: float = self.trajU[self.acc_idx, idx]
            delta_dot_target: float = self.trajU[self.ddelta_idx, idx]

        return v_target, a_target, delta_target, delta_dot_target

    # @util.timeme
    def get_controls(self) -> tuple[float, float, float, float]:

        v_target, a_target, delta_target, delta_dot_target = self.get_next_target_values()

        # v_target = max(-15.0, min(v_target, 15.0))

        # clip controls
        a_target = max(-self.params.solver.along_max,
                       min(a_target, self.params.solver.along_max))
        delta_target = max(-self.params.vehicle.delta_max,
                           min(delta_target, self.params.vehicle.delta_max))

        delta_dot_target = max(-self.params.solver.ddelta_max,
                               min(delta_dot_target, self.params.solver.ddelta_max))

        return v_target, a_target, delta_target, delta_dot_target

    # @util.timeme
    def is_traj_violated(self, x1: NDArray, x2: NDArray):
        """
        Analyze if two states x1 and x2 differ too much
        :return:
        """

        x_diff: float = x1[self.x_idx] - x2[self.x_idx]
        y_diff: float = x1[self.y_idx] - x2[self.y_idx]

        dist: float = math.sqrt(x_diff ** 2 + y_diff ** 2)
        v_diff = abs(x1[self.v_idx] - x2[self.v_idx])

        is_v_violated: bool = False
        is_dist_violated: bool = False
        if v_diff > self.params.general.v_diff_max:
            self.logger.info("Velocities differ too much, reset to x0_meas")
            is_v_violated = True
        if dist > self.params.general.cart_diff_max:
            self.logger.info("distance to traj differs too much, reset to x0_meas")
            is_dist_violated = True

        return is_v_violated or is_dist_violated

    # @util.timeme
    def predict_x0(self, x0_meas: NDArray):
        if not self.first_traj_planned:
            return x0_meas
        self.integrator.set("T", self.t_cycle)
        x0 = self.integrator.simulate(self.x0, self.u0)
        return x0

    # @util.timeme
    def set_x0(self, x0_meas: NDArray, is_automation_on: bool, new_path_received: bool):
        """
        Take x0_meas if used as a controller and x_pred if used as a trajectory planner, but reset to x_meas if too far away.

        :param is_automation_on:
        :param x0_meas:
        :return:
        """
        if new_path_received:
            self.x0 = x0_meas.copy()

        if (self.params.mode.is_trajectory_planner and
                not self.is_traj_violated(self.x0, x0_meas) and is_automation_on):
            self.x0 = self.predict_x0(x0_meas)
        else:
            self.x0 = x0_meas.copy()

    # @util.timeme
    def plan(self, path: Path, goal_pose: NDArray, goal_pose_only, pose_goal_tol: GoalTolerance,
             x0_meas: NDArray, obstacles: list, t_ros: float,
             t_elapsed: float, reinit: bool, is_automation_on: bool, trans_state: int) -> None:
        """
        Main function
        :param goal_pose:
        :param goal_pose_only:
        :param is_automation_on:
        :param trans_state:
        :param t_elapsed:
        :param pose_goal_tol:
        :param path:
        :param x0_meas:
        :param obstacles:
        :param t_ros:
        :param reinit:
        :return:
        """

        t_start = time.perf_counter()
        self.t_ros4vis = t_ros  # for vis

        # insert data
        self.t_cycle = t_elapsed
        new_path_received: bool = True if (self.path is not None and path.path_id != self.path.path_id) else False
        self.path = path

        self.goal_pose_only = goal_pose_only

        if goal_pose is not None:
            self.goal_state = np.array([goal_pose[0], goal_pose[1], goal_pose[2], 0, 0, 0])
        else:
            self.goal_state = None

        self.set_x0(x0_meas, is_automation_on, new_path_received)
        self.x0_meas = x0_meas
        self.obstacles = obstacles
        self.trans_state = trans_state

        if reinit or new_path_received:
            self.do_reinit = True

        if self.path is None:
            return

        self.goal_reached = self.is_goal_reached(pose_goal_tol)

        self.project_on_path()

        on_new_segment: bool = self.path.set_relevant_idx()
        if on_new_segment:
            self.logger.info("On new segment")
            self.do_reinit = True

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Planning~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.prep_mpc(t_ros)

        self.run_mpc()

        self.extract_traj(t_ros)

        t_end = time.perf_counter()
        self.t_plan = t_end - t_start

        # save history data
        self.check_sim_data(t_ros)

        self.runtimes.append(self.t_plan)
        self.cycle_times.append(self.t_cycle)

    def get_dist_xf_to_goal(self, stop_idx: int):
        last_x: float = self.trajX[self.x_idx, -1]
        last_y: float = self.trajX[self.y_idx, -1]
        if self.goal_state is not None:
            goal_x: float = self.goal_state[0]
            goal_y: float = self.goal_state[1]
        else:
            goal_x: float = self.path.x[stop_idx]
            goal_y: float = self.path.y[stop_idx]

        return math.sqrt((goal_x - last_x) ** 2 + (goal_y - last_y) ** 2)

    def get_theta_dist_to_path_end(self) -> float:
        theta: float = self.x0[self.theta_idx]
        theta_f: float = self.references.theta_f
        return theta_f - theta

    # @util.timeme
    def blend_weight_by_ego(self, w_default, w_goal) -> NDArray:
        """
        :return:
        """
        dist_goal_path_end: float = self.get_theta_dist_to_path_end()

        return blend_weights(dist_goal_path_end, np.asarray(w_default), np.asarray(w_goal),
                      self.params.solver.ego_sig_center, self.params.solver.ego_sig_steepness)

    # @util.timeme
    def project_goal_pose_on_path(self) -> Optional[int]:
        if self.goal_state is not None:
            goal_pose_idx, _, _, _ = get_closest_pose(self.goal_state[0], self.goal_state[1], self.goal_state[2],
                                                      self.path.x, self.path.y, self.path.phi, angle_lim=math.pi / 4,
                                                      angle_weight=0.0)
            return goal_pose_idx
        return None

    def set_cartesian_only_weights(self):
        self.param_setter.set("r", self.params.solver.r_goal)
        self.param_setter.set("r_e", self.params.solver.r_goal)
        self.param_setter.set("q_frenet", 0.0)
        self.param_setter.set("q_lat", 0.0)
        self.param_setter.set("q_bound", 0.0)
        self.param_setter.set("q_v_prox", 0.0)
        self.param_setter.set("q_right", 0.0)
        self.param_setter.set("q_frenet_e", 0.0)
        self.param_setter.set("q_frenet_outside", 0.0)
        self.param_setter.set("q_frenet_N", 0.0)
        self.param_setter.set("q_N", self.params.solver.q_N)
        self.param_setter.set("x_N", self.goal_state)

    def get_goal_pos_behind_path(self):
        if self.goal_state is not None:
            x = self.goal_state[self.x_idx]
            y = self.goal_state[self.y_idx]
            phi = self.goal_state[self.phi_idx]
            xr = self.path.get_at_end("x")
            yr = self.path.get_at_end("y")
            phir = self.path.get_at_end("phi")
            return util.get_frenet_differences(x, y, phi, xr, yr, phir)

        return 0.0, 0.0, 0.0

    def set_params(self, x0: NDArray):
        """
        Set params in solver
        :return:
        """
        delta_t_param, new_timestamp = util.set_and_measure(self.params.t_params_set)
        param_needed_by_time = True if delta_t_param > self.params.delta_t_set_params else False

        if not param_needed_by_time and not self.do_set_params:
            return

        # reset states
        self.do_set_params = False
        self.params.t_params_set = new_timestamp

        # save which params were used
        self.path.set_params_used()

        # Check if goal pose is relevant if it exists segment
        goal_idx: Optional[int] = self.project_goal_pose_on_path()
        goal_pose_relevant: bool = True if self.path.is_idx_relevant(goal_idx) else False
        # Find index to stop vehicle
        stop_idx: int = self.path.goal_index if goal_idx is None else min(self.path.goal_index, goal_idx)
        stop_idx = min(stop_idx, self.path.ego_data.segment_end_idx)

        # Place goal a few meters before
        if self.do_separate_planning and self.path.ego_data.segment_end_idx - goal_idx > 1:
            stop_idx -= 70

        # measure distance of goal behind path end to adapt v limits at path end
        theta_goal_behind: float = 0.0
        if goal_pose_relevant:
            theta_goal_behind, _, _ = self.get_goal_pos_behind_path()
            theta_goal_behind = abs(min(theta_goal_behind, 0.0))

        self.references.set(self.path, stop_idx, theta_goal_behind,
                            x0[self.v_idx], x0[self.theta_idx],
                            self.params)

        if self.goal_pose_only:
            self.references.v_max = 0.5

        # Get maximum theta depending on the position of the goal pose
        self.path.unwrap_angles(x0[self.phi_idx])

        # references for contouring
        direction: int = 1 if self.path.is_segment_forward() else -1
        self.param_setter.set("direction", direction)
        self.param_setter.set("theta_0", self.references.theta_0)
        self.param_setter.set("theta_f", self.references.theta_f)
        self.param_setter.set("x_ref", self.references.x)
        self.param_setter.set("y_ref", self.references.y)
        self.param_setter.set("phi_ref", self.references.phi)
        self.param_setter.set("kappa_ref", self.references.kappa)
        self.param_setter.set("v_max_ref", self.references.v_max)
        self.param_setter.set("v_min_ref", -self.references.v_max)
        self.param_setter.set("dtheta_max_ref", self.references.dtheta_max)
        self.param_setter.set("n_min_ref", self.references.n_min)
        self.param_setter.set("n_max_ref", self.references.n_max)

        # set vehicle metadata
        self.param_setter.set("lf", self.params.vehicle.lf)
        self.param_setter.set("lb", self.params.vehicle.lb)
        self.param_setter.set("width", self.params.vehicle.width)
        self.param_setter.set("delta_max", self.params.vehicle.delta_max)
        self.param_setter.set("along_max", self.params.solver.along_max)
        self.param_setter.set("ddelta_max", self.params.solver.ddelta_max)
        self.param_setter.set("wb", self.params.vehicle.wb)

        # blending params
        self.param_setter.set("sig_center", self.params.solver.sig_center)
        self.param_setter.set("sig_steepness", self.params.solver.sig_steepness)

        # ego independant weights
        self.param_setter.set("r", self.params.solver.r)
        self.param_setter.set("r_e", self.params.solver.r_e)
        self.param_setter.set("q", self.params.solver.q)
        self.param_setter.set("q_e", self.params.solver.q_e)
        self.param_setter.set("q_frenet", self.params.solver.q_frenet)
        self.param_setter.set("q_lat", self.params.solver.q_lat)
        self.param_setter.set("q_v_viol", self.params.solver.q_v_viol)
        self.param_setter.set("q_bound", self.params.solver.q_bound)
        self.param_setter.set("q_right", self.params.solver.q_right)
        self.param_setter.set("q_v_prox", self.params.solver.q_v_prox)
        self.param_setter.set("bound_dist", self.params.solver.bound_dist)
        self.param_setter.set("bound_dist_e", self.params.solver.bound_dist_e)
        self.param_setter.set("v_prox_bound_dist", self.params.solver.v_prox_bound_dist)

        # ego dependant weights and params
        # defaults
        self.param_setter.set("q_frenet_e", self.params.solver.q_frenet_e)
        self.param_setter.set("q_frenet_outside", self.params.solver.q_frenet_e)  # normally not possible but for safety
        self.param_setter.set("q_frenet_N", self.params.solver.q_frenet_N)
        self.param_setter.set("x_N", 0.0)
        self.param_setter.set("q_N", 0.0)

        # Handling of goal poses
        if self.goal_state is not None and goal_pose_relevant:
            # First baseline
            # Activate separated controller:
            if self.do_separate_planning:
                if self.path.ego_data.max_ego_idx >= stop_idx:
                    # only cart mpc on path end
                    self.set_cartesian_only_weights()

            # Second baseline
            elif self.do_hard_switch:
                if self.get_theta_dist_to_path_end() < self.params.solver.ego_sig_center:
                    # Activate cartesian only with hard switch
                    self.set_cartesian_only_weights()

            # Standard case: Dynamic Weight Allocation
            # goal maneuvering
            else:
                self.param_setter.set("x_N", self.goal_state)
                self.param_setter.set("r_e", self.params.solver.r_goal)
                # keep frenet coordinates in corridor...
                self.param_setter.set("q_frenet_e", self.params.solver.q_frenet_goal)
                # ...but ignore them outside
                self.param_setter.set("q_frenet_outside", self.params.solver.q_frenet_outside)

                # Activate weight for goal pose by ego positon
                q_N = self.blend_weight_by_ego(self.params.solver.q, self.params.solver.q_N)
                self.param_setter.set("q_N", q_N)

        # Cartesian only!
        if self.goal_pose_only and self.goal_state is not None:
            self.set_cartesian_only_weights()

        # React to moving objects: time dependant parameters
        theta_min: float = 0.0
        self.param_setter.set("theta_min", theta_min)
        theta_max: float = self.references.theta_f+self.res
        self.param_setter.set("theta_max", theta_max)

        # set param arr in solver
        for i_N in range(self.N + 1):
            # insert values
            self.param_setter.save_for_debug(i_N)
            self.solver.set(i_N, "p", self.param_setter.param_arr)

    # @util.timeme
    def check_sim_data(self, t_ros: float) -> None:
        """
        Record sim data
        :return:
        """
        # check for violations
        t_elapsed, t_now = util.set_and_measure(self.tstamp_viol_check)
        if t_elapsed > self.delta_t:
            self.tstamp_viol_check = t_now
            self.violation_checker.update(self.trajX, self.trajU, self.param_setter, self.extra_vals)
            if self.violation_checker.are_constraints_violated():
                self.logger.error("Violation triggered reinit of solver")
                self.do_reinit = True

        self.rec_sim_data(t_ros)

    # @util.timeme
    def rec_sim_data(self, t_ros: float) -> None:
        """
        Record data vor visualization
        :return:
        """

        t_elapsed = t_ros - self.tstamp_rec
        force: bool = t_elapsed > self.delta_t

        self.history.check_len()

        if self.history.sim_x:
            x_diff = self.x0[self.x_idx] - self.history.sim_x[-1]
            y_diff = self.x0[self.y_idx] - self.history.sim_y[-1]

        if force or not self.history.sim_x or math.hypot(x_diff, y_diff) >= self.res/2:
            # store relative time since start
            if not self.history.sim_t:
                self.history.first_t = t_ros
            self.history.sim_t.append(t_ros-self.history.first_t)

            self.history.sim_x.append(self.x0[self.x_idx])
            self.history.sim_y.append(self.x0[self.y_idx])
            self.history.sim_phi.append(self.x0[self.phi_idx])
            self.history.sim_v.append(self.x0[self.v_idx])
            self.history.sim_a.append(self.u0[self.acc_idx])

            kappa = math.tan(self.x0[self.delta_idx]) / self.params.vehicle.wb
            a_lat = self.x0[self.v_idx] ** 2 * kappa
            self.history.sim_a_lat.append(a_lat)
            self.history.sim_delta.append(self.x0[self.delta_idx])

            self.history.sim_ddelta.append(self.u0[self.ddelta_idx])
            self.history.sim_thetadot.append(self.u0[self.dtheta_idx])

            self.tstamp_rec = t_ros

    def integrate_and_project(self, x0: NDArray, u0: NDArray) -> [NDArray, NDArray]:
        """

        :param x0:
        :param u0:
        :return:
        """
        # get next state
        self.integrator.set("T", self.delta_t)
        x_next = self.integrator.simulate(x0, u0)

        # set theta and calculate dtheta
        x_next[self.theta_idx], _ = transform_point_cart2frenet(x_next[self.x_idx], x_next[self.y_idx],
                                                               self.path.get_segment("s"),
                                                               self.path.get_segment("x"),
                                                               self.path.get_segment("y"),
                                                               self.path.get_segment("phi"))

        # clip theta at path end
        theta_max: float = self.references.theta_f+self.res
        x_next[self.theta_idx] = min(x_next[self.theta_idx], theta_max)

        u0[self.dtheta_idx] = (x_next[self.theta_idx] - x0[self.theta_idx]) / self.delta_t

        return x_next, u0

    def init_zeros(self, x0: NDArray, start_idx: int = 0):
        """
        Reinit with zeros, only valid at low speeds or last resort reinit
        :param x0:
        :param start_idx:
        :return:
        """
        # start values
        u_init = np.zeros(self.nu_int)

        # initialize
        for i in range(start_idx, self.N):
            x_next, u0 = self.integrate_and_project(x0, u_init.copy())

            # set trajectory
            self.init_trajX[:, i] = x0
            self.init_trajU[:, i] = u0

            x0 = x_next.copy()

        # set last value
        self.init_trajX[:, self.N] = x0

    def get_init_acc(self, x0: NDArray, is_forward_segment: bool) -> float:
        """
        Get speed and acc from previously calculated v_init
        :param x0:
        :param is_forward_segment:
        :return:
        """
        v_is: float = x0[self.v_idx]
        v_is = max(v_is, 0.0) if is_forward_segment else min(v_is, 0.0)
        v_next: float = self.references.get_at_theta("v_init", x0[self.theta_idx])
        v_next = v_next if is_forward_segment else -v_next
        acc: float = (v_next - v_is) / self.delta_t
        acc = np.clip(acc, -self.params.general.init_acc, self.params.general.init_acc)
        return acc

    def get_init_ddelta(self, x0, x_vec, y_vec, phi_vec, is_forward_segment: bool) -> float:
        """
        Get delta from pure pursuit controller and clip the resulting values to the maximum possible steering angle
        and the maximum possible steer rate
        :param x0:
        :param x_vec:
        :param y_vec:
        :param phi_vec:
        :param is_forward_segment:
        :return:
        """
        # get steer by pure pursuit
        next_delta: float = self.pure_pursuit.get_delta(x0[self.x_idx],
                                                        x0[self.y_idx],
                                                        x0[self.phi_idx],
                                                        x0[self.v_idx],
                                                        x_vec, y_vec, phi_vec, is_forward_segment)
        # clip with system dynamic
        next_delta = np.clip(next_delta, -self.params.vehicle.delta_max, self.params.vehicle.delta_max)
        # correct ddelta
        is_delta: float = x0[self.delta_idx]
        u_ddelta: float = (next_delta - is_delta) / self.delta_t
        # clip with system dynamic
        return np.clip(u_ddelta, -self.params.solver.ddelta_max, self.params.solver.ddelta_max)

    def driving_reinit(self, x0: NDArray, start_idx: int = 0):
        """
        Reinit with beginning to drive
        :param x0:
        :param start_idx:
        :return:
        """


        # get forward movement by fixed acceleration
        is_forward_segment: bool = self.path.is_segment_forward()

        for i in range(start_idx, self.N):
            theta: float = x0[self.theta_idx]

            assert not np.isnan(x0[self.v_idx]), f"x0 {x0}"
            assert not np.isnan(theta), f"v {theta}"
            x_vec = self.path.get_forward_from_theta("x", theta, self.res)
            y_vec = self.path.get_forward_from_theta("y", theta, self.res)
            phi_vec = self.path.get_forward_from_theta("phi", theta, self.res)

            # get inputs with current state
            acc_appl: float = self.get_init_acc(x0, is_forward_segment)
            ddelta_appl: float = self.get_init_ddelta(x0, x_vec, y_vec, phi_vec, is_forward_segment)
            assert not np.isnan(acc_appl)
            assert not np.isnan(ddelta_appl)

            # integrate again with ddelta and acc
            u_int = np.array([acc_appl, ddelta_appl, 0.0])
            x_next, u0 = self.integrate_and_project(x0, u_int)

            # set trajectory
            self.init_trajX[:, i] = x0
            self.init_trajU[:, i] = u0

            x0 = x_next.copy()

        # set last value
        self.init_trajX[:, self.N] = x0

    def fill_steer(self, x0: NDArray) -> [float, int]:
        """
        Fill solver with steer change to steer defined by path curvature
        :return:
        """
        path_steer: float = np.arctan(self.path.get_at_ego("kappa") * self.params.vehicle.wb)
        path_steer = np.clip(path_steer, -self.params.vehicle.delta_max, self.params.vehicle.delta_max)
        is_steer: float = x0[self.delta_idx]
        steer_dir: int = 1 if path_steer > is_steer else -1
        ddelta: float = self.params.solver.ddelta_max * 0.25 * steer_dir

        # set ddelta to rotate
        u_init = np.zeros(self.nu_int)
        u_init[self.ddelta_idx] = ddelta

        if path_steer == is_steer:
            return x0, 0

        # fill with steer
        steer_reached: bool = False
        stage_idx: int = 0
        while not steer_reached:
            x_next, u0 = self.integrate_and_project(x0, u_init)

            # set trajectory
            self.init_trajX[:, stage_idx] = x0
            self.init_trajU[:, stage_idx] = u0

            x0 = x_next.copy()
            steer_reached = x0[self.delta_idx] > path_steer if ddelta > 0 else x0[self.delta_idx] < path_steer
            stage_idx += 1

            if stage_idx == self.N:
                break

        x0[self.delta_idx] = path_steer

        # correct u
        before_last_steer: float = self.solver.get(stage_idx - 1, "x")[self.delta_idx]
        last_ddelta = path_steer - before_last_steer
        u_init[self.ddelta_idx] = last_ddelta

        self.init_trajU[:, stage_idx-1] = u_init

        return x0, stage_idx

    def maneuver_init(self, x0: NDArray):
        """
        reinit with trajectory to begin moving along path, correct delta if curvature is wrong
        :return:
        """
        # correct delta
        x, stage_idx = self.fill_steer(x0)
        self.driving_reinit(x, stage_idx)

    def reset_internals(self):
        """
        Reset all internal values
        :return:
        """
        for i in range(self.N+1):
            tmp = self.solver.get(i, "lam")
            tmp[:] = 0.0
            self.solver.set(i, "lam", tmp)
            tmp = self.solver.get(i, "p")
            tmp[:] = 0.0
            self.solver.set(i, "p", tmp)

            if i < self.N:
                tmp = self.solver.get(i, "pi")
                tmp[:] = 0.0
                self.solver.set(i, "pi", tmp)
    def reset_solver(self):
        self.solver.reset(reset_qp_solver_mem=1)
        self.reset_internals()


    def apply_init(self):
        """
        Insert init traj into solver
        :return:
        """
        self.reset_solver()

        for i in range(self.N):
            self.solver.set(i, "x", self.init_trajX[:, i])
            self.solver.set(i, "u", self.init_trajU[:, i])

        self.solver.set(self.N, "x", self.init_trajX[:, self.N])
        self.logger.warning("Init applied")

    def init_solver(self, x0: NDArray, t_ros: float) -> None:
        """
        Init solver zeros as input
        :return:
        """

        if not self.do_reinit and not self.do_reinit_by_error:
            return

        if self.do_reinit_by_error and not self.do_reinit:
            # only reinit all T, to give the solver time to converge
            t_diff: float = t_ros - self.t_last_reinit
            if t_diff < self.t_reinit_period:
                return

        if abs(x0[self.v_idx]) < 0.2:
            self.maneuver_init(x0)
        else:
            self.driving_reinit(x0)

        # set solver values
        self.apply_init()

        self.t_last_reinit = t_ros
        self.do_reinit = False
        self.do_reinit_by_error = False
        self.do_set_params = True

    # @util.timeme
    def is_goal_reached(self, pose_tol: GoalTolerance) -> bool:

        if self.path is None:  # or tol is None:
            return False

        # v = self.x0[self.v_idx]

        x = self.x0[self.x_idx]
        y = self.x0[self.y_idx]
        phi = self.x0[self.phi_idx]

        # Reference
        if self.goal_state is not None:
            xr = self.goal_state[0]
            yr = self.goal_state[1]
            phir = self.goal_state[2]
            at_goal, (e_lag, e_cont, e_phi) = pose_tol.is_within_tol(x, y, phi, xr, yr, phir)
        else:
            xr = self.path.get_at_goal("x")
            yr = self.path.get_at_goal("y")
            phir = self.path.get_at_goal("phi")
            e_lag, e_cont, e_phi = util.get_frenet_differences(x, y, phi, xr, yr, phir)
            at_goal = self.path.is_ego_at_end()

        self.goal_frenet_error = np.asarray([e_lag, e_cont, np.deg2rad(e_phi)])

        if at_goal:  # and abs(self.x0[self.v_idx]) < 0.1:
            self.logger.info("Goal reached", throttle_duration_sec=2)
            self.logger.info("\te_lag = {:.6f} m".format(e_lag), throttle_duration_sec=2)
            self.logger.info("\te_cont = {:.6f} m".format(e_cont), throttle_duration_sec=2)
            self.logger.info("\te_phi = {:.6f} °".format(e_phi), throttle_duration_sec=2)
            return True
        return False

    # @util.timeme
    def project_on_path(self):
        """
        Projects the ego on the path.
        :return:
        """
        theta_ref = self.path.get_segment("s")
        self.x0[self.theta_idx], _ = transform_point_cart2frenet(self.x0[self.x_idx], self.x0[self.y_idx],
                                                                   theta_ref,
                                                                   self.path.get_segment("x"),
                                                                   self.path.get_segment("y"),
                                                                   self.path.get_segment("phi"))

        # clip to segment
        self.x0[self.theta_idx] = np.clip(self.x0[self.theta_idx], theta_ref[0], theta_ref[-1])

        # clip theta at path end
        theta_max: float = self.references.theta_f+self.res
        self.x0[self.theta_idx] = min(self.x0[self.theta_idx], theta_max)

        # get index
        ego_idx: int = int(self.x0[self.theta_idx] / self.res)
        self.references.param_offset = ego_idx * self.res - self.x0[self.theta_idx]
        self.path.ego_data.ego_idx = np.clip(ego_idx, self.path.ego_data.segment_start_idx,
                                             self.path.ego_data.segment_end_idx)

        # if near dir change, jump to change and reinit
        check_idx: int = self.params.general.switch_dist_idx
        if abs(self.x0[self.v_idx]) < 0.1:
            if self.path.shift_ego_near_switch(check_idx):
                self.logger.warning("Segment switched, manually")

        # set maximum reached index to track made progress on path
        self.path.ego_data.max_ego_idx = max(self.path.ego_data.max_ego_idx, ego_idx)


    def get_dead_time_x0(self):
        """
        Set predicted x0
        :return:
        """
        # dead time compensation
        dead_time: float = self.params.vehicle.get_min_dead_time()

        x0 = self.dead_time_comp.get_next_x0(self.x0.copy(), self.u0.copy(), self.integrator, dead_time, self.trans_state, self.platform_id)

        if np.isnan(x0).any():
            self.logger.error("x0 pred was nan!!!!!")
            x0 = self.x0.copy()

        # set correct theta
        theta_ref = self.path.get_segment("s")
        x0[self.theta_idx], _, _ = transform_pose_cart2frenet(x0[self.x_idx], x0[self.y_idx],
                                                              x0[self.phi_idx],
                                                              theta_ref,
                                                              self.path.get_segment("x"),
                                                              self.path.get_segment("y"),
                                                              self.path.get_segment("phi"))

        # clip theta at path end
        theta_max: float = self.references.theta_f+self.res
        x0[self.theta_idx] = min(self.x0[self.theta_idx], theta_max)

        # clip to segment
        x0[self.theta_idx] = np.clip(x0[self.theta_idx], theta_ref[0], theta_ref[-1])
        return x0

    def correct_traj_for_path(self):
        """
        If new path was received, modify theta to be valid on the current path again
        :return:
        """
        # correct theta to be on current path
        first_theta = self.solver.get(0, "x")[self.theta_idx]
        for i in range(self.N+1):
            # correct solver trajectory
            x = self.solver.get(i, "x")
            x[self.theta_idx] -= first_theta
            self.solver.set(i, "x", x)

            # correct also current trajectory
            self.trajX[:, i] = x

    # @util.timeme
    def prep_mpc(self, t_ros: float) -> None:
        """
        Setting parameter references and sets initial state
        :return:
        """

        # Set integrator model
        self.int_param_setter.set("wb", self.params.vehicle.wb)
        self.integrator.set("p", self.int_param_setter.param_arr)

        x0 = self.get_dead_time_x0() if self.params.mode.mpc_dead_time_comp and not self.params.mode.is_trajectory_planner else self.x0.copy()
        self.x0_dead_time = x0.copy()  # debug

        self.init_solver(x0, t_ros)

        # set start constraints
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set data
        self.set_params(x0)

    # @util.timeme
    def run_mpc(self):
        """
        Run solver
        :return:
        """

        # solve ocp
        status = self.solver.solve()
        if status != 0:
            # Acados status
            # 0 – success
            # 1 – failure
            # 2 – maximum number of iterations reached
            # 3 – minimum step size in QP solver reached
            # 4 – qp solver failed
            if status == 4:
                self.logger.error("Optimizer error 4: Reinit solver!")  # , throttle_duration_sec=1
                self.do_reinit_by_error = True
                self.optimizer_error = True
            if status == 1:
                self.logger.error("Optimizer error 1: Reinit solver!")  # , throttle_duration_sec=1
                self.do_reinit_by_error = True
                self.optimizer_error = True
        else:
            self.optimizer_error = False

        """
        res_stat: stationary residual
        res_eq: residual wrt equality constraints (dynamics)
        res_ineq: residual wrt inequality constraints (constraints)
        res_comp: residual wrt complementarity conditions
        """
        self.residuals = self.solver.get_residuals(recompute=False)
        """
        time_tot: total CPU time previous call
        time_lin: CPU time for linearization
        time_sim: CPU time for integrator
        time_sim_ad: CPU time for integrator contribution of external function calls
        time_sim_la: CPU time for integrator contribution of linear algebra
        time_qp: CPU time qp solution
        time_qp_solver_call: CPU time inside qp solver (without converting the QP)
        time_qp_xcond: time_glob: CPU time globalization
        time_solution_sensitivities: CPU time for previous call to eval_param_sens
        time_reg: CPU time regularization
        sqp_iter: number of SQP iterations
        qp_stat: status of QP solver
        qp_iter: vector of QP iterations for last SQP call
        statistics: table with info about last iteration
        stat_m: number of rows in statistics matrix
        stat_n: number of columns in statistics matrix
        residuals: residuals of last iterate
        alpha: step sizes of SQP iterations
        """
        self.time_tot = self.solver.get_stats("time_tot")

        # self.ocp_solver.print_statistics()

    # @util.timeme
    def extract_traj(self, t_ros: float):
        """
        Extract states, inputs and algebraic states
        :return:
        """

        # get trajectory
        for n in range(self.N):
            self.trajX[:, n] = self.solver.get(n, "x")
            self.trajU[:, n] = self.solver.get(n, "u")

        self.trajX[:, self.N] = self.solver.get(self.N, "x")
        self.t_traj_calc = t_ros

        # get input for next prediction and control vars
        self.u0 = self.trajU[:, 0]
        self.first_traj_planned = True

