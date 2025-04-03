import math
import subprocess
from datetime import datetime
from typing import Optional

import corridor_planning_lib.util as util
import imviz as viz
import matplotlib
import numpy as np
from numpy.typing import NDArray

import corridor_planning_lib.datasaver as data_saver
from corridor_planning_lib.transforms import transform_point_cart2frenet, transform_pose_frenet2cart_scalar

cmap = matplotlib.colormaps.get_cmap('turbo')

RED = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392]
GREEN = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313]
DARKGREEN = [0.1, 0.27, 0.1, 1.0]
BLUE = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765]
ORANGE = [1.0, 0.4980392156862745, 0.054901960784313725]


class VisModule:
    def __init__(self, cpNode):
        self.dockspace_ready: bool = False

        # get data
        self.cpNode = cpNode
        self.cp = cpNode.cp

        self.test_point = np.array([0, 0])

        # handling
        self.t_last_vis: float = -1.0
        self.delta_t_vis: float = 0.0
        self.fps: float = 15.0
        self.vis_period = 1 / self.fps
        self.map_locked: bool = True
        self.initialized: bool = False
        self.debug_vis_visible: bool = False

        self.follow_zoom: float = 3.5  # the higher, this value the further away is the view
        self.follow_vehicle: bool = False
        self.map_context_menu_open: bool = False

        self.redraw_boundaries: bool = True
        self.x_bound_left: Optional[NDArray] = None
        self.y_bound_left: Optional[NDArray] = None
        self.x_bound_right: Optional[NDArray] = None
        self.y_bound_right: Optional[NDArray] = None

        self.smooth_x_bound_left: Optional[NDArray] = None
        self.smooth_y_bound_left: Optional[NDArray] = None
        self.smooth_x_bound_right: Optional[NDArray] = None
        self.smooth_y_bound_right: Optional[NDArray] = None

        self.map_plot_size: tuple = (500, 500)

        # button control
        self.traj_details_visible: bool = True
        self.show_advanced_details: bool = False
        self.path_details_visible: bool = False
        self.param_in_map_visible: bool = False
        self.references_in_map_visible: bool = False
        self.n_true_visible: bool = False
        self.frenet_approx_visible: bool = False
        self.ellipsis_values_visible: bool = False
        self.runtimes_visible: bool = False
        self.debug_visible: bool = False
        self.param_visible: bool = False

        self.v_max_cm: float = 5
        self.pause: bool = False
        self.one_frame: bool = False
        self.create_plots: bool = False
        self.force_plot_creation: bool = False
        self.plots_path: str = "/home/schumann/mrm/projects/sandboxes/aduulm_sandbox/src/corridor_planning/publications/high_prec/plots"

        # Recording Params
        self.record_window: bool = True
        self.record: bool = False
        self.proc: Optional[subprocess.Popen] = None
        self.rec_process_started: bool = False

        # visualization data
        # trajectory
        self.delta_t: float = self.cp.Tf / self.cp.N
        self.t: NDArray = np.arange(0, self.cp.Tf, self.delta_t)
        # simulation

        self.t_sim: Optional[NDArray] = None

    def enable_follow_veh(self):
        viz.push_override_id(viz.get_plot_id())
        if viz.begin_popup("##PlotContext"):

            # if not self.map_context_menu_open:
            self.map_context_menu_open = True

            if viz.menu_item("Follow Vehicle"):
                self.follow_vehicle = True
                self.map_context_menu_open = False

            viz.end_menu()
        viz.pop_id()

    def enable_and_follow_vehicle(self):
        self.enable_follow_veh()

        if self.follow_vehicle:
            d = 10 * self.follow_zoom
            aspect_ratio = self.map_plot_size[1] / self.map_plot_size[0]
            plot_x_min = self.cp.x0[0] - d
            plot_x_max = self.cp.x0[0] + d
            plot_y_min = self.cp.x0[1] - d * aspect_ratio
            plot_y_max = self.cp.x0[1] + d * aspect_ratio

            viz.setup_axis_limits(viz.Axis.X1, plot_x_min, plot_x_max, flags=viz.PlotCond.ALWAYS)
            viz.setup_axis_limits(viz.Axis.Y1, plot_y_min, plot_y_max, flags=viz.PlotCond.ALWAYS)

            self.map_plot_size = viz.get_plot_size()
        else:
            if not self.map_locked or viz.get_scroll_events() or any(viz.get_mouse_drag_delta()):
                viz.setup_axis(viz.Axis.X1, "x in m")
                viz.setup_axis(viz.Axis.Y1, "y in m")
                self.map_locked = False
            else:
                viz.setup_axis(viz.Axis.X1, "x in m", flags=viz.PlotAxisFlags.AUTO_FIT)
                viz.setup_axis(viz.Axis.Y1, "y in m", flags=viz.PlotAxisFlags.AUTO_FIT)

        if viz.is_plot_hovered():
            for se in viz.get_scroll_events():
                if se.yoffset < 0:
                    self.follow_zoom *= abs(se.yoffset) * 1.1
                elif se.yoffset > 0:
                    self.follow_zoom /= abs(se.yoffset) * 1.1

            if not (viz.get_mouse_drag_delta() == 0.0).all():
                self.follow_vehicle = False

    def show_map(self):
        window_title = "map"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            if viz.begin_plot(window_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.EQUAL | viz.PlotFlags.NO_LEGEND):

                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                self.enable_and_follow_vehicle()

                try:
                    x0_new = viz.drag_point("drag_car", self.cpNode.vehicle.x0[0:2], radius=10.0, color=(0, 0, 0, 0))
                    if not np.equal(x0_new, self.cpNode.vehicle.x0[0:2]).all():
                        self.cpNode.vehicle.x0[3] = 0.0
                        self.cp.x0[3] = 0.0
                    self.cpNode.vehicle.x0[0:2] = x0_new
                except AttributeError:
                    pass

                if self.x_bound_left is not None:
                    viz.plot(self.x_bound_left, self.y_bound_left, fmt="-o", marker_size=2, color="black",
                             label="boundary_left")
                    viz.plot(self.x_bound_right, self.y_bound_right, fmt="-o", marker_size=2, color="black",
                             label="boundary_right")
                    # viz.plot(self.smooth_x_bound_left, self.smooth_y_bound_left, fmt="-o", marker_size=2, color='black',
                    #          label="smooth_bound_left")
                    # viz.plot(self.smooth_x_bound_right, self.smooth_y_bound_right, fmt="-o", marker_size=2,
                    #          color='black', label="smooth_bound_right")

                    viz.plot(self.cp.path.x, self.cp.path.y, label="reference path", fmt="-", color="black")

                for i, obs in enumerate(self.cp.obstacles):
                    self.show_obs_pred(obs)
                    viz.plot([obs.x], [obs.y], marker_size=2, label="obs", fmt=".o", color="black")
                    viz.plot(obs.vertices[0, :], obs.vertices[1, :], label="obs", fmt="-", color="black", line_weight=2)

                    # Print current time obs
                    # coords = util.get_ellipsis_coords(obs.x, obs.y, obs.phi, obs.l, obs.w,
                    #                                   self.cp.model.params.eps, self.cp.model.params.disk_r,
                    #                                   self.cp.model.params.n_se)
                    # viz.plot(coords[0, :], coords[1, :], line_weight=3, label="superell", fmt="-", color="blue")


                if self.references_in_map_visible:
                    x_ref = np.squeeze(self.cp.extra_vals.x_ref(self.cp.param_setter.get("all")))
                    y_ref = np.squeeze(self.cp.extra_vals.y_ref(self.cp.param_setter.get("all")))
                    viz.plot(x_ref, y_ref, marker_size=2, label="ref params", fmt=".o",
                             color="gray")

                    xr = np.squeeze(self.cp.extra_vals.xr(self.cp.trajX, self.cp.param_setter.get("all")))
                    yr = np.squeeze(self.cp.extra_vals.yr(self.cp.trajX, self.cp.param_setter.get("all")))
                    viz.plot(xr, yr, marker_size=5, label="ref pos", fmt=".o",
                             color="red")

                    if self.cp.path is not None:
                        # if len(self.cp.path.x) > 1 and len(self.cp.path.x) > self.cp.path.ego_data.ego_idx:
                        #     x, y, phi = transform_pose_frenet2cart_scalar(self.cp.x0[self.cp.theta_idx], 0, 0, self.cp.path.s, self.cp.path.x, self.cp.path.y, self.cp.path.phi)
                        #     viz.plot([x],[y],
                        #              label="proj_ego_pose", fmt="-o", marker_size=3, color="blue")

                        theta_disks = np.squeeze(self.cp.extra_vals.theta_disks(self.cp.trajX[:, 0], self.cp.param_setter.get("all", i=0)))
                        theta_0 = viz.autogui(float(self.cp.extra_vals.theta_0(self.cp.param_setter.get("all"))), "theta_0")

                        try:
                            x_list = []
                            y_list = []
                            for theta in theta_disks:
                                x, y, _ = transform_pose_frenet2cart_scalar(theta_0 + theta, 0, 0, self.cp.path.s, self.cp.path.x, self.cp.path.y, self.cp.path.phi)
                                x_list.append(x)
                                y_list.append(y)
                            viz.plot(x_list,y_list,label="thetas_on_path", fmt="-o", marker_size=5, color="black")
                        except Exception as e:
                            pass

                        # x_list = []
                        # y_list = []
                        # for i in range(self.cp.N):
                        #     x, y, _ = transform_pose_frenet2cart_scalar(self.cp.trajX[self.cp.theta_idx, i], 0, 0, self.cp.path.s, self.cp.path.x, self.cp.path.y, self.cp.path.phi)
                        #     x_list.append(x)
                        #     y_list.append(y)
                        # viz.plot(x_list,y_list,label="thetas_on_path", fmt="-o", marker_size=3, color="cyan")

                if len(self.cp.history.sim_v) > 0:
                    rel_v_sim = np.abs(np.asarray(self.cp.history.sim_v)) / self.v_max_cm
                    rgba_sim = cmap(rel_v_sim)
                #     viz.plot(self.cp.meas_x, self.cp.meas_y, label="driven meas trajectory", fmt="-o", marker_size=3,
                #              color="gray")
                    viz.plot(self.cp.history.sim_x, self.cp.history.sim_y, label="driven sim trajectory", fmt="-o", marker_size=3,
                             color=rgba_sim)


                # rgba = cmap(np.abs(self.cp.init_trajX[3, :]) / self.v_max_cm)
                # viz.plot(self.cp.init_trajX[0, :], self.cp.init_trajX[1, :], label="init_trajectory", fmt="-*", marker_size=5,
                #          color=rgba)

                rgba = cmap(np.abs(self.cp.trajX[3, :]) / self.v_max_cm)
                viz.plot(self.cp.trajX[0, :], self.cp.trajX[1, :], label="trajectory", fmt="-o", marker_size=3,
                         color=rgba)

                # viz.plot(self.cp.trajXY_front_axis[self.cp.x_idx, :], self.cp.trajXY_front_axis[self.cp.y_idx, :], label="trajectory_fa", fmt="-o", marker_size=3, color=rgba)

                # viz.plot(self.cp.pure_pursuit.tx_list, self.cp.pure_pursuit.ty_list, color="red", fmt="-o")
                self.show_vehicle(self.cp.x0_meas, "gray")
                self.show_vehicle(self.cp.goal_state, "red")
                self.show_vehicle(self.cp.x0, "blue", show_disks=True)


                if self.create_plots and self.cp.goal_reached and abs(self.cp.x0[self.cp.v_idx]) < 0.05 or self.force_plot_creation:
                    data_saver.save_map(self)

                # Create video
                if self.record:
                    self.pause = False
                    self.do_record()

                viz.end_plot()
            viz.end_window()

    def get_delta_from_state(self, x0: NDArray) -> float:
        if self.cp.delta_idx <= len(x0) - 1:
            return x0[self.cp.delta_idx]
        return 0.0

    def get_vertices_from_state(self, x0: NDArray) -> tuple[NDArray, NDArray]:
        x: float = x0[self.cp.x_idx]
        y: float = x0[self.cp.y_idx]
        yaw: float = x0[self.cp.phi_idx]

        ego_vertices = util.get_ego_vertices(x, y,
                                             self.cp.params.vehicle.width,
                                             self.cp.params.vehicle.lf, self.cp.params.vehicle.lb,
                                             yaw)
        return ego_vertices[0, :], ego_vertices[1, :]

    def get_wheel_vertices_from_state(self, x0: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        x: float = x0[self.cp.x_idx]
        y: float = x0[self.cp.y_idx]
        yaw: float = x0[self.cp.phi_idx]
        delta: float = self.get_delta_from_state(x0)

        track_width: float = self.cp.params.vehicle.width - 0.2
        return util.get_wheel_vertices(x, y, yaw, delta, self.cp.params.vehicle.wb, track_width)

    def show_vehicle(self, x0: NDArray, color: str = "blue", show_disks: bool = False):
        """
        Show vehicle with turned wheels
        :return:
        """
        if x0 is None:
            return

        veh_x, veh_y = self.get_vertices_from_state(x0)
        rl_wheel_bounds, rr_wheel_bounds, fl_wheel_bounds, fr_wheel_bounds = self.get_wheel_vertices_from_state(x0)

        # if show_disks:
        #     x_disks = np.squeeze(self.cp.extra_vals.x_disks(x0, self.cp.param_setter.get("all")))
        #     y_disks = np.squeeze(self.cp.extra_vals.y_disks(x0, self.cp.param_setter.get("all")))
        #     viz.plot(x_disks, y_disks, marker_size=2, label="disks", fmt=".o", color="black")

        viz.plot(veh_x, veh_y,
                 label="vehicle",
                 line_weight=2.0,
                 color=color)

        viz.plot(rr_wheel_bounds[:, 0], rr_wheel_bounds[:, 1],
                 label="rr",
                 line_weight=2.0,
                 color=color)

        viz.plot(rl_wheel_bounds[:, 0], rl_wheel_bounds[:, 1],
                 label="rl",
                 line_weight=2.0,
                 color=color)

        viz.plot(fr_wheel_bounds[:, 0], fr_wheel_bounds[:, 1],
                 label="fr",
                 line_weight=2.0,
                 color=color)

        viz.plot(fl_wheel_bounds[:, 0], fl_wheel_bounds[:, 1],
                 label="fl",
                 line_weight=2.0,
                 color=color)

    def show_menu(self):
        if viz.begin_main_menu_bar():
            if viz.begin_menu("show options"):

                if viz.menu_item("param_visible", selected=self.param_visible):
                    self.param_visible = not self.param_visible
                if viz.menu_item("debug_visible", selected=self.debug_visible):
                    self.debug_visible = not self.debug_visible
                if viz.menu_item("traj_details_visible", selected=self.traj_details_visible):
                    self.traj_details_visible = not self.traj_details_visible
                if viz.menu_item("show_advanced_details", selected=self.show_advanced_details):
                    self.show_advanced_details = not self.show_advanced_details
                if viz.menu_item("path_details_visible", selected=self.path_details_visible):
                    self.path_details_visible = not self.path_details_visible

                if viz.menu_item("references_in_map_visible", selected=self.references_in_map_visible):
                    self.references_in_map_visible = not self.references_in_map_visible

                if viz.menu_item("n_true_visible", selected=self.n_true_visible):
                    self.n_true_visible = not self.n_true_visible

                if viz.menu_item("frenet_approx_visible", selected=self.frenet_approx_visible):
                    self.frenet_approx_visible = not self.frenet_approx_visible
                if viz.menu_item("ellipsis_values_visible", selected=self.ellipsis_values_visible):
                    self.ellipsis_values_visible = not self.ellipsis_values_visible
                if viz.menu_item("runtimes_visible", selected=self.runtimes_visible):
                    self.runtimes_visible = not self.runtimes_visible

                viz.end_menu()

            if viz.button("create plot"):
                self.force_plot_creation = not self.force_plot_creation

            if viz.button("next frame"):
                self.one_frame = not self.one_frame

            if viz.button("pause"):
                self.pause = not self.pause

            if not self.record:
                if viz.button("record"):
                    self.record = not self.record
            else:
                if viz.button("stop"):
                    self.record = not self.record
                    self.proc.terminate()

            if viz.button("reinit"):
                self.cp.do_reinit = True

            viz.separator()

            viz.checkbox("error", self.cp.optimizer_error)
            viz.checkbox("goal reached", self.cp.goal_reached)

            viz.text("t_ros: " + str(round(self.cp.t_ros4vis, 2)) + " s")

            viz.end_menu_bar()

    def show_obs_pred(self, obs):
        viz.plot(obs.x_pred, obs.y_pred, marker_size=2, label="obs_pred", fmt=".o", color="black")
        s_idxs = (np.minimum(obs.s_pred, self.cp.path.s[-1]) / self.cp.res).astype(int)

        mask = abs(obs.n_pred) >= self.cp.params.general.obs_max_n
        s_idxs[mask] = 0
        viz.plot(self.cp.path.x[s_idxs], self.cp.path.y[s_idxs], marker_size=4, label="obs_s_pred", fmt=".o", color="black")


        # for i in range(0, self.cp.N, 2):
        #
        #     coords = util.get_ellipsis_coords(obs.x_pred[i], obs.y_pred[i], obs.phi_pred[i], obs.l, obs.w,
        #                                       self.cp.model.params.eps, self.cp.model.params.disk_r,
        #                                       self.cp.model.params.n_se)
        #     viz.plot(coords[0, :], coords[1, :], line_weight=2, label="superell", fmt="-", color="cyan")

    # def show_ellipsis_values(self):
    #     se_vals_r = np.squeeze(self.cp.extra_vals.se_vals_r(self.cp.trajX, self.cp.param_setter.all_param[:, 0]))
    #     se_vals_f = np.squeeze(self.cp.extra_vals.se_vals_f(self.cp.trajX, self.cp.param_setter.all_param[:, 0]))
    #
    #     window_title = "se_vals"
    #     if viz.begin_window(
    #             window_title,
    #             resize=True,
    #     ):
    #         plot_title = "se_vals"
    #         if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
    #
    #             viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
    #             viz.setup_axis(viz.Axis.Y1, "se_vals")  #, flags=viz.PlotAxisFlags.AUTO_FIT)
    #             viz.setup_axis_limits(viz.Axis.Y1, 0, 10)
    #
    #             viz.plot_hlines("constraint", [1])
    #             # of first obs
    #             colors = ["red", "blue", "green", "yellow", "black"]
    #             for i in range(self.cp.max_nb_obs):
    #                 viz.plot(self.t, se_vals_r[i, :], fmt="-o", marker_size=2, color=colors[i], label="se_vals_r")
    #                 viz.plot(self.t, se_vals_f[i, :], fmt="-o", marker_size=2, color=colors[i], label="se_vals_f")
    #
    #             viz.end_plot()
    #         viz.end_window()

    def show_frenet_approx(self):
        window_title = "frenet constraints"
        if viz.begin_window(
                window_title,
                resize=True,
        ):

            plot_title = window_title
            if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                viz.setup_axis(viz.Axis.X1, "t in s",
                               flags=viz.PlotAxisFlags.AUTO_FIT)
                viz.setup_axis(viz.Axis.Y1, "e_l & e_c in m", flags=viz.PlotAxisFlags.AUTO_FIT)
                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                e_conts_disks = np.asarray(self.cp.extra_vals.e_conts_disks(self.cp.trajX, self.cp.param_setter.get("all")))

                n_min = np.asarray(self.cp.extra_vals.n_min(self.cp.trajX, self.cp.param_setter.get("all")))
                n_max = np.asarray(self.cp.extra_vals.n_max(self.cp.trajX, self.cp.param_setter.get("all")))

                colors = ["blue", "red", "green", "yellow"]
                for i in range(e_conts_disks.shape[0]):
                    viz.plot(self.t, e_conts_disks[i], fmt="-o", marker_size=1, line_weight=1, color=colors[i], label="e_conts_" + str(i))

                    viz.plot(self.t, n_min[i], fmt="-o", marker_size=2, line_weight=2, color=colors[i], label="n_min_" + str(i))
                    viz.plot(self.t, n_max[i], fmt="-o", marker_size=2, line_weight=2, color=colors[i], label="n_max_" + str(i))

                if self.n_true_visible:
                    x_disks = np.squeeze(self.cp.extra_vals.x_disks(self.cp.trajX, self.cp.param_setter.get("all")))
                    y_disks = np.squeeze(self.cp.extra_vals.y_disks(self.cp.trajX, self.cp.param_setter.get("all")))

                    n_true = np.zeros((len(x_disks), self.cp.N))
                    for i_N in range(self.cp.N):
                        for i_disk in range(len(x_disks)):
                            _, n = transform_point_cart2frenet(x_disks[i_disk, i_N], y_disks[i_disk, i_N], self.cp.path.s, self.cp.path.x,
                                                                 self.cp.path.y, self.cp.path.phi)
                            n_true[i_disk, i_N] = n

                    for i_disk in range(len(x_disks)):
                        viz.plot(self.t, n_true[i_disk, :], fmt="--", line_weight=5, color=colors[i_disk], label="e_conts_true")

                viz.end_plot()
            viz.end_window()

    def show_runtimes(self):
        window_title = "runtimes"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            plot_title = "runtimes"
            if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                viz.setup_axis(viz.Axis.X1, "elements", flags=viz.PlotAxisFlags.AUTO_FIT)
                viz.setup_axis(viz.Axis.Y1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)

                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                viz.plot(self.cp.cycle_times, fmt="-o", marker_size=2, color="green", label="cycle_times")
                viz.plot(self.cp.runtimes, fmt="-o", marker_size=2, color="red", label="t_plan")

                viz.end_plot()
            viz.end_window()

    def show_trajectory_details(self):
        if self.show_advanced_details:
            window_title = "v_references"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "v_references"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                    viz.setup_axis(viz.Axis.X1, "idx", flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "v in m/s", flags=viz.PlotAxisFlags.AUTO_FIT)

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    viz.plot(self.cp.references.theta, self.cp.references.v_max_raw, fmt="-o", marker_size=2, color="red", label="v_max_raw")
                    viz.plot(self.cp.references.theta, self.cp.references.v_max, fmt="-o", marker_size=2, color=ORANGE, label="v_max")
                    viz.plot(self.cp.references.theta, self.cp.references.v_init, fmt="-o", marker_size=2, color="magenta", label="v_init")
                    viz.plot(self.cp.references.theta, self.cp.references.dtheta_max, fmt="-o", marker_size=2, color="green", label="dtheta_max")
                    viz.plot(self.cp.trajX[self.cp.theta_idx], self.cp.trajX[self.cp.v_idx], fmt="-o", marker_size=2, color="blue", label="traj")
                    viz.plot(self.cp.init_trajX[self.cp.theta_idx], self.cp.init_trajX[self.cp.v_idx], fmt="-o", marker_size=2, color="cyan", label="initTraj")

                    viz.end_plot()
                viz.end_window()

            window_title = "driven acc"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "driven acc"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):
                    viz.setup_axis(viz.Axis.X1, "t in s",
                                   flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "a in m/s²")

                    viz.setup_axis_limits(viz.Axis.Y1,
                                          min(-2, -self.cp.params.solver.along_max) * 1.1,
                                          max(2, self.cp.params.solver.along_max) * 1.1
                                          )

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    # viz.plot_hlines("", [-self.cp.params.solver.alat_max], color="blue", width=1)
                    viz.plot_hlines("", [-self.cp.params.solver.along_max], color="red", width=1)
                    # viz.plot_hlines("", [self.cp.params.solver.alat_max], color="blue", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.along_max], color="red", width=1)

                    viz.plot(self.cp.history.sim_t, self.cp.history.sim_a, fmt="-o", marker_size=2, color="red", label="driven a_long")
                    viz.plot(self.cp.history.sim_t, self.cp.history.sim_a_lat, fmt="-o", marker_size=2, color="blue", label="driven a_lat")
                    viz.end_plot()
                viz.end_window()

            window_title = "xy"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "xy"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.EQUAL):
                    viz.setup_axis(viz.Axis.X1, "x in m", flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "y in m", flags=viz.PlotAxisFlags.AUTO_FIT)

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    viz.plot(self.cp.trajX[self.cp.x_idx], self.cp.trajX[self.cp.y_idx], fmt="-o", marker_size=2,
                             color="blue", label="traj")
                    viz.plot(self.cp.init_trajX[self.cp.x_idx], self.cp.init_trajX[self.cp.y_idx], fmt="-o", marker_size=2,
                             color="gray", label="init_traj")

                    viz.end_plot()
                viz.end_window()

            window_title = "acc"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "acc"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                    viz.setup_axis(viz.Axis.X1, "t in s",
                                   flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "a in m/s²")

                    viz.setup_axis_limits(viz.Axis.Y1,
                                          min(-2, -self.cp.params.solver.along_max) * 1.1,
                                          max(2, self.cp.params.solver.along_max) * 1.1
                                          )

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    # viz.plot_hlines("", [-self.cp.params.solver.alat_max], color="blue", width=1)
                    viz.plot_hlines("", [-self.cp.params.solver.along_max], color="red", width=1)
                    # viz.plot_hlines("", [self.cp.params.solver.alat_max], color="blue", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.along_max], color="red", width=1)

                    a_lat = np.squeeze(self.cp.extra_vals.a_lat(self.cp.trajX, self.cp.param_setter.get("all")))

                    viz.plot(self.t, a_lat, fmt="-o", marker_size=2, color="blue", label="a_lat")
                    viz.plot(self.t, self.cp.trajU[self.cp.acc_idx], fmt="-o", marker_size=2, color="red", label="a_long")
                    viz.plot(self.t, self.cp.init_trajU[self.cp.acc_idx], fmt="-o", marker_size=2, color="gray", label="init_a_long")
                    viz.end_plot()
                viz.end_window()

            window_title = "v"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "v"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):  # | viz.PlotFlags.NO_LEGEND
                    viz.setup_axis(viz.Axis.X1, "t in s",
                                   flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "v in m/s", flags=viz.PlotAxisFlags.AUTO_FIT)
                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    v_max_r = np.squeeze(
                        self.cp.extra_vals.v_max_r(self.cp.trajX, self.cp.param_setter.get("all")))
                    v_min_r = np.squeeze(
                        self.cp.extra_vals.v_min_r(self.cp.trajX, self.cp.param_setter.get("all")))

                    viz.plot_hlines("0", [0], color="black", width=0.5)
                    viz.plot(self.t, self.cp.trajX[self.cp.v_idx, :], fmt="-o", marker_size=2, color="blue", label="v")
                    viz.plot(self.t, self.cp.init_trajX[self.cp.v_idx, :], fmt="-o", marker_size=2, color="gray", label="init_v")
                    # viz.plot(self.t, dtheta_max_r, fmt="-o", marker_size=2, color="green", label="dtheta_max")
                    viz.plot(self.t, v_max_r, fmt="-o", marker_size=2, color="red", label="v_max_ref")
                    viz.plot(self.t, v_min_r, fmt="-o", marker_size=2, color="yellow", label="v_min_ref")

                    viz.end_plot()
                viz.end_window()

            # window_title = "q"
            # if viz.begin_window(
            #         window_title,
            #         resize=True,
            # ):
            #     plot_title = "q"
            #     if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):  # | viz.PlotFlags.NO_LEGEND
            #         viz.setup_axis(viz.Axis.X1, "t in s",
            #                        flags=viz.PlotAxisFlags.AUTO_FIT)
            #         viz.setup_axis(viz.Axis.Y1, "value", flags=viz.PlotAxisFlags.AUTO_FIT)
            #         if viz.plot_selection_ended():
            #             viz.hard_cancel_plot_selection()
            #
            #         # weights = np.squeeze(self.cp.extra_vals.q_frenet_eff(self.cp.trajX, self.cp.param_setter.get("all")))
            #         # final_weights = np.squeeze(self.cp.extra_vals.q_f_eff(self.cp.trajX, self.cp.param_setter.get("all")))
            #         # viz.plot(self.t, weights[0, :], fmt="-o", marker_size=2, color="red", label="q_lag")
            #         # viz.plot(self.t, weights[1, :], fmt="-o", marker_size=2, color="blue", label="q_cont")
            #         # viz.plot(self.t, weights[2, :], fmt="-o", marker_size=2, color="green", label="q_angle")
            #
            #         # viz.plot(self.t, final_weights[0, :], fmt="-o", marker_size=2, color="red", label="qfx")
            #         # viz.plot(self.t, final_weights[1, :], fmt="-o", marker_size=2, color="blue", label="qfy")
            #         # viz.plot(self.t, final_weights[2, :], fmt="-o", marker_size=2, color="green", label="qfphi")
            #
            #         viz.end_plot()
            #     viz.end_window()


            # window_title = "driven phi"
            # if viz.begin_window(
            #         window_title,
            #         resize=True
            # ):
            #     plot_title = "driven phi"
            #     if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
            #         viz.setup_axis(viz.Axis.X1, "t in s",
            #                        flags=viz.PlotAxisFlags.AUTO_FIT)
            #         viz.setup_axis(viz.Axis.Y1, "phi in rad")
            #         viz.setup_axis_limits(viz.Axis.Y1, -math.pi, math.pi)
            #
            #         if viz.plot_selection_ended():
            #             viz.hard_cancel_plot_selection()
            #
            #         viz.plot_hlines("0", [0], color="black", width=0.5)
            #         viz.plot(self.t_sim, self.cp.sim_phi, fmt="-o", marker_size=2, color="red",
            #                  label="driven phi")
            #
            #         viz.end_plot()
            #     viz.end_window()
            #
            window_title = "phi"
            if viz.begin_window(
                    window_title,
                    resize=True
            ):
                plot_title = "phi"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                    viz.setup_axis(viz.Axis.X1, "t in s",
                                   flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "phi in rad")
                    viz.setup_axis_limits(viz.Axis.Y1, -math.pi, math.pi)

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    phir_vec = np.squeeze(
                        self.cp.extra_vals.phir(self.cp.trajX, self.cp.param_setter.get("all")))
                    viz.plot(self.t, self.cp.trajX[self.cp.phi_idx, :], fmt="-o", marker_size=2, color="blue", label="phi")
                    viz.plot(self.t, self.cp.init_trajX[self.cp.phi_idx, :], fmt="-o", marker_size=2, color="gray", label="init_phi")
                    viz.plot(self.t, phir_vec, fmt="-o", marker_size=2, color="black", label="phir")

                    viz.end_plot()

                viz.end_window()

            window_title = "delta"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "delta"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):

                    viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "delta in rad")
                    viz.setup_axis_limits(viz.Axis.Y1, -self.cp.params.vehicle.delta_max * 1.1,
                                          self.cp.params.vehicle.delta_max * 1.1)

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    viz.plot_hlines("", [-self.cp.params.vehicle.delta_max], color="blue", width=1)
                    viz.plot_hlines("", [self.cp.params.vehicle.delta_max], color="blue", width=1)
                    viz.plot(self.t, self.cp.trajX[self.cp.delta_idx], fmt="-o", marker_size=2, color="blue", label="delta")
                    viz.plot(self.t, self.cp.init_trajX[self.cp.delta_idx], fmt="-o", marker_size=2, color="gray", label="init_delta")

                    viz.end_plot()
                viz.end_window()

            window_title = "theta"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "theta"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):

                    viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
                    viz.setup_axis(viz.Axis.Y1, "theta in m", flags=viz.PlotAxisFlags.AUTO_FIT)

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    theta_max = [self.cp.extra_vals.theta_max(self.cp.param_setter.get("theta_max", i)) for i in range(self.cp.N)]
                    viz.plot(self.t, np.squeeze(theta_max), fmt="-o", marker_size=2, color="cyan", label="theta_max")
                    viz.plot(self.t, self.cp.trajX[self.cp.theta_idx], fmt="-o", marker_size=2, color="blue", label="theta")
                    viz.plot(self.t, self.cp.init_trajX[self.cp.theta_idx], fmt="-o", marker_size=2, color="gray", label="init_theta")

                    viz.end_plot()
                viz.end_window()

            # window_title = "driven acc"
            # if viz.begin_window(
            #         window_title,
            #         resize=True,
            # ):
            #     plot_title = "driven acc"
            #     if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):
            #         viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
            #         viz.setup_axis(viz.Axis.Y1, "acc in m/s²")
            #         viz.setup_axis_limits(viz.Axis.Y1, self.cp.model.along_min * 1.1, self.cp.model.along_max * 1.1)
            #
            #         if viz.plot_selection_ended():
            #             viz.hard_cancel_plot_selection()
            #
            #         viz.plot_hlines("", [self.cp.model.along_min], color="red", width=1)
            #         viz.plot_hlines("", [self.cp.model.along_max], color="red", width=1)
            #         viz.plot(self.t_sim, self.cp.sim_a, fmt="-o", marker_size=2, color="red",
            #                  label="driven acc")
            #
            #         viz.end_plot()
            #     viz.end_window()
            #
            # window_title = "acc"
            # if viz.begin_window(
            #         window_title,
            #         resize=True,
            # ):
            #     plot_title = "acc"
            #     if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):
            #         viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
            #         viz.setup_axis(viz.Axis.Y1, "acc in m/s²")
            #         viz.setup_axis_limits(viz.Axis.Y1, self.cp.model.along_min * 1.1, self.cp.model.along_max * 1.1)
            #
            #         if viz.plot_selection_ended():
            #             viz.hard_cancel_plot_selection()
            #
            #         viz.plot_hlines("", [self.cp.model.along_min], color="blue", width=1)
            #         viz.plot_hlines("", [self.cp.model.along_max], color="blue", width=1)
            #         viz.plot(self.t, self.cp.trajX[4], fmt="-o", marker_size=2, color="blue", label="acc")
            #
            #         viz.end_plot()
            #     viz.end_window()

            window_title = "driven u"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "driven u"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                    viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)

                    viz.setup_axis_limits(viz.Axis.Y1,
                                          min(-self.cp.params.solver.ddelta_max, -self.cp.params.solver.along_max) * 1.1,
                                          max(self.cp.params.solver.ddelta_max, self.cp.params.solver.along_max) * 1.1
                                          )

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    viz.plot_hlines("", [-self.cp.params.solver.along_max], color="red", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.along_max], color="red", width=1)
                    viz.plot_hlines("", [-self.cp.params.solver.ddelta_max], color="blue", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.ddelta_max], color="blue", width=1)
                    viz.plot(self.cp.history.sim_t, self.cp.history.sim_a, fmt="-o", marker_size=2, color="red",
                             label="a")
                    viz.plot(self.cp.history.sim_t, self.cp.history.sim_ddelta, fmt="-o", marker_size=2, color="blue",
                             label="ddelta")
                    viz.end_plot()
                viz.end_window()

            window_title = "u"
            if viz.begin_window(
                    window_title,
                    resize=True,
            ):
                plot_title = "u"
                if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                    viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)

                    viz.setup_axis_limits(viz.Axis.Y1,
                                          min(-self.cp.params.solver.ddelta_max, -self.cp.params.solver.along_max) * 1.1,
                                          max(self.cp.params.solver.ddelta_max, self.cp.params.solver.along_max) * 1.1
                                          )

                    if viz.plot_selection_ended():
                        viz.hard_cancel_plot_selection()

                    dtheta_max_ref = np.squeeze(
                        self.cp.extra_vals.dtheta_max_r(self.cp.trajX, self.cp.param_setter.get("all")))
                    viz.plot(self.t, dtheta_max_ref, fmt="-", marker_size=2, color="black", label="dtheta_max_ref")
                    viz.plot_hlines("", [-self.cp.params.solver.along_max], color="red", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.along_max], color="red", width=1)
                    viz.plot_hlines("", [-self.cp.params.solver.ddelta_max], color="blue", width=1)
                    viz.plot_hlines("", [self.cp.params.solver.ddelta_max], color="blue", width=1)
                    viz.plot(self.t, self.cp.trajU[self.cp.acc_idx, :], fmt="-o", marker_size=2, color="red", label="a")
                    viz.plot(self.t, self.cp.init_trajU[self.cp.acc_idx, :], fmt="-o", marker_size=2, color="magenta", label="init_a")
                    viz.plot(self.t, self.cp.trajU[self.cp.ddelta_idx, :], fmt="-o", marker_size=2, color="blue",
                             label="ddelta")
                    viz.plot(self.t, self.cp.init_trajU[self.cp.ddelta_idx, :], fmt="-o", marker_size=2, color="cyan",
                             label="init_ddelta")
                    viz.plot(self.t, self.cp.trajU[self.cp.dtheta_idx, :], fmt="-o", marker_size=2, color="green",
                             label="dtheta_dot")
                    viz.plot(self.t, self.cp.init_trajU[self.cp.dtheta_idx, :], fmt="-o", marker_size=2, color=DARKGREEN,
                             label="init_dtheta_dot")

                    viz.end_plot()
                viz.end_window()

        window_title = "driven v"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            plot_title = "driven v"
            if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):
                viz.setup_axis(viz.Axis.X1, "t in s",
                               flags=viz.PlotAxisFlags.AUTO_FIT)
                viz.setup_axis(viz.Axis.Y1, "v in m/s", flags=viz.PlotAxisFlags.AUTO_FIT)
                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                viz.plot_hlines("0", [0], color="black", width=0.5)
                viz.plot(self.cp.history.sim_t, self.cp.history.sim_v, fmt="-", marker_size=2, line_weight=2, color="blue", label="driven v")
                if self.cp.history.sim_t:
                    viz.plot(self.cp.history.sim_t[-1] + self.t, self.cp.trajX[self.cp.v_idx, :], fmt="-o", marker_size=2, color="cyan", label="v")
                # viz.plot(self.cp.sim_t, self.cp.sim_v_max, fmt="-o", marker_size=2, color="red", label="v_max")
                # viz.plot(self.t_sim, self.cp.sim_thetadot, fmt="-o", marker_size=2, color="green",
                #          label="sim_thetadot")

                viz.end_plot()
            viz.end_window()

        window_title = "driven delta"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            plot_title = "driven delta"
            if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE | viz.PlotFlags.NO_LEGEND):
                viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
                viz.setup_axis(viz.Axis.Y1, "delta in rad")
                viz.setup_axis_limits(viz.Axis.Y1, -self.cp.params.vehicle.delta_max * 1.1,
                                      self.cp.params.vehicle.delta_max * 1.1)

                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                viz.plot_hlines("", [-self.cp.params.vehicle.delta_max], color="red", width=1)
                viz.plot_hlines("", [self.cp.params.vehicle.delta_max], color="red", width=1)
                viz.plot(self.cp.history.sim_t, self.cp.history.sim_delta, fmt="-", line_weight=2, marker_size=2, color="red",
                         label="driven delta")
                if self.cp.history.sim_t:
                    viz.plot(self.cp.history.sim_t[-1] + self.t, self.cp.trajX[self.cp.delta_idx, :], fmt="-o", marker_size=2, color="magenta", label="delta")

                viz.end_plot()
            viz.end_window()

    def show_path_details(self):
        window_title = "Path details"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            plot_title = "Path details"
            if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
                viz.setup_axis(viz.Axis.X1, "s in m", flags=viz.PlotAxisFlags.AUTO_FIT)
                # viz.setup_axis(viz.Axis.Y1, "values", flags=viz.PlotAxisFlags.AUTO_FIT)

                if viz.plot_selection_ended():
                    viz.hard_cancel_plot_selection()

                # show switch points
                s_show = self.cp.path.s[self.cp.path.ego_data.ego_idx:]

                # viz.plot(s_show, self.cp.path.s[self.cp.path.ego_idx:], fmt="-o", marker_size=2, color="black", label="s")
                viz.plot(s_show, self.cp.path.get_segment_forward("phi"), fmt="-o", marker_size=2, color="green",
                         label="phi")
                viz.plot(s_show, self.cp.path.get_segment_forward("kappa"), fmt="-o", marker_size=2, color="yellow",
                         label="kappa")
                viz.plot(s_show, self.cp.path.get_segment_forward("directions"), fmt="-o", marker_size=2, color="blue",
                         label="directions")
                viz.plot(s_show, self.cp.path.get_segment_forward("v_max"), fmt="-o", marker_size=2, color="red",
                         label="v_max")

                if self.cp.path.dir_change_indices.size > 0:
                    switch_pos = self.cp.res * self.cp.path.dir_change_indices
                    if len(switch_pos.shape) > 1:
                        switch_pos = switch_pos.squeeze()
                    viz.plot(switch_pos, np.zeros_like(switch_pos), fmt="o", marker_size=5, color="black",
                             label="switching")

                viz.end_plot()
            viz.end_window()

    # @util.timeme
    def show_params(self):
        window_title = "params"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            viz.autogui(self.cp.params, "params")
            viz.autogui(self.cpNode, "Node")

            # vehicle properties
            disk_r = float(self.cp.extra_vals.disk_r(self.cp.x0, self.cp.param_setter.get("all")))
            disk_pos = np.squeeze(self.cp.extra_vals.disk_pos(self.cp.x0, self.cp.param_setter.get("all")))
            viz.autogui(disk_r, "disk_r")
            viz.autogui(disk_pos, "disk_pos")

            viz.end_window()

    def show_debug(self):
        window_title = "debug"
        if viz.begin_window(
                window_title,
                resize=True,
        ):
            self.cp.history.history_len = viz.input("history_len", self.cp.history.history_len)
            viz.autogui(self.cp.goal_pose_only, "goal_pose_only")
            viz.autogui(self.cp.x0, "x0")
            viz.autogui(self.cp.x0_dead_time, "x0_dead_time")
            viz.autogui(self.cp.goal_frenet_error, "goal_frenet_error")
            viz.autogui(self.cp.trajX[self.cp.theta_idx, 0:10], "thetas")
            viz.autogui(float(self.cp.extra_vals.theta_0(self.cp.param_setter.get("all"))), "theta_0")
            viz.autogui(float(self.cp.extra_vals.theta_rel(self.cp.trajX[:, 0], self.cp.param_setter.get("all", i=0))), "theta_rel at 0")

            if self.cp.path is not None:
                viz.text("ego index = " + str(self.cp.path.ego_data.ego_idx) + " / " + str(len(self.cp.path.x)))
                viz.text("goal index = " + str(self.cp.path.goal_index) + " / " + str(len(self.cp.path.x)))
                viz.text("segment = " + str(self.cp.path.ego_data.segment_start_idx) + " --> " + str(
                    self.cp.path.ego_data.segment_end_idx))
            viz.text("time_tot: " + str(self.cp.time_tot))
            viz.autogui(self.cp.residuals, "residuals")

            viz.text("vis fps: " + str(round(1 / self.delta_t_vis)))
            w_size = 10
            if len(self.cp.cycle_times) > w_size + 1 and len(self.cp.runtimes) > w_size + 1:
                data = np.asarray(self.cp.cycle_times)[-w_size:]
                mean_cycle = np.mean(data)
                min_cycle = np.min(data)
                max_cycle = np.max(data)
                mean_plan_runtime = np.mean(self.cp.runtimes[-w_size])
                viz.text("mean cycle runtime: " + str(mean_cycle))
                viz.text("max cycle runtime: " + str(max_cycle))
                viz.text("min cycle runtime: " + str(min_cycle))
                if mean_cycle != 0 and min_cycle != 0:
                    viz.text("mean cycle freq: " + str(round(1 / mean_cycle)))
                    viz.text("max cycle freq: " + str(round(1 / min_cycle)))
                    viz.text("min cycle freq: " + str(round(1 / max_cycle)))
                    viz.text("plan runtime: " + str(mean_plan_runtime))

            viz.text("\nConstraint Violations")
            for attr, value in self.cp.violation_checker.get_violations():
                color = "red" if value is not None else GREEN
                viz.text(attr + ": " + str(value), color)


            viz.end_window()

    # def show_dead_time(self):
    #
    #     window_title = "dead_time"
    #     if viz.begin_window(
    #             window_title,
    #             resize=True,
    #     ):
    #         plot_title = "a"
    #         if viz.begin_plot(plot_title, flags=viz.PlotFlags.NO_TITLE):
    #
    #             # viz.setup_axis(viz.Axis.X1, "t in s", flags=viz.PlotAxisFlags.AUTO_FIT)
    #             # viz.setup_axis(viz.Axis.Y1, "value", flags=viz.PlotAxisFlags.AUTO_FIT)
    #
    #             if viz.plot_selection_ended():
    #                 viz.hard_cancel_plot_selection()
    #
    #             viz.plot(self.cp.t_is, [x[self.cp.acc_idx] for x in self.cp.x0_is], color="red", fmt="-o", marker_size=2, label="a_is")
    #             viz.plot(self.cp.t_pred, [x[self.cp.acc_idx] for x in self.cp.x0_pred], color="blue", fmt="-o", marker_size=2, label="a_pred")
    #             viz.end_plot()
    #         viz.end_window()

    def setup(self):
        if self.dockspace_ready:
            return
        dockspace_id = viz.get_main_dockspace_id()

        viz.dock_builder_remove_node(dockspace_id)
        viz.dock_builder_add_node(dockspace_id)
        viz.dock_builder_set_node_size(dockspace_id, viz.get_window_size())

        left, middle = viz.dock_builder_split_node(dockspace_id, viz.Dir.LEFT, 0.2)
        left, left_bottom = viz.dock_builder_split_node(left, viz.Dir.UP, 0.7)
        middle, right = viz.dock_builder_split_node(middle, viz.Dir.LEFT, 0.5)
        right, right_bottom = viz.dock_builder_split_node(right, viz.Dir.UP, 0.5)
        middle, middle_middle = viz.dock_builder_split_node(middle, viz.Dir.UP, 0.6)
        middle_middle, middle_bottom = viz.dock_builder_split_node(middle_middle, viz.Dir.UP, 0.5)

        viz.dock_builder_dock_window("debug", left)
        viz.dock_builder_dock_window("map", middle)
        viz.dock_builder_dock_window("Path details", middle_bottom)
        viz.dock_builder_dock_window("driven acc", middle_bottom)
        viz.dock_builder_dock_window("xy", middle_bottom)
        viz.dock_builder_dock_window("v_references", middle_bottom)
        viz.dock_builder_dock_window("acc", middle_bottom)
        viz.dock_builder_dock_window("driven v", right)
        viz.dock_builder_dock_window("v", middle_bottom)
        viz.dock_builder_dock_window("phi", middle_bottom)
        viz.dock_builder_dock_window("driven delta", right_bottom)
        viz.dock_builder_dock_window("delta", middle_bottom)
        viz.dock_builder_dock_window("theta", middle_bottom)
        viz.dock_builder_dock_window("driven u", middle_bottom)
        viz.dock_builder_dock_window("u", middle_bottom)
        viz.dock_builder_dock_window("params", right)
        viz.dock_builder_dock_window("runtimes", left_bottom)
        viz.dock_builder_dock_window("frenet constraints", left_bottom)

        viz.dock_builder_finish(dockspace_id)
        self.dockspace_ready = True

    # @util.timeme
    def vis_step(self, show_vis: bool = True) -> bool:
        if not show_vis:
            if self.debug_vis_visible:
                viz.hide_main_window()
            self.initialized = False
            self.debug_vis_visible = False
            return self.pause

        self.delta_t_vis = self.cp.t_ros4vis - self.t_last_vis
        if self.delta_t_vis < self.vis_period and not self.pause:
            return self.pause

        self.t_last_vis = self.cp.t_ros4vis

        if not self.initialized:
            main_title = "Corridor Planning"
            viz.set_main_window_title(main_title)
            viz.style_colors_light()

            viz.set_main_window_size((10000, 10000))
            # viz.set_main_window_pos((1920, 1080))
            viz.show_main_window()
            self.setup()
            self.debug_vis_visible = True
            self.initialized = True

        if not viz.wait(vsync=False):
            return self.pause

        if self.redraw_boundaries and self.cp.path is not None:
            every_idx: int = 1
            self.x_bound_left = (self.cp.path.x - self.cp.path.border_min * np.sin(self.cp.path.phi))[::every_idx]
            self.y_bound_left = (self.cp.path.y + self.cp.path.border_min * np.cos(self.cp.path.phi))[::every_idx]
            self.x_bound_right = (self.cp.path.x - self.cp.path.border_max * np.sin(self.cp.path.phi))[::every_idx]
            self.y_bound_right = (self.cp.path.y + self.cp.path.border_max * np.cos(self.cp.path.phi))[::every_idx]

            self.smooth_x_bound_left = (self.cp.path.x - self.cp.path.smooth_border_min * np.sin(self.cp.path.phi))[
                                       ::every_idx]
            self.smooth_y_bound_left = (self.cp.path.y + self.cp.path.smooth_border_min * np.cos(self.cp.path.phi))[
                                       ::every_idx]
            self.smooth_x_bound_right = (self.cp.path.x - self.cp.path.smooth_border_max * np.sin(self.cp.path.phi))[
                                        ::every_idx]
            self.smooth_y_bound_right = (self.cp.path.y + self.cp.path.smooth_border_max * np.cos(self.cp.path.phi))[
                                        ::every_idx]

            self.redraw_boundaries = False

        viz.push_plot_style_var(
            viz.PlotStyleVar.FIT_PADDING,
            (0.5, 0.5))

        self.show_menu()

        self.show_map()

        if self.param_visible:
            self.show_params()

        # if self.ellipsis_values_visible:
        #     self.show_ellipsis_values()

        # self.show_dead_time()

        if self.frenet_approx_visible:
            self.show_frenet_approx()

        if self.runtimes_visible:
            self.show_runtimes()

        if self.traj_details_visible:
            self.show_trajectory_details()

        if self.path_details_visible:
            self.show_path_details()

        if self.debug_visible:
            self.show_debug()

        viz.pop_plot_style_var()

        if self.one_frame:
            self.one_frame = False
            return False

        return self.pause

    @staticmethod
    def ffmpeg_cmd(width: int, height: int, path, fps: float):
        return [
            '/usr/bin/ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-s', f'{int(width)}x{int(height)}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-vcodec', 'mpeg4',
            '-b:v', '10M',
            f'{path}'
        ]

    def do_record(self):

        if self.record_window:
            # _, y = viz.get_main_window_pos()
            x, y = 0, 0
            frame_width, frame_height = viz.get_main_window_size()
        else:
            x, y = viz.get_plot_pos()
            frame_width, frame_height = viz.get_plot_size()

        # get frame
        if frame_height % 2 == 1:
            frame_height -= 1
            frame_height = int(frame_height)

        frame = viz.get_pixels(x, y, frame_width, frame_height)[:, :, :3]

        if not self.rec_process_started:
            self.rec_process_started = True
            stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            scenario_name = "corridor_planner" + str(stamp)
            filename = "/home/schumann/mrm/projects/sandboxes/aduulm_sandbox/src/corridor_planning/videos/"

            self.proc = subprocess.Popen(
                self.ffmpeg_cmd(frame_width,
                                frame_height,
                                filename + scenario_name + ".mp4", self.fps),
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None)

        self.proc.stdin.write(frame.tobytes())

