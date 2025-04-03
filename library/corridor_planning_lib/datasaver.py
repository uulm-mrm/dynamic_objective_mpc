import os

import numpy as np
import pandas as pd


def save_map(vis):
    # data_prefix: str = input("Get data_prefix: ")
    data_prefix: str = "carla_test"
    if data_prefix:
        data_prefix += "_"

    driven_filename: str = vis.plots_path + "/map/" + data_prefix + "driven.csv"
    corridor_filename: str = vis.plots_path + "/map/" + data_prefix + "corridor.csv"
    goal_pose_filename: str = vis.plots_path + "/map/" + data_prefix + "goal_pose.csv"
    
    if True or not os.path.exists(driven_filename) or input("Do you want to overwrite the existing files? (yes/no): ").lower() in ["yes", "y"]:
        df1 = pd.DataFrame({
            "sim_x" : vis.cp.history.sim_x,
            "sim_y" : vis.cp.history.sim_y,
            "sim_t" : np.asarray(vis.cp.history.sim_t),
            "sim_v" : np.asarray(vis.cp.history.sim_v),
            "sim_v_abs" : np.abs(np.asarray(vis.cp.history.sim_v)),
            "sim_delta" : np.asarray(vis.cp.history.sim_delta),
            "sim_a" : np.asarray(vis.cp.history.sim_a),
            "sim_ddelta" : np.asarray(vis.cp.history.sim_ddelta)
        })
        print("Saving simulated states with length", len(vis.cp.history.sim_x))
        df2 = pd.DataFrame({
            "path_x" : vis.cp.path.x,
            "path_y" : vis.cp.path.y,
            "x_bound_left" : vis.x_bound_left,
            "y_bound_left" : vis.y_bound_left,
            "x_bound_right" : vis.x_bound_right,
            "y_bound_right" : vis.y_bound_right,
            "smooth_x_bound_left" : vis.smooth_x_bound_left,
            "smooth_y_bound_left" : vis.smooth_y_bound_left,
            "smooth_x_bound_right" : vis.smooth_x_bound_right,
            "smooth_y_bound_right" : vis.smooth_y_bound_right,
        })
    
    
        if vis.cp.goal_state is not None:
            veh_x, veh_y = vis.get_vertices_from_state(vis.cp.goal_state)
            rl_wheel_bounds, rr_wheel_bounds, fl_wheel_bounds, fr_wheel_bounds = vis.get_wheel_vertices_from_state(vis.cp.goal_state)
    
            df3 = pd.DataFrame({
                "x" : veh_x,
                "y" : veh_y,
                "rl_x": rl_wheel_bounds[:, 0],
                "rl_y": rl_wheel_bounds[:, 1],
                "rr_x": rr_wheel_bounds[:, 0],
                "rr_y": rr_wheel_bounds[:, 1],
                "fl_x": fl_wheel_bounds[:, 0],
                "fl_y": fl_wheel_bounds[:, 1],
                "fr_x": fr_wheel_bounds[:, 0],
                "fr_y": fr_wheel_bounds[:, 1],
            })
            df3.to_csv(goal_pose_filename, index=False)
    
        df1.to_csv(driven_filename, index=False)
        df2.to_csv(corridor_filename, index=False)
        print("File saved")