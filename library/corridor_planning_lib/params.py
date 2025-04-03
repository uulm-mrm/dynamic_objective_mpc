class SolverParams:
    def __init__(self):

        # Cartesian cost
        self.q_desc = "[x, y, phi, v, delta, theta]"
        self.q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.q_e = self.q.copy()
        # Cartesian penalty to fix last state on goal if goal pose exists
        self.q_N = [1e4, 1e4, 1e4, 0.0, 0.0, 0.0]

        # Input cost
        self.r_desc = "[a, deltadot, thetadot]"
        self.r = [1000.0, 100.0, -100.0]
        self.r_e = self.r.copy()
        self.r_goal = [1000.0, 100.0, 0.0]

        # Frenet cost
        self.q_frenet_desc = "[e_lag, e_cont, e_phi]"

        # blended
        self.q_frenet = [1e3, 1.0, 0.0]
        self.q_frenet_e = [1e3, 100.0, 10.0]
        self.q_frenet_N = [1e3, 0.0, 1e3]

        # Only stay on reference
        self.q_frenet_goal = [1e3, 0.0, 0.0]
        self.q_frenet_outside = [0.0, 0.0, 0.0]

        self.q_lat_desc = "[alat]"
        self.q_lat = 100.0
        self.q_v_viol = 100.0
        self.q_right = 0.0

        self.q_bound = 1e3
        self.bound_dist = 2.0
        self.bound_dist_e = 0.0
        self.q_v_prox = 20.0
        self.v_prox_bound_dist = 2.5

        # cost blending params
        self.ego_sig_center: float = 10.0
        self.ego_sig_steepness: float = 1.0
        self.sig_center: float = 10.0
        self.sig_steepness: float = 1.0

        # constraints
        self.along_max: float = 5.0
        self.ddelta_max: float = 1.0
        self.alat_max: float = 2.0


class VehicleParams:
    """
    Defaults only for sim, values will be set by platform metadata
    """
    def __init__(self,
                 width: float = 2.069,
                 wb: float = 2.875,
                 delta_max: float = 0.60,
                 distance_rear_axis_to_mass_center: float = 1.4375,
                 distance_rear_axis_to_geometric_center: float = 1.4375,
                 distance_frontmost_point_to_mass_center: float = 2.24,
                 length: float = 4.4804368019104,
                 dead_time_gas: float = 0.180,
                 dead_time_brake: float = 0.180,
                 dead_time_steer: float = 0.180
                 ):
        geo_center = length / 2

        self.width: float = width
        self.wb: float = wb
        self.delta_max: float = delta_max
        self.lf: float = distance_frontmost_point_to_mass_center + distance_rear_axis_to_mass_center
        self.lb: float = geo_center - distance_rear_axis_to_geometric_center
        self.dead_time_gas: float = dead_time_gas
        self.dead_time_brake: float = dead_time_brake
        self.dead_time_steer: float = dead_time_steer

    def get_min_dead_time(self) -> float:
        return min(self.dead_time_gas,
                   self.dead_time_brake,
                   self.dead_time_steer)


class PlanningModeParameters:
    def __init__(self):
        # set by launch file
        self.mpc_dead_time_comp: bool = True
        self.is_trajectory_planner: bool = False


class GeneralParameters:
    def __init__(self):
        self.switch_dist_idx: int = 1
        self.max_dist_prox2goal: float = 10.0  # at this dist 0.5 prox2goal is achieved
        self.max_dist_final_coord: float = 5.0

        self.v_diff_max: float = 10.0
        self.cart_diff_max: float = 1.0

        self.v_max_reverse: float = 10.0 # m/s
        self.init_acc: float = 0.5

        self.obs_max_n: float = 4.0
        self.s_safety: float = 2.0


class Params:
    def __init__(self, vehicle_params: VehicleParams = VehicleParams(),
                 solver_params: SolverParams = SolverParams(),
                 mode_params: PlanningModeParameters = PlanningModeParameters(),
                 general_params: GeneralParameters = GeneralParameters()):
        self.vehicle: VehicleParams = vehicle_params
        self.solver: SolverParams = solver_params
        self.mode: PlanningModeParameters = mode_params
        self.general: GeneralParameters = general_params

        self.t_params_set: float = 0.0
        self.delta_t_set_params: float = 0.1
