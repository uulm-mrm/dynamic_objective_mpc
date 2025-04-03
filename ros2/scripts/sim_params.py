from corridor_planning_lib.data_structures import GoalTolerance


class SimParams:
    def __init__(self):
        self.sim_delta_t: float = 0.01

        # track offsets
        self.offset: float = 0
        self.phi_rot_deg: float = 0

        # make plant and control model differ
        self.model_errors: bool = False
        self.runtime_model_errors: bool = False

        self.pose_goal_tol: GoalTolerance = GoalTolerance(0.05, 0.05, 2.0)

        # corridor defaults
        self.n_min: float = -4.0
        self.n_max: float = 4.0
        self.v_max: float = 10.0

        # params to set from command line
        self.track_switch: int = 6
        self.ignore_goal: bool = False
        self.baseline: int = -1
