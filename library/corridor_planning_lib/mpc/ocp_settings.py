import os
from casadi import *
from numpy.typing import NDArray

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, ACADOS_INFTY
    from corridor_planning_lib.mpc.file_checker import FileChecker

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    fc = FileChecker(__file__)

except ImportError:
    print("Running in direct mode, ignoring Acados")


# longer compile time but faster interface to the solver
USE_CYTHON = True

def get_voronoi_like_offset(val_min, val_max, rising_offset):
    # maximum half width to keep zero in middle
    width = val_max - val_min
    min_offset = width / 2
    voronoi_rising_offset = fmin(rising_offset, min_offset)
    return voronoi_rising_offset

    # overwrite with fixed distance to show advantage
    # return rising_offset * np.ones_like(voronoi_rising_offset).squeeze()


def get_bathtub_cost(val, val_min, val_max, rising_offset, m=1.0):
    right_bound = val_min + rising_offset
    left_bound = val_max - rising_offset
    min_viol = fmin(val - right_bound, 0)
    max_viol = fmin(left_bound - val, 0)
    return m*min_viol ** 2 + m*max_viol ** 2


def get_voronoi_bathtub_cost(val, val_min, val_max, const_rising_offset):
    rising_offset = get_voronoi_like_offset(val_min, val_max, const_rising_offset)

    # correction factor to reach same value at boundary
    shift = const_rising_offset - rising_offset  # shift by voronoi mode
    m = const_rising_offset**2/(const_rising_offset-shift)**2

    return get_bathtub_cost(val, val_min, val_max, rising_offset, m)


def right_driving_cost(val, val_min, val_max, rising_offset):
    rising_offset = get_voronoi_like_offset(val_min, val_max, rising_offset)
    right_bound = val_min + rising_offset
    return val - right_bound

def get_index(constraint_object, constraint: MX):
    for i, constr in enumerate(vertsplit(constraint_object)):
        if hash(constr) == hash(constraint):
            # print("hash", hash(constr))
            # print("hash", hash(constraint))
            return i

    raise Exception(f"constraint\n {str(constraint)} \nwas not found in constraint_object\n {constraint_object}")


def get_values(namespace: types.SimpleNamespace):
    return [namespace.__getattribute__(attr) for attr in vars(namespace)]


def get_disk_radius(l: float, w: float, nb_disks: int) -> float:
    return casadi.sqrt(l ** 2 / nb_disks ** 2 + w ** 2) / 2


def get_disk_dists(r: float, w: float) -> float:
    return 2 * casadi.sqrt(r ** 2 - w ** 2 / 4)


def get_disk_positions(r: float, nb_disks: int, w: float, offset: float):
    dist: float = get_disk_dists(r, w)
    first_pos: float = dist / 2 - offset
    disks_arange = np.arange(0, nb_disks, 1)
    return first_pos + disks_arange * dist


def interpolate(theta: MX, values: MX, N_refpath: int, res: float, extrapolate: bool = False):
    """

    :param theta:
    :param values:
    :param N_refpath:
    :param res:
    :return:

    Args:
        extrapolate:
    """
    # Interpolation function
    theta_rel = theta / res
    if not extrapolate:
        theta_rel = fmax(fmin(theta_rel, N_refpath - 1), 0)  # dont extrapolate
    theta_grid = np.linspace(0, N_refpath - 1, N_refpath)
    lut = interpolant("LUT", "linear", [theta_grid])
    return lut(theta_rel, values)


def index_access(theta: MX, values: MX, N_refpath: int, res: float) -> MX:
    """
    :param theta:
    :param values:
    :param N_refpath:
    :param res:
    :return:
    """
    # Interpolation function
    index = theta / res
    index = fmax(fmin(index, N_refpath - 1), 0)  # dont extrapolate
    return values[index]


def sigmoid(x, center, steepness):
    return 1 / (1 + exp(steepness * (x - center)))


def blend_weights(val, w_default, w_goal, sig_center, sig_steepness):
    factor: float = sigmoid(val, sig_center, sig_steepness)
    return factor * w_goal + (1 - factor) * w_default


def get_bicycle_model():
    # define structs
    constraint = types.SimpleNamespace()
    constraint.state = types.SimpleNamespace()
    constraint.state.upper_bounds = types.SimpleNamespace()
    constraint.state.lower_bounds = types.SimpleNamespace()
    constraint.state.variables = types.SimpleNamespace()
    constraint.add_state = types.SimpleNamespace()
    constraint.add_state.upper_bounds = types.SimpleNamespace()
    constraint.add_state.lower_bounds = types.SimpleNamespace()
    constraint.add_state.variables = types.SimpleNamespace()
    constraint.input = types.SimpleNamespace()
    constraint.input.upper_bounds = types.SimpleNamespace()
    constraint.input.lower_bounds = types.SimpleNamespace()
    constraint.input.variables = types.SimpleNamespace()

    extra_vals = types.SimpleNamespace()
    references = types.SimpleNamespace()
    model = types.SimpleNamespace()

    theta_res: float = 0.1

    model_name = "ocp_bicycle"

    nb_disks: int = 3

    # CasADi Model
    # set up states & controls
    x = MX.sym("x")
    y = MX.sym("y")
    phi = MX.sym("phi")
    v = MX.sym("v")
    delta = MX.sym("delta")
    theta = MX.sym("theta")
    state = vertcat(x, y, phi, v, delta, theta)

    # controls
    u1_a_long = MX.sym("u1_a_long")
    u2_delta_dot = MX.sym("u2_delta_dot")
    u3_theta_dot = MX.sym("u3_theta_dot")
    u = vertcat(u1_a_long, u2_delta_dot, u3_theta_dot)

    # xdot
    xdot = MX.sym("xdot")
    ydot = MX.sym("ydot")
    phidot = MX.sym("phidot")
    vdot = MX.sym("vdot")
    deltadot = MX.sym("deltadot")
    thetadot = MX.sym("thetadot")
    state_dot = vertcat(xdot, ydot, phidot, vdot, deltadot, thetadot)

    # super ellipsis constraints
    # max_nb_obs = 5
    # n_o = 5  # x, y, phi, b, a
    # n_se = 2  # super ellipse
    # eps = 0.1  # safety dist
    # obs_states = MX.sym("obs_states", n_o * max_nb_obs)
    # se_vals_r = MX.zeros(max_nb_obs)
    # se_vals_f = MX.zeros(max_nb_obs)

    direction = MX.sym("direction", 1)
    theta_0 = MX.sym("theta_0", 1)
    theta_f = MX.sym("theta_f", 1)
    theta_rel = fmax(theta - theta_0, 0.0)
    # references as parameters
    N_refpath = 700
    x_ref = MX.sym("x_ref", N_refpath)
    y_ref = MX.sym("y_ref", N_refpath)
    phi_ref = MX.sym("phi_ref", N_refpath)
    kappa_ref = MX.sym("kappa_ref", N_refpath)
    n_min_ref = MX.sym("n_min_ref", N_refpath)
    n_max_ref = MX.sym("n_max_ref", N_refpath)
    v_min_ref = MX.sym("v_min_ref", N_refpath)
    v_max_ref = MX.sym("v_max_ref", N_refpath)
    dtheta_max_ref = MX.sym("dtheta_max_ref", N_refpath)
    xr = interpolate(theta_rel, x_ref, N_refpath, theta_res, extrapolate=True)
    yr = interpolate(theta_rel, y_ref, N_refpath, theta_res, extrapolate=True)
    phir = index_access(theta_rel, phi_ref, N_refpath, theta_res)
    kappa_r = interpolate(theta_rel, kappa_ref, N_refpath, theta_res)
    v_min_r = interpolate(theta_rel, v_min_ref, N_refpath, theta_res)
    v_max_r = interpolate(theta_rel, v_max_ref, N_refpath, theta_res)
    dtheta_max_r = interpolate(theta_rel, dtheta_max_ref, N_refpath, theta_res)

    # weights for external cost func
    q = MX.sym("q", state.size()[0])
    q_N = MX.sym("q_N", state.size()[0])
    x_N = MX.sym("x_N", state.size()[0])
    q_e = MX.sym("q_e", state.size()[0])
    r = MX.sym("r", u.size()[0])
    r_e = MX.sym("r_e", u.size()[0])
    q_frenet = MX.sym("q_frenet", 3)
    q_frenet_outside = MX.sym("q_frenet_outside", 3)
    q_frenet_e = MX.sym("q_frenet_e", 3)
    q_frenet_N = MX.sym("q_frenet_N", 3)
    sig_steepness = MX.sym("sig_steepness", 1)
    sig_center = MX.sym("sig_center", 1)

    q_bound = MX.sym("q_bound", 1)
    bound_dist = MX.sym("bound_dist", 1)
    bound_dist_e = MX.sym("bound_dist_e", 1)
    v_prox_bound_dist = MX.sym("v_prox_bound_dist", 1)
    q_right = MX.sym("q_right", 1)
    q_lat = MX.sym("q_lat", 1)
    q_v_viol = MX.sym("q_v_viol", 1)
    q_v_prox = MX.sym("q_v_prox", 1)

    # vehicle params
    lf = MX.sym("lf", 1)
    lb = MX.sym("lb", 1)
    delta_max = MX.sym("delta_max", 1)
    along_max = MX.sym("along_max", 1)
    width = MX.sym("width", 1)
    wb = MX.sym("wb", 1)
    ddelta_max = MX.sym("ddelta_max", 1)
    theta_max = MX.sym("theta_max", 1)
    theta_min = MX.sym("theta_min", 1)

    p_list = [
        direction,
        theta_0,
        theta_f,
        x_ref,
        y_ref,
        phi_ref,
        kappa_ref,
        n_min_ref,
        n_max_ref,
        v_min_ref,
        v_max_ref,
        dtheta_max_ref,
        q,
        q_e,
        q_N,
        r,
        r_e,
        q_frenet,
        q_frenet_e,
        q_frenet_outside,
        q_frenet_N,
        sig_steepness,
        sig_center,
        q_lat,
        q_v_viol,
        q_bound,
        bound_dist,
        bound_dist_e,
        v_prox_bound_dist,
        q_right,
        q_v_prox,
        x_N,
        lf,
        lb,
        delta_max,
        width,
        wb,
        along_max,
        ddelta_max,
        theta_max,
        theta_min,
    ]

    p = vertcat(*p_list)
    N_parameters = p.shape[0]

    # Reference state
    state_r = vertcat(xr, yr, phir, 0, 0, 0)

    # Disk centers
    # TODO for ushift create third disk that is normally on rear disk but is placed manually if capsule is attached!
    # Or the freespace planner is not allowed to increase its number of disks but only change their size and pos, which
    # can be done here, too!
    length = lf + lb
    disk_r = get_disk_radius(length, width, nb_disks)
    disk_pos = get_disk_positions(disk_r, nb_disks, width, offset=lb)
    disk_pos = disk_pos[0]

    # coordinates of disk centers
    x_disks = x + disk_pos * cos(phi)
    y_disks = y + disk_pos * sin(phi)

    # obstacles avoidance
    # # superellipse constraint in frenet coordinates for every obstacle for both circles
    # for i in range(max_nb_obs):
    #     # Obstacle index
    #     i_obs = i * n_o
    #     # get ellipsis dim
    #     b_ell = obs_states[i_obs + 3]
    #     a_ell = obs_states[i_obs + 4]
    #
    #     obs_x = obs_states[i_obs + 0]
    #     obs_y = obs_states[i_obs + 1]
    #     obs_phi = -obs_states[i_obs + 2]
    #
    #     # rear disk
    #     # position of vehicle (point) relative to center of ellipse
    #     x_obs_diff_r = x_disk_r - obs_x
    #     y_obs_diff_r = y_disk_r - obs_y
    #
    #     # rotate coordinate in ellipsis principle axis system
    #     x_se_diff_r = x_obs_diff_r * cos(obs_phi) - y_obs_diff_r * sin(obs_phi)
    #     y_se_diff_r = x_obs_diff_r * sin(obs_phi) + y_obs_diff_r * cos(obs_phi)
    #
    #     # ellipsis value
    #     se_vals_r[i] = ((x_se_diff_r / (a_ell + disk_r + eps)) ** n_se +
    #                     (y_se_diff_r / (b_ell + disk_r + eps)) ** n_se) ** (1 / n_se)
    #
    #     # front disk
    #     # position of vehicle (point) relative to center of ellipse
    #     x_obs_diff_f = x_disk_f - obs_x
    #     y_obs_diff_f = y_disk_f - obs_y
    #     # rotate coordinate in ellipsis principle axis system
    #     x_se_diff_f = x_obs_diff_f * cos(obs_phi) - y_obs_diff_f * sin(obs_phi)
    #     y_se_diff_f = x_obs_diff_f * sin(obs_phi) + y_obs_diff_f * cos(obs_phi)
    #     # ellipsis value
    #     se_vals_f[i] = ((x_se_diff_f / (a_ell + disk_r + eps)) ** n_se +
    #                     (y_se_diff_f / (b_ell + disk_r + eps)) ** n_se) ** (1 / n_se)
    #
    # constraint.se_min_arr = np.ones(max_nb_obs)
    # constraint.se_max_arr = ACADOS_INFTY * np.ones(max_nb_obs)

    # relative to center of gravity
    # lb_cg = wb/2  # distance rear axis to center of gravity
    # beta = atan(lb_cg / wb * tan(delta))
    # x_dot = v * cos(phi + beta)
    # y_dot = v * sin(phi + beta)
    # phi_dot = v /wb * tan(delta) * cos(beta)

    # relative to rear axis
    x_dot = v * cos(phi)
    y_dot = v * sin(phi)
    phi_dot = v / wb * tan(delta)
    f_expl = vertcat(
        x_dot,  # x
        y_dot,  # y
        phi_dot,  # phi
        u1_a_long,  # a
        u2_delta_dot,  # delta
        u3_theta_dot,  # theta
    )

    # cartesian -> frenet approximation
    e_lag = (x - xr) * cos(phir) + (y - yr) * sin(phir)
    e_cont = -(x - xr) * sin(phir) + (y - yr) * cos(phir)
    a_diff = phi - phir
    e_phi = a_diff + if_else(a_diff > pi, -2 * pi, if_else(a_diff < -pi, 2 * pi, 0))
    x_frenet = vertcat(e_lag, e_cont, e_phi)

    # check thetas against nr
    # Direction is always forward, is wrong
    # TODO debug this with vis values, extract theta and n and visualize
    theta_offsets = cos(e_phi) * (direction * disk_pos) / (1 - kappa_r * e_cont)
    theta_disks = theta_rel + theta_offsets
    # Approximation of next frenet values by circular arc, not actual kappa TODO(Schumann) improve this
    e_conts_disks = e_cont + sin(e_phi) * disk_pos - kappa_r * theta_offsets

    n_min = interpolate(theta_disks.T, n_min_ref, N_refpath, theta_res, extrapolate=True).T
    n_max = interpolate(theta_disks.T, n_max_ref, N_refpath, theta_res, extrapolate=True).T

    # Add states
    kappa = tan(delta) / wb
    a_lat = v * v * kappa

    # blend metric
    # dist_to_goal_pose = sqrt((x - x_N[0])**2 + (y - x_N[1])**2)  # euclid distance
    dist_goal_path_end = fmax(theta_f - theta, 0)  # longitudinal distance

    # weight blending for frenet weights
    q_eff = blend_weights(dist_goal_path_end, q, q_e, sig_center, sig_steepness)
    r_eff = blend_weights(dist_goal_path_end, r, r_e, sig_center, sig_steepness)
    q_frenet_eff = blend_weights(dist_goal_path_end, q_frenet, q_frenet_e, sig_center, sig_steepness)
    q_right_eff = blend_weights(dist_goal_path_end, q_right, 0.0, sig_center, sig_steepness)
    bound_dist_eff = blend_weights(dist_goal_path_end, bound_dist, bound_dist_e, sig_center, sig_steepness)

    # complete switch with theta > path end
    # complete switch with theta > path end
    q_frenet_eff = if_else(dist_goal_path_end <= 0.0, q_frenet_outside, q_frenet_eff)
    q_frenet_N_eff = if_else(dist_goal_path_end <= 0.0, q_frenet_outside, q_frenet_N)
    q_bound_eff = if_else(dist_goal_path_end < 0.0, 0.0, q_bound)

    # TODO deactivate last frenet constraints of last 3 disks on goal pose either here, or in solver
    # this is buggy and not completely correct, maybe not even desired... In fact we dont know anything about the part behind, maybe keep the constraint
    # e_conts_disks[0] = if_else(dist_to_path_end <= 0.0, 0, e_conts_disks[0])
    # e_conts_disks[1] = if_else(dist_to_path_end <= 0.0, 0, e_conts_disks[1])
    # e_conts_disks[2] = if_else(dist_to_path_end <= 0.0, 0, e_conts_disks[2])

    # Define initial conditions
    model.x0 = np.zeros(state.size()[0])

    # nonlinear constraints
    # State constraints
    constraint.state.variables.v_min_diff = v - v_min_r  # > 0
    constraint.state.variables.v_max_diff = v_max_r - v  # > 0
    constraint.state.variables.delta_min_diff = delta + delta_max  # > 0
    constraint.state.variables.delta_max_diff = delta_max - delta  # > 0
    constraint.state.variables.theta_min_diff = theta - theta_min  # > 0
    constraint.state.variables.theta_max_diff = theta_max - theta  # > 0
    # Input constraints
    constraint.input.variables.along_min_diff = u1_a_long + along_max  # > 0
    constraint.input.variables.along_max_diff = along_max - u1_a_long  # > 0
    constraint.input.variables.ddelta_min_diff = u2_delta_dot + ddelta_max  # > 0
    constraint.input.variables.ddelta_max_diff = ddelta_max - u2_delta_dot  # > 0
    constraint.input.variables.dtheta_min_diff = u3_theta_dot + dtheta_max_r  # > 0
    constraint.input.variables.dtheta_max_diff = dtheta_max_r - u3_theta_dot  # > 0
    # Additional constraints
    # for i, (e_cont, nmin, nmax) in enumerate(zip(vertsplit(e_conts_disks), vertsplit(n_min), vertsplit(n_max))):
    #     setattr(constraint.add_state.variables, "n_min_diff_" + str(i), e_cont - nmin)
    #     setattr(constraint.add_state.variables, "n_max_diff_" + str(i), nmax - e_cont)
    #
    #     # TODO Only rear disk for the moment, is still wrong
    #     if i > 0:
    #         break

    # State boundaries
    constraint.state.lower_bounds.v = 0.0 * np.ones(2)
    constraint.state.upper_bounds.v = ACADOS_INFTY * np.ones(2)
    constraint.state.lower_bounds.delta = 0.0 * np.ones(2)
    constraint.state.upper_bounds.delta = ACADOS_INFTY * np.ones(2)
    constraint.state.lower_bounds.theta_max = 0.0 * np.ones(2)
    constraint.state.upper_bounds.theta_max = ACADOS_INFTY * np.ones(2)
    # Input boundaries
    constraint.input.lower_bounds.along = 0.0 * np.ones(2)
    constraint.input.upper_bounds.along = ACADOS_INFTY * np.ones(2)
    constraint.input.lower_bounds.ddelta = 0.0 * np.ones(2)
    constraint.input.upper_bounds.ddelta = ACADOS_INFTY * np.ones(2)
    constraint.input.lower_bounds.dtheta = 0.0 * np.ones(2)
    constraint.input.upper_bounds.dtheta = ACADOS_INFTY * np.ones(2)
    # add states
    # constraint.add_state.lower_bounds.alat = 0.0 * np.ones(2)  # maximum alat
    # constraint.add_state.upper_bounds.alat = ACADOS_INFTY * np.ones(2)  # maximum alat
    constraint.add_state.lower_bounds.n = 0.0 * np.ones(1 * 2)  # TODO is wrng nb_disks
    constraint.add_state.upper_bounds.n = ACADOS_INFTY * np.ones(1 * 2)  # TODO is wrong nb_disks

    # constraints at intermediate nodes
    constraint.expr = vertcat(
        *get_values(constraint.state.variables),
        *get_values(constraint.input.variables),
        # *get_values(constraint.add_state.variables),
    )

    # constraints at start node = only inputs
    constraint.expr_0 = vertcat(
        *get_values(constraint.input.variables),
        # *get_values(constraint.add_state.variables),
    )

    constraint.expr_e = vertcat(
        *get_values(constraint.state.variables),
        # *get_values(constraint.add_state.variables),
    )

    # define extra values to plot
    # to visualize this from outside
    extra_vals.theta_0 = Function("theta_0", [p], [theta_0])
    extra_vals.theta_max = Function("theta_max", [p], [theta_max])
    extra_vals.theta_rel = Function("theta_rel", [state, p], [theta_rel])
    extra_vals.theta_disks = Function("theta_disks", [state, p], [theta_disks])
    extra_vals.q_frenet_eff = Function("q_frenet_eff", [state, p], [q_frenet_eff])
    extra_vals.xr = Function("xr", [state, p], [xr])
    extra_vals.yr = Function("yr", [state, p], [yr])
    extra_vals.x_ref = Function("x_ref", [p], [x_ref])
    extra_vals.y_ref = Function("y_ref", [p], [y_ref])
    extra_vals.phir = Function("phir", [state, p], [phir])
    extra_vals.n_min = Function("n_min", [state, p], [n_min])
    extra_vals.n_max = Function("n_max", [state, p], [n_max])

    extra_vals.v_max_r = Function("v_max_r", [state, p], [v_max_r])
    extra_vals.v_min_r = Function("v_min_r", [state, p], [v_min_r])
    extra_vals.dtheta_max_r = Function("dtheta_max_r", [state, p],
                                       [dtheta_max_r])
    extra_vals.e_lag = Function("e_lag", [state, p], [e_lag])
    extra_vals.e_cont = Function("e_cont", [state, p], [e_cont])
    extra_vals.e_conts_disks = Function("e_conts_disks", [state, p], [e_conts_disks])
    extra_vals.e_phi = Function("e_phi", [state, p], [e_phi])
    extra_vals.a_lat = Function("a_lat", [state, p], [a_lat])

    # obstacle values
    extra_vals.x_disks = Function("x_disks", [state, p], [x_disks])
    extra_vals.y_disks = Function("y_disks", [state, p], [y_disks])
    extra_vals.disk_r = Function("disk_r", [state, p], [disk_r])
    extra_vals.disk_pos = Function("disk_pos", [state, p], [disk_pos])

    extra_vals.delta_max = Function("delta_max", [p], [delta_max])
    # extra_vals.se_vals_r = Function("se_vals_r", [state, p], [se_vals_r])
    # extra_vals.se_vals_f = Function("se_vals_f", [state, p], [se_vals_f])
    # extra_vals.obs_states = Function("obs_states", [p], [obs_states])

    # make constraint values readable
    # base_names: list = ["n_min_diff_", "n_max_diff_"]
    # for i in range(1):
    #     for name in base_names:
    #         var_name: str = name + str(i)
    #         setattr(extra_vals, var_name,
    #                 Function(var_name, [state, p], [getattr(constraint.add_state.variables, var_name)]))

    # state variables
    extra_vals.v_max_diff = Function("v_max_diff", [state, p], [constraint.state.variables.v_max_diff])
    extra_vals.v_min_diff = Function("v_min_diff", [state, p], [constraint.state.variables.v_min_diff])
    extra_vals.theta_min_diff = Function("theta_min_diff", [state, p], [constraint.state.variables.theta_min_diff])
    extra_vals.theta_max_diff = Function("theta_max_diff", [state, p], [constraint.state.variables.theta_max_diff])
    extra_vals.delta_min_diff = Function("delta_min_diff", [state, p], [constraint.state.variables.delta_min_diff])
    extra_vals.delta_max_diff = Function("delta_max_diff", [state, p], [constraint.state.variables.delta_max_diff])

    # input constraints
    extra_vals.along_min_diff = Function("along_min_diff", [state, u, p], [constraint.input.variables.along_min_diff])
    extra_vals.along_max_diff = Function("along_max_diff", [state, u, p], [constraint.input.variables.along_max_diff])
    extra_vals.ddelta_min_diff = Function("ddelta_min_diff", [state, u, p],
                                          [constraint.input.variables.ddelta_min_diff])
    extra_vals.ddelta_max_diff = Function("ddelta_max_diff", [state, u, p],
                                          [constraint.input.variables.ddelta_max_diff])
    extra_vals.dtheta_min_diff = Function("dtheta_min_diff", [state, u, p],
                                          [constraint.input.variables.dtheta_min_diff])
    extra_vals.dtheta_max_diff = Function("dtheta_max_diff", [state, u, p],
                                          [constraint.input.variables.dtheta_max_diff])

    # Define model struct
    params = types.SimpleNamespace()
    params.N_refpath = N_refpath
    params.N_parameters = N_parameters

    # set everything that is used in the cost function
    # params
    params.sig_center = sig_center
    params.sig_steepness = sig_steepness
    params.q_frenet_eff = q_frenet_eff
    params.q_frenet_N_eff = q_frenet_N_eff
    params.r_eff = r_eff
    params.q_eff = q_eff
    params.q_N = q_N
    params.q_v_viol = q_v_viol
    params.q_bound_eff = q_bound_eff
    params.q_right_eff = q_right_eff
    params.along_max = along_max
    params.bound_dist_eff = bound_dist_eff
    params.v_prox_bound_dist = v_prox_bound_dist
    params.q_lat = q_lat
    params.q_v_prox = q_v_prox

    params.disk_r = disk_r
    params.nb_disks = nb_disks
    # params.n_se = n_se
    # params.eps = eps
    # params.max_nb_obs = max_nb_obs
    # params.n_o = n_o

    # refs
    references.x_r = state_r
    references.x_N = x_N
    references.theta_f = theta_f
    references.n_min = n_min
    references.n_max = n_max
    references.v_min = v_min_r
    references.v_max = v_max_r

    # implicit dynamics for IRK
    model.f_impl_expr = vertcat(state_dot - f_expl)

    # explicit dynamics, for ERK
    model.f_expl_expr = f_expl

    model.x = state
    model.a_lat = a_lat
    model.x_frenet = x_frenet
    model.xdot = state_dot
    model.u = u
    model.p = p
    model.p_list = p_list
    model.name = model_name
    model.params = params
    model.references = references
    return model, constraint, extra_vals


def get_state_error(x: MX, x_r: NDArray) -> MX:
    e = x - x_r

    # angle workaround
    angle_diff = x[2] - x_r[2]
    e[2] = angle_diff + if_else(angle_diff > pi, -2 * pi, if_else(angle_diff < -pi, 2 * pi, 0))

    return e


def get_solver(model, constraint):
    # create render arguments
    ocp = AcadosOcp()
    suffix: str = "ocp"
    ocp.code_export_directory = "c_generated_code_" + suffix

    # define acados ODE
    model_ac = AcadosModel()
    # CasADi expression for the implicit dynamics 𝑓impl(𝑥˙,𝑥,𝑢,𝑧,𝑝)=0.
    # Used if acados_template.acados_ocp.AcadosOcpOptions.integrator_type == ‘IRK’.
    model_ac.f_impl_expr = model.f_impl_expr
    # CasADi expression for the explicit dynamics 𝑥˙=𝑓expl(𝑥,𝑢,𝑝).
    # Used if acados_template.acados_ocp.AcadosOcpOptions.integrator_type == ‘ERK’.
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x  # CasADi variable describing the state of the system;
    model_ac.xdot = model.xdot  # CasADi variable describing the derivative of the state wrt time;
    model_ac.u = model.u  # CasADi variable describing the input of the system;
    model_ac.p = model.p  # CasADi variable describing parameters of the DAE;
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr  # CasADi expression for the constraint ℎ
    model_ac.con_h_expr_0 = constraint.expr_0  # CasADi expression for the constraint ℎ
    model_ac.con_h_expr_e = constraint.expr_e  # CasADi expression for the constraint ℎ

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Weights
    Q = casadi.diag(model.params.q_eff)
    Q_frenet = casadi.diag(model.params.q_frenet_eff)
    Q_frenet_N = casadi.diag(model.params.q_frenet_N_eff)
    Q_N = casadi.diag(model.params.q_N)
    R = casadi.diag(model.params.r_eff)

    # Differences
    e = get_state_error(model.x, model.references.x_r)
    e_N = get_state_error(model.x, model.references.x_N)

    n_max = model.references.n_max[0]
    n_min = model.references.n_min[0]
    n = model.x_frenet[1]

    v_max = model.references.v_max[0]
    v_min = model.references.v_min[0]
    v = model.x[3]

    boundary_cost = model.params.q_bound_eff * get_voronoi_bathtub_cost(n,
                                                                        n_min, n_max,
                                                                        model.params.bound_dist_eff)

    v_viol_cost = model.params.q_v_viol * get_voronoi_bathtub_cost(v, v_min, v_max, 0.0)

    right_driving = model.params.q_right_eff * right_driving_cost(n,
                                                                  n_min, n_max,
                                                                  model.params.bound_dist_eff)

    corridor_prox = get_bathtub_cost(n, n_min, n_max, model.params.v_prox_bound_dist)
    velocity_proximity_cost = model.x[3]**2 * corridor_prox * model.params.q_v_prox

    # state cost
    cost_x = ((e.T @ Q @ e + model.x_frenet.T @ Q_frenet @ model.x_frenet) +
                  model.a_lat ** 2 * model.params.q_lat) + velocity_proximity_cost + v_viol_cost + boundary_cost + right_driving

    # final state cost
    cost_x_N = (e_N.T @ Q_N @ e_N + # reach final state if it exists
                        model.x_frenet.T @ Q_frenet_N @ model.x_frenet)  # keep path orientation at N

    # input cost
    acc_cost = model.u[0] ** 2 * R[0, 0]  # quadratic
    ddelta_cost = model.u[1] ** 2 * R[1, 1]  # quadratic
    progress_cost = model.u[2] * R[2, 2]  # linear progress reward
    cost_u = acc_cost + ddelta_cost + progress_cost

    # cost at intermediate stages
    ocp.model.cost_expr_ext_cost = cost_x + cost_u
    # cost at start and end
    ocp.model.cost_expr_ext_cost_0 = ocp.model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = cost_x + cost_x_N

    # lower bound for nonlinear inequalities at shooting nodes (1 to N-1)
    ocp.constraints.lh = np.hstack(
        [
            *get_values(constraint.state.lower_bounds),
            *get_values(constraint.input.lower_bounds),
            # *get_values(constraint.add_state.lower_bounds)
        ]
    )

    # constraints at first stage
    ocp.constraints.lh_0 = np.hstack(
        [
            *get_values(constraint.input.lower_bounds),
            # *get_values(constraint.add_state.lower_bounds),
        ]
    )
    ocp.constraints.lh_e = np.hstack(
        [
            *get_values(constraint.state.lower_bounds),
            # *get_values(constraint.add_state.lower_bounds)
        ]
    )
    # upper bound for nonlinear inequalities at shooting nodes (1 to N-1)
    ocp.constraints.uh = np.hstack(
        [
            *get_values(constraint.state.upper_bounds),
            *get_values(constraint.input.upper_bounds),
            # *get_values(constraint.add_state.upper_bounds)

        ]
    )
    # constraints at first stage
    ocp.constraints.uh_0 = np.hstack(
        [
            *get_values(constraint.input.upper_bounds),
            # *get_values(constraint.add_state.upper_bounds),
        ]
    )
    ocp.constraints.uh_e = np.hstack(
        [
            *get_values(constraint.state.upper_bounds),
            # *get_values(constraint.add_state.upper_bounds)
        ]
    )

    # slack on state constraints
    ocp.constraints.idxsbx = np.array([])
    nsbx = ocp.constraints.idxsbx.shape[0]

    ocp.constraints.idxsh = np.array([
        # get_index(constraint.expr, constraint.state.variables.v_min_diff),
        # get_index(constraint.expr, constraint.state.variables.v_max_diff),
        # get_index(constraint.expr, constraint.state.variables.theta_min_diff),
        # get_index(constraint.expr, constraint.state.variables.theta_max_diff),

        # get_index(constraint.expr, constraint.input.variables.along_min_diff),
        # get_index(constraint.expr, constraint.input.variables.along_max_diff),
        # get_index(constraint.expr, constraint.input.variables.ddelta_min_diff),
        # get_index(constraint.expr, constraint.input.variables.ddelta_max_diff),
        # get_index(constraint.expr, constraint.input.variables.dtheta_min_diff),
        # get_index(constraint.expr, constraint.input.variables.dtheta_max_diff),

        # get_index(constraint.expr, constraint.add_state.variables.n_min_diff_0),
        # get_index(constraint.expr, constraint.add_state.variables.n_max_diff_0),
        # get_index(constraint.expr, constraint.add_state.variables.n_min_diff_1),
        # get_index(constraint.expr, constraint.add_state.variables.n_max_diff_1),
        # get_index(constraint.expr, constraint.add_state.variables.n_min_diff_2),
        # get_index(constraint.expr, constraint.add_state.variables.n_max_diff_2),
    ])
    ocp.constraints.idxsh_0 = np.array([
        # get_index(constraint.expr_0, constraint.input.variables.along_min_diff),
        # get_index(constraint.expr_0, constraint.input.variables.along_max_diff),
        # get_index(constraint.expr_0, constraint.input.variables.ddelta_min_diff),
        # get_index(constraint.expr_0, constraint.input.variables.ddelta_max_diff),
        # get_index(constraint.expr_0, constraint.input.variables.dtheta_min_diff),
        # get_index(constraint.expr_0, constraint.input.variables.dtheta_max_diff),

        # get_index(constraint.expr_0, constraint.add_state.variables.n_min_diff_0),
        # get_index(constraint.expr_0, constraint.add_state.variables.n_max_diff_0),
        # get_index(constraint.expr_0, constraint.add_state.variables.n_min_diff_1),
        # get_index(constraint.expr_0, constraint.add_state.variables.n_max_diff_1),
        # get_index(constraint.expr_0, constraint.add_state.variables.n_min_diff_2),
        # get_index(constraint.expr_0, constraint.add_state.variables.n_max_diff_2),
    ])
    ocp.constraints.idxsh_e = np.array([
        # get_index(constraint.expr_e, constraint.state.variables.v_min_diff),
        # get_index(constraint.expr_e, constraint.state.variables.v_max_diff),
        # get_index(constraint.expr_e, constraint.state.variables.theta_min_diff),
        # get_index(constraint.expr_e, constraint.state.variables.theta_max_diff),

        # get_index(constraint.expr_e, constraint.add_state.variables.n_min_diff_0),
        # get_index(constraint.expr_e, constraint.add_state.variables.n_max_diff_0),
        # get_index(constraint.expr_e, constraint.add_state.variables.n_min_diff_1),
        # get_index(constraint.expr_e, constraint.add_state.variables.n_max_diff_1),
        # get_index(constraint.expr_e, constraint.add_state.variables.n_min_diff_2),
        # get_index(constraint.expr_e, constraint.add_state.variables.n_max_diff_2),
    ])
    nsh = ocp.constraints.idxsh.shape[0]
    nsh_0 = ocp.constraints.idxsh_0.shape[0]
    nsh_e = ocp.constraints.idxsh_e.shape[0]

    # slack gradients
    ns = nsh + nsbx
    ns_0 = nsh_0 + nsbx
    ns_e = nsh_e + nsbx

    grad_val: float = 100
    ocp.cost.zl = grad_val * np.ones((ns,))  # gradient wrt lower slack at intermediate shooting nodes
    ocp.cost.zu = grad_val * np.ones((ns,))  # gradient wrt upper slack at intermediate shooting nodes
    ocp.cost.Zl = 1 * np.ones((ns,))  # 𝑍𝑙 - diagonal of Hessian wrt lower slack at intermediate shooting nodes
    ocp.cost.Zu = 1 * np.ones((ns,))  # 𝑍𝑢 - diagonal of Hessian wrt upper slack at intermediate shooting nodes
    ocp.cost.zl_0 = grad_val * np.ones((ns_0,))  # gradient wrt lower slack at first shooting nodes
    ocp.cost.zu_0 = grad_val * np.ones((ns_0,))  # gradient wrt upper slack at first shooting nodes
    ocp.cost.Zl_0 = 1 * np.ones((ns_0,))  # 𝑍𝑙 - diagonal of Hessian wrt lower slack at first shooting nodes
    ocp.cost.Zu_0 = 1 * np.ones((ns_0,))  # 𝑍𝑢 - diagonal of Hessian wrt upper slack at first shooting nodes
    ocp.cost.zl_e = grad_val * np.ones((ns_e,))  # gradient wrt lower slack at final shooting nodes
    ocp.cost.zu_e = grad_val * np.ones((ns_e,))  # gradient wrt upper slack at final shooting nodes
    ocp.cost.Zl_e = 1 * np.ones((ns_e,))  # 𝑍𝑙 - diagonal of Hessian wrt lower slack at final shooting nodes
    ocp.cost.Zu_e = 1 * np.ones((ns_e,))  # 𝑍𝑢 - diagonal of Hessian wrt upper slack at final shooting nodes

    # set initial condition
    ocp.constraints.x0 = model.x0

    ocp.parameter_values = np.zeros(model.params.N_parameters)

    # set QP solver and integration
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpOptions
    ocp.solver_options.tf = 7.0
    ocp.solver_options.N_horizon = 70
    ocp.solver_options.ext_fun_compile_flags = "-O3 -march=native"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"  # GAUSS_NEWTON, EXACT
    ocp.solver_options.hpipm_mode = "BALANCE"  # String in (‘BALANCE’, ‘SPEED_ABS’, ‘SPEED’, ‘ROBUST’). Default: ‘BALANCE’.
    ocp.solver_options.integrator_type = "ERK"  # String in (‘ERK’, ‘IRK’, ‘GNSF’, ‘DISCRETE’, ‘LIFTED_IRK’). Default: ‘ERK’.
    ocp.solver_options.sim_method_num_stages = 1  # Number of stages in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 4
    ocp.solver_options.sim_method_num_steps = 1  # Number of steps in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 1

    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.regularize_method = "PROJECT"  # works, faster in paper
    # ocp.solver_options.regularize_method = "MIRROR"
    # ocp.solver_options.regularize_method = "NO_REGULARIZE"

    ocp.solver_options.qp_solver_cond_N = int(ocp.solver_options.N_horizon / 2)
    # ocp.solver_options.print_level = 1

    json_filename: str = "acados_" + suffix + ".json"
    if USE_CYTHON:
        if fc.check_rebuild():
            AcadosOcpSolver.generate(ocp, json_file=json_filename)
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            fc.write_stamp()
        ocp_solver = AcadosOcpSolver.create_cython_solver(json_file=json_filename)
    else:
        ocp_solver = AcadosOcpSolver(ocp, json_file=json_filename)

    return ocp_solver, ocp.solver_options.N_horizon, ocp.solver_options.tf
