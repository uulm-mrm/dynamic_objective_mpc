from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from casadi import *
from corridor_planning_lib.mpc.file_checker import FileChecker

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

fc = FileChecker(__file__)

# longer compile time but faster interface to the solver
USE_CYTHON = True


def get_integrator_bicycle_model():
    model = types.SimpleNamespace()

    model_name = "integrator_bicycle"

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

    # params
    wb = MX.sym("wb", 1)

    p_list = [
        wb,
    ]

    p = vertcat(*p_list)
    N_parameters = p.shape[0]

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

    # Define initial conditions
    model.x0 = np.zeros(state.size()[0])

    # Define model struct
    params = types.SimpleNamespace()
    params.N_parameters = N_parameters

    # implicit dynamics for IRK
    model.f_impl_expr = vertcat(state_dot - f_expl)

    # explicit dynamics, for ERK
    model.f_expl_expr = f_expl

    model.x = state
    model.xdot = state_dot
    model.u = u
    model.p = p
    model.p_list = p_list
    model.name = model_name
    model.params = params
    return model


def get_integrator(model):
    # create render arguments
    sim = AcadosSim()
    suffix: str = "integrator"
    sim.code_export_directory = "c_generated_code_" + suffix

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.xdot - model.f_expl_expr  # define manually to avoid algebraic variables in integrator
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x  # CasADi variable describing the state of the system;
    model_ac.xdot = model.xdot  # CasADi variable describing the derivative of the state wrt time;
    model_ac.u = model.u  # CasADi variable describing the input of the system;
    model_ac.p = model.p  # CasADi variable describing parameters of the DAE;
    model_ac.name = model.name
    sim.model = model_ac

    sim.parameter_values = np.zeros(model.params.N_parameters)

    sim.solver_options.ext_fun_compile_flags = "-O3 -march=native"
    sim.solver_options.T = 0.1  # will be set
    sim.solver_options.integrator_type = "ERK"  # String in (‘ERK’, ‘IRK’, ‘GNSF’, ‘DISCRETE’, ‘LIFTED_IRK’). Default: ‘ERK’.
    sim.solver_options.sim_method_num_stages = 1  # Number of stages in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 4
    sim.solver_options.sim_method_num_steps = 1  # Number of steps in the integrator. Type: int > 0 or ndarray of ints > 0 of shape (N,). Default: 1

    json_filename: str = "acados_" + suffix + ".json"
    if USE_CYTHON:
        if fc.check_rebuild():
            AcadosSimSolver.generate(sim, json_file=json_filename)
            AcadosSimSolver.build(sim.code_export_directory, with_cython=True)
            fc.write_stamp()
        sim_integrator = AcadosSimSolver.create_cython_solver(json_file=json_filename)
    else:
        sim_integrator = AcadosSimSolver(sim, json_file=json_filename)

    return sim_integrator
