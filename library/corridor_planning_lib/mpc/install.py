import os
import sys

if __name__ == "__main__":

    install_path = None
    if sys.argv:
        if len(sys.argv) == 2:
            if sys.argv[1]:
                install_path = sys.argv[1]

    if install_path is None:
        raise Exception("Install path not given cannot install optimizer")

    # add manually to python path to allow acces on first install
    sys.path.append(install_path)

    from corridor_planning_lib.mpc.real_vehicle_settings import get_real_integrator, get_real_bicycle_model
    from corridor_planning_lib.mpc.ocp_settings import get_solver, get_bicycle_model
    from corridor_planning_lib.mpc.integrator_settings import get_integrator, get_integrator_bicycle_model

    # to trigger files in install folder
    os.chdir(install_path + "/corridor_planning_lib/mpc/")

    real_model = get_real_bicycle_model()
    real_integrator = get_real_integrator(real_model)

    model, constraint, extra_vals = get_bicycle_model()
    acados_solver, N, Tf = get_solver(model, constraint)

    integrator_model = get_integrator_bicycle_model()
    integrator = get_integrator(integrator_model)
