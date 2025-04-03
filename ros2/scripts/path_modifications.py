import math

import numpy as np
from corridor_planning_lib.data_structures import Path


def add_forward_straight(path: Path, params, length: float = 20, res: float = 0.1, start_pose=None) -> None:
    """
    Create straight path
    :return:
        """
    nb_elements = int(length / res)

    if start_pose is None:
        phi: float = path.phi[-1]
        x: float = path.x[-1]
        y: float = path.y[-1]
        s: float = path.s[-1]
    else:
        phi: float = start_pose[2]
        x: float = start_pose[0]
        y: float = start_pose[1]
        s: float = 0.0

    s_rot = np.linspace(res, length, nb_elements)
    phi_ref = phi * np.ones(nb_elements)
    kappa_ref = np.zeros(nb_elements)
    v_max = params.v_max * np.ones(nb_elements)

    # rotate
    phi_rot = phi_ref[0]
    x_ref = s_rot * np.cos(phi_rot) + x
    y_ref = s_rot * np.sin(phi_rot) + y
    dir_ref = np.ones(nb_elements, dtype=bool)

    s_ref = s_rot + s
    add_path = Path(s_ref, x_ref, y_ref, phi_ref, kappa_ref,
                    dir_ref, v_max, params.n_min, params.n_max)

    path.concatenate(add_path)


def add_backward_straight(path: Path, params, length: float = 20, res: float = 0.1, start_pose=None) -> None:
    """
    Create path with switch
    :return:
    """
    nb_elements = int(length / res)

    if start_pose is None:
        phi: float = path.phi[-1]
        x: float = path.x[-1]
        y: float = path.y[-1]
        s: float = path.s[-1]
    else:
        phi: float = start_pose[2]
        x: float = start_pose[0]
        y: float = start_pose[1]
        s: float = 0.0

    s_rot = np.linspace(res, length, nb_elements)
    phi_ref = phi * np.ones(nb_elements)
    kappa_ref = np.zeros(nb_elements)
    v_max = params.v_max * np.ones(nb_elements)

    # rotate
    phi_rot = phi_ref[0] + math.pi  # go backwards
    x_ref = s_rot * np.cos(phi_rot) + x
    y_ref = s_rot * np.sin(phi_rot) + y
    dir_ref = np.zeros(nb_elements, dtype=bool)

    s_ref = s_rot + s
    add_path = Path(s_ref, x_ref, y_ref, phi_ref, kappa_ref,
                    dir_ref, v_max, params.n_min, params.n_max)

    path.concatenate(add_path)


def add_forward_curve(path: Path, params, kappa: float, length: float = 20, res: float = 0.1, start_pose=None) -> None:
    """
    Create path with switch
    :return:
    """
    nb_elements = int(length / res)

    s_rot = np.linspace(res, length, nb_elements)

    if start_pose is None:
        phi: float = path.phi[-1]
        x: float = path.x[-1]
        y: float = path.y[-1]
        s: float = path.s[-1]
    else:
        phi: float = start_pose[2]
        x: float = start_pose[0]
        y: float = start_pose[1]
        s: float = 0.0

    x_ref = []
    y_ref = []
    phi_ref = []
    kappa_ref = kappa * np.ones(nb_elements)
    dir_ref = np.ones(nb_elements, dtype=bool)
    v_max = params.v_max * np.ones(nb_elements)
    for _ in s_rot:
        x += res * np.cos(phi)
        y += res * np.sin(phi)
        phi += kappa * res

        x_ref.append(x)
        y_ref.append(y)
        phi_ref.append(phi)

    s_ref = s_rot + s
    add_path = Path(s_ref, x_ref, y_ref, phi_ref, kappa_ref,
                    dir_ref, v_max, params.n_min, params.n_max)

    path.concatenate(add_path)


def add_backward_curve(path: Path, params, kappa: float, length: float = 20, res: float = 0.1, start_pose=None) -> None:
    """
    Create path with switch
    :return:
    """
    nb_elements = int(length / res)

    s_rot = np.linspace(res, length, nb_elements)

    if start_pose is None:
        phi: float = path.phi[-1]
        x: float = path.x[-1]
        y: float = path.y[-1]
        s: float = path.s[-1]
    else:
        phi: float = start_pose[2]
        x: float = start_pose[0]
        y: float = start_pose[1]
        s: float = 0.0
    x_ref = []
    y_ref = []
    phi_ref = []
    kappa_ref = kappa * np.ones(nb_elements)
    v_max = params.v_max * np.ones(nb_elements)
    dir_ref = np.zeros(nb_elements, dtype=bool)
    for _ in s_rot:
        x += -res * np.cos(phi)
        y += -res * np.sin(phi)
        phi += kappa * -res

        x_ref.append(x)
        y_ref.append(y)
        phi_ref.append(phi)

    s_ref = s_rot + s
    add_path = Path(s_ref, x_ref, y_ref, phi_ref, kappa_ref,
                    dir_ref, v_max, params.n_min, params.n_max)

    path.concatenate(add_path)
