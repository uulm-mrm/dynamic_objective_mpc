from typing import Optional
import numba
import numpy as np

class Violations:
    def __init__(self):
        self.velocity: Optional[int] = None
        self.theta: Optional[int] = None
        self.delta: Optional[int] = None

        self.along: Optional[int] = None
        self.ddelta: Optional[int] = None
        self.dtheta: Optional[int] = None

        self.boundary: Optional[int] = None

class ViolationsChecker:
    def __init__(self, logger):
        self.logger = logger

        self.violations = Violations()

        self.trajX: Optional = None
        self.trajU: Optional = None
        self.param_setter: Optional = None
        self.extra_vals: Optional = None

    def get_violations(self):
        for attr, value in self.violations.__dict__.items():
            yield attr, value

    def set(self):
        # state
        self.violations.velocity = self.get_first_violation(["v_max_diff", "v_min_diff"])
        self.violations.delta = self.get_first_violation(["delta_max_diff", "delta_min_diff"])
        self.violations.theta = self.get_first_violation(["theta_max_diff", "theta_min_diff"])

        # input
        self.violations.along = self.get_first_violation(["along_max_diff", "along_min_diff"], is_input=True)
        self.violations.ddelta = self.get_first_violation(["ddelta_max_diff", "ddelta_min_diff"], is_input=True)
        # self.violations.dtheta = self.get_first_violation(["dtheta_max_diff", "dtheta_min_diff"], is_input=True)

        # frenet
        # self.violations.boundary = self.is_boundary_violated()

    def update_refs(self, trajX, trajU, param_setter, extra_vals):
        self.trajX = trajX
        self.trajU = trajU
        self.param_setter = param_setter
        self.extra_vals = extra_vals

    def update(self, trajX, trajU, param_setter, extra_vals):
        self.update_refs(trajX, trajU, param_setter, extra_vals)
        self.set()

    def get_current_constr_values(self, name: str, is_input: bool = False):
        constraint = getattr(self.extra_vals, name)
        if is_input:
            return np.squeeze(constraint(self.trajX[:, -1], self.trajU, self.param_setter.get("all")))
        return np.squeeze(constraint(self.trajX, self.param_setter.get("all")))

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def is_violated(constraint: list) -> Optional[int]:
        for i, val in enumerate(constraint):
            if val < -1e-1:
                return i
        return None

    def get_first_violation(self, constr_names: list[str], is_input: bool = False) ->Optional[int]:
        for name in constr_names:
            viol_idx: Optional[int] = self.is_violated(self.get_current_constr_values(name, is_input))
            if viol_idx is not None:
                return viol_idx
        return None

    def is_boundary_violated(self) -> Optional[int]:
        base_names: list = ["n_min_diff_", "n_max_diff_"]
        constr_names: list = []
        for i in range(1):  # TODO is wrong self.model.params.nb_disks
            for name in base_names:
                var_name = name + str(i)
                constr_names.append(var_name)

        return self.get_first_violation(constr_names)

    def are_constraints_violated(self):
        for attr, value in self.get_violations():
            if value is not None:
                self.logger.error(f"Constr. violation at var {attr} at {value}")
                return True
        return False