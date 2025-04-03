from collections import deque

import numpy as np
from numpy.typing import NDArray


class DeadTimeComp:
    def __init__(self, nu: int, mpc_period: float):
        self.mpc_period: float = mpc_period
        self.nu: int = nu
        self.u_fifo: deque = deque()

    def get_next_x0(self, x0: NDArray, u0: NDArray, integrator, dead_time: float, trans_state: int, platform_id: str) -> NDArray:

        # change size if needed
        nb_needed: int = int(dead_time / self.mpc_period) + 1
        nb_diff: int = nb_needed - len(self.u_fifo)
        must_add: bool = nb_diff > 0
        for i in range(abs(nb_diff)):
            if must_add:
                self.u_fifo.append(np.zeros(self.nu))
            else:
                if len(self.u_fifo) > 0:
                    self.u_fifo.popleft()

        # predict
        integrator.set("T", self.mpc_period)
        for u in self.u_fifo:
            x0 = integrator.simulate(x0, u)
            if platform_id != "ushift":
                # commands are only correct if gear is set correctly
                if trans_state == 1:
                    x0[3] = max(x0[3], 0.0)
                elif trans_state == -1:
                    x0[3] = min(x0[3], 0.0)
                else:
                    x0[3] = 0.0

        # insert u_0
        self.u_fifo.append(u0)

        # drop one element
        self.u_fifo.popleft()

        return x0
