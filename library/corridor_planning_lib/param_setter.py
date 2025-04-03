from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ParamType(str, Enum):
    p = "p_list"
    p_global = "p_global_list"


class ParamSetter:
    def __init__(self, model, ptype: ParamType = ParamType.p, N: int = 1):

        # print("\nSetup Param setter for", model.name)
        s_idx = 0
        for i, param in enumerate(getattr(model, ptype)):
            e_idx = s_idx + param.size()[0]
            # print("p", param.name(), " from ", s_idx, " to ", e_idx)
            setattr(self, param.name() + "_idxs", np.arange(s_idx, e_idx))
            s_idx = e_idx

        # print("param size", model.p.size())
        # print("index up to", e_idx)
        assert (model.p.size()[0] == e_idx)

        self.param_arr: NDArray = np.zeros(model.params.N_parameters)

        # only for debug, takes 1ms to populate
        self.all_param: NDArray = np.zeros((model.params.N_parameters, N + 1))

    def set(self, param_str: str, value, nb_extrapolate: int = 0) -> None:
        """
        set single param with given values and indices
        :param param_str:
        :param value:
        :param nb_extrapolate:
        :return:
        """

        indices = getattr(self, param_str + "_idxs")

        # is scalar
        if np.isscalar(value):
            self.param_arr[indices] = value
        else:
            # data lengths are equal
            if len(indices) == len(value):
                self.param_arr[indices] = value

            # data lengths differ
            else:
                # set minimum of actual length and size of param arr
                data_length: int = min(len(value), len(indices))
                # assign part of list
                self.param_arr[indices[:data_length]] = value[:data_length]

                remaining_idxs = indices[data_length:]
                if remaining_idxs.size > 0 and value.size > 0:
                    # set rest to last value
                    self.param_arr[remaining_idxs] = value[-1]

                    if nb_extrapolate != 0:
                        # do extrapolation
                        last_val = value[data_length-1]
                        before_last_val = value[data_length-2]
                        d_val = last_val - before_last_val

                        nb_extrapolate = min(nb_extrapolate, remaining_idxs.size)
                        for i in range(nb_extrapolate):
                            next_val = last_val + (i+1) * d_val
                            self.param_arr[remaining_idxs[i]] = next_val


    def get(self, param_str: str, i: int = 0) -> NDArray:
        if param_str == "all":
            return self.all_param[:, i]

        indices = getattr(self, param_str + "_idxs")

        return self.all_param[indices, i]

    def save_for_debug(self, i: int):
        self.all_param[:, i] = self.param_arr
