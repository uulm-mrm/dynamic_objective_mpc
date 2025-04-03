from collections import deque


class History:
    def __init__(self):
        # initialize data structs for history
        self.history_len: int = 1000
        self.prev_history_len: int = self.history_len
        self.first_t: float = 0.0
        self.sim_t: deque = deque(maxlen=self.history_len)
        self.sim_x: deque = deque(maxlen=self.history_len)
        self.sim_y: deque = deque(maxlen=self.history_len)
        self.sim_phi: deque = deque(maxlen=self.history_len)
        self.sim_v: deque = deque(maxlen=self.history_len)
        self.sim_a: deque = deque(maxlen=self.history_len)
        self.sim_a_lat: deque = deque(maxlen=self.history_len)
        self.sim_delta: deque = deque(maxlen=self.history_len)

        self.sim_ddelta: deque = deque(maxlen=self.history_len)
        self.sim_thetadot: deque = deque(maxlen=self.history_len)

    def check_len(self):

        if self.history_len == self.prev_history_len:
            return

        while len(self.sim_t) > self.history_len:
            self.sim_t.popleft()
            self.sim_x.popleft()
            self.sim_y.popleft()
            self.sim_phi.popleft()
            self.sim_v.popleft()
            self.sim_a.popleft()
            self.sim_a_lat.popleft()
            self.sim_delta.popleft()
            self.sim_ddelta.popleft()
            self.sim_thetadot.popleft()

        self.prev_history_len = self.history_len

