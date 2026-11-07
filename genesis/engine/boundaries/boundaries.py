import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import *
from genesis.utils.repr import brief


@qd.data_oriented
class CubeBoundary:
    def __init__(self, lower, upper, restitution=0.0):
        self.restitution = restitution

        self.upper = np.array(upper, dtype=gs.np_float)
        self.lower = np.array(lower, dtype=gs.np_float)
        assert (self.upper >= self.lower).all()

        self.upper_qd = qd.Vector(upper, dt=gs.qd_float)
        self.lower_qd = qd.Vector(lower, dt=gs.qd_float)

    @qd.func
    def impose_pos_vel(self, pos, vel):
        for i in qd.static(range(3)):
            if pos[i] >= self.upper_qd[i] and vel[i] >= 0:
                vel[i] *= -self.restitution
            elif pos[i] <= self.lower_qd[i] and vel[i] <= 0:
                vel[i] *= -self.restitution

        pos = qd.max(qd.min(pos, self.upper_qd), self.lower_qd)

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        pos = qd.max(qd.min(pos, self.upper_qd), self.lower_qd)
        return pos

    def is_inside(self, pos):
        return np.all(pos < self.upper) and np.all(pos > self.lower)

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"lower       : {brief(self.lower)}\n"
            f"upper       : {brief(self.upper)}\n"
            f"restitution : {brief(self.restitution)}"
        )
