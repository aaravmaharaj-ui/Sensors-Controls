
import numpy as np


def true_pose(sim):
    d, b = sim.d, sim.base_id
    R = d.xmat[b].reshape(3, 3)
    return np.array([d.xpos[b][0], d.xpos[b][1], np.arctan2(R[1, 0], R[0, 0])])
