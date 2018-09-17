from geometry.workspace import *
from motion.trajectory import *
import numpy as np


def collision_check_trajectory(workspace, trajectory):
    """ Check trajectory for collision """
    delta = workspace.box.diag() / 100.
    interpolated_traj = trajectory.continuous_trajectory()
    length = interpolated_traj.length()
    for s in np.linspace(0, 1, num=int(length / delta) + 1):
        if workspace.in_collision(
                interpolated_traj.configuration_at_parameter(s)):
            return True
    return False


def collision_check_linear_interpolation(workspace, p_init, p_goal):
    """ Check interior interpolation for collision """
    delta = workspace.box.diag() / 100.
    length = np.linalg.norm(p_init - p_goal)
    for s in np.linspace(0, 1, num=int(length / delta) + 1):
        p_interp = (1. - s) * p_init + s * p_goal
        if workspace.in_collision(p_interp):
            return True
    return False
