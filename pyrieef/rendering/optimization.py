import workspace_renderer as renderer
from motion.trajectory import Trajectory
import time
import numpy as np


class TrajectoryOptimizationViewer:

    def __init__(self, objective, draw):
        self.objective = objective
        self.viewer = None
        if draw:
            self.init_viewer()

    def init_viewer(self):
        self.viewer = renderer.WorkspaceRender(
            self.objective.workspace)
        self.viewer.draw_ws_background(self.objective.obstacle_costmap())

    def evaluate(self, x):
        return self.objective.objective(x)

    def gradient(self, x, draw=True):
        g = self.objective.objective.gradient(x)
        if draw and self.viewer is not None:
            q_init = self.objective.q_init
            trajectory = Trajectory(q_init=q_init, x=x)
            g_traj = Trajectory(q_init=q_init, x=-0.01 * g + x)
            for k in range(self.objective.T + 1):
                q = trajectory.configuration(k)
                self.viewer.draw_ws_circle(
                    .01, q, color=(0, 0, 1) if k == 0 else (0, 1, 0))
                self.viewer.draw_ws_line(q, g_traj.configuration(k))
            self.viewer.render()
            time.sleep(0.1)
        return g

    def hessian(self, x):
        return np.array(self.objective.objective.hessian(x))
