import demos_common_imports
from pyrieef.motion.objective import *
import matplotlib.pyplot as plt

dim = 1
trajectory = linear_interpolation_trajectory(
    q_init=np.zeros(dim),
    q_goal=np.ones(dim),
    T=20
)
objective = MotionOptimization2DCostMap(
    T=trajectory.T(),
    n=trajectory.n(),
    q_init=trajectory.initial_configuration(),
    q_goal=trajectory.final_configuration()
)
objective.create_clique_network()
objective.add_smoothness_terms(2)
objective.create_objective()

H1 = objective.objective.hessian(trajectory.active_segment())
np.set_printoptions(suppress=True, linewidth=200, precision=0,
                    formatter={'float_kind': '{:8.0f}'.format})
print H1[dim * 10:, dim * 10:]

H2 = objective.create_smoothness_metric()
np.set_printoptions(suppress=True, linewidth=200, precision=0,
                    formatter={'float_kind': '{:8.0f}'.format})


A_inv = np.linalg.inv(H2)
print np.max(A_inv)
A_inv /= np.max(A_inv)
plt.plot(A_inv)
plt.show()
