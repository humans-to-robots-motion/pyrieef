import demos_common_imports
from pyrieef.motion.motion_optimization import *
import matplotlib.pyplot as plt

motion = MotionOptimization2DCostMap(100, 1)
A = motion.create_smoothness_metric()
plt.plot(np.linalg.inv(A))
plt.show()
