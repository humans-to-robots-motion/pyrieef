import demos_common_imports
from pyrieef.motion.objective import *
import matplotlib.pyplot as plt

motion = MotionOptimization2DCostMap(22, 1)
A = motion.create_smoothness_metric()
A_inv = np.linalg.inv(A)
print np.max(A_inv)
A_inv /= np.max(A_inv)
plt.plot(A_inv)
plt.show()
