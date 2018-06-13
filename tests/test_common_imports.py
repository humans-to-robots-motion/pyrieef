import sys
import os
import numpy as np

driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory + os.sep + "../src")


# Returns True of all variable are close.
def check_is_close(a, b, tolerance=1e-10):
    results = np.isclose(
        np.array(a),
        np.array(b),
        atol=tolerance)
    return results.all()
