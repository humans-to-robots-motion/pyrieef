import os
import sys

driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory)
sys.path.insert(0, driectory + os.sep + "../pyrieef")

import numpy as np
from numpy.testing import assert_allclose
from itertools import product

print("import init")
