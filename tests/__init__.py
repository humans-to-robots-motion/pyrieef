import os
import sys

directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, directory)
sys.path.insert(0, directory + os.sep + "../pyrieef")

import numpy as np
from numpy.testing import assert_allclose
from itertools import product

print("import init")
