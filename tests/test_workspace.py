import test_common_imports
from geometry.workspace import *


def test_ellipse():

    ellipse = Ellipse()
    ellipse.a = 0.1
    ellipse.b = 0.2

    dist = ellipse.DistFromBorder(np.array([0.3, 0.0]))
    print "dist = ", dist
    assert np.fabs(dist - 0.2) < 1.e-06

    dist = ellipse.DistFromBorder(np.array([0.0, 0.3]))
    print "dist = ", dist
    assert np.fabs(dist - 0.1) < 1.e-06

test_ellipse()
