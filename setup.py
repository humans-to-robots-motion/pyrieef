from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        "pyrieef/geometry/differentiable_geometry.pyx",
        "pyrieef/geometry/workspace.pyx",
        "pyrieef/motion/cost_terms.pyx"])
)