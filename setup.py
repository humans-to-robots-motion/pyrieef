import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        "pyrieef/geometry/differentiable_geometry.pyx",
        "pyrieef/geometry/workspace.pyx",
        "pyrieef/motion/cost_terms.pyx"])
)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pyrieef',  
     version='0.1',
     scripts=['pyrieef'] ,
     author="Jim Mainprice",
     author_email="mainprice@gmail.com",
     description="Motion planning and control python package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/jmainpri/pyrieef",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
