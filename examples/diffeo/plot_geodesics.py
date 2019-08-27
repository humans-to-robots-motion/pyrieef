# http://cardona.co/math/geodesics/
# General imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display, Math

# Basic imports and functions
from sympy import latex, symbols, sin, cos, pi, simplify
from scipy.integrate import solve_ivp


from sympy.diffgeom import (
    Manifold, 
    Patch, 
    CoordSystem,
    metric_to_Christoffel_2nd,
    TensorProduct as TP
)

# def lprint(v):
#     display(Math(latex(v)))

# Create a manifold.
M = Manifold('M', 4)

# Create a patch.
patch = Patch('P', M)

# Basic symbols
c, r_s = symbols('c r_s')

# Coordinate system
schwarzchild_coord = CoordSystem(
    'schwarzchild', patch, ['t', 'r', 'theta', 'phi'])

# Get the coordinate functions
t, r, theta, phi = schwarzchild_coord.coord_functions()

# Get the base one forms.
dt, dr, dtheta, dphi = schwarzchild_coord.base_oneforms()

# Auxiliar terms for the metric.
dt_2 = TP(dt, dt)
dr_2 = TP(dr, dr)
dtheta_2 = TP(dtheta, dtheta)
dphi_2 = TP(dphi, dphi)
factor = (1 - r_s / r)

# Build the metric
metric = factor * c ** 2 * dt_2 - 1 / factor * dr_2 - r ** 2 * (
    dtheta_2 + sin(theta)**2 * dphi_2)
metric = factor * c ** 2 * dt_2 - 1 / factor * dr_2 - r ** 2 * (
    dtheta_2 + sin(theta)**2 * dphi_2)
metric = metric / c ** 2

# Get the Christoffel symbols of the second kind.
christoffel = metric_to_Christoffel_2nd(metric)

# Let's print this in an elegant way ;)
# for i in range(4):
#     for j in range(4):
#         for k in range(4):
#             if christoffel[i, j, k] != 0:
#                 display(Math('\Gamma^{}_{{{},{}}} = '.format(i, j, k) + latex(christoffel[i, j, k])))

## Specify c and r_s
christoffel_ = christoffel.subs({c: 2, r_s: 1})

def F(t, y):
    u = y[0:4]
    v = y[4:8]
    
    du = v
    dv = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dv[i] -= christoffel_.subs({r: u[1]})[i,j,k] * v[j] * v[k]
                
    return np.concatenate((du, dv))

T = 50
sol = solve_ivp(F, [0, T], [0, 10, 0, 0, 1, 0, 0, 0],
    t_eval=np.linspace(0, T, T * 123 + 1))

plt.figure(figsize=(14, 6),)
plt.plot(sol.y[0], sol.y[1])
ax = plt.gca()
ax.axhline(1, color="red", ls='--', lw=1)
plt.grid()
plt.ylim((0, 11))
plt.xlabel('t')
plt.ylabel('r')

T = 200
sol = solve_ivp(F, [0, T], [0, 10, 0, 0, 1, 0, 0.04, 0],
    t_eval=np.linspace(0, T, T * 123 + 1))

plt.figure(figsize=(14, 14),)
ax = plt.subplot(111, projection='polar')
ax.plot(sol.y[2], sol.y[1])
ax.grid(True)
plt.show()