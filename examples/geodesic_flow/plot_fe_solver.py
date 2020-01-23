import numpy as np
import sympy as sym
import time

x, t, a, L = sym.symbols('x t a L')
u = x * (L - x) * 5 * t


def pde(u):
    return sym.diff(u, t) - a * sym.diff(u, x, x)


f = sym.simplify(pde(u))
a = 0.5
L = 1.5
u_exact = sym.lambdify(
    [x, t], u.subs('L', L).subs('a', a), modules='numpy')
f = sym.lambdify(
    [x, t], f.subs('L', L).subs('a', a), modules='numpy')
I = lambda x: u_exact(x, 0)


def solver_FE_simple(I, a, f, L, dt, F, T):
    """
    Simplest expression of the computational algorithm
    using the Forward Euler method and explicit Python loops.
    For this method F <= 0.5 for stability.
    """
    t0 = time.clock()  # For measuring the CPU time

    Nt = int(round(T / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)   # Mesh points in time
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[i] = u_n[i] + F * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + \
                dt * f(x[i], t[n])

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0

        # Switch variables before next step
        # u_n[:] = u  # safe, but slow
        u_n, u = u, u_n

    t1 = time.clock()
    return u_n, x, t, t1 - t0  # u_n holds latest u


def solver_FE_vectorized(I, a, f, L, dt, F, T, version='scalar'):
    """
    Vectorized implementation of solver_FE_simple.
    """
    t0 = time.clock()  # for measuring the CPU time

    Nt = int(round(T / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)   # Mesh points in time
    dx = np.sqrt(a * dt / F)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros(Nx + 1)   # solution array
    u_n = np.zeros(Nx + 1)   # solution at t-dt

    # Set initial condition
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Update all inner points
        if version == 'scalar':
            for i in range(1, Nx):
                u[i] = u_n[i] +\
                    F * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) +\
                    dt * f(x[i], t[n])

        elif version == 'vectorized':
            u[1:Nx] = u_n[1:Nx] +  \
                F * (u_n[0:Nx - 1] - 2 * u_n[1:Nx] + u_n[2:Nx + 1]) +\
                dt * f(x[1:Nx], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0

        # Switch variables before next step
        u_n, u = u, u_n

    t1 = time.clock()
    return u_n, x, t, t1 - t0  # u_n holds latest u


def test_solver_FE():
    # Define u_exact, f, I as explained above

    dx = L / 3  # 3 cells
    F = 0.5
    dt = F * dx**2

    u, x, t, cpu = solver_FE_simple(
        I=I, a=a, f=f, L=L, dt=dt, F=F, T=2)
    u_e = u_exact(x, t[-1])
    diff = abs(u_e - u).max()
    print("diff : ", diff)
    print("cpu : ", cpu)
    # tol = 1E-14
    # assert diff < tol, 'max diff solver_FE_simple: %g' % diff

    u, x, t, cpu = solver_FE_vectorized(
        I=I, a=a, f=f, L=L, dt=dt, F=F, T=2, version='scalar')
    # user_action=None, version='scalar')
    u_e = u_exact(x, t[-1])
    diff = abs(u_e - u).max()
    print("diff : ", diff)
    print("cpu : ", cpu)
    # tol = 1E-14
    # assert diff < tol, 'max diff solver_FE, scalar: %g' % diff

    u, x, t, cpu = solver_FE_vectorized(
        I=I, a=a, f=f, L=L, dt=dt, F=F, T=2, version="vectorized")
    # user_action=None, version='vectorized')
    u_e = u_exact(x, t[-1])
    diff = abs(u_e - u).max()
    print("diff : ", diff)
    print("cpu : ", cpu)
    # tol = 1E-14
    # assert diff < tol, 'max diff solver_FE, vectorized: %g' % diff


if __name__ == '__main__':
    print("testing FE solvers...")
    test_solver_FE()
    print("Done.")
