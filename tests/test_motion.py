#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday June 13 2018

from __init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from motion.objective import *
from motion.control import *
import time
from numpy.linalg import norm
from numpy.testing import assert_allclose

np.random.seed(0)


def test_finite_differences():
    dim = 4
    acceleration = FiniteDifferencesAcceleration(dim, 1)
    print((acceleration.jacobian(np.zeros(dim * 3))))
    assert check_jacobian_against_finite_difference(acceleration)

    velocity = FiniteDifferencesVelocity(dim, 1)
    print((velocity.jacobian(np.zeros(dim * 3))))
    assert check_jacobian_against_finite_difference(acceleration)

    trajectory = Trajectory(T=10, n=2)
    trajectory.x()[:] = np.random.random(trajectory.x().size)
    print(trajectory)

    dt = 0.1
    vel_2d = FiniteDifferencesVelocity(2, dt)
    acc_2d = FiniteDifferencesAcceleration(2, dt)
    for t in range(1, trajectory.T()):
        """ Warning velocity FD are right sided in the trajectory class and
            left sided when compuded on the clique """
        v_t = trajectory.velocity(t, dt)
        a_t = trajectory.acceleration(t, dt)
        c_t = trajectory.clique(t)
        assert_allclose(v_t, vel_2d(c_t[:4]))
        assert_allclose(a_t, acc_2d(c_t))


def test_integration():

    dt = 0.1
    trajectory = Trajectory(T=20, n=2)
    trajectory.x()[:] = np.random.random(trajectory.x().size)
    # print trajectory

    """
    Test velocity integration
        we test that a kinematic trajectory can be recovered
        by integrating the finite difference velocity profile
        q_t1 = q_t + v_t * dt
        """
    trajectory_zero = Trajectory(T=20, n=2)
    trajectory_zero.x()[:] = np.zeros(trajectory.x().size)
    q_t0 = trajectory.configuration(0).copy()
    trajectory_zero.configuration(0)[:] = q_t0
    for t in range(1, trajectory.T() + 2):
        v_t = trajectory.velocity(t, dt)
        q_t = trajectory_zero.configuration(t - 1)
        q_t1 = q_t + v_t * dt
        trajectory_zero.configuration(t)[:] = q_t1
        assert_allclose(q_t1, trajectory.configuration(t))

    assert_allclose(trajectory_zero.x(), trajectory.x())

    """
    Test acceleration integration
        we test that a kinematic trajectory can be recovered
        by integrating the finite difference acceleration profile
        q_t1 = q_t + v_t * dt + a_t * (dt ** 2)
        """
    trajectory_zero = Trajectory(T=20, n=2)
    trajectory_zero.x()[:] = np.zeros(trajectory.x().size)
    q_t0 = trajectory.configuration(0).copy()
    trajectory_zero.configuration(0)[:] = q_t0
    for t in range(trajectory.T() + 1):
        q_t = trajectory_zero.configuration(t)
        v_t = trajectory_zero.velocity(t, dt)
        a_t = trajectory.acceleration(t, dt)
        q_t1 = q_t + v_t * dt + a_t * (dt ** 2)
        trajectory_zero.configuration(t + 1)[:] = q_t1
        assert_allclose(q_t1, trajectory.configuration(t + 1))


def test_cliques():
    A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cliques = [A[i:3 + i] for i in range(len(A) - 2)]
    print(A)
    print(cliques)
    assert len(cliques) == (len(A) - 2)

    dimension = 10
    network = CliquesFunctionNetwork(dimension, 1)
    x_0 = np.zeros(3)
    for _ in range(network.nb_cliques()):
        network.add_function(SquaredNorm(x_0))
    cliques = network.all_cliques(A)
    print(cliques)
    assert len(cliques) == (len(A) - 2)

    T = 10
    n = 2

    trajectory = Trajectory(T, n)
    trajectory.x()[:] = np.random.random(trajectory.x().size)
    print(trajectory)

    network = CliquesFunctionNetwork((n * (T + 2)), n)
    for i, c in enumerate(network.all_cliques(trajectory.x())):
        assert_allclose(c, trajectory.clique(i + 1))


def test_trajectory():
    T = 10
    n = 2

    traj = Trajectory(T)
    print((type(traj)))
    print(traj)

    size = n * (T + 2)  # This is the formula for n = 2

    traj.set(np.ones(size))
    print((type(traj)))
    print((str(traj)))

    traj.set(np.random.rand(size))
    print((type(traj)))
    print((str(traj)))

    print("final configuration : ")
    print((traj.final_configuration()))

    print("config 3 : ")
    print((traj.configuration(3)))

    print("clique 3 : ")
    print((traj.clique(3)))

    print("config 3 (ones) : ")
    traj.configuration(3)[:] = np.ones(n)
    print((traj.configuration(3)))

    print("final configuration (ones) : ")
    traj.final_configuration()[:] = np.ones(n)
    print((traj.final_configuration()))

    x_active = np.random.random(n * (T + 1))
    traj = Trajectory(q_init=np.zeros(n), x=x_active)
    print(("x_active : ", x_active))
    print(("traj.x : ", traj.x()))
    assert traj.x().size == size
    assert np.isclose(traj.x()[:2], np.zeros(n)).all()

    traj_continuous = traj.continuous_trajectory()
    assert traj_continuous.x().size == traj.x().size
    assert np.isclose(traj.x(), traj_continuous.x()).all()


def test_continuous_trajectory():
    q_init = np.random.random(2)
    q_goal = np.random.random(2)
    trajectory_1 = linear_interpolation_trajectory(q_init, q_goal, 10)
    trajectory_2 = ContinuousTrajectory(7, 2)
    trajectory_2.set(linear_interpolation_trajectory(q_init, q_goal, 7).x())
    for k, s in enumerate(np.linspace(0., 1., trajectory_1.T() + 1)):
        q_1 = trajectory_2.configuration_at_parameter(s)
        q_2 = trajectory_1.configuration(k)
        assert_allclose(q_1, q_2)

    q_init = np.random.random(2)
    q_goal = np.random.random(2)
    trajectory_1 = linear_interpolation_trajectory(q_init, q_goal, 100)
    trajectory_2 = linear_interpolation_trajectory(q_init, q_goal, 5)
    trajectory_2 = trajectory_2.continuous_trajectory()
    for t in range(trajectory_1.T() + 1):
        s = float(t) / float(trajectory_1.T())
        print(("s : {} , t : {}".format(s, t)))
        q_2 = trajectory_1.configuration(t)
        q_1 = trajectory_2.configuration_at_parameter(s)
        assert_allclose(q_1, q_2)


def test_constant_acceleration_trajectory():
    dt = 0.1
    T = 7
    n = 2
    q_init = np.random.random(n)
    q_goal = np.random.random(n)
    trajectory_1 = linear_interpolation_trajectory(q_init, q_goal, T)
    trajectory_2 = ConstantAccelerationTrajectory(T, n, dt)
    trajectory_2.set(trajectory_1.x())
    # for k, s in enumerate(np.linspace(0., 1., trajectory_1.T() + 1)):
    #     q_1 = trajectory_2.velocity_at_parameter(s)
    #     q_2 = trajectory_1.velocity(k, dt)
    #     assert_allclose(q_1, q_2)


def test_center_of_clique():
    config_dim = 2
    nb_way_points = 10
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(2),
        q_goal=np.ones(2),
        T=nb_way_points)
    network = CliquesFunctionNetwork(trajectory.x().size, config_dim)
    center_of_clique = network.center_of_clique_map()
    network.register_function_for_all_cliques(center_of_clique)
    for t, x_t in enumerate(network.all_cliques(trajectory.x())):
        assert (np.linalg.norm(network.clique_value(t, x_t) -
                               x_t[2:4]) < 1.e-10)


def test_obstacle_potential():

    np.random.seed(0)

    workspace = Workspace()
    for center, radius in sample_circles(nb_circles=10):
        workspace.obstacles.append(Circle(center, radius))
    sdf = SignedDistanceWorkspaceMap(workspace)
    phi = ObstaclePotential2D(sdf)
    print("Checkint Obstacle Potential")
    assert check_jacobian_against_finite_difference(phi)

    phi = SimplePotential2D(sdf)
    print("Checkint Simple Potential Gradient")
    assert check_jacobian_against_finite_difference(phi)
    print("Checkint Simple Potential Hessian")
    assert check_hessian_against_finite_difference(phi)

    phi = CostGridPotential2D(sdf, 10, 0.1, 1.)
    print("Checkint Grid Potential Gradient")
    assert check_jacobian_against_finite_difference(phi)
    print("Checkint Grid Potential Hessian")
    assert check_hessian_against_finite_difference(phi)


def test_squared_norm_derivatives():

    dt = 0.1
    n = 2

    f_v = SquaredNormVelocity(n, dt)

    print("Check SquaredNormVelocity (J implementation) : ")
    assert check_jacobian_against_finite_difference(f_v)

    print("Check SquaredNormVelocity (H implementation) : ")
    assert check_hessian_against_finite_difference(f_v)

    f_a = SquaredNormAcceleration(n, dt)

    print("Check SquaredNormAcceleration (J implementation) : ")
    assert check_jacobian_against_finite_difference(f_a)

    print("Check SquaredNormAcceleration (H implementation) : ")
    assert check_hessian_against_finite_difference(f_a)

    T = 20
    trajectory = Trajectory(T, n)
    trajectory.x()[:] = np.random.random(trajectory.x().size)

    f_v2 = Pullback(SquaredNorm(np.zeros(n)),
                    FiniteDifferencesVelocity(n, dt))

    f_a2 = Pullback(SquaredNorm(np.zeros(n)),
                    FiniteDifferencesAcceleration(n, dt))

    for t in range(1, trajectory.T() + 1):
        c_t = trajectory.clique(t)
        assert abs(f_v2(c_t[n:]) - f_v(c_t[n:])) < 1e-10
        assert abs(f_a2(c_t[0:]) - f_a(c_t[0:])) < 1e-10


def test_log_barrier():

    np.random.seed(0)

    f = LogBarrierFunction()

    print("Check BoundBarrier (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check BoundBarrier (H implementation) : ")
    assert check_hessian_against_finite_difference(f)


def test_bound_barrier():

    np.random.seed(0)

    v_lower = np.array([0, 0])
    v_upper = np.array([1, 1])
    f = BoundBarrier(v_lower, v_upper)

    print("Check BoundBarrier (J implementation) : ")
    assert check_jacobian_against_finite_difference(f)

    print("Check BoundBarrier (H implementation) : ")
    assert check_hessian_against_finite_difference(f)


def test_motion_optimimization_smoothness_metric():
    print("Checkint Motion Optimization")
    objective = MotionOptimization2DCostMap()
    A = objective.create_smoothness_metric()


def calculate_analytical_gradient_speedup(f, nb_points=10):
    samples = np.random.rand(nb_points, f.input_dimension())
    time1 = time.time()
    [f.gradient(x) for x in samples]
    time2 = time.time()
    t_analytic = (time2 - time1) * 1000.0
    print(('%s function took %0.3f ms' % ("analytic", t_analytic)))
    time1 = time.time()
    [finite_difference_jacobian(f, x) for x in samples]
    time2 = time.time()
    t_fd = (time2 - time1) * 1000.0
    print(('%s function took %0.3f ms' % ("finite diff", t_fd)))
    print((" -- speedup : {} x".format(int(round(t_fd / t_analytic)))))


def test_motion_optimimization_2d():

    np.random.seed(0)

    print("Check Motion Optimization (Derivatives)")
    problem = MotionOptimization2DCostMap()
    problem.set_test_objective()

    trajectory = Trajectory(problem.T)
    sum_acceleration = problem.cost(trajectory)
    print(("sum_acceleration : ", sum_acceleration))
    q_init = np.zeros(2)
    q_goal = np.ones(2)
    trajectory = linear_interpolation_trajectory(
        q_init, q_goal, problem.T)
    print(trajectory)
    print((trajectory.final_configuration()))
    sum_acceleration = problem.cost(trajectory)
    print(("sum_acceleration : ", sum_acceleration))

    print("Test J for trajectory")
    assert check_jacobian_against_finite_difference(
        problem.objective, False)

    # Check the hessian of the trajectory
    print("Test H for trajectory")
    is_close = check_hessian_against_finite_difference(
        problem.objective, False, tolerance=1e-2)

    xi = np.random.rand(problem.objective.input_dimension())
    H = problem.objective.hessian(xi)
    H_diff = finite_difference_hessian(problem.objective, xi)
    H_delta = H - H_diff
    print((" - H_delta dist = ", np.linalg.norm(H_delta, ord='fro')))
    print((" - H_delta maxi = ", np.max(np.absolute(H_delta))))

    assert is_close

    # Calulate speed up.
    # print "Calculat analytic gradient speedup"
    # calculate_analytical_gradient_speedup(objective.objective)


def test_linear_interpolation():
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(2),
        q_goal=np.ones(2),
        T=22
    )
    q_1 = trajectory.configuration(0)
    q_2 = trajectory.configuration(1)
    dist = norm(q_1 - q_2)
    for i in range(1, trajectory.T() + 1):
        q_1 = trajectory.configuration(i)
        q_2 = trajectory.configuration(i + 1)
        dist_next = norm(q_1 - q_2)
        assert abs(dist_next - dist) < 1.e-10


def test_linear_interpolation_velocity():
    dim = 2
    dt = 0.1
    deriv = SquaredNormVelocity(dim, dt)
    deriv_comp = Pullback(
        SquaredNorm(np.zeros(dim)), FiniteDifferencesVelocity(dim, dt))
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(dim),
        q_goal=np.ones(dim),
        T=22
    )
    q_1 = trajectory.configuration(0)
    q_2 = trajectory.configuration(1)
    g_traj = np.zeros(trajectory.x().shape)
    clique = np.append(q_1,  q_2)
    velocity = deriv(clique)
    gradient_1 = deriv.gradient(clique)
    for i in range(0, trajectory.T() + 1):
        q_1 = trajectory.configuration(i)
        q_2 = trajectory.configuration(i + 1)
        clique = np.append(q_1,  q_2)
        velocity_next = deriv(clique)
        gradient_2 = deriv.gradient(clique)
        # g_traj[i + 1: i + 1 + dim] += gradient_2
        print(("i = {}, g2 : {}".format(i, gradient_2)))
        assert abs(velocity - velocity_next) < 1.e-10
        assert norm(deriv_comp.gradient(clique) - gradient_2) < 1.e-10
        assert norm(gradient_1[0:2] + gradient_2[2:4]) < 1.e-10
    print(g_traj)


def test_linear_interpolation_optimal_potential():
    """ makes sure that the start and goal potentials
        are applied at the correct place """
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(2),
        q_goal=np.ones(2),
        T=10
    )
    objective = MotionOptimization2DCostMap(
        T=trajectory.T(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()
    )
    # print "q_init  : ", trajectory.initial_configuration()
    # print "q_goal  : ", trajectory.final_configuration()
    # objective.create_clique_network()
    # objective.add_init_and_terminal_terms()
    # objective.create_objective()
    # v = objective.objective.forward(trajectory.active_segment())
    # g = objective.objective.gradient(trajectory.active_segment())
    # print v
    # assert abs(v - 0.) < 1.e-10
    # assert np.isclose(g, np.zeros(trajectory.active_segment().shape)).all()

    # TODO test velocity profile. Gradient should correspond.
    # This is currently not the case.
    # assert check_jacobian_against_finite_difference(objective.objective,
    # False)
    objective.create_clique_network()
    objective.add_smoothness_terms(2)
    objective.create_objective()
    v = objective.objective.forward(trajectory.active_segment())
    g = objective.objective.gradient(trajectory.active_segment())
    g_diff = finite_difference_jacobian(
        objective.objective, trajectory.active_segment())
    print(("v : ", v))
    print(("x : ", trajectory.active_segment()))
    print(("g : ", g))
    print(("g_diff : ", g_diff))
    print((g.shape))
    assert np.isclose(
        g, np.zeros(trajectory.active_segment().shape), atol=1e-5).all()


def test_smoothness_metric():
    dim = 2
    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(dim),
        q_goal=np.ones(dim),
        T=10
    )
    objective = MotionOptimization2DCostMap(
        T=trajectory.T(),
        n=trajectory.n(),
        q_init=trajectory.initial_configuration(),
        q_goal=trajectory.final_configuration()
    )
    objective.set_scalars(acceleration_scalar=1.)
    objective.create_clique_network()
    objective.add_smoothness_terms(2)
    objective.create_objective()

    active_size = dim * (trajectory.T() - 1)

    H1 = objective.objective.hessian(trajectory.active_segment())
    H1 = H1[:active_size, :active_size]
    np.set_printoptions(suppress=True, linewidth=200, precision=0,
                        formatter={'float_kind': '{:8.0f}'.format})

    H2 = objective.create_smoothness_metric()
    H2 = H2[:active_size, :active_size]
    np.set_printoptions(suppress=True, linewidth=200, precision=0,
                        formatter={'float_kind': '{:8.0f}'.format})

    print((H1[:10, :10]))
    print((H2[:10, :10]))

    assert_allclose(H1, H2)


def test_trajectory_objective():
    q_init = np.zeros(2)
    problem = MotionOptimization2DCostMap(T=10, n=q_init.size)
    problem.set_test_objective()
    objective = TrajectoryObjectiveFunction(q_init, problem.function_network)
    assert check_jacobian_against_finite_difference(objective, False)
    assert check_hessian_against_finite_difference(objective, False, 1e-3)


def test_optimize():
    print("Check Motion Optimization (optimize)")
    q_init = np.zeros(2)
    objective = MotionOptimization2DCostMap()
    objective.optimize(q_init, nb_steps=5, optimizer="natural_gradient")
    objective.optimize(q_init, nb_steps=5, optimizer="newton")


def test_trajectory_following():
    dt = 0.1
    dim = 2
    trajectory = Trajectory(T=20, n=dim)
    trajectory.x()[:] = np.random.random(trajectory.x().size)
    lqr = KinematicTrajectoryFollowingLQR(dt, trajectory)
    lqr.solve_ricatti(1., 1., 1.)
    for i in range(trajectory.T()):
        x_t = trajectory.state(i, dt).reshape(dim * 2, 1)
        x_t += .1 * np.random.random(x_t.shape)
        u_t = lqr.policy(i * dt, x_t)
        q_t = trajectory.configuration(i)
        assert u_t.size == q_t.size


if __name__ == "__main__":
    # test_finite_differences()
    # test_integration()
    # test_cliques()
    # test_trajectory()
    # test_continuous_trajectory()
    test_constant_acceleration_trajectory()
    # test_squared_norm_derivatives()
    # test_log_barrier()
    # test_bound_barrier()
    # test_obstacle_potential()
    # test_motion_optimimization_2d()
    # test_motion_optimimization_smoothness_metric()
    # test_center_of_clique()
    # test_linear_interpolation()
    # test_linear_interpolation_velocity()
    # test_linear_interpolation_optimal_potential()
    # test_smoothness_metric()
    # test_trajectory_objective()
    # test_optimize()
    # test_trajectory_following()
