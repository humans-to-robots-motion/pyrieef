#!/usr/bin/env python

# Copyright (c) 2015 Max Planck Institute
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
# Jim Mainprice on Sunday June 17 2018
from __init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from motion.objective import *
import time


def test_finite_differences():
    dim = 4
    acceleration = FiniteDifferencesAcceleration(dim, 1)
    print acceleration.jacobian(np.zeros(dim * 3))
    assert check_jacobian_against_finite_difference(acceleration)

    velocity = FiniteDifferencesVelocity(dim, 1)
    print velocity.jacobian(np.zeros(dim * 3))
    assert check_jacobian_against_finite_difference(acceleration)


def test_cliques():
    A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cliques = [A[i:3 + i] for i in range(len(A) - 2)]
    print A
    print cliques
    assert len(cliques) == (len(A) - 2)

    dimension = 10
    network = CliquesFunctionNetwork(dimension, 1)
    x_0 = np.zeros(3)
    for _ in range(network.nb_cliques()):
        network.add_function(SquaredNorm(x_0))
    cliques = network.all_cliques(A)
    print cliques
    assert len(cliques) == (len(A) - 2)


def test_trajectory():
    T = 10

    traj = Trajectory(T)
    print type(traj)
    print traj

    size = 2 * (T + 2)  # This is the formula for n = 2

    traj.set(np.ones(size))
    print type(traj)
    print str(traj)

    traj.set(np.random.rand(size))
    print type(traj)
    print str(traj)

    print "final configuration : "
    print traj.final_configuration()

    print "config 3 : "
    print traj.configuration(3)

    print "clique 3 : "
    print traj.clique(3)

    print "config 3 (ones) : "
    traj.configuration(3)[:] = np.ones(2)
    print traj.configuration(3)

    print "final configuration (ones) : "
    traj.final_configuration()[:] = np.ones(2)
    print traj.final_configuration()


def test_continuous_trajectory():
    q_init = np.random.random(2)
    q_goal = np.random.random(2)
    trajectory_1 = linear_interpolation_trajectory(q_init, q_goal, 10)
    trajectory_2 = ContinuousTrajectory(7, 2)
    trajectory_2.set(linear_interpolation_trajectory(q_init, q_goal, 7).x())
    for k, s in enumerate(np.linspace(0., 1., 11)):
        q_1 = trajectory_2.configuration_at_parameter(s)
        q_2 = trajectory_1.configuration(k)
        assert_allclose(q_1, q_2)


def test_obstacle_potential():
    workspace = Workspace()
    for center, radius in sample_circles(nb_circles=10):
        workspace.obstacles.append(Circle(center, radius))
    sdf = SignedDistanceWorkspaceMap(workspace)
    phi = ObstaclePotential2D(sdf)
    print "Checkint Obstacle Potential"
    assert check_jacobian_against_finite_difference(phi)

    phi = SimplePotential2D(sdf)
    print "Checkint Simple Potential"
    assert check_jacobian_against_finite_difference(phi)

    phi = CostGridPotential2D(sdf, 10, 0.1, 1.)
    print "Checkint Grid Potential"
    assert check_jacobian_against_finite_difference(phi)


def test_squared_norm_derivatives():

    f = SquaredNormVelocity(2, dt=0.1)

    print "Check SquaredNormVelocity (J implementation) : "
    assert check_jacobian_against_finite_difference(f)

    print "Check SquaredNormVelocity (H implementation) : "
    assert check_hessian_against_finite_difference(f)

    f = SquaredNormAcceleration(2, dt=0.1)

    print "Check SquaredNormAcceleration (J implementation) : "
    assert check_jacobian_against_finite_difference(f)

    print "Check SquaredNormAcceleration (H implementation) : "
    assert check_hessian_against_finite_difference(f)


def calculate_analytical_gradient_speedup(f, nb_points=10):
    samples = np.random.rand(nb_points, f.input_dimension())
    time1 = time.time()
    [f.gradient(x) for x in samples]
    time2 = time.time()
    t_analytic = (time2 - time1) * 1000.0
    print '%s function took %0.3f ms' % ("analytic", t_analytic)
    time1 = time.time()
    [finite_difference_jacobian(f, x) for x in samples]
    time2 = time.time()
    t_fd = (time2 - time1) * 1000.0
    print '%s function took %0.3f ms' % ("finite diff", t_fd)
    print " -- speedup : {} x".format(int(round(t_fd / t_analytic)))


def test_motion_optimimization_2d():
    print "Checkint Motion Optimization"
    objective = MotionOptimization2DCostMap()
    trajectory = Trajectory(objective.T)
    sum_acceleration = objective.cost(trajectory)
    print "sum_acceleration : ", sum_acceleration
    q_init = np.zeros(2)
    q_goal = np.ones(2)
    trajectory = linear_interpolation_trajectory(
        q_init, q_goal, objective.T)
    print trajectory
    print trajectory.final_configuration()
    sum_acceleration = objective.cost(trajectory)
    print "sum_acceleration : ", sum_acceleration
    assert check_jacobian_against_finite_difference(
        objective.objective)

    # Calulate speed up.
    # print "Calculat analytic gradient speedup"
    # calculate_analytical_gradient_speedup(objective.objective)


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
        assert (np.linalg.norm(network.function_on_clique(t, x_t) -
                               x_t[2:4]) < 1.e-10)


def test_motion_optimimization_smoothness_metric():
    print "Checkint Motion Optimization"
    objective = MotionOptimization2DCostMap()
    A = objective.create_smoothness_metric()


def test_optimize():
    print "Checkint Motion Optimization"
    q_init = np.zeros(2)
    objective = MotionOptimization2DCostMap()
    objective.optimize(q_init)

if __name__ == "__main__":
    test_trajectory()
    test_continuous_trajectory()
    test_cliques()
    test_finite_differences()
    test_squared_norm_derivatives()
    test_obstacle_potential()
    test_motion_optimimization_2d()
    test_motion_optimimization_smoothness_metric()
    test_optimize()
    test_center_of_clique()
