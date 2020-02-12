#!/usr/bin/env python

# Copyright (c) 2020, University of Stuttgart
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
#                                 Jim Mainprice on Wednesday February 12 2020

import __init__
from geometry.interpolation import *


def test_1d_solution():
    """
    # Matlab test:

     % Data
     X = [-1, .5, 0, .5, 1]'
     Y = [.1, .2, .3, .4, .5]'
     x_query = .3
     lambda = .1

     % Augment with constant feature
     Xa = [X, ones(size(X,1),1)]

     % Calculate linear function
     W = diag(exp(-.5*(x_query - X).^2))
     beta = inv(Xa'*W*Xa+lambda*eye(2,2))*Xa'*W*Y

     beta =
        0.166173
        0.257212

     % Evaluate qt query point
     beta'*[x_query; 1]
     ans = 0.307063690815270
     """

    N = 5
    X = np.array([-1, .5, 0, .5, 1])
    Y = np.array([.1, .2, .3, .4, .5])

    x_query = .3
    ridge_lambda = .1

    X.shape = (N, 1)
    Y.shape = (N, 1)
    Xa = np.hstack((X, np.ones((N, 1))))

    W = np.empty(N)
    for i in range(W.size):
        W[i] = np.exp(-.5 * (x_query - X[i]) ** 2)
    W = np.diag(W)

    H = np.dot(np.dot(Xa.T, W), Xa)  # Xa.T * W * Xa
    Pinv = H + ridge_lambda * np.eye(2)
    beta = np.dot(np.linalg.inv(Pinv), np.dot(Xa.T, np.dot(W, Y)))
    assert np.abs(beta[0] - 0.166173) < 1e-5
    assert np.abs(beta[1] - 0.257212) < 1e-5

    # Evaluate qt query point
    value = np.dot(beta.T, np.array([x_query, 1]))
    assert np.abs(value - 0.307063690815270) < 1e-7


def test_lwr():
    """
    See above : test_1d_solution
    """
    # Setup data.
    X = np.array([-1, .5, 0, .5, 1])
    Y = np.array([.1, .2, .3, .4, .5])
    x_query = np.array([.3])
    D = np.ones(1)
    ridge_lambda = .1
    value = locally_weighted_regression(x_query, X, Y, D, ridge_lambda)
    print(value)
    assert np.abs(value - 0.307063690815270) < 1e-7


def test_mahalanobis_tools_basic_square_distance():
    verbose = False

    D = np.eye(3)
    x1 = np.random.rand(3)
    x2 = np.random.rand(3)

    expected_square_distance = np.linalg.norm(x1 - x2) ** 2
    calculated_square_distance = mahalanobis_square_distance(x1, x2, D)

    if verbose:
        print(expected_square_distance)
        print(calculated_square_distance)
    assert np.abs(expected_square_distance - calculated_square_distance) < 1e-7


def random_mahalanobis_matrix(dim):
    return np.diag(np.absolute(np.random.rand(dim)))


def test_mahalanobis_tools_neighborhood_distance_threshold():
    verbose = False
    D = random_mahalanobis_matrix(3)

    x_data = np.random.rand(3)
    x = np.random.rand(3)

    weight_at_x = lwr_weight(x, x_data, D)
    equiv_threshold_dist = distance_at_weight_threshold(weight_at_x)
    dist = mahalanobis_distance(x, x_data, D)
    if verbose:
        print(equiv_threshold_dist)
        print(dist)
    assert np.abs(equiv_threshold_dist - dist) < 1e-7


def test_mahalanobis_tools_test_rescale_mahalanobis():
    verbose = True
    D1 = random_mahalanobis_matrix(3)
    x_data = np.random.rand(3)
    x = np.random.rand(3)
    initial_distance = mahalanobis_distance(x, x_data, D1)
    desired_weight = .001
    if verbose:
        print(D1)

    D2 = rescale_mahalanobis_metric(initial_distance, desired_weight, D1)
    if verbose:
        print(D2)

    scaled_weight = lwr_weight(x, x_data, D2)
    if verbose:
        print(desired_weight)
        print(scaled_weight)
    assert np.abs(desired_weight - scaled_weight) < 1e-7


# def test_signed_distance_interpolation():
#     workspace = Workspace()
#     workspace.obstacles = [Circle(origin=[.0, .0], radius=0.1)]
#     sdf = SignedDistanceWorkspaceMap(workspace)
#     grid = workspace.box.stacked_meshgrid(20)
#     X = np.empty((20 ** 2, 2))
#     Y = np.empty((20 ** 2))
#     k = 0
#     for i, j in itertools.product(range(grid.shape[1]),
#                 range(grid.shape[2])):
#         X[k, :] = grid[:, i, j]
#         Y[k] = sdf(X[k, :])
#         k += 1
#     sdf_inter = LWR(1, 2)
#     sdf_inter.X = [X]
#     sdf_inter.Y = [Y]
#     sdf_inter.D = [np.eye(2)]
#     sdf_inter.ridge_lambda = [.1]


if __name__ == "__main__":
    test_1d_solution()
    test_lwr()
    test_mahalanobis_tools_basic_square_distance()
    test_mahalanobis_tools_neighborhood_distance_threshold()
    test_mahalanobis_tools_test_rescale_mahalanobis()
