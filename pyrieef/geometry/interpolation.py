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

import numpy as np


def locally_weighted_regression(x_query, X, Y, D, ridge_lambda):
    """
    VectorXd & x_query
    MatrixXd & X
    VectorXd & Y
    MatrixXd & D
    ridge_lambda

    Calculates the locally weighted regression at the query point.
     Parameters:
      x_query is a column vector with the query point.
      X's rows contain domain points.
      Y's entries contain corresponding targets.
      D gives the Mahalanobis metric as:
      dist(x_query, x) = sqrt( (x_query - x)'D(x_query - x) )
      ridge_lambda is the regression regularizer, denoted lambda in the
      calculation below

      Calculates:
          beta^* = argmin 1/2 |Y - X beta|_W^2 + lambda/2 |w|^2

          with W a diagonal matrix with elements
                        w_i = \exp{ -1/2 |x_query - x_i|_D^2

      Solution: beta^* = inv(X'WX + lambda I)X'WY.
      Final returned value: beta^*'x_query.

      Note that all points are augmented with an extra
      constant feature to handle the bias.
   """
    # Default value is 0. The calculation uses ridge regression with a finite
    # regularizer, so the values should diminish smoothly to 0 away from the
    # data set anyway.
    if Y.size == 0:
        return 0.

    # Compatibility with 1 dim arrays
    X.shape = (X.size, 1) if X.ndim == 1 else X.shape

    # The "augmented" version of X has an extra constant
    # feature to represent the bias.
    Xaug = np.ones((X.shape[0], X.shape[1] + 1))
    Xaug[:, :-1] = X

    x_query_aug = np.ones((x_query.size + 1))
    x_query_aug[:-1] = x_query

    # Compute weighted points:
    # WX, where W is the diagonal matrix of weights.
    WX = np.empty(Xaug.shape)
    for i in range(X.shape[0]):
        w = lwr_weight(X[i, :].T, x_query, D)
        WX[i, :] = w * Xaug[i, :]

    # Fit plane to the weighted data
    diag = np.diag(ridge_lambda * np.ones(Xaug.shape[1]))

    # Calculate Pinv=X'WX + lambda I.
    # P = inv(Pinv) is then P = inv(X'WX + lambda I).
    beta = np.dot(np.linalg.inv(np.dot(WX.T, Xaug) + diag), np.dot(WX.T, Y))
    # beta=inv(X'WX + lambda I)WX'Y
    # Return inner product between plane and querrie point
    return np.dot(beta.T, x_query_aug)


def mahalanobis_square_distance(x1, x2, D):
    diff = (x1 - x2)
    return np.dot(diff.T, np.dot(D, diff))


def mahalanobis_distance(x1, x2, D):
    return np.sqrt(mahalanobis_square_distance(x1, x2, D))


def lwr_weight_from_dist(square_distance):
    return np.exp(-.5 * square_distance)


def lwr_weight(x, x_data, D):
    return lwr_weight_from_dist(mahalanobis_square_distance(x, x_data, D))


def distance_at_weight_threshold(weight_threshold):
    return np.sqrt(-2. * np.log(weight_threshold))


def rescale_mahalanobis_metric(
        distance_threshold,
        corresponding_weight_threshold, D):
    """
     Rescale the Mahalanobis metric so that the weight
     at the specified distance is the given threshold.
     D is both an input parameter specifying the
     Mahalanobis metric and an output parameter storing the scaled matrix.
     It's assumed that distance_threshold
     is in units given by the Mahalanobis metric.

     Algebra: exp{ -s d^2/2} = w,
     solve for s (d is the distance threshold, w is
     the corresponding weight threshold).
     Solution: s = -2/d^2 log(w)
    """
    d_squared = distance_threshold ** 2
    return D * (-2. / d_squared) * np.log(corresponding_weight_threshold)
