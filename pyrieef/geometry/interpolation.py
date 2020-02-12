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
#                                    Jim Mainprice on Thursday January 23 2020

import numpy as np


def MahalanobisSquareDistance(x1, x2, D):
    diff = (x1 - x2)
    return diff.T * D * diff


def MahalanobisDistance(x1, x2, D):
    return np.sqrt(MahalanobisSquareDistance(x1, x2, D))


def LwrWeightFromDist(square_distance):
    return np.exp(-.5 * square_distance)


def LwrWeight(x, x_data, D):
    return LwrWeightFromDist(MahalanobisSquareDistance(x, x_data, D))


def LocallyWeightedRegression(x_query, X, Y, D, ridge_lambda):
    """
    VectorXd & x_query
    MatrixXd & X
    VectorXd & Y
    MatrixXd & D
    ridge_lambda

    Calculates the locally weighted regression at the query point.
     Parameters:
      x_query is a column vector with the query point.
      X's rows contain domain points. Y's entries contain
      corresponding targets.
      D gives the Mahalanobis metric as:
      dist(x_query, x) = sqrt( (x_query - x)'D(x_query - x) )
      ridge_lambda is the regression regularizer, denoted lambda in the
      calculation below

      Calculates:
          beta^* = \argmin 1/2 |Y - X beta|_W^2 + lambda/2 |w|^2

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

    # The "augmented" version of X has an extra constant
    # feature to represent the bias.
    Xaug = np.zeros((X.shape[0], X.shape[1] + 1))
    Xaug[:, :-1] = X
    Xaug[:, -1] = np.ones(Xaug.shape[0])

    x_query_aug = np.zeros((x_query.size() + 1))
    x_query_aug[:-1] = x_query
    x_query_aug[-1] = 1

    # Compute weighted points:
    # WX, where W is the diagonal matrix of weights.
    WX = np.array(Xaug.shape[0], Xaug.shape[1])
    for i in range(X.shape[0]):
        WX[i, :] = LwrWeight(X[i, :].T, x_query, D) * Xaug[i, :]  # check

    # Fit plane to the weighted data
    diag = np.diag((ridge_lambda * np.ones(Xaug.shape[1])))

    # Calculate Pinv=X'WX + lambda I. P = inv(Pinv) is then
    # P = inv(X'WX + lambda I).
    Pinv = np.dot(WX.T, Xaug) + diag
    P = np.linalg.inv(Pinv)
    beta = np.dot(np.dot(P * WX.T) * Y)
    # beta=inv(X'WX + lambda I)WX'Y
    # Return inner product between plane and querrie point
    return np.dot(beta.T, x_query_aug)
