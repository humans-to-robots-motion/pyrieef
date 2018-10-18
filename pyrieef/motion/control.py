
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

import scipy
import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigvals
from numpy import dot
import trajectory


def controller_lqr_discrete_time(A, B, Q, R):
    """
    Solve the constdiscrete time LQR controller for a
    discrete time constant system.

        A and B are system matrices, describing the systems dynamics:

            x_{t+1} = A x_t + B u_t

    The controller minimizes the
    infinite horizon quadratic cost function:

            cost = sum x_t^T Q x_t + u_t^T R u_t

        where Q is a positive semidefinite matrix,
        and R is positive definite matrix.

    Parameters
    ----------
        A, B, Q, R : numpy matrices

    Returns
    ----------
        K : gain the optimal gain K,
        P : the solution matrix X, and the closed loop system 
        eigVals: eigenvalues.

        The optimal input is then computed as:
             input: u = -K*x
    """
    assert type(A) == np.matrix
    assert type(B) == np.matrix
    assert type(Q) == np.matrix
    assert type(R) == np.matrix

    print "A : ", A.shape
    print "B : ", B.shape
    print "Q : ", Q.shape
    print "R : ", R.shape

    # first, try to solve the ricatti equation
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = np.matrix(inv(B.T * P * B + R) * (B.T * P * A))

    # Compute eigen values
    eigVals = eigvals(A - B * K)

    return K, P, eigVals


class KinematicTrajectoryFollowingLQR:

    def __init__(self, dt, trajectory):
        self._dt = dt
        self._n = trajectory.n()
        self._trajectory = trajectory
        self._K_matrix = None

    def solve_ricatti(self, Q_p, Q_v, R_a):
        """
        State and action are pulerly kinematic and defined as :

        q_t : configuration
        v_t = \dot q_t : velocity
        a_t = \ddot q_t : acceleration

        x_t = [q_t, v_t]
        u_t = a_t

        the dynamical system is described as:

            q_{t+1} = q_t + v_t * dt + a_t * dt^2
            v_{t+1} = v_t + a_t * dt

        the policy should be:

            a_{t+1} = K (x_t - x_td) + a_td

        where x_td and a_td are the state and acceleration
        along the trajectory.

        Parameters
        ----------

        Q_p : float, position gain
        Q_v : float, velocity gain
        R_a : float, control cost

        """
        # dynamics matrix
        A = np.matrix(np.vstack([
            np.hstack([np.eye(self._n), self._dt * np.eye(self._n)]),
            np.hstack([np.zeros((self._n, self._n)), np.eye(self._n)])]))

        # control matrix
        B = np.matrix(np.vstack([
            (self._dt ** 2) * np.eye(self._n), self._dt * np.eye(self._n)]))

        # state gain
        Q = np.matrix(np.zeros((2 * self._n, 2 * self._n)))
        Q[:self._n, :self._n] = Q_p * np.eye(self._n)
        Q[self._n:, self._n:] = Q_v * np.eye(self._n)

        # control gain
        R = np.matrix(R_a * np.eye(self._n))

        # solve Ricatti equation
        self._K_matrix, X, eigVals = controller_lqr_discrete_time(A, B, Q, R)

    def policy(self, t, x_t):
        """
        Computes the desired acceleration given an LQR error term gain.

            u = K (x - x_d) + u_d

            where x_d and u_d are the state and control along the trajectory

        Parameters
        ----------
            t : float
            x_t : ndarray
        """
        assert self._K_matrix is not None
        assert x_t.shape[0] == self._n * 2
        assert x_t.shape[1] == 1
        i = int(t / self._dt)  # index on trajectory
        x_d = self._trajectory.state(i, self._dt).reshape(self._n * 2, 1)
        a_t = self._trajectory.acceleration(i, self._dt)
        e_t = x_t - x_d
        u_t = self._K_matrix * e_t + a_t.reshape(self._n, 1)
        return u_t
