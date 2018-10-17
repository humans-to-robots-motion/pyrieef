
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
import np
from np.linalg import inv
from np.linalg import eigvals
from np import dot
import trajectory


def controller_lqr_discrete_time(A, B, Q, R):
    """
    Solve the constdiscrete time LQR controller for a 
    discrete time constant system.

        A and B are system matrices, describing the systems dynamics:

            x_{t+1} = A x_t + B u_t

        The controller minimizes the 
        infinite horizon quadratic cost function:

            cost = sum x_r^T Q x_t + u_t^T R u_t

        where Q is a positive semidefinite matrix, 
        and R is positive definite matrix.

        Returns K, X, eigVals:
            Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
            The optimal input is then computed as:
             input: u = -K*x
    """
    assert type(A) == np.matrix
    assert type(B) == np.matrix
    assert type(Q) == np.matrix
    assert type(R) == np.matrix

    # first, try to solve the ricatti equation
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = np.matrix(inv(B.T * P * B + R) * (B.T * P * A))

    # Compute eigen values
    eigVals = eigvals(A - B * K)

    return K, X, eigVals


class KinematicTrajectoryFollowingLQR:

    def __init__(self):
        return None

    def solve_ricatti(self, n, dt, Q_p, Q_v, R_a):
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
        """
        # dynamics matrix
        A = np.matrix(np.vstack([
            np.hstack([np.eye(n), dt * np.eye(n)]),
            np.hstack([np.zeros((n, n)), np.eye(n)])]))

        # control matrix
        B = np.matrix(np.vstack([(dt ** 2) * np.eye(n), dt * np.eye(n)]))

        # state gain
        Q = np.matrix(np.vstack([Q_p * np.eye(n), Q_v * np.eye(n)]))

        # control gain
        R = np.matrix(R_a * np.eye(n))

        # solve Ricatti equation
        K, X, eigVals = controller_lqr_discrete_time(A, B, Q, R)

        return K

    def policy(self, t, x_t, K, trajectory, dt):
        """
        Computes the desired acceleration given an LQR error term gain.

            u = K (x - x_d) + u_d

            where x_d and u_d are the state and control along the trajectory
        """
        s_t = trajectory.state(t, dt)
        a_t = trajectory.acceleration(t, dt)
        e_t = x_t - np.vstack([s_t(0), s_t(1)])
        u_t = K * e_t + a_t
        return u_t
