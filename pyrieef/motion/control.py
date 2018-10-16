
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
import trajectory


def lqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.

        x_{t+1} = A x_t + B u_t

        cost = sum x_t^T Q x_t + u_t^T R u_t
    """
    # ref Bertsekas, p.151

    # first, try to solve the Ricatti equation
    try:
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    except LinAlgError:
        print "Error in LQR"

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


def trajectory_following_lqr(trajectory, dt):
    """
    State and action are pulerly kinematic and defined as :

        q_t : configuration
        v_t = \dot q_t : velocity
        a_t = \ddot q_t : acceleration

        x_t = [q_t, v_t]
        u_t = a_t

        the dynamical system is described as:

            q_t+1 = q_t + v_t * dt + a_t * dt^2
            v_t+1 = v_t + a_t * dt

    """
    n = trajectory.n()

    # Dynamics matrix
    A = np.matrix(np.vstack([
        np.hstack([np.eye(n), dt * np.eye(n)]),
        np.hstack([np.zeros((n, n)), np.eye(n)])]))

    # Control matrix
    B = np.matrix(np.vstack([(dt ** 2) * np.eye(n), dt * np.eye(n)]))
