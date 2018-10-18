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

import numpy as np
from .workspace import *


class ChargeSimulation:

    def __init__(self):
        self.charged_points_ = []
        self.q = None

    def PotentialDueToAPoint(self, p1, p2, charge):
        k = 1
        # The distance offset depends on the discretization of the mesh
        # in 2d add 1 cm
        return k * charge / (np.linalg.norm(p1 - p2) + 0.001)

    def PotentialCausedByObject(self, p):
        potential = 0.
        if self.q_ is None:
            print("charges not initialized")
            return potential
        for i in range(len(self.charged_points_)):
            potential += self.PotentialDueToAPoint(
                p, self.charged_points_[i], self.q_[i])
        return potential

    def ChargeMatrix(self, charged_points):
        nb_charges = len(charged_points)
        A = np.matrix(np.zeros((nb_charges, nb_charges)))
        print(("nb_charges : ", nb_charges))
        print(("start filling charge matrix (" + str(A.shape) + ")"))
        for i in range(nb_charges):
            for j in range(nb_charges):
                if i != j:
                    p1 = charged_points[i]
                    p2 = charged_points[j]
                    # print "p1 : ", p1 , " , p2 : ", p2
                    A[i, j] = self.PotentialDueToAPoint(p1, p2, 1)
                else:
                    A[i, j] = 10000  # in 2d 100000
        return A

    def Run(self):
        A = self.ChargeMatrix(self.charged_points_)
        B = np.array(np.ones(A.shape[0]))
        # print "A : ", A
        print("Solve linear system...")
        self.q_ = np.linalg.solve(A, B)
        # print "q : ", self.q_

# run the server
if __name__ == "__main__":
    simulation = ChargeSimulation()
    simulation.charged_points_ = Circle().sampled_points()
    simulation.Run()
