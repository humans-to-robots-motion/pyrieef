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

from .__init__ import *
from motion.trajectory import *
from motion.cost_terms import *
from optimization.optimization import *
from geometry.differentiable_geometry import *
from geometry.workspace import *
from scipy import optimize


class MotionOptimization2DCostMap:

    def __init__(self, T=10, n=2,
                 box=EnvBox(np.array([0., 0.]), np.array([2., 2.])),
                 signed_distance_field=None,
                 costmap=None,
                 q_init=None,
                 q_goal=None):
        self.verbose = False
        self.config_space_dim = n       # nb of dofs
        self.T = T                      # time steps
        self.dt = 0.1                   # sample rate
        self.trajectory_space_dim = (self.config_space_dim * (self.T + 2))
        self.workspace = None
        self.signed_distance_field = signed_distance_field
        self.costmap = costmap
        self.objective = None

        self.q_goal = q_goal if q_goal is not None else .3 * np.ones(2)
        self.q_init = q_init if q_init is not None else np.zeros(2)

        self._eta = 10.
        self._obstacle_scalar = 1.
        self._init_potential_scalar = 0.
        self._term_potential_scalar = 10000000.
        self._term_velocity_scalar = 100000.
        self._velocity_scalar = 5.
        self._acceleration_scalar = 20.
        self._attractor_stdev = .1

        # We only need the signed distance field
        # to create a trajectory optimization problem
        self.box = box
        self.extent = box.extent()
        if self.signed_distance_field is None and self.costmap is None:
            self.create_sdf_hardcoded_workspace()

        # Create metric for natural gradient descent
        self.create_smoothness_metric()
        self.obstacle_potential_from_sdf()

        # Here we combine everything to make an objective
        # TODO see why n==1 doesn't work...
        if self.config_space_dim > 1:
            # Creates a differentiable clique function.
            self.create_clique_network()
            self.add_all_terms()
            self.create_objective()

    def set_problem(self, workspace, trajectory, obstacle_potential):
        self.workspace = workspace
        self.box = workspace.box
        self.extent = workspace.box.extent()
        self.signed_distance_field = SignedDistanceWorkspaceMap(workspace)
        self.obstacle_potential = obstacle_potential
        self.q_init = trajectory.initial_configuration()
        self.q_goal = trajectory.final_configuration()
        self.create_clique_network()
        self.add_all_terms()
        self.add_attractor(trajectory)
        self.create_objective()

    def set_scalars(self,
                    obstacle_scalar=1.,
                    init_potential_scalar=0.,
                    term_potential_scalar=10000000.,
                    velocity_scalar=1.,
                    acceleration_scalar=1):
        self._obstacle_scalar = obstacle_scalar
        self._init_potential_scalar = init_potential_scalar
        self._term_potential_scalar = term_potential_scalar
        self._velocity_scalar = velocity_scalar
        self._acceleration_scalar = acceleration_scalar
        self._term_velocity_scalar = 100000.

    def set_test_objective(self):
        """ This objective does not collide with the enviroment"""
        self.create_sdf_test_workspace()
        self.obstacle_potential_from_sdf()
        self.create_clique_network()
        self.create_objective()

    def set_eta(self, eta):
        self._eta = eta

    def obstacle_potential_from_sdf(self):
        self.obstacle_potential = SimplePotential2D(self.signed_distance_field)
        # self.obstacle_potential = CostGridPotential2D(
        #     self.signed_distance_field,
        #                            alpha=10.,
        #                            margin=.03,
        #                            offset=1.)
        return self.obstacle_potential

    def cost(self, trajectory):
        """ compute sum of acceleration """
        return self.objective.forward(trajectory.active_segment())

    def create_sdf_hardcoded_workspace(self):
        workspace = Workspace(self.box)
        p1 = np.array([0.2, .15])
        p2 = np.array([-.1, .15])
        workspace.obstacles.append(Circle(p1, .1))
        workspace.obstacles.append(Circle(p2, .1))
        # workspace.obstacles.append(Box(p2, .1))
        self.signed_distance_field = SignedDistanceWorkspaceMap(workspace)
        self.workspace = workspace
        return workspace

    def create_sdf_test_workspace(self):
        workspace = Workspace(self.box)
        p1 = np.array([2, 2])
        p2 = np.array([-2, 2])
        workspace.obstacles.append(Circle(p1, .1))
        workspace.obstacles.append(Circle(p2, .1))
        self.signed_distance_field = SignedDistanceWorkspaceMap(workspace)
        self.workspace = workspace
        return workspace

    def create_smoothness_metric(self):
        """ TODO this does not seem to work at all... """
        a = FiniteDifferencesAcceleration(1, self.dt).a()
        # print "a : "
        # print a
        no_variance = True
        K_dof = np.matrix(np.zeros((self.T + 1, self.T + 1)))
        for i in range(0, self.T + 1):
            if i == 0:
                K_dof[i, i:i + 2] = a[0, 1:3]
                # if no_variance:
                #     K_dof[i, i] *= 1000  # No variance at end points
            elif i == self.T:
                K_dof[i, i - 1:i + 1] = a[0, 0:2]
                if no_variance:
                    K_dof[i, i] *= 1000  # No variance at end points
            elif i > 0:
                K_dof[i, i - 1:i + 2] = a
        A_dof = K_dof.T * K_dof
        # print K_dof
        # print A_dof

        # represented in the form :  \xi = [q_0 ; q_1; ... ; q_2]
        K_full = np.matrix(np.zeros((
            self.config_space_dim * (self.T + 1),
            self.config_space_dim * (self.T + 1))))
        for dof in range(self.config_space_dim):
            for (i, j), K_ij in np.ndenumerate(K_dof):
                id_row = i * self.config_space_dim + dof
                id_col = j * self.config_space_dim + dof
                if id_row < K_full.shape[0] and id_col < K_full.shape[1]:
                    K_full[id_row, id_col] = K_ij
        # print K_full
        # print K_full.shape
        A = K_full.T * K_full
        self.metric = A
        return A

    def add_attractor(self, trajectory):
        """ Add an attractor to each clique scalled by the distance
            to the goal, it ensures that the trajectory does not slow down
            in time as it progresses towards the goal.
            This is Model Predictive Control grounded scheme.
            TODO check the literature to set this appropriatly. """
        alphas = np.zeros(trajectory.T())
        for t in range(1, trajectory.T()):
            dist = np.linalg.norm(self.q_goal - trajectory.configuration(t))
            alphas[t] = np.exp(-dist / (self._attractor_stdev ** 2))
        alphas /= alphas.sum()

        for t in range(1, trajectory.T()):
            potential = Pullback(
                SquaredNorm(self.q_goal),
                self.function_network.center_of_clique_map())
            self.function_network.register_function_for_clique(
                t, Scale(potential, alphas[t] * self._term_potential_scalar))

    def add_init_and_terminal_terms(self):

        if self._init_potential_scalar > 0.:
            initial_potential = Pullback(
                SquaredNorm(self.q_init),
                self.function_network.left_most_of_clique_map())
            self.function_network.register_function_for_clique(
                0, Scale(initial_potential, self._init_potential_scalar))

        terminal_potential = Pullback(
            SquaredNorm(self.q_goal),
            self.function_network.center_of_clique_map())
        self.function_network.register_function_last_clique(
            Scale(terminal_potential, self._term_potential_scalar))

    def add_waypoint_terms(self, q_waypoint, i, scalar):

        initial_potential = Pullback(
            SquaredNorm(q_waypoint),
            self.function_network.left_most_of_clique_map())
        self.function_network.register_function_for_clique(
            i, Scale(initial_potential, scalar))

    def add_final_velocity_terms(self):

        derivative = Pullback(SquaredNormVelocity(
            self.config_space_dim, self.dt),
            self.function_network.left_of_clique_map())

        self.function_network.register_function_last_clique(
            Scale(derivative, self._term_velocity_scalar))

    def add_smoothness_terms(self, deriv_order=2):

        if deriv_order == 1:
            derivative = Pullback(SquaredNormVelocity(
                self.config_space_dim, self.dt),
                self.function_network.left_of_clique_map())
            self.function_network.register_function_for_all_cliques(
                Scale(derivative, self._velocity_scalar))
            # TODO change the last clique to have 0 velocity change
            # when linearly interpolating

        elif deriv_order == 2:
            derivative = SquaredNormAcceleration(
                self.config_space_dim, self.dt)
            self.function_network.register_function_for_all_cliques(
                Scale(derivative, self._acceleration_scalar))
        else:
            raise ValueError("deriv_order ({}) not suported".format(
                deriv_order))

    def add_isometric_potential_to_all_cliques(self, potential, scalar):
        """
        Apply the following euqation to all cliques:

                c(x_t) | d/dt x_t |

            The resulting Riemanian metric is isometric. TODO see paper.
            Introduced in CHOMP, Ratliff et al. 2009.
        """
        cost = Pullback(
            potential,
            self.function_network.center_of_clique_map())
        squared_norm_vel = Pullback(
            SquaredNormVelocity(self.config_space_dim, self.dt),
            self.function_network.right_of_clique_map())

        self.function_network.register_function_for_all_cliques(
            Scale(ProductFunction(cost, squared_norm_vel), scalar))

    def add_obstacle_barrier(self):
        """ obstacle barrier function """
        if self.signed_distance_field is None:
            return
        barrier = LogBarrierFunction()
        barrier.set_mu(20.)
        potential = Compose(barrier, self.signed_distance_field)
        # self.obstacle_potential = potential
        self.function_network.register_function_for_all_cliques(
            Pullback(
                potential,
                self.function_network.center_of_clique_map()))

    def add_obstacle_terms(self, geodesic=False):
        """ Takes a matrix and adds a isometric potential term
            to all cliques """
        assert self.obstacle_potential is not None
        self.add_isometric_potential_to_all_cliques(
            self.obstacle_potential, self._obstacle_scalar)

    def add_costgrid_terms(self, scalar=1.):
        """ Takes a matrix and adds a isometric potential term
            to all cliques """
        assert self.costmap is not None
        self.add_isometric_potential_to_all_cliques(self.costmap, scalar)

    def add_box_limits(self):
        v_lower = np.array([self.extent.x_min, self.extent.y_min])
        v_upper = np.array([self.extent.x_max, self.extent.y_max])
        box_limits = BoundBarrier(v_lower, v_upper)
        self.function_network.register_function_for_all_cliques(Pullback(
            box_limits, self.function_network.center_of_clique_map()))

    def create_clique_network(self):

        self.function_network = CliquesFunctionNetwork(
            self.trajectory_space_dim,
            self.config_space_dim)

    def create_objective(self):
        """ resets the objective """
        self.objective = TrajectoryObjectiveFunction(
            self.q_init, self.function_network)

    def add_all_terms(self):
        self.add_final_velocity_terms()
        self.add_smoothness_terms(1)
        self.add_smoothness_terms(2)
        self.add_obstacle_terms()
        self.add_box_limits()
        self.add_init_and_terminal_terms()
        self.add_obstacle_barrier()

    def optimize(self,
                 q_init,
                 nb_steps=100,
                 trajectory=None,
                 optimizer="natural_gradient"):

        if trajectory is None:
            trajectory = linear_interpolation_trajectory(
                q_init, self.q_goal, self.T)

        xi = trajectory.active_segment()

        if optimizer is "natural_gradient":
            optimizer = NaturalGradientDescent(self.objective, self.metric)
            optimizer.set_eta(self._eta)

            dist = float("inf")
            gradient = xi
            delta = xi
            for i in range(nb_steps):
                xi = optimizer.one_step(xi)
                trajectory.active_segment()[:] = xi
                dist = np.linalg.norm(
                    trajectory.final_configuration() - self.q_goal)
                print("dist[{}] : {}, objective : {}, gnorm {}".format(
                    i, dist, optimizer.objective(xi),
                    np.linalg.norm(optimizer.gradient(xi))))
                gradient = optimizer.gradient(xi)
                delta = optimizer.delta(xi)

        elif optimizer is "newton":
            res = optimize.minimize(
                x0=np.array(xi),
                method='Newton-CG',
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
            trajectory.active_segment()[:] = res.x
            gradient = res.jac
            delta = res.jac
            dist = np.linalg.norm(
                trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                print(("gradient norm : ", np.linalg.norm(res.jac)))
        else:
            raise ValueError

        return [dist < 1.e-3, trajectory, gradient, delta]
