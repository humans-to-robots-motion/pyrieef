import numpy as np
from pyrieef.motion.trajectory import Trajectory


class IterativeLQR(object):

    def __init__(self, clique_hessians, clique_jacobians, T_length, dt):
        self.clique_hessians = clique_hessians
        self.clique_jacobians = clique_jacobians
        self.T_length = T_length
        self.dt = dt

        self.K_t = np.zeros((self.T_length + 1, 2, 4))
        self.k_t = np.zeros((self.T_length + 1, 2, 1))

        # dynamics matrix
        A = np.matrix(np.vstack([
            np.hstack([np.eye(2), self.dt * np.eye(2)]),
            np.hstack([np.zeros((2, 2)), np.eye(2)])]))

        # control matrix
        B = np.matrix(np.vstack([
            (self.dt ** 2) * np.eye(2), self.dt * np.eye(2)]))

        self.F = np.matrix(np.hstack([A, B]))

    def backward_pass(self):

        V_t = np.matrix(np.zeros((4, 4)))
        v_t = np.matrix(np.zeros((4, 1)))

        for j in range(self.T_length - 1, 0, -1):
            Q_t = np.matrix(self.clique_hessians[j]) + self.F.T * V_t * self.F
            q_t = np.matrix(self.clique_jacobians[j]) + self.F.T * v_t

            Q_xx = Q_t[:4, :4]
            Q_xu = Q_t[:4, -2:]
            Q_ux = Q_t[-2:, :4]
            Q_uu = Q_t[-2:, -2:]

            q_x = q_t[:4]
            q_u = q_t[-2:]

            self.K_t[j] = -np.linalg.inv(Q_uu) * Q_ux
            self.k_t[j] = -np.linalg.inv(Q_uu) * q_u

            V_t = Q_xx + Q_xu * \
                self.K_t[j] + self.K_t[j].T * Q_ux + \
                self.K_t[j].T * Q_uu * self.K_t[j]
            v_t = q_x + Q_xu * self.k_t[j] + self.K_t[j].T * \
                q_u + self.K_t[j].T * Q_uu * self.k_t[j]

    def forward_pass(self, start_point, x_d, u_d):

        trajectory = Trajectory(T=self.T_length, n=2)
        x_t = np.hstack([start_point, np.zeros(
            start_point.size)]).reshape(4, 1)
        for i in range(self.T_length + 1):
            # self.K_t[i] = np.matrix([[7.60447174, 0.,         4.21720096, 0.        ],
            # 							[0.,         7.60447174, 0.,         4.21720096]])
            # 1) compute acceleration
            error = np.matrix(x_t - x_d[i].reshape(4, 1))
            # print("error: "+str(error)+ " iteration: "+str(i)+ ", K_t: "+str(self.K_t[i]))
            u_t = u_d[i].reshape(2, 1) + self.K_t[i].dot(error) + self.k_t[i]
            a_t = np.array(u_t).reshape((2,))
            a_t = u_t
            v_t = x_t[2:]
            q_t = x_t[:2]

            trajectory.configuration(i)[:] = q_t.reshape((2,))

            # 2) integrate forward and update state
            q_t1 = q_t + v_t * self.dt + a_t.reshape(2, 1) * (self.dt ** 2)
            v_t1 = v_t + a_t.reshape(2, 1) * self.dt
            x_t = np.hstack([q_t1.reshape(2,), v_t1.reshape(2,)]).T

        return trajectory
