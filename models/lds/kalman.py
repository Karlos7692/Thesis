from models.models import LDS, Axis
from models import probability as prob
from scipy.linalg import pinv
import numpy as np


class KalmanFilter(LDS):
    def __init__(self, init_params, init_mu, init_V, fixed=True):
        (A, B, C, D, Q, R) = init_params
        state_space_size = A.shape[Axis.rows]
        observations_size = C.shape[Axis.rows]

        # Properties
        self.fixed_params = fixed
        self.state_size = state_space_size
        self.observations_size = observations_size

        # Kalman parameters.
        self.A_0tT = np.empty((state_space_size, state_space_size, 0))
        self.B_0tT = np.empty((state_space_size, state_space_size, 0))
        self.C_0tT = np.empty((observations_size, state_space_size, 0))
        self.D_0tT = np.empty((observations_size, state_space_size, 0))
        self.Q_0tT = np.empty((state_space_size, state_space_size, 0))
        self.R_0tT = np.empty((observations_size, observations_size, 0))

        # Kalman State
        self.mu_0tT = np.empty((state_space_size, 1, 0))
        self.V_tt0 = np.empty((state_space_size, state_space_size, 0))

        # Initialize initial values
        self.initialize(init_params, init_mu, init_V)

        # History:
        self.y_0tT = np.empty((observations_size, 1, 0))
        self.u_0tT = np.empty((state_space_size, 1, 0))

    def initialize(self, init_params, init_mu, init_V):
        (A, B, C, D, Q, R) = init_params
        self.update_parameters(self.init_t, A, B, C, D, Q, R)
        self.update_state(self.init_t, init_mu, init_V)

    # Train using EM maximisation algorithm init_data, init_conditional_inputs
    def fit(self, observations, conditional_inputs):
        if observations.shape[int(Axis.rows)] != self.observations_size:
            raise Exception("Observations size is supposed to be ({s}xn) but found ({s2}x{s3})"
                            .format(s=self.observations_size, s2=observations.shape[int(Axis.rows)],
                                    s3=observations.shape[int(Axis.cols)]))

        if conditional_inputs.shape[int(Axis.rows)] != self.state_size:
            raise Exception("Conditional inputs size is supposed to be ({s}xn) but found ({s2}x{s3})"
                            .format(s=self.state_size, s2=conditional_inputs.shape[int(Axis.rows)],
                                    s3=conditional_inputs.shape[int(Axis.cols)]))

    def predict(self, A, B, C, D, mu_t, u_t):
        mu_pred = self.predict_state(A, B, mu_t, u_t)
        y_pred = self.predict_observable(C, D, mu_pred, u_t)
        return y_pred, mu_pred

    def predict_observable(self, C, D, mu_pred, u_t):
        return C @ mu_pred + D @ u_t

    def predict_state(self, A, B, mu_t, u_t, init=False):
        return A @ mu_t + B @ u_t

    def predict_covariance(self, A, V, Q, init=False):
        return A @ V @ A.T + Q

    def kalman_filter(self, likelihood=False):
        """
        Filter values from observed data points.
        :param likelihood:
        :return: y_t, online predictions, updated mu's
        """
        # Initial values
        ll_sum = 0
        T = self.n_observations()
        y_pred_online = np.zeros((self.observations_size, 1, T))

        # Kalman Filter and update over all observations t in 0..T
        for t in range(T):

            # t-1 parameters and state
            (A, B, C, D, Q, R) = self.parameters(t - 1)
            (mu_tm1, V_tm1) = self.state(t - 1)
            (y_t, u_t) = self.data(t)

            # Predict
            (y_pred, mu_pred) = self.predict(A, B, C, D, mu_tm1, u_t)
            V_pred = self.predict_covariance(A, V_tm1, Q)
            y_pred_online[:, :, t] = y_pred

            (y_t, _) = self.data(t)

            ll = self.kalman_update(t, mu_pred, V_pred, y_t, y_pred, likelihood)

            if likelihood:
                ll_sum += ll

        return y_pred_online, ll_sum

    def kalman_update(self, t, mu_pred, V_pred, y_t, y_pred, compute_likelihood=False):
        (A, B, C, D, Q, R) = self.parameters(t - 1)

        # Residuals
        eps_tp1 = y_t - y_pred

        # RLS
        S = C @ V_pred @ C.T + R
        K = V_pred @ C.T @ pinv(S)

        mu_new = mu_pred + K @ eps_tp1
        V_new = (np.identity(self.state_size) - K @ C) @ V_pred

        likelihood = None
        if compute_likelihood:
            zcm = np.zeros((self.observations_size, 1))
            likelihood = prob.mvn_likelihood(eps_tp1, zcm, S)

        self.update_state(t, mu_new, V_new)

        return likelihood

    def kalman_smooth(self, likelihood=False):
        # Time arguments
        init_t = self.init_t
        final_t = self.y_0tT.shape[Axis.time] - 1

        (y_pred_online, ll_sum) = self.kalman_filter(likelihood)
        V_smooth_tp1_ts = np.empty(self.V_tt0[:, :, 1:].shape)

        # Start smoothing from t+1 given t
        for t in range(final_t - 1, init_t, -1):
            u_t = self.u_0tT[:, :, t]
            mu_pred_tp1, V_pred_tp1 = self.state(t + 1)
            mu_pred_t, V_pred_t = self.state(t)
            V_smooth_tp1_ts[:, :, t] = self.kalman_smooth_update(t, u_t, mu_pred_tp1, V_pred_tp1, mu_pred_t, V_pred_t)

        return y_pred_online, ll_sum, V_smooth_tp1_ts

    # Given prediction t+1, smooth prediction at t
    def kalman_smooth_update(self, t, u_t, mu_pred_tp1, V_pred_tp1, mu_pred, V_pred):
        (A, B, C, D, Q, R) = self.parameters(t)

        # E[E[t| t+1, ys]] and Cov(t+1| t (prediction))
        mu_pred_tp1_t = self.predict_state(A, B, mu_pred, u_t)
        V_pred_tp1_t = self.predict_covariance(A, V_pred, Q)

        # Smoothed Gain matrix
        J = V_pred @ A.T @ pinv(V_pred_tp1_t)

        # Smooth E[t| t+1] and Cov(t| t+1)
        mu_smooth = mu_pred + J @ (mu_pred_tp1 - mu_pred_tp1_t)
        V_smooth = V_pred + J @ (V_pred_tp1 - V_pred_tp1_t)
        V_smooth_tp1_t = J @ V_pred_tp1

        self.update_state(t, mu_smooth, V_smooth)

        return V_smooth_tp1_t

    def parameters(self, t):
        """
        Observations start from 0 where parameters include an initial fit. n observations <=> n+1 parameters
        If the model is fixed we will return the initial parameters.
        :param t: The time with respect to the observations.
        :return: The parameters (tuple) at time t with respect to the observations
        """
        if self.fixed_params:
            return (self.A_0tT[:, :, 0], self.B_0tT[:, :, 0], self.C_0tT[:, :, 0], self.D_0tT[:, :, 0],
                    self.Q_0tT[:, :, 0], self.R_0tT[:, :, 0])

        return (self.A_0tT[:, :, t + 1], self.B_0tT[:, :, t + 1], self.C_0tT[:, :, t + 1], self.D_0tT[:, :, t + 1],
                self.Q_0tT[:, :, t + 1], self.R_0tT[:, :, t + 1])

    def state(self, t):
        """
        Observations start from 0 where parameters include an initial fit. n observations <=> n+1 states.
        To get the inital value pass t = -1 into the state function
        :param t: The time with respect to the observations.
        :return: The cached state (tuple) at time t with respect to the observations
        """
        return self.mu_0tT[:, :, t + 1], self.V_tt0[:, :, t + 1]

    def update_parameters(self, t, A_t, B_t, C_t, D_t, Q_t, R_t):
        """
        :param t: The time with respect to data. Updated state contains initial t=0. Thus we index at t+1
        :param A_t: A_t respect to time t
        :param B_t: B_t respect to time t
        :param C_t: C_t respect to time t
        :param D_t: D_t respect to time t
        :param Q_t: Q_t respect to time t
        :param R_t: R_t respect to time t
        :return:
        """
        if self.fixed_params and self.A_0tT.shape[Axis.time] == 1:
            self.A_0tT[:, :, 0] = A_t
            self.B_0tT[:, :, 0] = B_t
            self.C_0tT[:, :, 0] = C_t
            self.D_0tT[:, :, 0] = D_t
            self.Q_0tT[:, :, 0] = Q_t
            self.R_0tT[:, :, 0] = R_t
            return

        # Not fixed, adjust parameters at t
        if t + 1 < self.A_0tT.shape[Axis.time]:
            self.A_0tT[:, :, t+1] = A_t
            self.B_0tT[:, :, t+1] = B_t
            self.C_0tT[:, :, t+1] = C_t
            self.D_0tT[:, :, t+1] = D_t
            self.Q_0tT[:, :, t+1] = Q_t
            self.R_0tT[:, :, t+1] = R_t
            return

        self.A_0tT = np.insert(self.A_0tT, t+1, A_t, axis=2)
        self.B_0tT = np.insert(self.B_0tT, t+1, B_t, axis=2)
        self.C_0tT = np.insert(self.C_0tT, t+1, C_t, axis=2)
        self.D_0tT = np.insert(self.D_0tT, t+1, D_t, axis=2)
        self.Q_0tT = np.insert(self.Q_0tT, t+1, Q_t, axis=2)
        self.R_0tT = np.insert(self.R_0tT, t+1, R_t, axis=2)

    def update_state(self, t, mu_t, V_t):
        self.mu_0tT = np.insert(self.mu_0tT, t + 1, mu_t, axis=2)
        self.V_tt0 = np.insert(self.V_tt0, t + 1, V_t, axis=2)

    def data(self, t):
        if t == self.init_t:
            return None, np.zeros((self.state_size, 1))
        return self.y_0tT[:, :, t], self.u_0tT[:, :, t]

    def observe(self, t, y_t, u_t):
        self.y_0tT = np.insert(self.y_0tT, t, y_t, axis=2)
        self.u_0tT = np.insert(self.u_0tT, t, u_t, axis=2)

    def observe_point(self, y_t):
        t = self.n_observations() - 1
        self.y_0tT = np.insert(self.y_0tT, t, y_t, axis=2)

    def observe_conditional(self, u_t):
         t = self.u_0tT.shape[Axis.time] - 1
         self.u_0tT = np.insert(self.u_0tT, t, u_t, axis=2)

    def n_observations(self):
        return self.y_0tT.shape[Axis.time]


class KalmanStateManager(object):
    pass