from collections import OrderedDict

from models.models import LDS, Axis
from models import probability as prob
from scipy.linalg import pinv
import os
import numpy as np
import json
import utils


def g_kalman_prms(obs_size, state_size):
    """
    Parameters of the following form. You can fit kalman filters for hidden states specified by changing the hidden
    model, to (quadratic, cubic etc).
    :param obs_size: The number of observations
    :param state_size:
    :return:
    """
    A = np.eye(state_size, state_size) + 0.01 * gauss(state_size, state_size)
    B = np.eye(state_size, state_size)
    C = np.eye(obs_size, state_size) + 0.001 * gauss(obs_size, state_size)
    D = np.eye(obs_size, state_size)
    Q = np.eye(state_size, state_size)
    R = uniform(obs_size, obs_size)

    init_mu = np.array([[1 if obs_size < r == state_size -1 else 0] for r in range(state_size)])
    init_V = np.eye(state_size, state_size)

    return (A, B, C, D, Q, R), init_mu, init_V

CACHE_EXT = ".json"


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
        self.As = np.empty((state_space_size, state_space_size, 0))
        self.Bs = np.empty((state_space_size, state_space_size, 0))
        self.Cs = np.empty((observations_size, state_space_size, 0))
        self.Ds = np.empty((observations_size, state_space_size, 0))
        self.Qs = np.empty((state_space_size, state_space_size, 0))
        self.Rs = np.empty((observations_size, observations_size, 0))

        # Kalman State
        self.mus = np.empty((state_space_size, 1, 0))
        self.Vs = np.empty((state_space_size, state_space_size, 0))

        # Initialize initial values
        self.initialize(init_params, init_mu, init_V)

        # History:
        self.ys = np.empty((observations_size, 1, 0))
        self.us = np.empty((state_space_size, 1, 0))

    def initialize(self, init_params, init_mu, init_V):
        (A, B, C, D, Q, R) = init_params
        self.update_parameters(self.init_t, A, B, C, D, Q, R)
        self.update_state(self.init_t, init_mu, init_V)

    def reset(self):
        # Kalman parameters.
        self.As = np.empty((self.state_size, self.state_size, 0))
        self.Bs = np.empty((self.state_size, self.state_size, 0))
        self.Cs = np.empty((self.observations_size, self.state_size, 0))
        self.Ds = np.empty((self.observations_size, self.state_size, 0))
        self.Qs = np.empty((self.state_size, self.state_size, 0))
        self.Rs = np.empty((self.observations_size, self.observations_size, 0))

        # Kalman State
        self.mus = np.empty((self.state_size, 1, 0))
        self.Vs = np.empty((self.state_size, self.state_size, 0))

        # History:
        self.ys = np.empty((self.observations_size, 1, 0))
        self.us = np.empty((self.state_size, 1, 0))

    # Train using EM maximisation algorithm init_data, init_conditional_inputs
    # Note that conditional variables implicitly give the number of latent variables.
    @classmethod
    def fit(cls, ys, us, initial_kalman_params, iters=2, use_last_to_init=False, debug_limit=10):
        (init_params, init_mu, init_V) = initial_kalman_params
        try:
            kf = cls(init_params, init_mu, init_V)
            ll_hists = []
            for i in range(iters):

                (C1_ts, C2_ts, P_ts, P_t_tm1s, P_tm1s, G_ts, ll_hist) = kf.__e__(ys, us)
                (init_params, init_mu, init_V) = kf.__m__(C1_ts, C2_ts, P_ts, P_t_tm1s, P_tm1s, G_ts)

                kf.reset()
                kf.initialize(init_params, init_mu, init_V)
                ll_hists.append(ll_hist)

            if use_last_to_init:
                kf.ys = ys
                kf.us = us
                kf.filter()
                t = ys.shape[Axis.time] - 1
                init_params = kf.parameters(t)
                (mu_t, V_t) = kf.state(t)
                kf.reset()
                kf.initialize(init_params, mu_t, V_t)
        except ValueError:
            print("There is a problem with the kalman stability.")
            raise ValueError("There is a problem with the kalman stability.")

        return kf, ll_hists

    def __e__(self, ys, us):
        """
        We will be fitting these parameters we will throw away the first observation, to work out the prior in this
        for the hidden state in this sequence.
        :param ys:
        :param us:
        :return:
        """

        # Initial values
        init_y = ys[:, :, 0]
        init_mu = self.mus[:, :, 0]
        init_V = self.Vs[:, :, 0]

        self.ys = ys[:, :, 1:]
        self.us = us[:, :, 1:]

        # Note that we set the ys and us to be one less observation
        n = self.n_observations() + 1

        # These values will be with respect to actual observations t. Not filter_t, see comments below.
        C1_ts = np.zeros((self.observations_size, self.state_size, n))
        C2_ts = np.zeros((self.observations_size, self.observations_size, n))
        P_ts = np.zeros((self.state_size, self.state_size, n))
        P_t_tm1s = np.zeros((self.state_size, self.state_size, n))
        P_tm1s = np.zeros((self.state_size, self.state_size, n))
        G_ts = np.zeros((self.state_size, self.observations_size, n))

        (y_pred_online, ll_hist, V_smooth_tp1_ts) = self.smooth_filter(likelihood=True)

        # Note that these values are the smoothed mus from the backward algorithm.

        # Calculate initial Expectations.
        P_ts[:, :, 0] = init_V + init_mu @ init_mu.T

        # No P_tm1 or P_t_tm1 at t = 0

        # Calculate initial values needed for maximisation step
        C1_ts[:, :, 0] = init_y @ init_mu.T
        C2_ts[:, :, 0] = init_y @ init_y.T
        G_ts[:, :, 0] = init_mu @ init_y.T

        # Initial t index is out by 1 since we removed the first observation from filtering, to use as initial prior.
        # The data observed by the filter will thus be out by 1.
        # Let t represent the actual observation index and filter_t be the index observed by the filter. ie:
        # filter_t = 0..n-1 <=> t = 1..n
        for filter_t in range(n-1):
            t = filter_t + 1
            (y_t, u_t) = self.data(filter_t)
            (mu_t, V_t) = self.state(filter_t)
            C1_ts[:, :, t] = y_t @ mu_t.T
            C2_ts[:, :, t] = y_t @ y_t.T
            G_ts[:, :, t] = mu_t @ y_t.T
            P_ts[:, :, t] = V_t + mu_t @ mu_t.T

            (mu_tm1, V_tm1) = self.state(filter_t-1)
            P_t_tm1s[:, :, t] = V_smooth_tp1_ts[:, :, filter_t] + mu_t @ mu_tm1.T
            P_tm1s[:, :, t] = V_tm1 + mu_tm1 @ mu_tm1.T

        return C1_ts, C2_ts, P_ts, P_t_tm1s, P_tm1s, G_ts, ll_hist

    def __m__(self, C1_ts, C2_ts, P_ts, P_t_tm1s, P_tm1s, G_ts):

        # Same as e step, we used initial observation to calculate the prior.
        n = self.n_observations() + 1

        C1_sum = np.sum(C1_ts, axis=2)
        P_t_sum = np.sum(P_ts, axis=2)
        P_t_tm1_sum_1tT = np.sum(P_t_tm1s[:, :, :], axis=2)
        P_tm1_sum_1tT = np.sum(P_tm1s[:, :, :], axis=2)

        # Output matrix fit.
        C = C1_sum @ pinv(P_t_sum)

        # Observation covariance fit.
        R = np.zeros((self.observations_size, self.observations_size))
        for t in range(n):
            R += C2_ts[:, :, t] - C @ G_ts[:, :, t]
        R *= 1.0/n

        # State dynamics
        A = P_t_tm1_sum_1tT @ pinv(P_tm1_sum_1tT)

        # Hidden Noise
        Q = 1.0/(n-1) * (np.sum(P_ts[:, :, 1:], axis=2) - A @ P_t_tm1_sum_1tT)

        # Control signal
        B = self.Bs[:, :, 0]
        D = self.Ds[:, :, 0]

        # Initial state
        init_mu = self.mus[:, :, 0]
        init_V = P_ts[:, :, 0] - init_mu @ init_mu.T

        return (A, B, C, D, Q, R), init_mu, init_V

    # TODO Clean up predict names such that predict online becomes the default predict method
    def predict_online(self, u_t):
        t = self.mus.shape[Axis.time] - 1
        (mu_tm1, V_tm1) = self.state(t-1)
        (A, B, C, D, Q, R) = self.parameters(t-1)
        (y_pred, mu_pred) = self.predict(A, B, C, D, mu_tm1, u_t)
        return y_pred

    def predict(self, A, B, C, D, mu_t, u_t):
        mu_pred = self.predict_state(A, B, mu_t, u_t)
        y_pred = self.predict_observable(C, D, mu_pred, u_t)
        return y_pred, mu_pred

    def predict_observable(self, C, D, mu_pred, u_t):
        return C @ mu_pred + D @ u_t

    def predict_state(self, A, B, mu_t, u_t):
        return A @ mu_t + B @ u_t

    def predict_covariance(self, A, V, Q):
        return A @ V @ A.T + Q

    def project(self, n):
        y_preds = np.zeros((self.observations_size, 1, n))
        last_observed_t = self.n_observations() -1
        (mu_t, V_t) = self.state(last_observed_t)
        for t in range(n):
            (A, B, C, D, Q, R) = self.parameters(last_observed_t)
            (y_pred, mu_t) = self.predict(A, B, C, D, mu_t, np.zeros((self.state_size, 1)))
            y_preds[:, :, t] = self.predict_observable(C, D, mu_t, np.zeros((self.state_size, 1)))
        return y_preds

    def filter(self, likelihood=False, calculate_backwards_inital=False):
        """
        Filter values from observed data points.
        :param likelihood:
        :return: y_t, online predictions, updated mu's
        """
        # Initial values
        ll_hist = []
        T = self.n_observations()
        y_pred_online = np.zeros((self.observations_size, 1, T))
        final_gain_partial = None

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

            ll = self.update(t, mu_pred, V_pred, y_t, y_pred, likelihood)

            if likelihood:
                ll_hist.append(ll)

            if calculate_backwards_inital and t == T-1:
                # Occurs after update, for V_tm1 given 1..t-1
                (mu_tm1_new, V_tm1_new) = self.state(t-1)
                S = C @ V_pred @ C.T + R
                K = V_pred @ C.T @ pinv(S)
                (A, B, C, D, Q, R) = self.parameters(t - 1)
                final_gain_partial = V_tm1_new - K @ C @ A @ V_tm1_new

        return y_pred_online, ll_hist, final_gain_partial

    def update(self, t, mu_pred, V_pred, y_t, y_pred, compute_likelihood=False):
        (A, B, C, D, Q, R) = self.parameters(t - 1)

        # Residuals
        eps_tp1 = y_t - y_pred

        # RLS
        S = C @ V_pred @ C.T + R
        K = V_pred @ C.T @ pinv(S)

        mu_new = mu_pred + K @ eps_tp1
        V_new = V_pred - K @ C @ V_pred

        likelihood = None
        if compute_likelihood:
            zcm = np.zeros((self.observations_size, 1))
            likelihood = prob.mvn_likelihood(eps_tp1, zcm, S)

        self.update_state(t, mu_new, V_new)

        return likelihood

    def smooth_filter(self, likelihood=False):
        # Time arguments
        init_t = self.init_t
        n = self.n_observations()

        (y_pred_online, ll_hist, backwards_smooth_inital) = self.filter(likelihood, calculate_backwards_inital=True)
        V_smooth_t_tm1s = np.empty(self.Vs[:, :, 1:].shape)

        # Start smoothing from t+1 given t. Also change indexed for tm1+1 = j, see backwards algorithm for explanation
        J_t = None
        V_smooth_t_tm1 = backwards_smooth_inital
        for t in range(n - 1, init_t, -1):
            u_t = self.us[:, :, t]
            (mu_t, V_t) = self.state(t)
            (mu_tm1, V_tm1) = self.state(t-1)
            (V_smooth_t_tm1, J_tm1) = self.smooth_update(t, u_t, mu_t, V_t, mu_tm1, V_tm1, J_t, V_smooth_t_tm1)

            # Only n-1 smoothed values, so there are n-1 pairs
            V_smooth_t_tm1s[:, :, t-1] = V_smooth_t_tm1
            # iterate over index for Js
            J_t = J_tm1

        return y_pred_online, ll_hist, V_smooth_t_tm1s

    # Backwards algorithm, smooth mu and Vs
    def smooth_update(self, t, u_t, mu_t, V_t, mu_tm1, V_tm1, J_t, V_t_tm1_given_T):
        (A, B, C, D, Q, R) = self.parameters(t-1)

        n = self.n_observations()

        # E[mu_t, mu_tm1 | y1..t]
        mu_t_tm1 = self.predict_state(A, B, mu_tm1, u_t)
        V_t_tm1 = self.predict_covariance(A, V_tm1, Q)

        # Smoothed Gain matrix
        J_tm1 = V_tm1 @ A.T @ pinv(V_t_tm1)

        # Smooth E[t-1 | t..T ] and Cov(t-1 | t..T)
        mu_smooth = mu_tm1 + J_tm1 @ (mu_t - mu_t_tm1)
        V_smooth = V_tm1 + J_tm1 @ (V_t - V_t_tm1) @ J_tm1.T

        # Delay computation so that t given tm1 is the same when t = T. The rest is computing the
        # (t, tm1) = (T-1, T-2) onwards, pairs
        if t < n-1:
            V_smooth_t_tm1 = V_t @ J_tm1.T + J_t @ (V_t_tm1_given_T.T - A @ V_t) @ J_tm1.T
        else:
            # Delayed computation
            V_smooth_t_tm1 = V_t_tm1_given_T

        self.update_state(t-1, mu_smooth, V_smooth)

        return V_smooth_t_tm1, J_tm1

    def parameters(self, t):
        """
        Observations start from 0 where parameters include an initial fit. n observations <=> n+1 parameters
        If the model is fixed we will return the initial parameters.
        :param t: The time with respect to the observations.
        :return: The parameters (tuple) at time t with respect to the observations
        """
        if self.fixed_params:
            return (self.As[:, :, 0], self.Bs[:, :, 0], self.Cs[:, :, 0], self.Ds[:, :, 0],
                    self.Qs[:, :, 0], self.Rs[:, :, 0])

        return (self.As[:, :, t + 1], self.Bs[:, :, t + 1], self.Cs[:, :, t + 1], self.Ds[:, :, t + 1],
                self.Qs[:, :, t + 1], self.Rs[:, :, t + 1])

    def state(self, t):
        """
        Observations start from 0 where parameters include an initial fit. n observations <=> n+1 states.
        To get the inital value pass t = -1 into the state function
        :param t: The time with respect to the observations.
        :return: The cached state (tuple) at time t with respect to the observations
        """
        return self.mus[:, :, t + 1], self.Vs[:, :, t + 1]

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
        if self.fixed_params and self.As.shape[Axis.time] == 1:
            self.As[:, :, 0] = A_t
            self.Bs[:, :, 0] = B_t
            self.Cs[:, :, 0] = C_t
            self.Ds[:, :, 0] = D_t
            self.Qs[:, :, 0] = Q_t
            self.Rs[:, :, 0] = R_t
            return

        # Not fixed, adjust parameters at t
        if t + 1 < self.As.shape[Axis.time]:
            self.As[:, :, t + 1] = A_t
            self.Bs[:, :, t + 1] = B_t
            self.Cs[:, :, t + 1] = C_t
            self.Ds[:, :, t + 1] = D_t
            self.Qs[:, :, t + 1] = Q_t
            self.Rs[:, :, t + 1] = R_t
            return

        self.As = np.insert(self.As, t + 1, A_t, axis=2)
        self.Bs = np.insert(self.Bs, t + 1, B_t, axis=2)
        self.Cs = np.insert(self.Cs, t + 1, C_t, axis=2)
        self.Ds = np.insert(self.Ds, t + 1, D_t, axis=2)
        self.Qs = np.insert(self.Qs, t + 1, Q_t, axis=2)
        self.Rs = np.insert(self.Rs, t + 1, R_t, axis=2)

    def update_state(self, t, mu_t, V_t):

        if t + 1 < self.mus.shape[Axis.time]:
            self.mus[:, :, t+1] = mu_t
            self.Vs[:, :, t+1] = V_t
            return

        self.mus = np.insert(self.mus, t + 1, mu_t, axis=2)
        self.Vs = np.insert(self.Vs, t + 1, V_t, axis=2)

    def data(self, t):
        if t == self.init_t:
            return None, np.zeros((self.state_size, 1))
        return self.ys[:, :, t], self.us[:, :, t]

    def observe(self, t, y_t, u_t):
        self.ys = np.insert(self.ys, t, y_t, axis=2)
        self.us = np.insert(self.us, t, u_t, axis=2)

    def observe_point(self, y_t):
        t = self.n_observations() - 1
        self.ys = np.insert(self.ys, t, y_t, axis=2)

    def observe_conditional(self, u_t):
         t = self.us.shape[Axis.time] - 1
         self.us = np.insert(self.us, t, u_t, axis=2)

    def n_observations(self):
        return self.ys.shape[Axis.time]

    """
    Disk Cache Functions
    """
    @classmethod
    def restore(cls, from_file, cache_dir=utils.get_default_cache_dir()):
        cached = [f for f in os.listdir(cache_dir) if str(f) == from_file]
        if not cached:
            raise Exception("Kalman Filter: Could not restore {from_file} from {dir}".format(from_file=from_file, dir=cache_dir))

        kf_json_file = os.path.join(cache_dir, cached.pop(0))
        (init_params, init_mu, init_V, fixed_params) = json_to_params(kf_json_file)
        return cls(init_params, init_mu, init_V, fixed=fixed_params)

    def cache(self, id, features, description, cache_dir=utils.get_default_cache_dir()):
        cached = [f for f in os.listdir(cache_dir) if str(f).replace(CACHE_EXT, "") == id]
        if cached:
            raise Exception("Kalman Filter: Could not cache {id} in {dir}, because it already exists!"
                            .format(id=id, dir=cache_dir))

        filename = id + CACHE_EXT
        features_str = ",".join(features)
        fixed_params = self.fixed_params
        observation_size = self.observations_size
        state_size = self.state_size
        A = matrix_to_str(self.As[:, :, 0])
        B = matrix_to_str(self.Bs[:, :, 0])
        C = matrix_to_str(self.Cs[:, :, 0])
        D = matrix_to_str(self.Ds[:, :, 0])
        Q = matrix_to_str(self.Qs[:, :, 0])
        R = matrix_to_str(self.Rs[:, :, 0])

        mu = matrix_to_str(self.mus[:, :, 0])
        V = matrix_to_str(self.Vs[:, :, 0])

        data = OrderedDict([("name", id), ("description", description), ("features", features_str),
                            ("fixed params", fixed_params), ("observation size", observation_size),
                            ("state size", state_size), ("A", A), ("B", B), ("C", C), ("D", D), ("Q", Q), ("R", R),
                            ("mu", mu), ("V", V)])

        with open(os.path.join(cache_dir, filename), 'w') as f:
            json.dump(data, f)
            f.close()


# TODO Allow for time cached values
def json_to_params(kf_json_file):
    with open(kf_json_file, 'r') as f:
        kalman_dict = json.load(f)
        obs_size = int(kalman_dict['observation size'])
        state_size = int(kalman_dict['state size'])
        A = get_matrix(kalman_dict['A'], state_size, state_size)
        B = get_matrix(kalman_dict['B'], state_size, state_size)
        C = get_matrix(kalman_dict['C'], obs_size, state_size)
        D = get_matrix(kalman_dict['D'], obs_size, state_size)
        Q = get_matrix(kalman_dict['Q'], state_size, state_size)
        R = get_matrix(kalman_dict['R'], obs_size, obs_size)

        mu = get_matrix(kalman_dict['mu'], state_size, 1)
        V = get_matrix(kalman_dict['V'], state_size, state_size)

        fixed_params = bool(kalman_dict['fixed params'])

        f.close()
    return (A, B, C, D, Q, R), mu, V, fixed_params


def get_matrix(matrix_str, rs, cs):
    return np.fromstring(matrix_str, sep=' ').reshape((rs, cs))


def matrix_to_str(matrix):
    return np.array2string(matrix).replace('[', "").replace(']', "")


def gauss(m, n):
    return np.array([[np.random.standard_normal() for c in range(n)] for r in range(m)])


def uniform(m, n):
    return np.array([[np.random.uniform(0, 1) for c in range(n)] for r in range(m)])
