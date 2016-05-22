import numpy as np
from models.ml import Gym
import sys
import random

from models.models import Axis


def gauss(m, n):
    return np.array([[np.random.standard_normal() for c in range(n)] for r in range(m)])


class LDSGym(Gym):

    def __init__(self, lds_class):
        self.lds_class = lds_class

    def __generate_model_params__(self, obs_size, n_latent, snf=0.001):
        state_size = obs_size + n_latent
        A = np.array([[0.98 if c==r or c==r+1 else 0 for c in range(state_size)] for r in range(state_size)])
        A = A + A * snf * gauss(state_size, state_size)
        B = np.eye(state_size, state_size)
        C = np.eye(obs_size, state_size)
        D = np.eye(obs_size, state_size)
        Q = np.eye(state_size, state_size)
        R = np.eye(obs_size, obs_size)

        return A, B, C, D, Q, R

    def __generate__init_state__(self, ys, n_latent):
        state_size = ys.shape[Axis.rows] + n_latent
        ob = ys[:, :, 0]
        latent = np.zeros((n_latent, 1))
        init_mu = np.vstack((ob, latent))
        init_mu[-1, :] = 0.1
        init_V = np.eye(state_size, state_size)

        return init_mu, init_V

    def train(self, ys, us, kalman_params, iters=5, use_last_to_init=False):
        try:
            model_hist = self.lds_class.fit(ys, us, kalman_params, iters=iters, use_last_to_init=use_last_to_init)
            return model_hist
        except:
            return (None, None)

    def select_best_model(self, ys, us, iters=5, min_fit=30, n_models=100, run_title=None):
        run_title = " for {run}".format(run=run_title) or ""

        if ys.shape[Axis.time] < min_fit:
            raise ValueError("Oops, there is not enough data to even do a min fit.")

        obs_size = ys.shape[Axis.rows]
        state_size = us.shape[Axis.rows]
        n_latent = state_size - obs_size

        # Make the max mse equal to the max possible se_total/observations so that invalid models arent checked
        best_mse = sys.float_info.max / float(ys.shape[Axis.time] - min_fit)
        best_model = None
        print("Running models{run_msg}...".format(run_msg=run_title))
        for m_index in range(n_models):
            se_total = 0
            print("Running model {i}...".format(i=m_index))
            init_params = self.__generate_model_params__(obs_size, n_latent)
            (init_mu, init_V) = self.__generate__init_state__(ys, n_latent)
            kalman_params = (init_params, init_mu, init_V)
            for n_points in range(min_fit, ys.shape[Axis.time]-1):
                ys_to_fit = ys[:, :, :n_points]
                us_to_fit = us[:, :, :n_points]
                y_t = ys[:, :, n_points]
                u_t = us[:, :, n_points]
                (model, hist) = self.train(ys_to_fit, us_to_fit, kalman_params, iters=iters, use_last_to_init=True)

                if model is None:
                    se_total = sys.float_info.max
                    break

                y_pred = model.predict_online(u_t)
                se_total += (y_t - y_pred).T @ (y_t - y_pred)

            mse = se_total / float(ys.shape[Axis.time] - min_fit)

            if mse < best_mse:
                # Guard against any invalid models (on last data point)
                (model, hist) = self.train(ys, us, kalman_params, iters=iters, use_last_to_init=True)
                if model is not None:
                    best_mse = mse
                    best_model = model
                    print(best_mse)

        return best_model
