from abc import abstractmethod
import os
import numpy as np
import statsmodels.api as sm
from models.models import Model
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
import json

# Similar format as other models in this module.


class TimeSeries(Model):
    def __init__(self, model_info, fitted_model):
        self.model_info = model_info
        self.model = fitted_model

    @classmethod
    def fit(cls, data, order):
        pass

    @abstractmethod
    def predict(self):
        pass

    def update(self, y_t: np.array, y_pred: np.array):
        y_t = y_t.flatten()
        y_pred = y_pred.flatten()
        n_obs = self.model_info.endog.shape[0]
        m = self.model

        # Residual Calculation
        eps_t = y_t - y_pred
        m.resid = np.insert(m.resid, n_obs, eps_t, axis=0)

        # Add data point
        self.model_info.endog = np.insert(self.model_info.endog, n_obs, y_t, axis=0)

    def print_stats(self):
        print(self.model_info) if self.model_info else print("We dont not have any results!")


class ARMA(TimeSeries):
    def __init__(self, model_info: sm.tsa.ARMA, fitted_model):
        super().__init__(model_info, fitted_model)

    @classmethod
    def fit(cls, data, order):
        arma = sm.tsa.ARMA(data, order)
        fitted_results = arma.fit(disp=0)
        return cls(arma, fitted_results)

    def predict(self):
        m = self.model
        data = self.model_info.endog
        return _arma_predict_out_of_sample(m.params, 1, m.resid, m.k_ar, m.k_ma, m.k_trend, m.k_exog, endog=data,
                                           exog=None, start=data.shape[0])

    @classmethod
    def restore(cls, from_file, cache_dir=os.getcwd()):
        # TODO
        pass

    def cache(self):
        pass


def arma_restore(ts_cls, from_file, cache_dir):
    with open(from_file, 'r') as f:
        json_obj = json.load(f)
        order = json_obj['order']
        # GOT TO ALSO RESTORE THE FITTED PARAMS!!!!!!
        # "name": "ARMA",
        # "description": "ARMA description",
        # "features": "Something",
        # "order": "p,q",
        # "k ar": "k_ar",
        # "k ma": "k_ma",
        # "rv sequence": "np.array(p vals)",
        # "res sequence": "np.array(q vals)",
        # "params": "params"


class ARIMA(TimeSeries):
    def __init__(self, model_info: sm.tsa.ARIMA, fitted_model):
        super().__init__(model_info, fitted_model)

    @classmethod
    def fit(cls, data, order):
        arima = sm.tsa.ARIMA(data, order)
        fitted_results = arima.fit(disp=0)
        return cls(arima, fitted_results)

    def predict(self):
        pass


# arparams = np.array([.75, -.25])
# maparams = np.array([.65, .35])
# ar = np.r_[1, -arparams] # add zero-lag and negate
# ma = np.r_[1, maparams] # add zero-lag
# ys = sm.tsa.arma_generate_sample(ar, ma, 250)
#
# a_22 = ARMA.fit(ys, (2,2))
# print(a_22.model.params)
#
#
# #results = arma_22.predict()
# ys2 = sm.tsa.arma_generate_sample(ar, ma, 250)
# results = np.zeros(ys2.shape)
# for t in range(ys2.shape[0]):
#     y_p = a_22.predict()
#     results[t] = y_p
#     y = ys2[t]
#     a_22.update(y, y_p)
#
#
# #result = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=ys2, exog=None, start=0)
#
# from matplotlib import pyplot as plt
# plt.plot(ys2, c='Red')
# plt.plot(results, c='Blue')
# plt.show()
