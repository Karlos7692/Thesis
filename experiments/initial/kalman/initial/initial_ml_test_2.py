from data.commsec import CommsecColumns as ccs, FeatureTypes as ft, CommsecDataManager
from models.lds.kalman import KalmanFilter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from renderers.renderers import GraphWriter
import os

selection = [ccs.high, ccs.low]
types = [ft.median_price] * 2
names = ['High', 'Low']
rmd = CommsecDataManager('RMD', selection, types, names)
rmd_allprices = CommsecDataManager('RMD', [ccs.high, ccs.low], [ft.price, ft.price], ['High', 'Low'])
rmd_close = CommsecDataManager('RMD', [ccs.close], [ft.price], ['Close'])

price_name = 'Median RMD High, RMD Low'

# Setup data
(dates, obs) = rmd[:]
obs = obs.T
obs = obs.reshape((obs.shape[0], 1, obs.shape[1]))
conds = np.zeros((4, 1, obs.shape[2]))

def gen_params(obs_size, state_size):
    # Setup Kalman Filter
    A = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    C = np.array([[1, 0, 0, 0]])
    D = np.array([[0, 0, 0, 0]])
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    R = np.array([[1]])

    init_mu = np.array([[5.32],
                        [0],
                        [0],
                        [0]])
    init_V = np.eye(A.shape[0], A.shape[1])
    return (A, B, C, D, Q, R), init_mu, init_V

# Split the data into first half and second half
(r, c, ts) = obs.shape
training_start = 0
test_start = 252
n_data_points = 252
test_end = test_start + n_data_points
training_ys = obs[:, :, training_start:test_start]
training_us = conds[:, :, training_start:test_start]
test_ys = obs[:, :, test_start:test_end]
test_us = conds[:, :, test_start:test_end]

(kf, ll_hists) = KalmanFilter.fit(training_ys, training_us, gen_params(1, 4), iters=4)

# Show Kalman Filter hist
# for l in ll_hists:
#     print(l, sum(l))


# Get test values
kf.ys = obs[:, :, :test_end]
kf.us = conds[:, :, :test_end]

(y_pred, ll, _) = kf.filter()
y_pred = y_pred[:, :, test_start:test_end]

(x, y, pred_ts) = y_pred.shape
pred = pd.DataFrame(y_pred.reshape((x, pred_ts)).T, columns=['Predicted Prices'])

# Projected prices
# (projection, V_projections) = kf.project(30)
# plt.plot(projection[1, :, :].flatten(), c='r')
# plt.show()

# Ordinary Least Squares Calculation
residuals = rmd.data.values[test_start:test_end, 1] - pred.values[:, 0]
ols = sum(residuals * residuals)
print('Least Mean Squares', ols)

gwr = GraphWriter(os.getcwd())
gwr.residuals_plot('RMD', residuals, dates[test_start], dates[test_end-1])

below_high = np.less_equal(pred.values[:, 0], rmd_allprices.data[test_start:test_end]['RMD High'].values[:])
above_low = np.less_equal(rmd_allprices.data[test_start:test_end]['RMD Low'].values[:], pred.values[:, 0])
between_high_low = np.logical_and(below_high, above_low)

print('Probability (fr) Prediction between high and low:', sum(between_high_low) / (between_high_low.size))

# Profitability of model:
# Long/Short at close, Short/Long at median price if within high-low else close
# If not executed liquidate position. Sell/Buy back at next close
position = np.sign(pred.values[1:, 0] - rmd_close.data[test_start:test_end]['RMD Close'].values[:-1])
executed = position * between_high_low[1:] * (pred.values[1:, 0] - rmd_close.data[test_start:test_end]['RMD Close'].values[:-1])
not_executed = position * np.logical_not(between_high_low[1:]) * (rmd_close.data[test_start:test_end]['RMD Close'].values[1:] - rmd_close.data[test_start:test_end]['RMD Close'].values[:-1])

point_profit = executed + not_executed
cum_profit = np.zeros(executed.shape)
for i in range(1, executed.shape[0]):
    cum_profit[i] = cum_profit[i-1] + point_profit[i]


df_exec = pd.DataFrame({'Date': rmd.data[test_start:test_end][ccs.date.value].values[1:],
                        'RMD High': rmd_allprices.data[test_start:test_end]['RMD High'].values[1:],
                        'RMD Low': rmd_allprices.data[test_start:test_end]['RMD Low'].values[1:], 'Executed': executed,
                        'Not Executed': not_executed, 'Cum Profit': cum_profit})
df_exec.plot(x='Date', y=['RMD High', 'RMD Low', 'Cum Profit', 'Executed', 'Not Executed'])
profit = np.sum(executed) + np.sum(not_executed)
print(profit)
plt.show()

# Show forward projection of the Kalman Filter
# project_n_points = 200
# next_ys_index = test_end
# y_proj = kf.project(project_n_points)
# plt.plot(np.concatenate((y_pred, y_proj), axis=2)[0, :].flatten())
# plt.plot(obs[0, :, test_start:test_end+project_n_points].flatten(), c='purple')
# # plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD High'].values[:], 'g')
# # plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD Low'].values[:], 'r')
# plt.show()

# Show Kalman Filter fitted params
# print('A:')
# print(np.around(kf.As[:, :, 0], decimals=4))
# print('C:')
# print(np.around(kf.Cs[:, :, 0], decimals=4))
# print('Q')
# print(np.around(kf.Qs[:, :, 0], decimals=4))
# print('R')
# print(np.around(kf.Rs[:, :, 0], decimals=4))

