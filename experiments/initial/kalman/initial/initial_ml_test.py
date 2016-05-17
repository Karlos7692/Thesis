from data.commsec import CommsecColumns as ccs, FeatureTypes as ft, CommsecDataManager
from models.lds.kalman import KalmanFilter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4, threshold=3, suppress=True)

selection = [ccs.high, ccs.low, ccs.volume]
types = [ft.median_price] * 2 + [ft.signed_volume]
names = ['High', 'Low', '(+/-) x volume']
rmd = CommsecDataManager('RMD', selection, types, names)
rmd_allprices = CommsecDataManager('RMD', [ccs.high, ccs.low], [ft.price, ft.price], ['High', 'Low'])
rmd_close = CommsecDataManager('RMD', [ccs.close], [ft.price], ['Close'])

price_name = 'Median RMD High, RMD Low'

# Setup data
(dates, obs) = rmd[:]
obs = obs.T
obs = obs.reshape((obs.shape[0], 1, obs.shape[1]))
conds = np.zeros((5, 1, obs.shape[2]))

def gen_params(obs_size, state_size):
    # Setup Kalman Filter
    A = np.array([[1, 0, 0, 0, 0],
                  [0.01, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    C = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0.1]])
    D = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    Q = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    R = np.array([[1, 0],
                  [0, 1]])

    init_mu = np.array([[0],
                        [6],
                        [0],
                        [0],
                        [0]])
    init_V = np.eye(A.shape[0], A.shape[1])
    return (A, B, C, D, Q, R), init_mu, init_V

# Split the data into first half and second half
(r, c, ts) = obs.shape

test_start = int(ts/2) - 300
n_data_points = 100
test_end = test_start + n_data_points + 1
training_ys = obs[:, :, :test_start]
training_us = conds[:, :, :test_start]
test_ys = obs[:, :, test_start:test_end]
test_us = conds[:, :, test_start:test_end]

(kf, ll_hists) = KalmanFilter.fit(training_ys, training_us, gen_params(2, 5), iters=10)
for l in ll_hists:
    print(l, sum(l))

kf.mus[:, :, 0] = np.array([[0],
                            [6],
                            [0],
                            [0],
                            [0]])
init_V = np.eye(kf.state_size, kf.state_size) * 10
# kf.mus[:, :, 0] = init_mu
kf.Vs[:, :, 0] = init_V
kf.ys = test_ys
kf.us = test_us


(y_pred, ll, _) = kf.filter()
(x, y, t) = y_pred.shape
pred = pd.DataFrame(y_pred.reshape((x, t)).T, columns=['Predicted Liquidity', 'Predicted Prices'])

# Projected prices
# (projection, V_projections) = kf.project(30)
# plt.plot(projection[1, :, :].flatten(), c='r')
# plt.show()

# Ordinary Least Squares Calculation
residuals = rmd.data.values[test_start:test_end, 2] - pred.values[:, 1]
ols = sum(residuals)
print(ols)

residuals_mean = np.mean(residuals) * np.ones(residuals.shape)
residuals_df = pd.DataFrame({'Nums': list(range(dates[test_start:test_end].shape[0])), 'Residuals': residuals, 'Mean': residuals_mean})
rax = residuals_df.plot(x='Nums', y='Residuals', kind='scatter')
residuals_df.plot(x='Nums', y='Mean', c='Red', ax=rax)
plt.show()

# Plot results
#area_ols if pred < low -> +, low <= pred <= high -> 0. high < pred -> +
ax = pred.plot(x=rmd.data[test_start:test_end][ccs.date.value], y=['Predicted Prices'])
#ax = rmd.data.plot(x=ccs.date.value, y=rmd.features(), ax=ax)
ax = rmd_allprices.data[test_start:test_end].plot(x=ccs.date.value, y=rmd_allprices.features(), ax=ax)
residual_spikes = pd.DataFrame((rmd.data.values[test_start:test_end, 2] - pred.values[:, 1]) * (rmd.data.values[test_start:test_end, 2] - pred.values[:, 1]), columns=['Residuals'])
residual_spikes.plot(x=rmd.data[test_start:test_end][ccs.date.value], y=['Residuals'], ax=ax)
plt.show()

below_high = np.less_equal(pred.values[:, 1], rmd_allprices.data[test_start:test_end]['RMD High'].values[:])
above_low = np.less_equal(rmd_allprices.data[test_start:test_end]['RMD Low'].values[:], pred.values[:, 1])
between_high_low = np.logical_and(below_high, above_low)

print('Probability (fr) Prediction between high and low:', sum(between_high_low) / (between_high_low.size))

# Profitability of model:
# Long/Short at close, Short/Long at median price if within high-low else close
# If not executed liquidate position. Sell/Buy back at next close
position = np.sign(pred.values[1:, 1] - rmd_close.data[test_start:test_end]['RMD Close'].values[:-1])
executed = position * between_high_low[1:] * (pred.values[1:, 1] - rmd_close.data[test_start:test_end]['RMD Close'].values[:-1])
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

project_n_points = 20
next_ys_index = test_end
y_proj = kf.project(project_n_points)
plt.plot(np.concatenate((y_pred, y_proj), axis=2)[1, :].flatten())
plt.plot(obs[1, :, test_start:test_end+project_n_points].flatten(), c='purple')
# plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD High'].values[:], 'g')
# plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD Low'].values[:], 'r')
plt.show()

print('A:')
print(np.around(kf.As[:, :, 0], decimals=4))
print('C:')
print(np.around(kf.Cs[:, :, 0], decimals=4))
print('Q')
print(np.around(kf.Qs[:, :, 0], decimals=4))
print('R')
print(np.around(kf.Rs[:, :, 0], decimals=4))

