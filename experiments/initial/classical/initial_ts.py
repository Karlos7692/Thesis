from data.commsec import CommsecColumns as ccs, FeatureTypes as ft, CommsecDataManager
from models.ts.ts import ARMA, ARIMA
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4, threshold=3, suppress=True)

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


# Split the data into first half and second half
(r, c, ts) = obs.shape

test_start = 252
n_data_points = 2000
test_end = test_start + n_data_points
training_ys = obs[:, :, :test_start].reshape(test_start)
test_ys = obs[:, :, test_start:test_end]


# TODO Internally flatten result
arma = ARMA.fit(training_ys, (1, 1, 1))

y_pred = np.zeros((r, c, n_data_points))
for t in range(n_data_points):
    y_t = test_ys[:, :, t]
    y_pred[:, :, t] = y_p = arma.predict()
    arma.update(y_t, y_p)


(x, y, t) = y_pred.shape
pred = pd.DataFrame(y_pred.reshape((x, t)).T, columns=['Predicted Prices'])

# Projected prices
# (projection, V_projections) = kf.project(30)
# plt.plot(projection[1, :, :].flatten(), c='r')
# plt.show()

# Ordinary Least Squares Calculation
residuals = rmd.data.values[test_start:test_end, 1] - pred.values[:, 0]
ols = sum(residuals * residuals)
print('Least Mean Squares', ols)

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
residual_spikes = pd.DataFrame((rmd.data.values[test_start:test_end, 1] - pred.values[:, 0]) * (rmd.data.values[test_start:test_end, 1] - pred.values[:, 0]), columns=['Residuals'])
residual_spikes.plot(x=rmd.data[test_start:test_end][ccs.date.value], y=['Residuals'], ax=ax)
plt.show()

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


