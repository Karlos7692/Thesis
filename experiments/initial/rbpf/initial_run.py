from data.commsec import CommsecColumns as ccs, FeatureTypes as ft, CommsecDataManager
from models.mcmc.pf import ParticleFilter, KalmanParticle
import pandas as pd
import numpy as np

from models.models import Axis

MISSPEC = 0.01
ZERO_MISSPEC = 0.0001
MISSPEC_VAR = 0.001
def gauss(sigma):
    return np.random.normal(0, sigma)


def compute_n_unique_particles(particles):
    return len({p.label for p in particles})


def kalman_particle_factory():
    # Setup Kalman Filter
    A = np.array([[1 + gauss(MISSPEC), 0 + gauss(ZERO_MISSPEC), 0.1 + gauss(MISSPEC), 0 + gauss(ZERO_MISSPEC), 0 + gauss(ZERO_MISSPEC)],
                  [0.001 + gauss(MISSPEC), 1 + gauss(MISSPEC), 0 + gauss(MISSPEC), 0.8 + gauss(MISSPEC), 0 + gauss(MISSPEC)],
                  [0 + gauss(ZERO_MISSPEC), 0 + gauss(ZERO_MISSPEC), 1 + gauss(MISSPEC), 0 + gauss(ZERO_MISSPEC), 0 + gauss(ZERO_MISSPEC)],
                  [0 + gauss(MISSPEC), 0 + gauss(MISSPEC), 0 + gauss(MISSPEC), 1 + gauss(MISSPEC), 0 + gauss(MISSPEC)],
                  [0 + gauss(ZERO_MISSPEC), 0 + + gauss(ZERO_MISSPEC), 0 + + gauss(ZERO_MISSPEC), 0 + gauss(ZERO_MISSPEC), 1 + gauss(MISSPEC)]])
    B = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    C = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0]])
    D = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])
    Q = np.array([[0.8817 + gauss(MISSPEC_VAR), 0, 0, 0, 0],
                  [0, 17.81 + gauss(MISSPEC_VAR), 0, 0, 0],
                  [0, 0, 1 + gauss(MISSPEC_VAR), 0, 0],
                  [0, 0, 0, 0 + gauss(MISSPEC_VAR), 0],
                  [0, 0, 0, 0, 1 + gauss(MISSPEC_VAR)]])
    R = np.array([[1 + gauss(MISSPEC_VAR), 0],
                  [0, 1 + gauss(MISSPEC_VAR)]])

    init_mu = np.array([[0 + gauss(ZERO_MISSPEC)],
                        [5 + gauss(MISSPEC)],
                        [0 + gauss(ZERO_MISSPEC)],
                        [0 + gauss(ZERO_MISSPEC)],
                        [0 + gauss(ZERO_MISSPEC)]])

    init_V = np.array([[100000000000, 0, 0, 0, 0],
                       [0, 100000000000, 0, 0, 0],
                       [0, 0, 100000000000, 0, 0],
                       [0, 0, 0, 100000000000, 0],
                       [0, 0, 0, 0, 100000000000]])

    return KalmanParticle((A, B, C, D, Q, R), init_mu, init_V)

selection = [ccs.open, ccs.high, ccs.low, ccs.close, ccs.volume]
types = [ft.median_price] * 4 + [ft.signed_volume]
names = ['Open', 'High', 'Low', 'Close', '(+/-) x volume']
rmd = CommsecDataManager('RMD', selection, types, names)
rmd_allprices = CommsecDataManager('RMD', [ccs.high, ccs.low], [ft.price, ft.price], ['High', 'Low'])
rmd_close = CommsecDataManager('RMD', [ccs.close], [ft.price], ['Close'])

price_name = 'Median RMD High, RMD Low'

# Setup data
(dates, obs) = rmd[:]
obs = obs.T
obs = obs.reshape((obs.shape[0], 1, obs.shape[1]))
conds = np.zeros((5, 1, obs.shape[2]))

N = 300
init_rbpf_particles = [kalman_particle_factory().set_label("p={i}".format(i=i)) for i in range(N)]
rbpf = ParticleFilter(init_rbpf_particles, 30)

# Running rbpf
print("Running RBPF")
y_pred = np.zeros(obs.shape)
n_unique_particles = []
for t in range(obs.shape[Axis.time]):
    n_unique_particles.append(compute_n_unique_particles(rbpf.draw()))
    print("Processing point t={t}".format(t=dates[t]))
    y_pred[:, :, t] = rbpf.predict(conds[:, :, t])
    rbpf.observe(obs[:, :, t], conds[:, :, t])

(x, y, t) = y_pred.shape
pred = pd.DataFrame(y_pred.reshape((x, t)).T, columns=['Predicted Liquidity', 'Predicted Prices'])

# Ordinary Least Squares Calculation
ols = sum((rmd.data.values[:, 2] - pred.values[:, 1]) * (rmd.data.values[:, 2] - pred.values[:, 1])) / 2
print(ols)

from matplotlib import pyplot as plt
plt.plot(dates, n_unique_particles, color='red')
plt.show()

# Plot results
#area_ols if pred < low -> +, low <= pred <= high -> 0. high < pred -> +
ax = pred.plot(x=rmd.data[ccs.date.value], y=['Predicted Prices'])
#ax = rmd.data.plot(x=ccs.date.value, y=rmd.features(), ax=ax)
ax = rmd_allprices.data.plot(x=ccs.date.value, y=rmd_allprices.features(), ax=ax)
residual_spikes = pd.DataFrame((rmd.data.values[:, 2] - pred.values[:, 1]) * (rmd.data.values[:, 2] - pred.values[:, 1]), columns=['Residuals'])
residual_spikes.plot(x=rmd.data[ccs.date.value], y=['Residuals'], ax=ax)
plt.show()


below_high = np.less_equal(pred.values[:, 1], rmd_allprices.data['RMD High'].values[:])
above_low = np.less_equal(rmd_allprices.data['RMD Low'].values[:], pred.values[:, 1])
between_high_low = np.logical_and(below_high, above_low)

print('Probability (fr) Prediction between high and low:', sum(between_high_low) / (between_high_low.size))

# Profitability of model:
# Long/Short at close, Short/Long at median price if within high-low else close
# If not executed liquidate position. Sell/Buy back at next close
position = np.sign(pred.values[1:, 1] - rmd_close.data['RMD Close'].values[:-1])
executed = position * between_high_low[1:] * (pred.values[1:, 1] - rmd_close.data['RMD Close'].values[:-1])
not_executed = position * np.logical_not(between_high_low[1:]) * (rmd_close.data['RMD Close'].values[1:] - rmd_close.data['RMD Close'].values[:-1])

point_profit = executed + not_executed
cum_profit = np.zeros(executed.shape)
for i in range(1, executed.shape[0]):
    cum_profit[i] = cum_profit[i-1] + point_profit[i]


df_exec = pd.DataFrame({'Date': rmd.data[ccs.date.value].values[1:],
                        'RMD High': rmd_allprices.data['RMD High'].values[1:],
                        'RMD Low': rmd_allprices.data['RMD Low'].values[1:], 'Executed': executed,
                        'Not Executed': not_executed, 'Cum Profit': cum_profit})
df_exec.plot(x='Date', y=['RMD High', 'RMD Low', 'Cum Profit'])
profit = np.sum(executed) + np.sum(not_executed)
print(profit)
plt.show()