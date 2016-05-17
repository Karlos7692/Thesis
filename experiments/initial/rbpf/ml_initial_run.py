from data.commsec import CommsecColumns as ccs, FeatureTypes as ft, CommsecDataManager
from models.mcmc.pf import KalmanParticle, ParticleFilter, Axis
import pandas as pd
import numpy as np


def compute_n_unique_particles(particles):
    return len({p.label for p in particles})


def gauss(m, n):
    return np.array([[np.random.standard_normal() for c in range(n)] for r in range(m)])

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
                  [0, 0, 0, 1]]) +  0.01 * gauss(state_size, state_size)
    B = np.zeros((state_size, state_size))
    C = np.array([[1, 0, 0, 0]]) + 0.001 * gauss(obs_size, state_size)
    D = np.zeros((obs_size, state_size))
    Q = np.eye(state_size, state_size)
    R = np.eye(obs_size, obs_size)

    init_mu = np.array([[6],
                        [0],
                        [0],
                        [0]]) + 0.01 * gauss(state_size, 1)

    init_V = np.eye(A.shape[0], A.shape[1])
    return (A, B, C, D, Q, R), init_mu, init_V


def gen_particle(i, recursion_limit=10):
    print("Generating particle {i}...".format(i=i))
    try :
        (kp, ll_hists) = KalmanParticle.fit(training_ys, training_us, gen_params(1, 4), iters=5, use_last_to_init=True)
        return kp
    except ValueError:
        if recursion_limit == 0:
            raise ValueError("There seems to be a problem with the model!")
        return gen_particle(i, recursion_limit=recursion_limit-1)


# Split the data into first half and second half
(r, c, ts) = obs.shape

test_start = 252
n_data_points = 2000
test_end = test_start + n_data_points + 1
training_ys = obs[:, :, :test_start]
training_us = conds[:, :, :test_start]
test_ys = obs[:, :, test_start:test_end]
test_us = conds[:, :, test_start:test_end]

N = 1000
init_rbpf_particles = [gen_particle(i).set_label("p={i}".format(i=i)) for i in range(N)]
rbpf = ParticleFilter(init_rbpf_particles, 20, eta=3.5)

# Running rbpf
print("Running RBPF")
y_pred = np.zeros(test_ys.shape)
n_unique_particles = []
for t in range(test_ys.shape[Axis.time]):
    n_unique_particles.append(compute_n_unique_particles(rbpf.draw()))
    print("Processing point t={t}".format(t=dates[t]))
    y_pred[:, :, t] = rbpf.predict(test_us[:, :, t])
    rbpf.observe(test_ys[:, :, t], test_us[:, :, t])


(x, y, t) = y_pred.shape
pred = pd.DataFrame(y_pred.reshape((x, t)).T, columns=['Predicted Prices'])

# Ordinary Least Squares Calculation
residuals = rmd.data.values[test_start:test_end, 1] - pred.values[:, 0]
ols = sum(residuals * residuals)
print('Least Mean Squares', ols)

from matplotlib import pyplot as plt
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

point_profit = executed #+ not_executed
cum_profit = np.zeros(executed.shape)
for i in range(1, executed.shape[0]):
    cum_profit[i] = cum_profit[i-1] + point_profit[i]


df_exec = pd.DataFrame({'Date': rmd.data[test_start:test_end][ccs.date.value].values[1:],
                        'RMD High': rmd_allprices.data[test_start:test_end]['RMD High'].values[1:],
                        'RMD Low': rmd_allprices.data[test_start:test_end]['RMD Low'].values[1:], 'Executed': executed,
                        'Not Executed': not_executed, 'Cum Profit': cum_profit})
df_exec.plot(x='Date', y=['RMD High', 'RMD Low', 'Cum Profit', 'Executed', 'Not Executed'])
profit = np.sum(executed) #+ np.sum(not_executed)
print(profit)
plt.show()

# project_n_points = 30
# next_ys_index = test_end
# y_proj = rbpf.project(project_n_points)
# plt.plot(np.concatenate((y_pred, y_proj), axis=2)[0, :].flatten())
# plt.plot(obs[1, :, test_start:test_end+project_n_points].flatten(), c='purple')
# plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD High'].values[:], 'g')
# plt.plot(rmd_allprices.data[test_start:test_end+project_n_points]['RMD Low'].values[:], 'r')
# plt.show()
