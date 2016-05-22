import os
from data.commsec import CommsecDataManager, CommsecColumns, FeatureTypes
from models.lds.gym import LDSGym
from models.lds.kalman import KalmanFilter
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import json
from cache import cache_utils
from models.ts.ts import ARMA
from models import ml
import utils

#TODO Simplify cache_dir
# TODO Add matching observation size and state size features to the mix
from renderers.renderers import GraphWriter, CSVReportWriter


def kalman_identified(kalman_file, features, fixed_params, cache_dir=os.getcwd()):
    with open(os.path.join(cache_dir, kalman_file), 'r') as f:
        kalman = json.load(f)
        f.close()
        return kalman['features'] == ",".join(features) and fixed_params == kalman['fixed params']


def log(log_file, msg):
    mode = 'a' if os.path.exists(log_file) else 'w'
    today = datetime.now()
    with open(log_file, mode) as f:
        f.write("Run, {today} {msg}\n".format(today=today, msg=msg))
        f.close()

START_DATE = datetime.strptime('20060307', '%Y%m%d')
DELTA_SAN = relativedelta(months=6)
N_SAN = 10 * 2
KEY_DATES = [START_DATE + k * DELTA_SAN for k in range(0, N_SAN + 1)]

CACHE_DIR = os.path.join(os.getcwd(), "..", "..", "cache/exp1")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
LOG_FILE = os.path.join(os.getcwd(), "experiment1.log")

# ALL of these stocks are between 20060307-20160307
STOCKS = ['ANN', 'BKL', 'CCL', 'COH', 'CSL', 'CTX', 'GNC', 'MTS', 'ORG', 'OSH', 'RHC', 'RIC', 'RMD', 'SHV', 'SIP', 'SOL',
          'SPL', 'SRX', 'STO', 'WOW', 'WPL']

SELECTION = [CommsecColumns.high, CommsecColumns.low]
AS_TYPES = [FeatureTypes.median_price, FeatureTypes.median_price]
WITH_NAMES = ['High', 'Low']

training_start_dates = KEY_DATES[:-2]
test_start_dates = KEY_DATES[1:-1]
test_end_dates = KEY_DATES[2:]
n_periods = len(test_end_dates)

N_LATENT = 3

# Define a period as 6 months.
for stock in STOCKS[:7]:
    data_manager = CommsecDataManager(stock, SELECTION, AS_TYPES, WITH_NAMES)
    # Split into training sets and test sets for each period
    for period in range(n_periods):
        training_data = data_manager.get_data_set(training_start_dates[period], test_start_dates[period])
        test_data = data_manager.get_data_set(test_start_dates[period], test_end_dates[period], end_inclusive=True)

        # Experiment for 1 period
        training_dates = training_data[CommsecColumns.date.value]
        test_dates = test_data[CommsecColumns.date.value]

        training_set = training_data[data_manager.features()].values
        test_set = test_data[data_manager.features()].values

        # Reshape data for models to use.
        training_ys = training_set.T.reshape((training_set.shape[1], 1, training_set.shape[0]))
        test_ys = test_set.T.reshape((test_set.shape[1], 1, test_set.shape[0]))

        # Kalman Filter Management
        # Kalman Filter control input, Also we can choose the number of latent variables
        training_us = np.zeros((1 + N_LATENT, 1, training_ys.shape[2]))
        test_us = np.zeros((1 + N_LATENT, 1, test_ys.shape[2]))

        # Train Kalman Filter
        n_obs = training_ys.shape[2]
        kalman_gym = LDSGym(KalmanFilter)

        # Check if we have already trained a model for this.
        # TODO Move to a caching module, to hide the cache and restore kalman filters.
        # TODO Simplify Caching code
        possible_files = cache_utils.get_matching_ids("kalman", stock, training_start_dates[period],
                                                      test_start_dates[period], True, cache_dir=CACHE_DIR)
        identified_cached_kalman = [kalman_identified(f, data_manager.features(), True, cache_dir=CACHE_DIR)
                                    for f in possible_files]
        if any(identified_cached_kalman):
            # Debug purposes
            if len(identified_cached_kalman) > 1:
                raise Exception("There is a problem with the cache!")
            # Get the identified:
            kalman_files = [possible_files[i] for i, is_match in enumerate(identified_cached_kalman) if is_match]
            cached_kalman = kalman_files.pop(0)
            kf = KalmanFilter.restore(cached_kalman, cache_dir=CACHE_DIR)
            print("RESTORED {cached}.".format(cached=cached_kalman))
        else:

            # Train a new kalman filter and cache it
            kf = None
            for n_its in range(8, 0, -1):
                run_title = "{stock} from period {start} - {finish} with {n_its} EM iterations"\
                    .format(stock=stock, start=utils.pretty_date_str(training_start_dates[period]),
                            finish=utils.pretty_date_str(test_start_dates[period]), n_its=n_its)
                kf = kalman_gym.select_best_model(training_ys, training_us, iters=n_its, min_fit=n_obs - 10,
                                                  n_models=25, run_title=run_title)
                if kf is not None:
                    break

            if kf is None:
                raise Exception("Could not fit an LDS")

            kf_cache_name = cache_utils.get_new_cache_name("kalman", stock, training_start_dates[period],
                                                           test_start_dates[period], True)
            kf.cache(kf_cache_name, data_manager.features(), "Kalman filter (Best Model) trained from {start}-{end}"
                     .format(start=training_start_dates[period], end=test_start_dates[period]), cache_dir=CACHE_DIR)
