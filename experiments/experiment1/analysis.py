import os
import csv

RESULTS_FILE = os.path.join("results", "experiment_results.csv")


with open(RESULTS_FILE, 'r') as rf:
    data = rf.readlines()
    rf.close()

DATE_FROM = 0
DATE_TILL = 1
STOCK = 2
ALGORITHM = 3
AAE = 4
RMSE = 5
MAPE = 6
MSPE = 7

RESULTS_OFFSET = 4
KALMAN_FILTER = 'Kalman Filter'
ARMA = "ARMA(1,0)"
ARMA_KEY = "AR"

results = {}
for line in data[1:]:
    #Date From, Till Date, Stock, Algorithm, AAE, RMSE, MAPE, MSPE
    line = line.replace(ARMA, ARMA_KEY)
    values = line.split(",")
    date_key = values[DATE_FROM] + "-" + values[DATE_TILL]
    stock = values[STOCK]
    algorithm = values[ALGORITHM]
    if (date_key, stock) not in results:
        results[(date_key, stock)] = {algorithm: (values[AAE], values[RMSE], values[MAPE], values[MSPE])}
    else:
        results[(date_key, stock)][algorithm] = (values[AAE], values[RMSE], values[MAPE], values[MSPE])

# Calculate the frequency that the Kalman Filter is better than the ARMA model.
kalman_better_freq = 0.0
print('Times Kalman Filter performs better than ARMA')
for date_stock_key in results:
    algorithm_res = results[date_stock_key]
    if algorithm_res[KALMAN_FILTER][RMSE-RESULTS_OFFSET] < algorithm_res[ARMA_KEY][RMSE-RESULTS_OFFSET]:
        print(date_stock_key[0], date_stock_key[1], 'KF', algorithm_res[KALMAN_FILTER][RMSE-RESULTS_OFFSET],
              'ARMA', algorithm_res[ARMA_KEY][RMSE-RESULTS_OFFSET])
        kalman_better_freq += 1.0


total_comparisons = len(results)

print("Percentage of times that the fitted Kalman Filter was better than the ARMA(1,1): ",
      kalman_better_freq/total_comparisons * 100)