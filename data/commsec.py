import utils
import os
import csv as csvutils
import itertools
from data import timeutils
from data.datamgmt import DailyDataManager
from typing import List
import numpy as np


DATA_DIR = utils.res(utils.DATA, "commsec")
OPENING_PRICE = 0
HIGH_SALES_PRICE = 1
LOW_SALES_PRICE = 2
CLOSING_PRICE = 3
TOTAL_VOLUME_TRADED = 4


class CommsecDataManager(DailyDataManager):

    def __init__(self, ticker):
        super(CommsecDataManager, self).__init__(ticker)
        self.point_definition = ()

    def read_data(self, start_date=None, end_date=None):
        ticker = self.get_ticker()
        commsec_csvs = filter(lambda f: ticker in str(f), os.listdir(DATA_DIR))
        commsec_csvs = list(map(lambda f: os.path.abspath(os.path.join(DATA_DIR, f)), commsec_csvs))

        if not commsec_csvs:
            raise Exception("There are no csvs in the {data_dir} that matched the ticker {ticker}".format(data_dir=DATA_DIR, ticker=ticker))

        list(map(self.read_commsec_csv, commsec_csvs))

    def read_commsec_csv(self, commsec_csv):
        with open(commsec_csv, 'r') as csv:
            for i, l in enumerate(csvutils.reader(csv.readlines())):
                if i is 0:
                    self.point_definition = (l[1:])
                else:
                    self.parse_point(l[1:])
            csv.close()
            self.reorder(sorted(self.data.keys()))

    def parse_point(self, point: iter):
        # Format: Date, Opening Price, High Sale Price, Low Sale Price, Closing Price, Total Volume Traded
        self[timeutils.get_closing_time(point[0])] = (float(point[1]), float(point[2]), float(point[3]), float(point[4]),
                                                      int(point[5]))

    # TODO Find paper dealing with missing data points in time series
    def clean(self):
        for i, (d, point) in enumerate(self):
            zeroed_prices = [v == 0 for v in list(point)[0:4]]

    def daily_return(self, r: range) -> List[np.array]:
        values = [np.array(list(itertools.islice(point, r.start, r.stop, r.step))) for (key, point) in self]
        return [ (values[i] - values[i-1])/values[i-1] for i in range(0, len(values))]



