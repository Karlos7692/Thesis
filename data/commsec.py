import utils
import os
import csv as csvutils
import data import

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

        map(lambda commsec_csv: read_commsec_csv(self, commsec_csv), commsec_csvs)


def read_commsec_csv(dm: CommsecDataManager, commsec_csv):
    with open(commsec_csv, 'r') as csv:
        for i, l in enumerate(csvutils.reader(csv.readlines())):
            if i is 0:
                dm.point_definition = (l[1:])
            else:
                parse_point(dm, l[1:])
        csv.close()
        sorted(dm)


def parse_point(dm: CommsecDataManager, point: iter):
    # Format: Date, Opening Price, High Sale Price, Low Sale Price, Closing Price, Total Volume Traded
    dm[timeutils.get_closing_time(point[0])] = (float(point[1]), float(point[2]), float(point[3]), float(point[4]),
                                                 int(point[5]))

wow = CommsecDataManager('WOW')
wow.read_data()
for d,v in wow:
    print(d,v)