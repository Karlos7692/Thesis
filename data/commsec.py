from typing import List, abstractmethod
from data.datamgmt import AbstractDataManager, AbstractDataReaderManager, AbstractMergeManager, FeatureContract
from enum import Enum
from collections import OrderedDict
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils
import os

DATA_DIR = utils.res(utils.DATA, "commsec")


class CommsecColumns(Enum):
    date = 'Date'
    open = 'Opening Price'
    high = 'High Sale Price'
    low = 'Low Sale Price'
    close = 'Closing Price'
    volume = 'Total Volume Traded'

    @staticmethod
    def values():
        return [e.value for e in iter(CommsecColumns)]

    @staticmethod
    def prices():
        return [CommsecColumns.open.value, CommsecColumns.high.value, CommsecColumns.low.value,
                CommsecColumns.close.value]


class FeatureTypes(Enum):
    price = "price"
    median_price = "median price",
    returns = "returns"
    median_returns = "median returns"
    volume = "volume"
    signed_volume = "signed volume"


def get_ticker_name(ticker, name):
    return "{ticker} {name}".format(ticker, name)

FILE_FORMAT = '{ticker}.{min_date}-{max_date}.csv'


def get_csv(ticker):
    commsec_csvs = filter(lambda f: ticker in str(f), os.listdir(DATA_DIR))
    commsec_csvs = list(map(lambda f: os.path.abspath(os.path.join(DATA_DIR, f)), commsec_csvs))

    if not commsec_csvs:
        raise Exception('There do not exist any csvs with ticker {ticker}'.format(ticker=ticker))

    if len(commsec_csvs) != 1:
        raise Exception("Unhandled more than one csv {csvs}".format(csvs=commsec_csvs))

    return commsec_csvs.pop()


def get_signed_volume(volumes: np.array):
    signs = np.concatenate((np.array([0]), np.sign(volumes[1:] - volumes[:-1])))
    return signs * volumes


# TODO implement lagged components, diffed components to feature selection.
class AbstractCommsecDataManager(AbstractDataManager):

    def __init__(self, select_columns: List[CommsecColumns], as_types: List[FeatureTypes], with_names: List[str]):
        super(AbstractDataManager, self).__init__()

        if len(select_columns) != len(as_types) != len(with_names):
            raise Exception("Column selection must have the same length as the types of features and their names.")

        self.select_columns = [col.value for col in select_columns]
        self.as_types = as_types
        self.with_names = with_names

    @abstractmethod
    def __clean_data__(self):
        pass

    def __select_features__(self):
        super().__select_features__()
        # TODO Prices first then returns then other data
        features = OrderedDict([(CommsecColumns.date.value, self.data[CommsecColumns.date.value])])
        median_prices = OrderedDict([])
        median_returns = OrderedDict([])
        for (i, col) in enumerate(self.select_columns):
            type = self.as_types[i]
            name = self.with_names[i]

            if type == FeatureTypes.price or type == FeatureTypes.volume:
                features[name] = self.data[col]
                continue

            if type == FeatureTypes.median_price:
                median_prices[name] = self.data[col]
                continue

            if type == FeatureTypes.returns:
                features[name] = self.data[col].pct_change(1)
                continue

            if type == FeatureTypes.median_returns:
                median_returns[name] = self.data[col].pct_change(1)
                continue

            if type == FeatureTypes.signed_volume:
                features[name] = get_signed_volume(self.data[col].values)
                continue

            raise Exception("Unsupported Feature Type {type}.".format(type=type))

        if median_prices:
            name = "Median of " + ", ".join([name for name in median_prices])
            features[name] = pd.DataFrame(data=median_prices).median(axis=1)

        if median_returns:
            name = "Median of " + ", ".join([name for name in median_returns])
            features[name] = pd.DataFrame(data=median_returns).median(axis=1)

        self.data = pd.DataFrame(data=features)


class CommsecDataManager(AbstractCommsecDataManager, AbstractDataReaderManager):

    def __init__(self, ticker: str, select_columns: List[CommsecColumns], as_types: List[FeatureTypes],
                 with_names: List[str]):
        with_names = ["{ticker} {name}".format(ticker=ticker, name=name) for name in with_names]
        AbstractCommsecDataManager.__init__(self, select_columns, as_types, with_names)
        AbstractDataReaderManager.__init__(self, ticker)

        self.start_date = None
        self.end_date = None

    def __read_data__(self, ticker: str):
        ccs = CommsecColumns
        csv = get_csv(ticker)
        self.data = pd.read_csv(csv, parse_dates=[ccs.date.value], dayfirst=True, usecols=ccs.values())
        self.data.sort_values(by=CommsecColumns.date.value, inplace=True)
        self.start_date = self.data[ccs.date.value].min()
        self.start_date = self.data[ccs.date.value].max()

    def __clean_data__(self):
        self.data.drop_duplicates(subset=[CommsecColumns.date.value])

        # Replace any zero prices with previous days prices.
        [self.data[col].replace(to_replace=0, method='ffill', inplace=True) for col in CommsecColumns.prices()]

        # Replace 0 traded volume with mean because we are trying to assess the added value of liquidity.
        self.data.replace(to_replace=0, value=self.data[CommsecColumns.volume.value].mean(), inplace=True)

    def __transform_data__(self):
        # Normalize volume
        vmean = self.data[CommsecColumns.volume.value].mean()
        vsd = self.data[CommsecColumns.volume.value].std()
        self.data[CommsecColumns.volume.value] = self.data[CommsecColumns.volume.value]\
            .apply(func=lambda v:  (v - vmean)/vsd)

    def __validate_selection__(self):
        for (i, col) in enumerate(self.select_columns):
            type = self.as_types[i]
            if col in CommsecColumns.prices() and type != FeatureTypes.price and type != FeatureTypes.returns and\
                            type != FeatureTypes.median_price and type != FeatureTypes.median_returns:
                raise Exception("column {column} is a price however we found {type}".format(column=col, type=type))

            if col == CommsecColumns.volume and type != FeatureTypes.volume and type != FeatureTypes.signed_volume:
                raise Exception("column {column} is not a {type}".format(column=col, type=type))

            if col == CommsecColumns.date:
                raise Exception("Date is not a supported feature")

    def features(self):
        return [ft for ft in super().features() if ft != CommsecColumns.date.value]

    def plot(self):
        self.data.plot(x=CommsecColumns.date.value, y=self.features())
        return self

    def show(self):
        plt.show()

    """
    Feature Management and Iteration
    """
    def __getitem__(self, item):
        if isinstance(item, int):
            date = self.data.iloc[item][CommsecColumns.date.value]
            obs = np.array([self.data.iloc[item][self.features()]])
            return date, obs
        data_slice = self.data[item]
        return data_slice[CommsecColumns.date.value].values, data_slice[self.features()].values

    """
    Data Set Management
    """
    def get_data_set(self, start_date, end_date, end_inclusive=False):
        if end_inclusive:
            mask = (start_date <= self.data[CommsecColumns.date.value]) & (self.data[CommsecColumns.date.value] <= end_date)
        else:
            mask = (start_date <= self.data[CommsecColumns.date.value]) & (self.data[CommsecColumns.date.value] < end_date)
        return self.data.loc[mask]



class CommsecMergeManager(AbstractCommsecDataManager, AbstractMergeManager):

    def __init__(self, data_managers: List[AbstractDataManager], select_columns: List[CommsecColumns],
                 as_types: List[FeatureTypes], with_names: List[str]):
        self.data_managers = data_managers
        AbstractCommsecDataManager.__init__(self, select_columns, as_types, with_names)
        AbstractMergeManager.__init__(self)

    def __clean_data__(self):
        pass

    def __merge__(self):
        # Do not assume order of feature names in selection. Load all features into one sigular data frame
        features = [(fname, dm.data[fname]) for fname in self.select_columns for dm in self.data_managers
                   if fname in dm.features()]
        self.data = pd.DataFrame(data=OrderedDict(features))

    def __select_features__(self):
        super(AbstractCommsecDataManager, self).__select_features__()



# from matplotlib import pyplot as plt
#
# c = CommsecColumns
# f = FeatureTypes
# dwes = CommsecDataManager('WES', [c.open, c.high, c.low, c.close, c.volume],
#                           [f.median_price, f.median_price, f.median_price, f.median_price, f.signed_volume],
#                           ['Open', 'High', 'Low', 'Close', 'Volume'])
# dwow = CommsecDataManager('WOW', [c.open, c.high, c.low, c.close, c.volume],
#                           [f.median_price, f.median_price, f.median_price, f.median_price, f.signed_volume],
#                           ['Open', 'High', 'Low', 'Close', 'Volume'])
# dtwe = CommsecDataManager('TWE', [c.open, c.high, c.low, c.close, c.volume],
#                           [f.median_price, f.median_price, f.median_price, f.median_price, f.signed_volume],
#                           ['Open', 'High', 'Low', 'Close', 'Volume'])
# dccl = CommsecDataManager('CCL', [c.open, c.high, c.low, c.close, c.volume],
#                           [f.median_price, f.median_price, f.median_price, f.median_price, f.signed_volume],
#                           ['Open', 'High', 'Low', 'Close', 'Volume'])
# ax = dwes.data.plot(x=CommsecColumns.date.value, y=['Median of WES Open, WES High, WES Low, WES Close', 'WES Volume'])
# ax = dwow.data.plot(x=CommsecColumns.date.value, y=['Median of WOW Open, WOW High, WOW Low, WOW Close', 'WOW Volume'], ax=ax)
# ax = dtwe.data.plot(x=CommsecColumns.date.value, y=['Median of TWE Open, TWE High, TWE Low, TWE Close', 'TWE Volume'], ax=ax)
# ax = dccl.data.plot(x=CommsecColumns.date.value, y=['Median of CCL Open, CCL High, CCL Low, CCL Close', 'CCL Volume'], ax=ax)
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.show()
