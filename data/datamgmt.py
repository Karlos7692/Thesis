from abc import abstractmethod, ABCMeta
from datetime import time
from collections import OrderedDict

"""
Abstract Data Manager classes. To be implemented in each data manager type.
"""


# Abstract Data Manager. The default class handling any data.
class AbstractDataManager(metaclass=ABCMeta):

    INVALID_POSITION = -1

    @abstractmethod
    def read_data(self, start_date=None, end_date=None):
        pass

    def __init__(self, ticker):
        # date -> data point (tuple)
        self.ticker = ticker
        self.data = OrderedDict()
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            self.reset()
            raise StopIteration("The data does not have a next.")
        ret = list(self.data.items())[self.position]
        self.position += 1
        return ret

    def __setitem__(self, key, value):
        if key in self.data and self.data[key] != value:
            raise Exception("Added key {key} exists, expected {expected} but found {value}"
                            .format(key=key, expected=self.data[key], value=value))

        if key in self.data:
            return

        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __after_last__(self):
        if self.position > len(self.data):
            return True
        return False

    def get_ticker(self):
        return self.ticker

    def has_next(self,) -> bool:
        if self.position < len(self.data):
            return True
        return False

    def get_position(self) -> int:
        return self.position

    def reset(self) -> bool:
        self.position = 0

    # TODO Change to slice
    def filter_values(self, value_range: range):
        for d, v in self.data:
            tuple([v[i] for i in value_range])

    def first(self):
        if len(self) == 0:
            raise Exception("DataManager must have items to retrieve first item")
        return list(self.data.items())[0]

    def last(self):
        if len(self) == 0:
            raise Exception("DataManager must have items to retrieve first item")
        return list(self.data.items())[len(self)-1]


class DailyDataManager(AbstractDataManager):

    @abstractmethod
    def read_data(self, start_date=None, end_date=None):
        pass

    def get_point(self, date):
        if date in self.data:
            return self.data[date]
        return None

    def get_range(self, start_date: time, end_date=None):
        ran = []
        if start_date not in self.data:
            raise Exception("Invalid start date {start_date}".format(start_date=start_date))

        for point in list(self.data.items()):
            d, v = point
            if end_date is None and start_date <= d:
                ran.append(point)
                continue

            if end_date is not None and start_date <= d <= end_date:
                ran.append(point)
                continue

            if d > end_date:
                break

        return ran




