from abc import abstractmethod, ABCMeta
import pandas as pd
import json

class AbstractDataManager(metaclass=ABCMeta):

    def __init__(self):
        self.data = pd.DataFrame([])

    @abstractmethod
    def __clean_data__(self):
        pass

    @abstractmethod
    def __select_features__(self):
        pass

    def features(self):
        return self.data.columns.values


class AbstractDataReaderManager(AbstractDataManager):

    def __init__(self, ticker):
        super().__init__()
        self.ticker = ticker
        self.__read_data__(ticker)
        self.__clean_data__()
        self.__transform_data__()
        self.__validate_selection__()
        self.__select_features__()

    @abstractmethod
    def __read_data__(self, ticker: str):
        pass

    @abstractmethod
    def __clean_data__(self):
        pass

    @abstractmethod
    def __transform_data__(self):
        pass

    @abstractmethod
    def __validate_selection__(self):
        pass

    @abstractmethod
    def __select_features__(self):
        pass


class AbstractMergeManager(AbstractDataManager):

    def __init__(self):
        super().__init__()
        self.__merge__()
        self.__select_features__()

    @abstractmethod
    def __clean_data__(self):
        pass

    @abstractmethod
    def __merge__(self):
        pass

    @abstractmethod
    def __select_features__(self):
        pass


class FeatureContract(object):

    def __init__(self, column_names):
        self.column_names = column_names


# TODO
class FeatureProcessor(object):
    """
    Create a Data Manager with all relevant features.
    Specification: Json of hierarchy of operations needed to transform all relevant data into clean features.
    {
        {
            operation: 'name'
            args: [args needed for operation]
            data: [operations] or {operation} (Note read does not require data)
            provider: Data manager class used to perform this operation
        }
    }
    """
    pass


# TODO
class ExogenousVariableProcessor(object):
    """
    Create a ExoVarManager with all the relevant features.
    Specification: Json of hierarchy of operations needed to transform exogenous variables.
    """
    pass