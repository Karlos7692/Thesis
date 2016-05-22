import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(__file__)
RESOURCES = os.path.join(ROOT, "resources")
CODES = os.path.join(ROOT, "codes")
DATA = os.path.join(RESOURCES, "data")
CACHE_DIR = 'cache'
DATE_FORMAT = 'yyyy'
def res(*paths):
    f = os.path.join(RESOURCES, *paths)
    if not os.path.exists(f):
        raise Exception("File path {abspath} does not exist.".format(abspath=os.path.abspath(f)))
    return f


def get_default_cache_dir():
    return os.path.join(ROOT, CACHE_DIR)


def pretty_date_str(date):
    format = '%d/%m/%Y'
    return pd.to_datetime(str(date)).strftime(format)


def date_file_str(date):
    format = '%d%m%Y'
    return pd.to_datetime(str(date)).strftime(format)


def arr11_str(d1xd1_array):
    return np.array2string(d1xd1_array).replace("[", "").replace("]", "")