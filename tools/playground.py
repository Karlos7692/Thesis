# import os
#
# import tools.codes as cs
# import utils
# from data import commsec
#
# COMMSEC = utils.res("data", "commsec")
#
#
#
# def get_filenames(secs):
#     tickers = cs.get_all_codes().keys()
#     tickers = list(filter(lambda n: n != 'CSV', tickers))
#     l = []
#     for t in tickers:
#         for s in secs:
#             if t in str(s):
#                 l.append(t)
#     fnames = []
#     for t in l:
#         dm = commsec.CommsecDataManager(t)
#         dm.read_data()
#         d, _ = dm.first()
#         min_date = '%d%02d%02d' % (d.year, d.month, d.day)
#         d, _ = dm.last()
#         max_date = '%d%02d%02d' % (d.year, d.month, d.day)
#         fnames.append("{ticker}.{min_date}-{max_date}.csv".format(ticker=t, min_date=min_date, max_date=max_date))
#     return fnames
#
# secs = os.listdir(COMMSEC)
# fnames = get_filenames(secs)
#
# print(list(set(secs)-set(fnames)))
# print(get_filenames(list(set(secs)-set(fnames)))

# from data import commsec
#
# c = commsec.CommsecDataManager('WOW')
# c.read_data()
#
# npa = commsec.daily_return(c, range(0, commsec.TOTAL_VOLUME_TRADED))
# for i,p in enumerate(npa):
#     if any(list(map(lambda x: x==0, p))):
#         print(i, p)
#
# c.position = 1088
# print(next(c))

import pandas
from pandas import Series, DataFrame
import math
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot


df = pandas.read_csv('resources/data/commsec/WOW.20060307-20160307.csv', parse_dates=['Date'], dayfirst=True,
                     usecols=['Date', 'Opening Price', 'High Sale Price',  'Low Sale Price', 'Closing Price',
                              'Total Volume Traded'])

df.sort_values(by='Date', inplace=True)

df['Opening Price'].replace(to_replace=0, method='ffill', inplace=True)
df['High Sale Price'].replace(to_replace=0, method='ffill', inplace=True)
df['Low Sale Price'].replace(to_replace=0, method='ffill', inplace=True)
df['Closing Price'].replace(to_replace=0, method='ffill', inplace=True)
df['Total Volume Traded'].replace(to_replace=0, value=df['Total Volume Traded'].mean(), inplace=True)
df['Total Volume Traded'] = df['Total Volume Traded'].apply(func=math.log1p)

tvmean = df['Total Volume Traded'].mean()
tvsd = df['Total Volume Traded'].std()


def normalize(v):
    return (v-tvmean)/tvsd


# df['Total Volume Traded'] = df['Total Volume Traded'].apply(func=normalize)
# df.hist(column='Total Volume Traded', figsize=(15,7), bins=15)
# df.plot(x='Date', y='Total Volume Traded', title='ln Traded Volume')
# plt.show()
#
# autocorrelation_plot(df['Opening Price'])
# autocorrelation_plot(df['High Sale Price'])
# autocorrelation_plot(df['Low Sale Price'])
# autocorrelation_plot(df['Closing Price'])
# autocorrelation_plot(df['Total Volume Traded'])
# df.plot(x='Date', y=['Opening Price', 'Closing Price', 'High Sale Price', 'Low Sale Price'], color=['blue', 'purple', 'green', 'red'])
# plt.show()
#
# oprs = df['Opening Price'].pct_change(25)
# ohprs = df['High Sale Price'].pct_change(25)
# olprs = df['Low Sale Price'].pct_change(25)
# ocprs = df['Closing Price'].pct_change(25)
# autocorrelation_plot(oprs)
# autocorrelation_plot(ohprs)
# autocorrelation_plot(olprs)
# autocorrelation_plot(ocprs)
# plt.show()
#
# rdf = DataFrame(data={'Date': df['Date'], 'Open': oprs, 'High': ohprs, 'Low': olprs, 'Close': ocprs})
# rdf.plot(x='Date', y=['Open', 'Close', 'High', 'Low'], color=['blue', 'purple', 'green', 'red'])
# rdf.hist(column='Open', figsize=(15,7), bins=15)
# rdf.hist(column='Close', figsize=(15,7), bins=15)
# rdf.hist(column='High', figsize=(15,7), bins=15)
# rdf.hist(column='Low', figsize=(15,7), bins=15)
# plt.show()
#
#
# data = {'Date': df['Date'], 'Open': df['Opening Price'], 'High': df['High Sale Price'], 'Close': df['Closing Price']}
# med = pandas.DataFrame(data=data)
# med = med.median(axis=1)
# print(med)


