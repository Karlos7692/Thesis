from data.commsec import CommsecDataManager
import utils
import os
import re

COMMSEC = utils.res("data", "commsec")

def reformat_files():
    csvs = [f for f in os.listdir(COMMSEC) if str(os.path.abspath(f)).endswith(".csv")
            and not re.search(r'\.\d{8}-\d{8}', f)]

    ticker_to_csv = {str(csv).replace('CSV-', '').replace('.csv', ''): csv for csv in csvs}

    for ticker in ticker_to_csv:
        dm = CommsecDataManager(ticker, [], [], [])
        start = '%d%02d%02d' % (dm.start_date.year, dm.start_date.month, dm.start_date.day)
        end = '%d%02d%02d' % (dm.end_date.year, dm.end_date.month, dm.end_date.year.day)
        csv_new_name = "{ticker}.{start}-{end}.csv".format(ticker=ticker, start=start, end=end)
        csv_old_name = ticker_to_csv[ticker]
        os.rename("/".join([COMMSEC, csv_old_name]), "/".join([COMMSEC, csv_new_name]))



csvs = [f for f in os.listdir(COMMSEC) if str(os.path.abspath(f)).endswith(".csv") and "20060307" in str(f) and "20160307" in str(f)]
for csv in csvs:
    print(re.split("\.",csv)[0])