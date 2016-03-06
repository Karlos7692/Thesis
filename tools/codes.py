import argparse
import os
import re
import csv
import setup
from typing import List, Mapping

parser = argparse.ArgumentParser(description="Code parser to find the codes of specific securities")
parser.add_argument(
    '-t', '--tickers', dest='tickers', nargs='+', help='The tickers you wish to retrieve.'
)

parser.add_argument(
        '-si', '--isins_only', action='store_true', dest='isins_only'
)

SPLIT_PATTERN = re.compile(r'([A-Z_0-9 ]+)|("[a-zA-Z_0-9, ]+")')


def code_line_identity(csv_line: List[str]):
    if not csv_line:
        return False

    return re.match(r'^[A-Z0-9]+', csv_line[0]) is not None


def get_code(csv_file) -> Mapping[str,str]:
    with open(csv_file, 'r') as csvf:
        csv_lines = [l for l in csv.reader(csvf.readlines())]
        sanatised_lines = filter(code_line_identity, csv_lines)
        indexed_lines = map(tuple, sanatised_lines)
        return {ticker: isin for (ticker, _, _, isin) in indexed_lines}


def get_all_codes() -> Mapping[str, str]:
    codes = {}
    csvfs = filter(lambda f: str(f).endswith(".csv"), os.listdir(setup.res("codes")))
    for csvf in csvfs:
        codes = {**codes, **get_code(setup.res("codes", csvf))}
    return codes


def print_mapping(tickers: List[str], codes: Mapping[str, str], isins_only: bool):
    for ticker in tickers:
        if ticker not in codes:
            raise Exception("Ticker {ticker} is not in codes.".format(ticker=ticker))
        if isins_only:
            print(codes[ticker])
        else:
            print("{ticker} -> {isin}".format(ticker=ticker, isin=codes[ticker]))


def main():
    args = parser.parse_args()
    codes = get_all_codes()
    print_mapping(args.tickers, codes, args.isins_only)

if __name__ == '__main__':
    main()
