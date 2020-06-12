import argparse
import csv
import itertools
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='merges two csv files'
    )
    parser.add_argument('in1', help='first csv to merge')
    parser.add_argument('in2', help='second csv to merge')
    parser.add_argument(
        '--output', help='where to save merged csv file', default='merged.csv')
    parser.add_argument(
        '--on', help='what field to merge on', default='ID')

    args = parser.parse_args()
    merge_csvs(args)

def merge_csvs(args):
    cases_df = pd.read_csv(args.in1)
    labels_df = pd.read_csv(args.in2)
    merged_df = pd.merge(cases_df, labels_df, how='outer', on=args.on)
    merged_df.to_csv(args.output)


if __name__ == '__main__':
    main()
