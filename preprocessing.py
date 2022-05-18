import logging 
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler



def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def verify_data():
    """
    1. The age of a cow == (sampling date - birthday).
    2. sampling date <= recording date.
    3. #days of lactation == sampling date - last labour date
    4. The birthday should agree with the one in birth.csv.
    5. A cow should have exactly one birthday.
    6. The serial number should not appear in report.csv after the cow died.
    """
    report = pd.read_csv('data/report.csv', skiprows=1, header=None, 
                parse_dates=[7, 11, 12, 14])
    birth  = pd.read_csv('data/birth.csv',  skiprows=1, header=None)
    spec   = pd.read_csv('data/spec.csv',   skiprows=1, header=None)
    spec_died = spec.loc[spec[4].isin(['死亡', '木乃伊', '淘汰'])]
    for i, row in report.iterrows():
        if diff_month(row[12], row[7]) != row[13]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 8, 13, 14')
        if row[12] > row[14]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 13, 15')
        if row[12].year != row[1] or row[12].month != row[2]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 2, 3, 13')
        if row[9] != (row[12] - row[11]).days:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 9, 11, 12')
        oracle = birth.loc[(birth[0] == row[4]) & (birth[1] == row[11])]
        if oracle.shape[0] != 1:        # One and only one.
            logging.debug(f'Not found ID = {row[0]:<6d} in birth.csv')
        elif row[8] != oracle.iloc[0][8]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 9')
    multi_birthday = report.drop_duplicates(subset=[4,7]).duplicated(subset=4)
    for serial, flag in multi_birthday.iteritems():
        if flag == True:
            logging.warning(f'Multi-birthday: Serial = {serial}')
    for i, row in spec_died.iterrows():
        oracle = report.loc[report[4] == row[0]]
        if oracle.shape[0] == 0:
            logging.debug(f'Not found: Serial = {row[0]} in report.csv')
        else:
            logging.warning(f'Died: Serial = {row[0]}')
    return


def get_all_features():
    report = pd.read_csv('data/report.csv', skiprows=1,
                names=['ID', 'year', 'month', 'ranch', 'serial', 
                       'delivery', 'lactation', 'yield', 'age'],
                usecols=[0, 1, 2, 3, 4, 8, 9, 10, 13])
    # A. From the result of verifing data.
    report.drop(report.index[report['ID']==6960], inplace=True)
    report.drop(report.index[report['ID']==16714], inplace=True)
    # B. Use int64 if possible.
    report.ranch = LabelEncoder().fit_transform(report.ranch).astype('int64')
    report.delivery = report.delivery.astype('int64')
    report.lactation = report.lactation.astype('int64')
    # D. TODO Encoding serial.
    # G. TODO Add (col 3 - col 2) from birth.csv.
    # H. TODO col 16~21 might be found from breed.csv
    train = report.loc[report['yield'].notnull()]
    test  = report.loc[report['yield'].isnull()]
    logging.info(f'In train: Number of unique:\n{train.nunique()}')
    return train, test

# TODO
def remove_outlier(train):
    pass


if __name__ == '__main__':
    train, test = get_all_features()
