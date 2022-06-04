import logging 
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler



def RMSE(yt, yp):
    return np.sum((yt - yp) ** 2) / yp.shape[0]


def get_all_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    report = pd.read_csv('data/report.csv', skiprows=1,
                names=[
                    'ID', 'year', 'month', 'ranch',
                    'serial', 'father','mother', 'birthday',
                    'delivery', 'lactation', 'volume', 'age'],
                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
                dtype={'father': str, 'mother': str},
                parse_dates=['birthday'])
    # A. From the result of verifing data.
    report.drop(report.index[report['ID'] == 6960], inplace=True)
    report.drop(report.index[report['ID'] == 16714], inplace=True)
    # B. Encode father, mother, ranch.
    report.father = LabelEncoder().fit_transform(report.father)
    report.mother = LabelEncoder().fit_transform(report.mother)
    report.ranch = LabelEncoder().fit_transform(report.ranch)
    # C. Encode serial based on birthday.
    unique_serial = report.sort_values(by='birthday').serial.unique()
    new_serial_map = {old: i for i, old in enumerate(unique_serial)}
    report.serial = report.serial.map(new_serial_map)
    # Don't need birthday now.
    report.drop('birthday', axis=1, inplace=True)
    report.lactation = report.lactation.astype('int64')
    train = report.loc[report['volume'].notnull()]
    test  = report.loc[report['volume'].isnull()]
    logging.info(f'In train: Number of unique:\n{train.nunique()}')
    return train, test

# TODO
def remove_outlier(train):
    pass


if __name__ == '__main__':
    train, test = get_all_features()
