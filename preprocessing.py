import logging 
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVR

def main():
    train, test = get_all_features()
    # Remove 64 data with volume == 0. 
    train.drop(train.index[train.volume == 0], inplace=True)
    # Realistic validation set based on number of new cows
    valid = train.query('year == 2018 & month > 9').copy()
    train.query('year < 2018 | month <= 9', inplace=True)
    # Just like in real case, we don't know all the mean and std.
    add_mean_std(train, valid)
    # If we have mean and std, drop other features
    drop_list = ['ID', 'ranch', 'serial', 'father', 'mother']
    # Play with other features
    drop_list.extend(['year', 'month'])
    train.drop(drop_list, axis=1, inplace=True)
    valid.drop(drop_list, axis=1, inplace=True)

    x = train.drop('volume', axis=1).to_numpy()
    xv = valid.drop('volume', axis=1).to_numpy()
    y = train['volume'].to_numpy()
    yv = valid['volume'].to_numpy()

    regr = SVR()
    regr.fit(x, y)
    yp = regr.predict(xv)
    print('RMSE =', RMSE(yv, yp))
    plt.scatter(yv, yp, label='SVR')
    plt.scatter(yv, valid['mean'], label='Mean')
    plt.axis([0, 60, 0, 60])
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.legend()
    plt.show()
    return


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
    report.drop(report.index[report['age'] < 21], inplace=True)
    # B. Simple encoding
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


def add_mean_std(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Add the average volume of each cow at each deliveray.
    """
    for i in train['serial'].unique():
        for j in train.loc[train.serial == i, 'delivery'].unique():
            index = train.index[(train.serial == i) & (train.delivery == j)]
            train.loc[index, 'mean'] = train.loc[index, 'volume'].mean()
            train.loc[index, 'std'] = train.loc[index, 'volume'].std()
    train['std'].fillna(value=0, inplace=True)
    # For cows in the test set, we try to find the same cow from training set.
    # If the same cow can be found, yet it got new delivery, 
    # we should fill the mean/std from last delivery from itself.
    for i in test['serial'].unique():
        for j in sorted(test.loc[test.serial == i, 'delivery'].unique()):
            same_delivery = train.query(f'serial == {i} & delivery == {j}')
            if same_delivery.shape[0] == 0:
                same_delivery = train.query(f'serial=={i} & delivery=={j-1}')
            index = test.index[(test.serial == i) & (test.delivery == j)]
            test.loc[index, 'mean'] = same_delivery['volume'].mean()
            test.loc[index, 'std'] = same_delivery['volume'].std()
    logging.warning(
        " Number of new cows in test: " +
        f"{test.drop_duplicates(subset='serial')['mean'].isnull().sum()}"
    )
    valid_old_cows = test.dropna().copy()
    test['mean'].fillna(value=train['volume'].mean(), inplace=True)
    test['std'].fillna(value=0, inplace=True)
    return valid_old_cows


if __name__ == '__main__':
    main()
