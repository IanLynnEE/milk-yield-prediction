import logging 
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVR

def main():
    report = get_all_features()
    train, valid, test = split_data(report, valid_year=2018, valid_month=10)

    yv = valid['volume'].to_numpy()
    valid['volume'] = pd.NA
    report = pd.concat([train, valid])
    add_mean_std(report)

    train, _, valid = split_data(report, valid_year=2018, valid_month=10)

    drop_list = ['serial', 'father', 'mother', 'year', 'month']
    train.drop(drop_list, axis=1, inplace=True)
    valid.drop(drop_list, axis=1, inplace=True)

    x = train.drop('volume', axis=1).to_numpy()
    xv = valid.drop('volume', axis=1).to_numpy()
    y = train['volume'].to_numpy()

    regr = SVR()
    regr.fit(x, y)
    yp = regr.predict(xv)
    print('RMSE =', RMSE(yv, yp))
    print('RMSE by mean =', RMSE(yv, valid['mean']))
    plt.scatter(yp, yv, label='SVR')
    plt.scatter(valid['mean'], yv, label='Mean')
    plt.axis([0, 60, 0, 60])
    plt.xlabel('Predict')
    plt.ylabel('Truth')
    plt.legend()
    plt.show()
    return


def RMSE(yt, yp):
    return np.sum((yt - yp) ** 2) / yp.shape[0]


def get_all_features() -> pd.DataFrame:
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
    report = pd.get_dummies(report, columns=['ranch'])
    # C. Encode serial based on birthday.
    unique_serial = report.sort_values(by='birthday').serial.unique()
    new_serial_map = {old: i for i, old in enumerate(unique_serial)}
    report.serial = report.serial.map(new_serial_map)
    # Don't need birthday and ID now.
    report.drop(columns=['ID', 'birthday'], inplace=True)
    # After dropping data with NaN, use integer.
    report.lactation = report.lactation.astype('int64')
    return report


def split_data(df: pd.DataFrame, valid_year: int, valid_month: int) -> tuple:
    # Get training and test set.
    train = pd.DataFrame.copy(df.loc[df['volume'].notnull()])
    test  = pd.DataFrame.copy(df.loc[df['volume'].isnull()])
    # Remove 64 data with volume == 0.
    train.drop(train.index[train.volume == 0], inplace=True)
    # A realistic validation set based on number of new cows
    valid = train.query(f'year == {valid_year} & month > {valid_month}').copy()
    train.query(f'year < {valid_year} | month <= {valid_month}', inplace=True)
    logging.info(f' Number of unique in train:\n{train.nunique()}')
    return train, valid, test


def add_mean_std(df: pd.DataFrame):
    """
    Add mean and std of each cow at each deliveray.
    Kind of like TargetEncoder, but it uses two columns to encode.
    Moreover, it gives std as well.
    For known cows with new delivery, the last delivery data will be used.
    """
    for i in df['serial'].unique():
        for j in sorted(df.loc[df.serial == i, 'delivery'].unique()):
            index = df.index[(df.serial == i) & (df.delivery == j)]
            ref = df.loc[index]
            if ref['volume'].isnull().all():
                ref = df.query(f'serial == {i} & delivery == {j-1}')
            df.loc[index, 'mean'] = ref['volume'].mean()
            df.loc[index, 'std'] = ref['volume'].std()
    # TODO What should we fill for new cows?
    df['mean'].fillna(value=df['volume'].mean(), inplace=True)
    df['std'].fillna(value=df['volume'].std(), inplace=True)
    return


if __name__ == '__main__':
    main()
