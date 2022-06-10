import logging
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def main():
    report = get_all_features()
    # Remove 64 training data with volume == 0.
    report.drop(report.index[report.volume == 0], inplace=True)
    add_mean_std(report)
    report['diff'] = report['volume'] - report['mean']
    # Remove 'volume' for the validation set.
    train, valid, test = split_data(report, valid_year=2018, valid_month=0)
    yv = valid['diff'].to_numpy()
    valid['diff'] = pd.NA
    report = pd.concat([train, valid])

    report = pd.get_dummies(report, columns=['year', 'month'])
    report[['lactation', 'age']] = StandardScaler().fit_transform(report[['lactation', 'age']])
    add_mean_std(report)

    train, valid = split_data(report, col='diff')

    drop_list = ['serial', 'volume']
    train.drop(drop_list, axis=1, inplace=True)
    valid.drop(drop_list, axis=1, inplace=True)

    x = train.drop('diff', axis=1)
    xv = valid.drop('diff', axis=1)
    y = train['diff'].to_numpy()

    regr = RandomForestRegressor()
    regr.fit(x, y)
    yp = regr.predict(xv)
    print('RMSE =', RMSE(yv, yp))
    print(regr.feature_names_in_)
    print(regr.feature_importances_)
    plt.scatter(yp, yv, label='SVR')
    plt.axis([0, 60, 0, 60])
    plt.xlabel('Predict')
    plt.ylabel('Truth')
    plt.legend()
    plt.show()
    return


def RMSE(yt, yp):
    return np.sum((yt - yp) ** 2) / yp.shape[0]


def get_all_features() -> pd.DataFrame:
    df = pd.read_csv('data/report.csv', skiprows=1,
                names=[
                    'ID', 'year', 'month', 'ranch',
                    'serial', 'father','mother', 'birthday',
                    'delivery', 'lactation', 'volume', 'age', 'breeding'],
                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17],
                dtype={'father': str, 'mother': str},
                parse_dates=['birthday'])
    # Drop some data based on the result of verifing data.
    df.drop(df.index[df.ID == 6960], inplace=True)
    df.drop(df.index[df.ID == 16714], inplace=True)
    df.drop(df.index[df.age < 21], inplace=True)
    # Frequency encoding
    df.father = df.father.map(df.father.value_counts().to_dict())
    df.mother = df.mother.map(df.mother.value_counts().to_dict())
    df.father.fillna(value=df.father.mean(), inplace=True)
    df.mother.fillna(value=df.mother.mean(), inplace=True)
    df['serial_freq'] = df.serial.map(df.serial.value_counts().to_dict())
    # One hot encoding
    df = pd.get_dummies(df, columns=['ranch'])
    # Ascending serial based on birthday.
    unique_serial = df.sort_values(by='birthday').serial.unique()
    serial_map = {old: i for i, old in enumerate(unique_serial)}
    df.serial = df.serial.map(serial_map)
    # Drop birthday and ID now.
    df.drop(columns=['ID', 'birthday'], inplace=True)
    # Use integer.
    df.lactation = df.lactation.astype('int64')
    logging.info(f'Number of unique\n{df.dtypes}')
    return df


def split_data(df: pd.DataFrame, col='volume', valid_year=0, valid_month=0):
    # Get training and test set.
    train = pd.DataFrame.copy(df.loc[df[col].notnull()])
    test  = pd.DataFrame.copy(df.loc[df[col].isnull()])
    if valid_year == 0:
        return train, test
    # A realistic validation set based on number of new cows
    valid = train.query(f'year == {valid_year} & month > {valid_month}').copy()
    train.query(f'year < {valid_year} | month <= {valid_month}', inplace=True)
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
