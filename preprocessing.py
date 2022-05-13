import logging 
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score 



def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def verify_features():
    report = pd.read_csv('data/report.csv', skiprows=1, header=None, 
                parse_dates=[7, 12, 14])
    birth  = pd.read_csv('data/birth.csv',  skiprows=1, header=None)
    spec   = pd.read_csv('data/spec.csv',   skiprows=1, header=None)
    spec_died = spec.loc[spec[4].isin(['死亡', '木乃伊', '淘汰'])]
    for i, row in report.iterrows():
        # A. col 14 == col 13 - col 8.
        if diff_month(row[12], row[7]) != row[13]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 8, 13, 14')
        # B. col 13 <= col 15.
        if row[12] > row[14]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 13, 15')
        # C. col 2&3 == col 13.
        if row[12].year != row[1] or row[12].month != row[2]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 2, 3, 13')
        # D. With col 12, col 9 should agree with col 9 in birth.csv.
        oracle = birth.loc[(birth[0] == row[4]) & (birth[1] == row[11])]
        if oracle.shape[0] != 1:        # One and only one.
            logging.debug(f'Not found ID = {row[0]:<6d} in birth.csv')
        elif row[8] != oracle.iloc[0][8]:
            logging.warning(f'Conflict: ID = {row[0]:<6d}: col 9')
    # E. A cow should have exactly one birthday.
    serial = report.drop_duplicates(subset=4)[4].tolist()
    for i in serial:
        oracle = report.loc[report[4] == i].drop_duplicates(subset=7)
        if oracle.shape[0] != 1:
            logging.warning(f'Multi-birthday: Serial = {i}')
    # F. The died cows' serial numbers are re-used.
    for i, row in spec_died.iterrows():
        oracle = report.loc[report[4] == row[0]]
        if oracle.shape[0] == 0:
            logging.debug(f'Not found: Serial = {row[0]} in report.csv')
        else:
            logging.warning(f'Died: Serial = {row[0]}')
    return


# Cannot get mother's ID (col 6) from birth.csv.
# As no corresponding calf serial in birth.csv can be found. 
# Consequently, cannot get father's ID (col 7) from breed.csv.
def get_all_features():
    report = pd.read_csv('data/report.csv', skiprows=1,
                names=['year', 'month', 'ranch', 'serial', 'labours', 
                       'lactation', 'yield', 'last_labour', 'sample', 'age'],
                usecols=[1, 2, 3, 4, 8, 9, 10, 11, 12, 13],
                parse_dates=['last_labour', 'sample'])
    # A. Remove ID=6960: Conflict sample date; ID=16713: missing features.
    report = report.drop([6960, 16713])
    # B. Use int64 if possible.
    report['ranch'] = LabelEncoder().fit_transform(report.ranch)
    report['labours'] = report['labours'].astype('int64')
    report['lactation'] = report['lactation'].astype('int64')
    # C. Calculate number of days from parturition.
    report['last_labour'] = (report['sample'] - report['last_labour']).dt.days
    report = report.drop('sample', axis=1)
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


def play_plot(train):
    g = sns.JointGrid(data=train, x='labours', y='yield', space=0)
    g.plot_joint(sns.histplot, discrete=(True, False), cmap='crest')
    g.plot_marginals(sns.histplot, discrete=(True, False))
    plt.savefig('images/labours.png')
    plt.clf()
    g = sns.JointGrid(data=train, x='age', y='yield', space=0)
    g.plot_joint(sns.histplot, discrete=(True, False), cmap='crest')
    g.plot_marginals(sns.histplot, discrete=(True, False))
    plt.savefig('images/age.png')
    plt.clf()
    g = sns.JointGrid(data=train, x='lactation', y='yield', 
            space=0, xlim=(0, 600))
    g.plot_joint(sns.histplot, cmap='crest')
    g.plot_marginals(sns.histplot)
    plt.savefig('images/lactation.png')
    plt.clf()
    sns.violinplot(data=train, x='month', y='yield')
    plt.savefig('images/month.png')
    plt.clf()
    g = sns.JointGrid(data=train, x='ranch', y='yield', space=0)
    g.plot_joint(sns.violinplot)
    g.plot_marginals(sns.histplot, discrete=(True, False))
    plt.savefig('images/ranch.png')
    return


def play_SVR(train):
    X = train.drop(['yield', 'serial'], axis=1).to_numpy()
    Y = train['yield'].to_numpy()
    x, xt, y, yt = train_test_split(X, Y, train_size=0.8)
    regr = make_pipeline(StandardScaler(), SVR())
    regr.fit(x, y)
    yp = regr.predict(xt)
    print(mean_squared_error(yt, yp))
    return


if __name__ == '__main__':
    verify_features() 
    train, test = get_all_features() 
    play_plot(train)
