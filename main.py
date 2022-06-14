import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import read_report, drop_data, label_encoding
from preprocessing import frequency_encoding, add_time_different, add_mean_std


def main():
    logging.basicConfig(level=logging.INFO)
    # ==================== Select Features Here ==================== #
    report = read_report()
    drop_data(report)
    label_encoding(report)
    # frequency_encoding(report)
    # one hot encoding
    # report = pd.get_dummies(report, columns=['ranch'])

    drop_cols = [
        'ID',
        'birthday', 'deliveryDate','samplingDate', 'recordDate',
        'lastBreedDate', 'lastDeliveryDate', 'firstBreedDate',
        'lastSemen', 'firstSemen'
    ]
    report.drop(columns=drop_cols, inplace=True)

    # ==================== Select Scaler Here ====================== #
    features_scaler = MinMaxScaler()

    # ==================== Select Model Here ======================= #
    regr = MLPRegressor(
        hidden_layer_sizes=(150,100,50),
        max_iter = 300,
        activation = 'relu',
        solver = 'adam'
    )

    test  = report.loc[report['volume'].isnull()]
    train = report.loc[report['volume'].notnull()]
    logging.info(f' Number of test data: {test.shape[0]}')

    check_model(train, features_scaler, regr)

    yp = predict_test(report, features_scaler, regr)
    write_answer(yp)
    return


def check_model(train: pd.DataFrame, features_scaler, regr):
    Y = train['volume']
    X = train.drop(columns='volume')
    X[:] = features_scaler.fit_transform(X)

    # Special validation set
    # index = train.index[(train.year == 2018) & (train.month >= 7)]
    # xv = X.loc[index].to_numpy()
    # yv = Y.loc[index].to_numpy()
    # x = X.drop(index).to_numpy()
    # y = Y.drop(index).to_numpy()

    # Random validation set
    x, xv, y, yv = train_test_split(X, Y, test_size=0.12, random_state=0)
    
    logging.info(f' Number of validation data: {xv.shape[0]}')
    
    regr.fit(x, y)
    yp = regr.predict(xv)
    print('RMSE =', np.sqrt(mean_squared_error(yv, yp)))
    return


def predict_test(report: pd.DataFrame, features_scaler, regr) -> np.ndarray:
    # Train feature scaler again.
    X = report.drop(columns='volume')
    X[:] = features_scaler.fit_transform(X)

    test  = report.loc[report['volume'].isnull()]
    train = report.loc[report['volume'].notnull()]

    x = train.drop(columns='volume')
    xt = test.drop(columns='volume')
    y = train['volume']

    x_scaled = features_scaler.transform(x)
    xt_scaled = features_scaler.transform(xt)

    regr.fit(x_scaled, y)
    return regr.predict(xt_scaled)


def write_answer(yp: np.ndarray):
    report = read_report()
    test  = report.loc[report['volume'].isnull()]
    answer = test[['ID', 'volume']].copy()
    # Check all IDs are in correct order.
    template = pd.read_csv('./data/submission.csv')['ID'].to_numpy()
    for i, ID in enumerate(answer['ID'].to_numpy()):
        if template[i] != ID:
            print('ERROR! Anser for ID =', ID)
    if answer.shape[0] != yp.shape[0]:
        print('ERROR! Number of prediction incorrect!')
        print('yp.shape[0]:', yp.shape[0])
        return
    answer.volume = yp
    answer.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    main()
