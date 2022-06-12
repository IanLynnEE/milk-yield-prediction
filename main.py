import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_squared_error

from preprocessing import read_report, drop_data, label_encoding
from preprocessing import frequency_encoding, add_mean_std
from preprocessing import add_delivery_season
from preprocessing import add_time_different


def main():
    logging.basicConfig(level=logging.INFO)
    # ==================== Select Features Here ==================== #
    report = read_report()
    drop_data(report)
    label_encoding(report)
    # frequency_encoding(report)
    # one hot encoding
    # report = pd.get_dummies(report, columns=['ranch'])
    add_delivery_season(report)
    # report.dropna(subset='intervalA', inplace=True)
    add_time_different(report)
    # add_mean_std(report)
    # report.dropna(subset='std', inplace=True)

    drop_cols = [
        'ID', 
        'birthday', 'deliveryDate','samplingDate', 'recordDate',
        'lastBreedDate', 'lastDeliveryDate', 'firstBreedDate',
        'lastSemen', 'firstSemen',
        'serial',
        'breeding',
        'father', 'mother', 'serial_freq',
        'intervalA',
        'intervalB',
        'intervalC',
        'intervalD',
        'intervalE',
        'deliverySeason'
    ]

    # ==================== Select Scaler Here ====================== #
    features_scaler = MinMaxScaler()

    # ==================== Select Model Here ======================= #
    regr = SVR(C=10)

    test  = report.loc[report['volume'].isnull()]
    train = report.loc[report['volume'].notnull()]
    logging.info(f' Number of test data: {test.shape[0]}')

    rmse = check_model(regr, features_scaler, train, drop_cols, add_mean=False)
    print(f'C = 10, n_support_ = {regr.n_support_}, RMSE = {rmse:1.4f}')

    # add_mean_std(report)
    ID = test['ID'].to_numpy()
    report.drop(columns=drop_cols, inplace=True)
    yp = predict_test(regr, features_scaler, report)

    np.save('ID1.npy', ID)
    np.save('answer1.npy', yp)
    write_answer(yp)
    return


def check_model(regr, scaler, report, drop_cols, *,
                add_mean=False, special_valid=True):
    train = report.copy()
    if add_mean:
        logging.warning(' Using the special validation set.')
        index = train.index[(train.year == 2018) & (train.month < 7)]
        # Backup and clean validation set.
        y = train.drop(index)['volume'].to_numpy()
        yv = train.loc[index, 'volume'].to_numpy()
        train.loc[index, 'volume'] = pd.NA
        # Now mean and std can be added
        add_mean_std(train)

    train.drop(columns=drop_cols, inplace=True)

    Y = train['volume']
    X = train.drop(columns='volume')
    X[:] = scaler.fit_transform(X[:])
    X.info()

    if add_mean:
        xv = X.loc[index].to_numpy()
        x = X.drop(index).to_numpy()
    elif special_valid:
        index = train.index[(train.year == 2018) & (train.month < 7)]
        xv = X.loc[index].to_numpy()
        yv = Y.loc[index].to_numpy()
        x = X.drop(index).to_numpy()
        y = Y.drop(index).to_numpy()
    else:
        x, xv, y, yv = train_test_split(X, Y, test_size=0.12, random_state=0)

    logging.info(f' Number of validation data: {xv.shape[0]}')

    regr.fit(x, y)
    yp = regr.predict(xv)
    return np.sqrt(mean_squared_error(yv, yp))


def predict_test(regr, features_scaler, report):
    # Train feature scaler again.
    X = report.drop(columns='volume')
    X[:] = features_scaler.fit_transform(X[:])

    test  = report.loc[report['volume'].isnull()]
    train = report.loc[report['volume'].notnull()]

    x = train.drop(columns='volume')
    xt = test.drop(columns='volume')
    y = train['volume']

    x_scaled = features_scaler.transform(x)
    xt_scaled = features_scaler.transform(xt)

    regr.fit(x_scaled, y)
    return regr.predict(xt_scaled)


def merge_answer():
    x1 = np.load('ID1.npy')
    x2 = np.load('ID2.npy')
    y1 = np.load('answer1.npy')
    y2 = np.load('answer2.npy')
    ID_list = pd.read_csv('./data/submission.csv')['ID'].to_numpy().tolist()
    yp = np.zeros(len(ID_list))
    for i, ID in enumerate(ID_list):
        idx1 = np.where(x1 == ID)
        idx2 = np.where(x2 == ID)
        if len(idx1[0]) == 0:
            if len(idx2) == 0:
                logging.error(f'Cannot find ID = {ID}')
            else:
                yp[i] = y2[idx2[0][0]]
        else:
            yp[i] = y1[idx1[0][0]]
    return yp


def write_answer(yp: np.ndarray):
    report = read_report()
    test  = report.loc[report['volume'].isnull()]
    answer = test[['ID', 'volume']].copy()
    # Check all IDs are in correct order.
    template = pd.read_csv('./data/submission.csv')['ID'].to_numpy()
    for i, ID in enumerate(answer['ID'].to_numpy()):
        if template[i] != ID:
            logging.error(' Anser ID mismatched: ID =', ID)
    if answer.shape[0] != yp.shape[0]:
        logging.error(' Number of prediction incorrect!')
        logging.error(f' yp.shape[0] = {yp.shape[0]}')
        return
    answer.volume = yp
    answer.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    main()
    # yp = merge_answer()
    # write_answer(yp)
