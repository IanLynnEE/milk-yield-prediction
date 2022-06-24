import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, NuSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from preprocessing import read_report, drop_data, label_encoding
from preprocessing import frequency_encoding
from preprocessing import add_delivery_season
from preprocessing import add_time_different, add_mean_std


def main():
    logging.basicConfig(level=logging.INFO)
    # ==================== Select Features Here ==================== #
    report = read_report()
    drop_data(report)
    label_encoding(report)
    frequency_encoding(report)
    add_delivery_season(report)

    # add_time_different(report)
    # report.dropna(subset='intervalA', inplace=True)

    # add_mean_std(report)
    # report.dropna(subset='std', inplace=True)

    use_cols = [
        'year',
        'month',
        'ranch',
        'serial',
        'father', 'mother',
        # 'firstSemen', 'lastSemen',
        'delivery', 'lactation', 'age',
        'breeding',
        'deliverySeason',
        # 'intervalA', 'intervalB', 'intervalC', 'intervalD', 'intervalE',
        # 'mean', 'std',
        'volume'
    ]
    data = report[use_cols]
    data = pd.get_dummies(data, columns=['ranch'])
    data.info()

    # ==================== Select Scaler Here ====================== #
    features_scaler = MinMaxScaler()

    # ==================== Select Model Here ======================= #
    regr = RandomForestRegressor(
        n_estimators=1000,
        criterion='squared_error',
        max_features='sqrt',
        # random_state=1,
        n_jobs=-1
    )

    test  = report.loc[report['volume'].isnull()]
    train = data.loc[data['volume'].notnull()]
    logging.info(f' Number of test data: {test.shape[0]}')

    rmse = np.empty(10)
    for i in range(len(rmse)):
        rmse[i] = check_model(train, features_scaler, regr, time_valid=False)
    print(f'RMSE = {rmse.mean():2.3f}, max = {rmse.max():2.3f}')

    yp = predict_test(data, features_scaler, regr)
    write_answer(yp)
    ID = test['ID'].to_numpy()
    np.save('ID1.npy', ID)
    np.save('answer1.npy', yp)
    return


def check_model(train: pd.DataFrame, features_scaler, regr, time_valid=True):
    Y = train['volume']
    X = train.drop(columns='volume')
    X[:] = features_scaler.fit_transform(X)

    if time_valid:
        index = train.index[(train.year == 2018) & (train.month >= 7)]
        xv = X.loc[index]
        yv = Y.loc[index]
        x = X.drop(index)
        y = Y.drop(index)
    else:
        x, xv, y, yv = train_test_split(X, Y, test_size=0.12)
    
    logging.info(f' Number of validation data: {xv.shape[0]}')
    
    regr.fit(x, y)
    yp = regr.predict(xv)
    return np.sqrt(mean_squared_error(yv, yp))


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
    np.save('answer.npy', yp)
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
