import logging
# logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def main():
    report = read_report()
    drop_data(report)
    label_encoding(report)
    frequency_encoding(report)
    add_time_different(report)
    add_mean_std(report)
    # one hot encoding
    report = pd.get_dummies(report, columns=['ranch', 'year', 'month'])
    # Split data
    train = report.loc[report['volume'].notnull()]
    test  = report.loc[report['volume'].isnull()]
    train.to_csv('data/train.csv')
    test.to_csv('data/test.csv')
    return


def read_report() -> pd.DataFrame:
    cols = [
        'ID', 'year', 'month', 'ranch', 'serial',
        'father','mother', 'birthday', 'delivery', 'lactation',
        'volume', 'deliveryDate', 'samplingDate', 'age', 'recordDate',
        'lastBreedDate', 'lastSemen', 'breeding',
        'lastDeliveryDate','firstBreedDate','firstSemen'
        ]
    report = pd.read_csv(
        './data/report.csv',
        skiprows=1,
        names=cols,
        dtype={'father': str, 'mother': str},
        parse_dates=[
            'birthday', 'deliveryDate', 'samplingDate', 'recordDate',
            'lastBreedDate', 'lastDeliveryDate', 'firstBreedDate'
        ]
    )
    # The column name are given casually.
    # breeding       is not really times of breeding.
    # lastBreedDate  is not really last breeding date before sampling.
    # firstBreedDate is not really first breeding date either.
    # Check them by excel and you'll know what I'm saying..
    return report


def drop_data(df: pd.DataFrame):
    logging.info(' Droping data with ID == 6960.')
    logging.info(' Droping data with ID == 16714.')
    logging.info(' Droping data with age < 21.')
    logging.info(' Droping data with volume == 0.')
    # Drop some data based on the result of verifing data.
    df.drop(df.index[df.ID == 6960], inplace=True)
    df.drop(df.index[df.ID == 16714], inplace=True)
    df.drop(df.index[df.age < 21], inplace=True)
    # Remove 64 training data with volume == 0.
    df.drop(df.index[df.volume == 0], inplace=True)
    # After dropping unknown, we can use integer.
    df.lactation = df.lactation.astype('int64')
    return


def label_encoding(df: pd.DataFrame):
    logging.info(' Label encoding father, mother, ranch.')
    logging.info(' Label encoding serial based on brithday.')
    # Mark unknow
    df.father.fillna(value='unknown', inplace=True)
    df.mother.fillna(value='unknown', inplace=True)
    # Label encoding on father and mother
    df.father = LabelEncoder().fit_transform(df.father)
    df.mother = LabelEncoder().fit_transform(df.mother)
    # Label encoding on ranch
    df.ranch = LabelEncoder().fit_transform(df.ranch)
    # Ascending serial based on birthday.
    unique_serial = df.sort_values(by='birthday').serial.unique()
    serial_map = {old: i for i, old in enumerate(unique_serial)}
    df.serial = df.serial.map(serial_map)
    return


def frequency_encoding(df: pd.DataFrame):
    logging.info(' Frequency encoding father, mother.')
    df.father = df.father.map(df.father.value_counts().to_dict())
    df.mother = df.mother.map(df.mother.value_counts().to_dict())
    # df['serial_freq'] = df.serial.map(df.serial.value_counts().to_dict())
    return


def add_time_different(df: pd.DataFrame):
    logging.info(' Adding more features based on datetime features.')
    logging.warning(' Will generate more NaN for data with missing values.')
    # lastBreedDate is not really last breeding date before sampling.
    # firstBreedDate is not really first breeding date either.
    oneday = np.timedelta64(1, 'D')
    df['intervalA'] = (df.lastBreedDate - df.firstBreedDate) / oneday
    df['intervalB'] = (df.samplingDate - df.lastBreedDate)   / oneday
    df['intervalC'] = (df.samplingDate - df.firstBreedDate)  / oneday
    df['intervalD'] = (df.lastBreedDate - df.deliveryDate)   / oneday
    df['intervalE'] = (df.firstBreedDate - df.deliveryDate)  / oneday
    return


def add_mean_std(df: pd.DataFrame):
    """
    Add mean and std of each cow at each deliveray.
    Kind of like TargetEncoder, but it uses two columns to encode.
    Moreover, it gives std as well.
    For known cows with new delivery, the last delivery data will be used.
    """
    logging.warning(' For validation set, volume needs to be NaN.')
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
