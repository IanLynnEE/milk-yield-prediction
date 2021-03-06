import logging 

import pandas as pd


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def verify_data():
    """
    1. The age of a cow == (sampling date - birthday).
    2. sampling date <= recording date.
    3. #days of lactation == sampling date - last delivery date
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