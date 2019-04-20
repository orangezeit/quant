# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import pandas as pd


def data_import(date, option='call'):

    df = pd.read_csv('SPY_20190301_to_20190329.csv', usecols=[5, 7, 8, 9, 11, 12]) # 5996
    df['price'] = (df['bid'] + df['ask']) / 2
    df = df[df['quotedate'] == date]
    df = df[df['type'] == option]
    # print(df)
    return df.shape[0], np.array(df['time']).reshape(-1), \
           np.array(df['strike']).reshape(-1), np.array(df['price']).reshape(-1)
