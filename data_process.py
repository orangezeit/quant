# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import numpy as np
import pandas as pd
import datetime
import warnings


def data_analyze(volume=0, mode='sabr', p=4):

    """
        Import data from csv file
            0: underlying price(1)         1: type - call or put(5)
            2: expiration(6)               3: quotedate(7)
            4: strike(8)                   5, 6: bid, ask(10, 11)
            7: volume(12)                  8: implied volatility at expiry(14)
        SABR: 0, 2, 3, 4, 7, 8
        Heston: 0, 1, 2, 3, 4, 5, 6, 7
    """

    # warnings.filterwarnings('ignore')   # suppress the running time errors

    # initialization
    if mode == 'sabr':
        sigma, alpha, beta, rho, s, r, t = 0.1, 1, 0.5, -0.5, 250, 0.024, 1
        sabr = stochastic.SABR(sigma, alpha, beta, rho, s, r, t)
        cols = [1, 6, 7, 8, 12, 14]
        idx = 2
        durations = np.zeros(252 * 5)
    elif mode == 'heston':
        sigma, kappa, theta, xi, rho = 0.2, 0.2, 0.2, 0.2, 0
        s, r, t = 250, 0.024, 1
        heston = stochastic.Heston(sigma, kappa, theta, xi, rho, s, r, t)
        cols = [1, 5, 6, 7, 8, 10, 11, 12]
        idx = 3

    ds = [''] * (252 * 5)
    parameters = np.zeros((252 * 5, p))

    k = 0

    for year in range(2014, 2019):
        y = str(year)
        df = pd.read_csv('SPY_' + y + '.csv', usecols=cols)   # import each
        df = df[df['volume'] > volume]

        for month in range(1, 13):
            for day in range(1, 32):

                m, d = str(month), str(day)
                dates = ['/'.join([m, d, y]), '/'.join(['0' + m, d, y]),
                         '/'.join([m, '0' + d, y]), '/'.join(['0' + m, '0' + d, y])]

                for date in dates:
                    if (df['quotedate'] == date).any():
                        new_df = df[(df['quotedate'] == date)]
                        break
                else:
                    continue

                if mode == 'sabr':

                    sabr.sigma, sabr.alpha, sabr.beta, sabr.rho = 0.1, 1, 0.5, -0.5

                    duration = 72
                    flag = True

                    while flag:

                        expiry = datetime.datetime.strptime(new_df.iloc[0, 2], '%m/%d/%Y') + datetime.timedelta(days=duration)

                        for e in [expiry.strftime('%m/%d/%Y'), expiry.strftime('%#m/#%d/%Y')]:
                            if (new_df['expiration'] == e).any():
                                new_df = new_df[new_df['expiration'] == e]
                                flag = False
                        else:
                            duration += 1

                    n, s, t, km, sigma_m = new_df.shape[0], new_df.iloc[0, 0], duration / 365,\
                                           new_df['strike'].values, new_df['impliedvol'].values
                    # print(s, t, km, sigma_m)
                    sabr.s = s
                    sabr.t = t
                    # print(km, sigma_m)
                    parameters[k] = sabr.calibrate(n, sigma_m, km)
                    print(dates[0], parameters[k])
                    durations[k] = duration
                    ds[k] = dates[0]
                    k += 1

                elif mode == 'heston':

                    print('a')
                    new_df = new_df[new_df['type'] == 'call']
                    new_df['time'] = (pd.to_datetime(new_df['expiration']) - pd.to_datetime(new_df['quotedate'])).dt.days
                    new_df['payoff'] = (new_df['bid'] + new_df['ask']) / 2

                    n, s, tm, km, cm = new_df.shape[0], new_df.iloc[0, 0], new_df['time'].values / 365, \
                                       new_df['strike'].values, new_df['payoff'].values
                    print(n)
                    print(s)
                    print(tm)
                    print(km)
                    print(cm)

                    heston.s = s

                    parameters[k] = heston.calibrate(n, tm, km, cm)
                    ds[k] = dates[0]
                    k += 1
                    print(dates[0])

    pm_df = pd.DataFrame(parameters)

    pm_df.columns = ['alpha', 'beta', 'rho', 'sigma']
    pm_df['Duration'] = pd.Series(durations)
    pm_df['Date'] = pd.Series(ds)
    pm_df.set_index('Date', inplace=True)

    pm_df.to_csv('parameters3.csv')

data_analyze(volume=100, mode='sabr')