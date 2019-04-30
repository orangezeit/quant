# Creator: Yunfei Luo
# Date: Apr 29, 2019  10:07 PM

import stochastic
import numpy as np
import pandas as pd
import datetime


"""
    Import data from csv file
        0: underlying price(1)         1: type - call or put(5)
        2: expiration(6)               3: quotedate(7)
        4: strike(8)                   5, 6: bid, ask(10, 11)
        7: volume(12)                  8: implied volatility at expiry(14)
"""


def generate_sabr_parameters(volume=100):

    """
        SABR: 0, 2, 3, 4, 7, 8
        Heston: 0, 1, 2, 3, 4, 5, 6, 7
    """

    # initialization

    sabr = stochastic.SABR(0.1, 1, 0.5, -0.5, 250, 0.024, 1)
    durations = np.zeros(252 * 5)
    ds = [''] * (252 * 5)
    parameters = np.zeros((252 * 5, p))
    stock_prices = np.zeros(252 * 5)
    k = 0

    for year in range(2014, 2019):

        # import csv file
        df = pd.read_csv('SPY_{:d}.csv'.format(year), usecols=[1, 6, 7, 8, 12, 14])
        df = df[df['volume'] > volume]

        for month in range(1, 13):
            for day in range(1, 32):

                dates = ['{:d}/{:d}/{:d}'.format(month, day, year), '{:02d}/{:02d}/{:d}'.format(month, day, year)]

                for date in dates:
                    if (df['quotedate'] == date).any():
                        new_df = df[(df['quotedate'] == date)]
                        break
                else:
                    continue

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

                n, sabr.s, sabr.t, km, sigma_m = new_df.shape[0], new_df.iloc[0, 0], duration / 365, \
                                                 new_df['strike'].values, new_df['impliedvol'].values

                parameters[k] = sabr.calibrate(n, sigma_m, km)
                print(dates[0], parameters[k])

                durations[k] = duration
                stock_prices[k] = s
                ds[k] = dates[0]
                k += 1

    pm_df = pd.DataFrame(parameters)
    pm_df.columns = ['alpha', 'beta', 'rho', 'sigma']
    pm_df['Duration'] = pd.Series(durations)
    pm_df['StockPrice'] = pd.Series(stock_prices)
    pm_df['Date'] = pd.Series(ds)
    pm_df.set_index('Date', inplace=True)
    pm_df.to_csv('parameters_price.csv')


def generate_heston_parameters():

    pass


def validate_predicted_parameters():

    beta = 0.5

    df = pd.read_csv('svr.csv', usecols=[0,1,2,3,4,5,6,7,8,9])
    prices = pd.read_csv('parameters_price.csv', usecols=[0, 6]).iloc[1010:]
    spy = pd.read_csv('SPY_2018.csv', usecols=[7, 8, 10, 11])
    spy['price'] = (spy['bid'] + spy['ask']) / 2
    data = np.zeros((150, 4))
    k = 0

    for i in range(1, 4): # 247

        alpha, rho, sigma = df.iloc[i, 1:4]
        alpha2, rho2, sigma2 = df.iloc[i, 4:7]
        alpha3, rho3, sigma3 = df.iloc[i-1, 7:10]

        s = prices.iloc[i, 1]
        spy_copy = spy[spy['quotedate'] == prices.iloc[i, 0]]

        rows = spy_copy.shape[0]

        sabr = stochastic.SABR(sigma, alpha, beta, rho, s, 0.024, 1 / 4)
        sabr2 = stochastic.SABR(sigma2, alpha2, beta, rho2, s, 0.024, 1 / 4)
        sabr3 = stochastic.SABR(sigma3, alpha3, beta, rho3, s, 0.024, 1 / 4)

        sim = np.array([sabr.simulate() for _ in range(1000)])
        sim2 = np.array([sabr2.simulate() for _ in range(1000)])
        sim3 = np.array([sabr3.simulate() for _ in range(1000)])

        for j in range(0, rows, 100):

            sim_copy = sim.copy()
            sim_copy -= spy_copy['strike'].iloc[j]
            sim_copy[sim_copy <= 0] = 0

            sim_copy2 = sim2.copy()
            sim_copy2 -= spy_copy['strike'].iloc[j]
            sim_copy2[sim_copy2 <= 0] = 0

            sim_copy3 = sim3.copy()
            sim_copy3 -= spy_copy['strike'].iloc[j]
            sim_copy3[sim_copy3 <= 0] = 0

            data[k] = np.mean(sim_copy), np.mean(sim_copy2), np.mean(sim_copy3), spy_copy['price'].iloc[j]
            k += 1

    df2 = pd.DataFrame({'svr': data[:, 0], 'varma': data[:, 1], 'last day': data[:, 2], 'real': data[:, 3]})
    df2.to_csv('pred.csv')
