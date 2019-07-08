import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import portfolio
import statsmodels.api as sm


def import_data(ticks):

    stocks = pd.concat([pd.read_csv(tick + '.csv', header=0, names=[tick], usecols=[5], nrows=3137) for tick in ticks], 1)
    spy = pd.read_csv('SPY.csv', header=0, names=['SPY'], usecols=[5])

    return stocks, spy


def portfolio_ew(df, spy, long=1.0, short=0.0, cash=10000000.0):

    nums = cash * long / df.shape[1] / np.array(df.iloc[0, :]) // 1

    for i in range(df.shape[1]):
        df.iloc[:, i] *= nums[i]

    df['sum'] = df.sum(axis=1)
    spy['SPY'] *= -cash * short / spy['SPY'][0] // 1
    df['sum'] = df['sum'] + spy['SPY']

    df['pct'] = (df['sum'] + cash - df['sum'][0]) / cash / (1 + short) - 1

    plt.plot(df['pct'])


def portfolio_markowitz(stocks, spy, long=0.8, cash=100000000.0):

    rates = stocks.pct_change().shift(-1).dropna()
    labels = ['SPY']

    # SPY plot
    ts_spy = [0.0 if i == 0 else spy['SPY'][i - 1] / spy['SPY'][0] - 1 for i in range(0, stocks.shape[0], 21)]
    plt.plot(ts_spy)

    for r, bounds in ((-2, None), (-1, (-0.1, -1))):
        for short in (0.0, ):

            shares = cash * long / stocks.shape[1] / stocks.iloc[0, :] // 1
            share_spy = cash * short / spy['SPY'][0] // 1

            ts = [0.0]
            leftover = cash - shares @ stocks.iloc[0, :]

            for i in range(21, stocks.shape[0], 21):

                vr = np.array(rates[(i - 21):i].mean())
                sigma = np.array(rates[(i - 21):i].cov())

                weights = portfolio.markovitz(sigma, vr, r, None, bounds).allocate()

                total = shares @ stocks.iloc[(i - 1), :] + leftover                        # total value

                temp = total * long * weights / stocks.iloc[(i - 1), :] // 1             # allocation
                temp2 = total * short / spy['SPY'][i - 1] // 1
                leftover = total - temp @ stocks.iloc[(i - 1), :]                        # new remain

                leftover -= abs(shares - temp) @ stocks.iloc[(i-1), :] * 0.0001
                leftover -= abs(share_spy - temp2) * spy['SPY'][i - 1] * 0.0001

                # test liquidity
                if leftover < 0:
                    print(leftover)

                shares = temp
                share_spy = temp2

                ts.append(total / cash / (1 + short) - 1)

            plt.plot(ts)

            if r == -2:
                labels.append('{}, {}, {}'.format(short, r, 21))
            else:
                labels.append('{}, {}, {}, {}, {}'.format(short, r, 21, -0.1, -1))

    plt.legend(labels)
    plt.show()


def bayes_stein(df, spy, cash=100000000.0):

    rates = df.pct_change().shift(-1).dropna()
    labels = ['SPY']
    ts = np.empty(10)


    for i in range(21, 3138, 21):
        mus = np.array(rates[(i - 21):i].mean())
        ts = np.vstack((ts, mus))
    print(ts[:,0])
    """
    test = arima.ARMA(ts[:, 0], (1, 1))
    mod = test.fit()
    print(mod.summary())
    """
    arma = sm.tsa.ARMA(ts[:,0], (1,1)).fit(disp=False)
    res = arma.resid
    sm.graphics.tsa.plot_acf(ts[:,0], lags=20)
    plt.show()


def sensitivity_analysis(df, c=2):

    rates = df.pct_change().shift(-1).dropna()
    mus = np.array(rates.mean())
    stds = np.array(rates.std()) * np.sqrt(c)

    for i in range(1, 3137):
        for j in range(df.shape[1]):
            df.iloc[i, j] = df.iloc[i-1, j] * (1 + mus[j] + stds[j] * np.random.normal())

    return df


def timing_factor(df, spy):

    rates = df.pct_change().shift(-1).dropna()
    z = 1

    # momentum
    rates['long_signal'] = (rates['BA'] > 0).rolling(3).sum()
    rates['short_signal'] = (rates['BA'] < 0).rolling(3).sum()

    # break through
    df['long_break'] = (df['BA'] - df['BA'].rolling(3).mean() - z * df['BA'].rolling(3).std()) > 0
    df['short_break'] = (df['BA'] - df['BA'].rolling(3).mean() + z * df['BA'].rolling(3).std()) < 0

    # aggregate
    k = 0
    x = 0.05
    df['test'] = 0

    for i in range(df.shape[0]):

        if df.iloc[i, 0] / df.iloc[k, 0] - 1 > x:
            df.iloc[i, 12] = 1
            k = i
        elif df.iloc[i, 0] / df.iloc[k, 0] - 1 < -x:
            df.iloc[i, 12] = -1
            k = i

    # variance adjustment

    df['std'] = 1 / df['BA'].rolling(3).std()
    plt.plot(df['std'])
    plt.show()

    print(df)
    # print(df['test'].cumsum())    # short much more than long


if __name__ == '__main__':

    stocks, spy = import_data(['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM'])
    """
    stocks = sensitivity_analysis(stocks, 5)
    spy = sensitivity_analysis(spy, 5)

    for long, short in ((0.8, 0.0), (1.0, 0.0), (1.0, 0.3), (1.0, 0.15)):
        portfolio_ew(stocks.copy(), spy.copy(), long, short)

    plt.legend(['0.8 0.0', '1.0 0.0', '1.0 0.3', '1.0 0.15'])
    plt.show()
    """

    portfolio_markowitz(stocks, spy)
    #bayes_stein(stocks, spy)
    #plt.plot(stocks)
    #timing_factor(stocks, spy)
    """
    plt.plot()
    plt.legend(['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM'])
    plt.show()
    """



