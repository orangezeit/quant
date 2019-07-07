import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import portfolio


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

    df['return'] = df['sum'] - df['sum'][0]
    df['pct'] = (df['sum'] + cash - df['sum'][0]) / cash / (1 + short) - 1

    plt.plot(df['pct'])


def portfolio_markowitz(df, spy, cash=100000000.0):

    rates = df.pct_change().shift(-1).dropna()
    labels = []
    flag = True
    flag2 = False
    s2 = [0.0]
    long = 0.8

    nums_spy_c = cash * long / spy['SPY'][0] // 1
    for r in (-2, -1):
        sl = -0.1
        st = -10000
        for st in (-10, -5, -2, -1, -0.5, -0.2, -0.1, 0):
            if r == -2 and flag2:
                continue
            for short in (0.0, ):

                nums = cash * long / df.shape[1] / np.array(df.iloc[0, :]) // 1
                nums_spy = cash * short / spy['SPY'][0] // 1

                s = [0.0]

                leftover = cash * (1 - long)

                for i in range(21, 3138, 21):
                    transact = 0
                    rates_month = rates[(i-21):i]

                    mus = np.array(rates_month.mean())
                    covs = np.array(rates_month.cov())

                    if r == -2:

                        weights = portfolio.markovitz(covs, mus, r).allocate()
                    else:
                        weights = portfolio.markovitz(covs, mus, r, None, (sl, st)).allocate()

                    temp = (s[-1] + 1) * cash * long * weights / np.array(df.iloc[(i-1), :]) // 1
                    nums -= temp

                    temp2 = (s2[-1] + 1) * cash * short / spy['SPY'][(i - 1)] // 1
                    nums_spy -= temp2

                    for j in range(df.shape[1]):
                        leftover += (nums[j]) * df.iloc[(i-1), j]
                        transact += abs((nums[j]) * df.iloc[(i-1), j]) * 0.0002

                    leftover += nums_spy * spy['SPY'][(i-1)]
                    transact += abs(nums_spy * spy['SPY'][(i-1)]) * 0.0002
                    leftover -= transact
                    if leftover < 0:
                        print(leftover)

                    nums = temp
                    nums_spy = temp2
                    # a = np.vstack((a, msr.T))

                    sum_money = 0
                    for j in range(df.shape[1]):
                        sum_money += nums[j] * df.iloc[(i -1), j]
                    sum_money += leftover
                    sum_money -= nums_spy * spy['SPY'][(i-1)]

                    s.append(sum_money / 100000000 - 1)

                    if flag:
                        s2.append(nums_spy_c * spy['SPY'][(i-1)] / 100000000 - 1)
                flag = False
                flag2 = True
                plt.plot(s)
                if r == -2:
                    labels.append('{}, {}, {}'.format(short, r, 21))
                else:
                    labels.append('{}, {}, {}, {}, {}'.format(short, r, 21, '-inf', st))
                #plt.show()


    plt.plot(s2)
    labels.append('spy')

    plt.legend(labels)
    plt.show()


def bayes_stein(df, spy, cash=100000000.0):



    pass


def sensitivity_analysis(df, c=2):

    rates = df.pct_change().shift(-1).dropna()
    mus = np.array(rates.mean())
    stds = np.array(rates.std()) * np.sqrt(c)

    for i in range(1, 3137):
        for j in range(df.shape[1]):
            df.iloc[i, j] = df.iloc[i-1, j] * (1 + mus[j] + stds[j] * np.random.normal())

    return df



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
    #plt.plot(stocks)
    """
    plt.plot()
    plt.legend(['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM'])
    plt.show()
    """



