import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import portfolio


def import_data(tickers, label='SPY'):

    stocks = pd.concat([pd.read_csv(f'data\{ticker}.csv', header=0, names=[ticker], usecols=[5]) for ticker in tickers],
                       1)
    benchmark = pd.read_csv(f'data\{label}.csv', header=0, names=[label], usecols=[5 if label == 'SPY' else 1])

    # check data integrity
    if np.isnan(stocks.values).any():

        print('Warning: Stock data are missing.')
        stocks = stocks.ffill()

        if np.isnan(stocks.values).any():
            stocks = stocks.bfill()

    if np.isnan(benchmark.values).any():

        print('Warning: Benchmark data are missing.')
        benchmark = benchmark.ffill()

        if np.isnan(benchmark.values).any():
            benchmark = benchmark.bfill()

    return stocks, benchmark


def backtest_stat(ts):

    j = np.argmax(np.maximum.accumulate(ts) - ts)
    i = np.argmax(ts[:j])
    max_retract = (ts[j] - ts[i]) / (1 + ts[i])
    v = np.var(np.diff(np.log(np.array(ts) + 1)))

    # maximum retract, returns, volatility
    print(max_retract, ts[-1], v)


def portfolio_passive(stocks, benchmark, long_ratio=0.8, cash=100000000.0):

    ts_bm = [benchmark.iloc[i - 1, 0] / benchmark.iloc[0, 0] - 1 for i in range(1, benchmark.shape[0])]
    backtest_stat(ts_bm)
    plt.plot(ts_bm)

    nums = cash * long_ratio / stocks.shape[1] / np.array(stocks.iloc[0, :]) // 1
    stocks *= nums
    stocks['sum'] = stocks.sum(1)
    stocks['pct'] = (stocks['sum'] + cash - stocks['sum'][0]) / cash - 1

    backtest_stat(stocks['pct'].values)
    plt.plot(stocks['pct'].values)


def portfolio_active(rates, stocks, benchmark, indices, label='SPY', empty=False, long_ratio=0.8, cash=100000000.0):

    if empty:
        benchmark[f'{label}_mean'] = benchmark[f'{label}'].rolling(21).mean()

    #(-2, None),
    for r, bound in ((-2, (0, np.inf)), (-1, (0, np.inf)), (-3, None), (-4, None), (-5, None)):

        ts = []
        shares = cash * long_ratio / stocks.shape[1] / stocks.iloc[0].values // 1
        leftover = cash - shares @ stocks.iloc[0].values

        for day in range(1, benchmark.shape[0]):  # stocks
            # print(day)
            short_signal = benchmark.iloc[day, 0] < benchmark.iloc[day, 1] if day >= 20 and empty else False

            if short_signal:
                ts.append(ts[-1])
            else:
                prices = stocks.iloc[day - 1].values

                if day % 21:
                    ts.append((shares @ prices + leftover) / cash - 1)
                else:

                    vr, sigma = portfolio.estimate(rates[day - 21:day])

                    if bound:
                        weights = portfolio.Markowitz(sigma, vr, r, bounds=(bound,)).allocate()
                    else:
                        if r == -3:
                            weights = portfolio.RiskParity(rates[day - 21:day]).ivp()
                        elif r == -4:
                            weights = portfolio.RiskParity(rates[day - 21:day]).hrp()
                        elif r == -5:
                            weights = portfolio.RiskParity(rates[day - 21:day]).trp()
                        else:
                            weights = portfolio.Markowitz(sigma, vr, r).allocate()

                    # recalculate total value, reallocate assets and calculate new leftover
                    total = shares @ prices + leftover
                    temp = total * long_ratio * weights / prices // 1
                    leftover = total - temp @ prices - abs(shares - temp) @ prices * 0.0001

                    # test liquidity
                    if leftover < 0:
                        print(leftover)

                    shares = temp
                    ts.append(total / cash - 1)

        plt.plot(ts)
        backtest_stat(ts)

    labels = [label, 'EW', 'GMV + Long Only', 'MSR + Long Only', 'IVP', 'HRP', 'TRP']
    plt.legend(labels)
    plt.title('From Day {:d} to Day {:d} {} Hedging'.format(*indices, 'With' if empty else 'Without'))
    plt.show()


def sensitivity_analysis(df, c=2):

    rates = df.pct_change().shift(-1).dropna()
    mus = np.array(rates.mean())
    stds = np.array(rates.std()) * np.sqrt(c)

    for i in range(1, 3137):
        for j in range(df.shape[1]):
            df.iloc[i, j] = df.iloc[i-1, j] * (1 + mus[j] + stds[j] * np.random.normal())

    return df


def timing_factor(rates, stocks, ticker, d=21, x=0.05):

    rates['momentum_{:s}'.format(ticker)] = (rates[ticker] > 0).rolling(d).sum() / d

    rates['deviation_{:s}'.format(ticker)] = (stocks[ticker] -
                                            stocks[ticker].rolling(d).mean()) / stocks[ticker].rolling(d).std()

    rates['aggregation_{:s}'.format(ticker)] = 0

    for j in range(10):
        k = 0
        for i in range(stocks.shape[0]):

            if stocks.iloc[i, j] / stocks.iloc[k, j] - 1 > x:
                rates.iloc[i, 12] += 1
                k = i
            elif stocks.iloc[i, j] / stocks.iloc[k, j] - 1 < -x:
                rates.iloc[i, 12] -= 1
                k = i

    return rates.iloc[:, 10:12]


if __name__ == '__main__':

    mark = 'ETF50'
    tks = ['601398.SS', '600028.SS', '600019.SS', '600018.SS', '600050.SS',
           '600519.SS', '000063.SZ', '002024.SZ', '000839.SZ', '600177.SS']

    #mark = 'SPY'
    #tks = ['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM']

    stks, bm = import_data(tks, mark)

    rates = stks.pct_change().shift(-1).dropna()
    stocks = stks.shift(-1).dropna()
    bm = bm.shift(-1).dropna()

    print(len(rates))
    print(len(stocks))
    idxs = (0, 3028)

    portfolio_passive(stks.copy(), bm.copy())
    portfolio_active(rates, stks, bm, idxs, mark, True)





