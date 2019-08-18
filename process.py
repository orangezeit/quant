# Creator: Yunfei Luo
# Date: Apr 29, 2019  10:07 PM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import stochastic
import portfolio

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


def no_arbitrage_test(num=16):

    """
        test the no-arbitrage assumption of the SABR model
        return none, show plots of risk-neutral density
        extracted from the B method
    """

    df = pd.read_csv('parameters_price.csv')
    h = 2

    rn = np.random.choice(df.shape[0], num)

    for n in rn:

        alpha, beta, rho, sigma, _, s = df.iloc[n, 1:7]
        sabr = stochastic.SABR(sigma, alpha, beta, rho, s, 0.024, 0.25)

        sim_st = [sabr.simulate() for _ in range(10000)]
        strikes = np.linspace(s - 3 * sigma, s + 3 * sigma, num=100)
        density = np.zeros(100)

        for i, k in enumerate(strikes):

            sim_test = sim_st.copy()
            sim_test -= (k - h)
            payoff_right = np.mean(sim_test[sim_test > 0])

            sim_test -= h
            payoff_middle = np.mean(sim_test[sim_test > 0])

            sim_test -= h
            payoff_left = np.mean(sim_test[sim_test > 0])

            density[i] = (payoff_left + payoff_right - 2 * payoff_middle) / h ** 2

        plt.plot(strikes, density)
        plt.show()


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


if __name__ == '__main__':

    mark = 'ETF50'
    tks = ['601398.SS', '600028.SS', '600019.SS', '600018.SS', '600050.SS',
           '600519.SS', '000063.SZ', '002024.SZ', '000839.SZ', '600177.SS']

    #mark = 'SPY'
    #tks = ['BA', 'CSCO', 'DHI', 'DIS', 'JNJ', 'JPM', 'KO', 'MSFT', 'NEE', 'XOM']

    stks, bm = import_data(tks, mark)

    rs = stks.pct_change().shift(-1).dropna()
    stks = stks.shift(-1).dropna()
    bm = bm.shift(-1).dropna()

    idxs = (0, 3028)

    portfolio_passive(stks.copy(), bm.copy())
    portfolio_active(rs, stks, bm, idxs, mark, True)