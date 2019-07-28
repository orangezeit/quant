
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cxiv_simulation(parameters, n=100000):

    mu, mu_y, sigma_y, lbd, alpha, beta, rho, sigma_v, rho_j, mu_v = parameters
    returns, volatility, jump_returns, jump_volatility = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    j = np.random.binomial(1, lbd, n)
    ey, et = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n).T

    zt = np.random.exponential(1 / mu_v, n)
    mu_zt = np.mean(zt)
    zy = np.random.normal(mu_y + rho_j * mu_zt, sigma_y, n)

    for i in range(n):

        sigma_y = max(0, sigma_y)
        volatility[i] = sigma_y
        jump_returns[i] = zy[i] * j[i]
        returns[i] = mu + np.sqrt(sigma_y) * ey[i] + jump_returns[i]
        jump_volatility[i] = zt[i] * j[i]
        sigma_y = alpha + beta * sigma_y + sigma_v * np.sqrt(sigma_y) * et[i] + jump_volatility[i]

    return returns, volatility, jump_returns, jump_volatility


p1 = [0.0334, -0.1484, 2.7169, 0.0426, 0.0117, -0.1694, 0.2645, 0.0103, -0.2519, 0.7482]    # crix 14-17
p11 = [0.0368, -0.1941, 2.4057, 0.0454, 0.0099, -0.1220, 0.3327, 0.0107, -0.5918, 0.6433] # 14-19
strikes1 = range(12000, 14001, 100)

p2 = [0.0279, 0.1083, 1.7987, 0.0529, 0.0094, -0.1655, 0.1241, 0.0068, -0.8082, 0.6765]     # bitcoin 14-17
p22 = [0.0298, -0.5911, 1.9912, 0.0336, 0.0095, -0.0973, 0.1026, 0.0106, -0.1818, 0.7514] # 14-19
strikes2 = range(4000, 6001, 100)


def plots():

    for p in (p1, p2):

        r, v, jr, jv = cxiv_simulation(p)

        plt.plot(r)
        plt.title('plot of log-returns')
        plt.show()

        plt.plot(v)
        plt.title('plot of volatilities')
        plt.show()

        plt.plot(jr)
        plt.title('plot of jumps in log-returns')
        plt.show()

        plt.plot(jv)
        plt.title('plot of jumps in volatilities')
        plt.show()


def price_sim(s, p, days=30, n=10000):

    st = np.zeros(n)

    for i in range(n):
        log_rets = cxiv_simulation(p, days)[0]
        st[i] = s * np.exp(sum(log_rets) / 252)

    return st


def option_pricing(s0, p, strikes, durations=(1, 7, 30, 60, 90, 180, 360, 720)):

    data_set = np.zeros((len(strikes), len(durations)))

    for i, d in enumerate(durations):
        st = price_sim(s0, p, d)
        for j, k in enumerate(strikes):
            # print(j, k)
            st_copy = st.copy()
            st_copy -= k
            st_copy[st_copy < 0] = 0
            data_set[j, i] = np.mean(st_copy)

    return data_set

ds2 = pd.DataFrame(option_pricing(5161.100098, p2, strikes2))
ds2.to_csv('tep.csv')