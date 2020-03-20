import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.fftpack import fft


def svcj_simulation(parameters, n=100000):

    mu, mu_y, sigma_y, lbd, alpha, beta, rho, sigma_v, rho_j, mu_v = parameters
    returns = np.zeros(n)

    j = np.random.binomial(1, lbd, n)
    ey, et = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n).T

    zt = np.random.exponential(1 / mu_v, n)
    zy = np.random.normal(mu_y + rho_j * zt, sigma_y)

    for i in range(n):

        returns[i] = mu + np.sqrt(sigma_y) * ey[i] + zy[i] * j[i]
        sigma_y = max(0, alpha + beta * sigma_y + sigma_v * np.sqrt(sigma_y) * et[i] + zt[i] * j[i])

    return returns


def option_pricing(s0, p, strikes, durations=(1, 7, 30, 60, 90, 180, 360, 720)):

    n = 100000

    data_set = np.zeros((len(strikes), len(durations)))

    for i, d in enumerate(durations):

        st = np.array([s0 * np.exp(sum(cxiv_simulation(p, d)[0]) / 252) for _ in range(n)])

        for j, k in enumerate(strikes):

            st_copy = st.copy()
            st_copy -= k
            st_copy[st_copy < 0] = 0
            data_set[j, i] = np.mean(st_copy)

    return data_set


def moments(lr):

    return lr.mean().values[0], lr.var().values[0], lr.skew().values[0], lr.kurt().values[0]


def pdf(x, a, b, d, m):

    return (2 * np.cos(b / 2)) ** (2 * d) / (2 * a * np.pi * gamma(2 * d)) * np.exp(b * (x - m) / a) * abs(
            gamma(d + (x - m) / a * 1j)) ** 2


def mle(lr):

    def obj(p):
        return -np.log([pdf(i, *p) for i in lr.values]).sum()

    x0 = np.array([0.05492887,  1.33707347,  0.48707681, -0.01988082])
    b = np.array([(0.0001, None), (-np.pi, np.pi), (0.0001, None), (None, None)])
    return minimize(obj, x0, method='L-BFGS-B', bounds=b).x


def pdf_plot(lr, pm, tick):

    p_mle = mle(lr)
    plt.hist(lr.values, 100, density=True)
    rates = np.linspace(-0.3, 0.3, 10000)
    plt.plot(rates, [pdf(r, *p_mle) for r in rates])
    plt.plot(rates, [pdf(r, *pm) for r in rates])
    plt.title(f'PDF of Log-Returns of {tick}')
    plt.legend(['Maximum Likelihood Estimation', 'Moment Estimation'])
    plt.show()


def m_fft(lr, t, s, n=65536, dk=0.005, r=0.024, alpha=2.0):

    a, b, d, m = mle(lr)
    m = r - 2 * d * np.log(np.cos(0.5 * b) / np.cos(0.5 * (a + b)))

    def psi(x):

        x = x - (alpha + 1) * 1j
        phi = (np.cos(0.5 * b) / np.cosh(0.5 * (a * x - b * 1j))) ** (2 * d * t) * np.exp(m * t * x * 1j)
        x = x + (alpha + 1) * 1j

        return np.exp(-r * t) * phi / (alpha + x * 1j) / (alpha + x * 1j + 1) * np.exp(1j * (x - (alpha + 1) * 1j)) * np.log(s)

    dv = 2 * np.pi / (n * dk)
    v = np.array([i * dv for i in range(n)])
    w = np.array([dv / 2 if i == 0 else dv for i in range(n)])
    beta = np.log(s) - dk * n / 2
    x = np.exp(-1j * v * beta) * psi(v) * w
    y = fft(x)
    k = np.array([beta + i * dv for i in range(n)])
    price = np.exp(-alpha * k) * y.real / np.pi

    return price, np.exp(k)


def payoff(k, combo):

    payoffs, strikes = combo

    left = 0
    right = len(strikes) - 1

    while right - left > 1:

        mid = (left + right) // 2

        if strikes[mid] <= k:
            left = mid
        else:
            right = mid

    slope = (payoffs[right] - payoffs[left]) / (strikes[right] - strikes[left])

    return payoffs[left] + slope * (k - strikes[left])


if __name__ == '__main__':

    p11 = [0.0368, -0.1941, 2.4057, 0.0454, 0.0099, -0.1220, 0.3327, 0.0107, -0.5918, 0.6433]  # 14-19
    strikes1 = range(12000, 14001, 100)

    p22 = [0.0298, -0.5911, 1.9912, 0.0336, 0.0095, -0.0973, 0.1026, 0.0106, -0.1818, 0.7514]  # 14-19
    strikes2 = range(4000, 6001, 100)

    df1 = pd.read_csv('BTC-USD.csv', usecols=[0, 5], index_col=0)
    df2 = pd.read_csv('crix.csv', index_col=0)

    lr1 = np.log(df1).diff().dropna()  # btc
    lr2 = np.log(df2).diff().dropna()  # crix

    p_moments_btc = [0.10314447262338783, -0.2214755731812241, 0.28257906198879670, -0.0019990791448741]
    p_moments_crix = [0.08723110162240453, 0.585616465664008, 0.35822484818297800, -0.007925857740947]

    # pdf_plot(lr2, p_moments_crix, 'CRIX')

    cb = m_fft(lr1, 30/252, 5161.100098)
    print(cb)
    p = payoff(4100.67, cb)
    print(p)
