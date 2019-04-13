import numpy as np
import pandas as pd
from scipy.stats import norm

class stochastic:

    def __init__(self, s, r, sigma, t, q=0.0):

        """
            The general stochastic process can be written as

            dx = mu(x,t) * dt + v(x,t) * dW

            where
                x is the object of the process
                mu is the drift
                v is the volatility

            s: the (initial) price of underlying assets
            r: the (initial) risk-free interest rate
            sigma: the volatility
            t: the expiry
            q: the dividend
        """

        self.s = s
        self.r = r-q
        self.sigma = sigma
        self.t = t
        self.q = q


class ornstein_uhlenbeck(stochastic):

    """ Vasicek process (interest rate model) """

    def __init__(self, r, sigma, t, kappa, theta, q=0.0):

        stochastic.__init__(self, 0, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate_1d(self, z, n=1000, type='num', obj='r', xi=0.0):

        """
            Usually the OU process is an interest rate model
            it could be used to simulate the volatility in 2d model, in which we need
            xi as the volatility of volatility (skewness)

            n: the number of randoms
            z: sample from the standard normal distribution (possibly correlated with other distribution)
            type: whether the output is a final value or a path
            obj: the object of the process, 'r' in default, could change to 'sigma'
            xi: required if obj is 'sigma'
        """

        dt = self.t / n

        if obj == 'r':
            x = self.r
            sigma = self.sigma
        else:
            x = self.sigma
            sigma = xi

        if type == 'num':
            for i in range(n):
                x += self.kappa * (self.theta - x) * dt + sigma * np.sqrt(x) np.sqrt(dt) * z[i]
            return x
        else:
            record = np.zeros(n+1)
            record[0] = x
            for i in range(1, n+1):
                record[i] += self.kappa * (self.theta - record[i-1]) * dt + sigma * np.sqrt(dt) * z[i]
            return record



class cox_intergell_ross(stochastic):

    """ interest rate model """

    def __init__(self, r, sigma, t, kappa, theta, q=0.0):

        stochastic.__init__(self, 0, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate_1d(self, z, n=1000, type='num', obj='r', xi=0.0, method='cutoff'):

        """
            Usually the OU process is an interest rate model
            it could be used to simulate the volatility in 2d model, in which we need
            xi as the volatility of volatility (skewness)

            n: the number of randoms
            z: sample from the standard normal distribution (possibly correlated with other distribution)
            type: whether the output is a final value or a path
            obj: the object of the process, 'r' in default, could change to 'sigma'
            xi: required if obj is 'sigma'
            method: decision to make if simulated volatility is less than 0
        """

        dt = self.t / n

        if obj == 'r':
            x = self.r
            sigma = self.sigma
        else:
            x = self.sigma
            sigma = xi

        if type == 'num':
            for i in range(n):
                x += self.kappa * (self.theta - x) * dt + sigma * np.sqrt(max(x,0)) * np.sqrt(dt) * z[i]
            return x
        else:
            record = np.zeros(n+1)
            record[0] = x
            for i in range(1, n+1):
                record[i] += self.kappa * (self.theta - record[i-1]) * dt + sigma * np.sqrt(max(x,0)) * np.sqrt(dt) * z[i]
            return record


class cev(stochastic):

    def __init__(self, s, r, sigma, t, beta, k, q=0.0):

        """ model for the price of the underlying asset """

        stochastic.__init__(self, s, r, sigma, t, q)
        self.beta = beta
        self.k = k

    def simulation_1d(self, z, n=1000, type='num'):

        dt = self.t / n
        x = self.s

        if type == 'num':
            for i in range(n):
                x += self.r * x * dt + sigma * self.sigma * np.sqrt(dt) * z[i] * x ** beta
            return x
        else:
            record = np.zeros(n+1)
            record[0] = x
            for i in range(1, n+1):
                record[i] += self.r * record[i-1] * dt + sigma * self.sigma * np.sqrt(dt) * z[i] * record[i-1] ** beta
            return record


class black_scholes(cev):

    def __init__(self, s, r, sigma, t, k, q=0.0):

        cev.__init__(self, s, r, sigma, t, 1, k, q)

    def euro_option(self, option='call', output='value'):    # black scholes formula

        d1 = (np.log(self.s / self.k) + (self.r + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)

        if option == 'call':
            if output == 'value':
                return norm.cdf(d1) * self.s * np.exp(-self.q * self.t) - norm.cdf(d2) * self.k * np.exp(-self.r * self.t)
        else:  # put
            return 1


class heston(cox_intergell_ross):

    def __init__(self, sigma, kappa, theta, xi, rho, s, r, t, alpha=2, q=0.0):

        """
            s: the initial underlying asset price              known
            r: the risk-free interest rate                     known - The Treasury Bill rate
            t: the expiry                                      known

            sigma(nu0): the market volatility                  unknown - iv:
            kappa:
            theta:
            xi:
            rho: the correlation between two motions

            alpha: the damping factor
            q: the dividend rate
        """

        stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta
        self.xi = xi # sigma
        self.rho = rho
        self.alpha = alpha   # damping factor

    def psi(self, x):

        """ the characteristic function """

        x -= (self.alpha + 1) * 1j

        a = x ** 2 + 1j * x
        b = self.kappa - 1j * self.rho * self.xi * x
        lbd = np.sqrt(self.xi ** 2 * a + b ** 2)
        c = np.sinh(lbd * self.t / 2)
        d = np.cosh(lbd * self.t / 2)

        phi = np.exp(1j * x * (np.log(self.s) + self.r * self.t) + self.kappa * self.theta * self.t * b / self.xi ** 2
                     - a * self.sigma / (lbd * d / c + b)) / (d + b * c / lbd) ** (2 * self.kappa * self.theta / self.xi ** 2)

        x += (self.alpha + 1) * 1j

        return np.exp(-self.r * self.t) * phi / (self.alpha + x * 1j) / (self.alpha + x * 1j + 1)

    def fourier(self, n=65536, dv=0.01):

        """ FFT """

        weights = [dv / 2 if i == 0 else dv for i in range(n)]
        x = [np.exp(-1j * i * (dv * np.log(self.s) - np.pi)) * self.psi(i * dv) * weights[i] for i in range(n)]

        y = np.fft.fft(x)

        k = [np.log(self.s) + 2 * np.pi / (n * dv) * (i - n / 2) for i in range(n)]
        payoff = [np.exp(-self.alpha * k[i]) / np.pi * y[i].real for i in range(n)]

        return payoff, np.exp(k)

    def optimize(self, n, s_m, k_m, payoff_m, t_m, method='equal_weight'):

        """
            n: int / the number of
            k_m: np.array / the market strikes
            payoff_m: np.array / the market prices for options
            t_m: np.array / the time / increasing order
        """

        if method == 'equal_weights':
            weights = [1/n] * n

        def obj(x):

            s = 0

            """ x = [ s, k, payoff, t, sigma, nu0, kappa, rho, theta ] """
            n = 65536

            p1, k1 = Heston(s0, 0.13, x[0], r, q, x[1], x[2], x[3], x[4], alpha).fourier()
            p2, k2 = Heston(s0, 0.38, x[0], r, q, x[1], x[2], x[3], x[4], alpha).fourier()
            p3, k3 = Heston(s0, 0.56, x[0], r, q, x[1], x[2], x[3], x[4], alpha).fourier()

            for i in range(len(prices)):

                if i < 9:  # t = 0.13
                    errs = [abs(strikes[i] - k1[j]) for j in range(n)]
                    s += weights[i] * (p1[errs.index(min(errs))] - prices[i]) ** 2
                elif i < 25:  # t = 0.38
                    errs = [abs(strikes[i] - k2[j]) for j in range(n)]
                    s += weights[i] * (p2[errs.index(min(errs))] - prices[i]) ** 2
                else:  # t = 0.59
                    errs = [abs(strikes[i] - k3[j]) for j in range(n)]
                    s += weights[i] * (p3[errs.index(min(errs))] - prices[i]) ** 2

            return s


class sabr(cev):

    def __init__(self):

        cev.__init__(self, s, r, sigma, t, beta, k, q)
        self.alpha = alpha
        self.rho = rho

    def calibration(self, n, sigma_m, s_m, k_m, t, method='equal_weight'):

        """
            calibrate for every expiry
            sigma_m: the market volatility
            k_m: the market strike
            method: equal weights or weighted

            Note that self.sigma is the initial volatility (how to determine)
        """

        def obj(x):

            self.alpha, self.beta, self.rho = x
            s = 0

            for i in range(n):
                p = (s_m[i] * k_m[i]) ** ((1 - self.beta) / 2)
                q = np.log(s_m[i] / k_m[i])
                z = self.alpha * p * q / self.sigma

                x = np.log((np.sqrt(1 - 2 * self.rho * z + z * z) + z - self.rho) / (1 - self.rho))
                a = 1 + ((1 - self.beta) ** 2 * self.sigma ** 2 / (
                            24 * p ** 2) + self.rho * self.alpha * self.beta * self.sigma
                         / (4 * p) + self.alpha ** 2 * (2 - 3 * self.rho ** 2) / 24) * t
                b = 1 + (1 - self.beta) ** 2 * q ** 2 / 24 + (1 - self.beta) ** 4 * q ** 4 / 1920

                var_k = self.alpha * q / x * a / b

                s += (var_k - sigma_m[i]) ** 2

            return s

        x1 = self.alpha, self.beta, self.rho

        sol = minimize(obj, x1, method='SLSQP')

        print(sol.x)






sigma = 2.3467724
nu0 = 0.05211403
kappa = 3.1926385
rho = -0.72705056
theta = 0.08327063
s = 282
t = 1
r = 0.015-0.0177
k1 = 285
k2 = 315

# (0.12540801386207912-0.30499349480606175j)
h = heston(s, r, nu0, t, kappa, theta, sigma, rho, k1)
payoff, strikes = h.fourier()
print(payoff[len(payoff) // 2])

