import numpy as np
import pandas as pd
from scipy.stats import norm

class stochastic:

    def __init__(self, s, r, sigma, t, q=0.0):

        """
            The general stochastic process can be written as

                dx = mu(x, t) * dt + v(x, t) * dW

            where
            
                x is the object of the process
                mu(x, t) is the drift
                v(x, t) is the volatility

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

    """
        also known as Vasicek process
        mainly used as the interest rate model
    """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):
        
        """
            kappa: the speed of the mean reversion
            theta: the level of the mean reversion
        """
        
        stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate_1d(self, z, n=1000, output='num', obj='r', xi=0.0):

        """
            mostly an interest rate model
            could be used to simulate the volatility in 2d model, where
            xi is required as the volatility of volatility (skewness)

            z: sample from the standard normal distribution
                possibly correlated with other distributions and that's why it is an input
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path')
                'num' in default
            obj: the object of the process, the interest rate 'r' or the volatility 'sigma'
                'r' in default
            xi: skewness, required if obj is 'sigma', 0.0 in default
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

    """ mainly used as the interest rate model """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):

        """
            kappa: the speed of the mean reversion
            theta: the level of the mean reversion
        """
        
        stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate_1d(self, z, n=1000, output='num', obj='r', xi=0.0, method='cutoff'):

        """
            mostly an interest rate model
            could be used to simulate the volatility in 2d model, where
            xi is required as the volatility of volatility (skewness)

            z: sample from the standard normal distribution
                possibly correlated with other distributions and that's why it is an input
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path')
                'num' in default
            obj: the object of the process, the interest rate 'r' or the volatility 'sigma'
                'r' in default
            xi: skewness, required if obj is 'sigma', 0.0 in default
            method: decision to make if simulated object is less than 0
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
    
    """ model for the price of the underlying asset """

    def __init__(self, s, r, sigma, t, beta, k, q=0.0):

        """
            beta: skewness of volatility surface
                if beta = 0, 
                if beta = 1, Black-Scholes model
            k: the strike
        """
        
        stochastic.__init__(self, s, r, sigma, t, q)
        self.beta = beta
        self.k = k

    def simulate_1d(self, z, n=1000, output='num'):

        dt = self.t / n

        if type == 'num':
            x = self.s
            for i in range(n):
                x += self.r * x * dt + sigma * self.sigma * np.sqrt(dt) * z[i] * x ** beta
            return x
        else:
            record = np.zeros(n+1)
            record[0] = self.s
            for i in range(1, n+1):
                record[i] += self.r * record[i-1] * dt + sigma * self.sigma * np.sqrt(dt) * z[i] * record[i-1] ** beta
            return record


class black_scholes(cev):

    def __init__(self, s, r, sigma, t, k, q=0.0):
        
        """ d1 and d2 are constants for closed-form solution """
        
        cev.__init__(self, s, r, sigma, t, 1, k, q)
        self.d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        self.d2 = self.d1 - sigma * np.sqrt(t)

    def euro_option(self, option='call', output='value'):    # black scholes formula

        if option == 'call':
            if output == 'value':
                return norm.cdf(self.d1) * self.s - norm.cdf(self.d2) * self.k * np.exp(-self.r * self.t)
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

        cox_intergell_ross.__init__(self, r, sigma, t, kappa, theta, s, q)
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

    def optimize(self, n, s, t, k_m, c_m, method='equal_weight'):

        """
            optimization principle: given every (s, t), we only generate fft once

            n: int / the length of the payoffs (or the strikes)
            s: the initial price
            t: the expiry

            k_m: np.array / the market strikes
            c_m: np.array / the market prices for options

            method: how to add errors, equal weights in default
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
    
    def simulate_2d(self, n=1000, output='num'):
        
        dt = self.t / n
        z1, z2 = 
        vol_process = self.simulate_1d


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

