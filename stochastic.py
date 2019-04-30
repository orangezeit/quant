# Creator: Yunfei Luo
# Date: Apr 29, 2019  10:07 PM

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def _generate(mu, v, dt, z, method='Euler', dv=0.0):

    """
        simulation generator

        mu: the drift
        v: the volatility
        dt: time mash
        z: random generated from (possibly correlated) standard normal distribution

        The general stochastic process can be written as

            dx = mu(x, t) * dt + v(x, t) * dW

        where

            x is the object of the process
            mu(x, t) is the drift
            v(x, t) is the volatility

        method: the simulation method, 'Euler' in default

            Euler method: delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z
            Milstein method: delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z
                                                          + 1 / 2 * v(x, t) * delta_v(x, t) * (z ** 2 - 1) * delta_t

        dv: volatility mesh, required if Milstein method is chosen
    """

    if method == 'Euler':
        return mu * dt + v * np.sqrt(dt) * z
    elif method == 'Milstein':
        return mu * dt + v * np.sqrt(dt) * z + v * dv * dt * (z * z - 1) / 2


def _modify(v, method='truncate'):

    """
        modify volatility if negative during simulation

        vol: the volatility
        method: adjust volatility, 'truncate' in default

            truncate: assume vol=0 if vol < 0
                cause volatility to be less
            reflect: assume vol=-vol if vol < 0
                cause volatility to be more
    """

    if v >= 0:
        return v

    if method == 'truncate':
        return 0
    elif method == 'reflect':
        return -v


class Stochastic:
    
    """ general stochastic process """
    
    def __init__(self, s, r, sigma, t, q=0.0):

        """
            s: the (initial) price of underlying assets
            r: the (initial) interest rate / rate of return
            sigma: the volatility
            t: the expiry
            q: the dividend
        """

        self.s = s
        self.r = r-q
        self.sigma = sigma
        self.t = t


class OrnsteinUhlenbeck(Stochastic):

    """
        interest rate model, also known as Vasicek process
        mainly used as the interest rate model
    """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):
        
        """
            kappa: the speed of the mean reversion
            theta: the level of the mean reversion
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=1000, output='num'):

        """
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path'), 'num' in default
        """

        dt = self.t / n
        z = np.random.normal(size=n)

        if output == 'num':
            r = self.r
            for i in range(n):
                r += _generate(self.kappa * (self.theta - r), self.sigma, dt, z[i])
            return r
        else:
            record = np.zeros(n+1)
            record[0] = self.r
            for i in range(n):
                record[i+1] = record[i] + _generate(self.kappa * (self.theta - record[i]), self.sigma, dt, z[i])
            return record


class CoxIntergellRoss(Stochastic):

    """ interest rate model """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):

        """
            kappa: the speed of the mean reversion
            theta: the level of the mean reversion
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=1000, output='num'):

        """
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path'), 'num' in default
        """

        dt = self.t / n
        z = np.random.normal(size=n)

        if output == 'num':
            r = self.r
            for i in range(n):
                r += _generate(self.kappa * (self.theta - r), self.sigma * np.sqrt(_modify(r)), dt, z[i])
            return r
        else:
            record = np.zeros(n+1)
            record[0] = self.r
            for i in range(n):
                record[i+1] = record[i] + _generate(self.kappa * (self.theta - record[i]),
                                                    self.sigma * np.sqrt(_modify(record[i])), dt, z[i])
            return record


class CEV(Stochastic):
    
    """ model for the price of the underlying asset """

    def __init__(self, s, r, sigma, t, beta, q=0.0):

        """
            beta: skewness of volatility surface
                stocks: beta between 0 and 1, inclusive
                commodity: beta greater than 0
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.beta = beta

    def simulate(self, n=1000, output='num'):

        """
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path'), 'num' in default
        """

        dt = self.t / n
        z = np.random.normal(size=n)

        if output == 'num':
            s = self.s
            for i in range(n):
                s += _generate(self.r * s, self.sigma * s ** self.beta, dt, z[i])
            return s
        else:
            record = np.zeros(n+1)
            record[0] = self.s
            for i in range(n):
                record[i+1] = record[i] + _generate(self.r * record[i], self.sigma * record[i] ** self.beta, dt, z[i])
            return record

    def pde(self, ht, hs, smax, k1, k2=0, option='Euro', method='Euler-Explicit'):

        n = self.t // ht
        m = smax // hs

        c = np.zeros((m-1, 1))

        # payoff


class Bachelier(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Bachelier is CEV with beta = 0 """

        CEV.__init__(self, s, r, sigma, t, 0, q)


class BlackScholes(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Black Scholes is CEV with beta = 1 """

        CEV.__init__(self, s, r, sigma, t, 1, q)

    def _d1(self, k):

        """ constant required for the close form solutions with strike k """

        return (np.log(self.s / k) + (self.r + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))

    def _d2(self, k):

        """ constant required for the close form solutions with strike k """

        return self._d1(k) - self.sigma * np.sqrt(self.t)

    def euro_value(self, k, option='call'):

        """
            closed-form solutions for price of European options

            k: the strike
            option: 'call' or 'put'
        """

        if option == 'call':
            return norm.cdf(self._d1(k)) * self.s - norm.cdf(self._d2(k)) * np.exp(-self.r * self.t)
        elif option == 'put':
            return 1

    def euro_delta(self, k, output='value'):

        """
            closed-form solutions for deltas of European options

            k: the strike
            option: 'call' or 'put'
        """

        pass


class Heston(CoxIntergellRoss):

    def __init__(self, sigma, kappa, theta, xi, rho, s, r, t=1, alpha=2, q=0.0):

        """
            the price is log-normal
            the volatility follows Cox-Intergell-Ross process

            xi: the skewness
            rho: the correlation between two motions
            alpha: the damping factor
        """

        CoxIntergellRoss.__init__(self, r, sigma, t, kappa, theta, s, q)
        self.xi = xi
        self.rho = rho
        self.alpha = alpha

    def psi(self, x):

        """ the characteristic function """

        x -= (self.alpha + 1) * 1j

        # intermediate variables
        a = x ** 2 + 1j * x
        b = self.kappa - 1j * self.rho * self.xi * x
        c = self.kappa * self.theta / self.xi ** 2
        lbd = np.sqrt(self.xi ** 2 * a + b ** 2)
        d = np.sinh(lbd * self.t / 2)
        e = np.cosh(lbd * self.t / 2)

        phi = np.exp(1j * x * (np.log(self.s) + self.r * self.t) + self.t * b * c
                     - a * self.sigma / (lbd * e / d + b)) / (e + b * d / lbd) ** (2 * c)

        x += (self.alpha + 1) * 1j

        return np.exp(-self.r * self.t) * phi / (self.alpha + x * 1j) / (self.alpha + x * 1j + 1)

    def fft(self, n=65536, dv=0.01):

        """ Fast Fourier Transform """

        weights = [dv / 2 if i == 0 else dv for i in range(n)]
        x = [np.exp(-1j * i * (dv * np.log(self.s) - np.pi)) * self.psi(i * dv) * weights[i] for i in range(n)]

        y = np.fft.fft(x)

        k = [np.log(self.s) + 2 * np.pi / (n * dv) * (i - n / 2) for i in range(n)]
        payoffs = [np.exp(-self.alpha * k[i]) / np.pi for i in range(n)]

        return payoffs * y.real, np.exp(k)
    
    def payoff(self, k, combo=tuple()):
        
        """ 
            Given the payoffs and strikes (combo) from FFT, find the model price for the strike k
            
                Find the desired strikes range by the binary search
                Linearly interpolate in the corresponding payoffs range and output the price
        """
        
        payoffs, strikes = combo if combo else self.fft()
        
        left = 0
        right = len(strikes)-1

        while right - left > 1:
            
            mid = (left + right) // 2

            if strikes[mid] <= k:
                left = mid
            else:
                right = mid
        
        slope = (payoffs[right] - payoffs[left]) / (strikes[right] - strikes[left])
        
        return payoffs[left] + slope * (k - strikes[left])
        
    def calibrate(self, n, tm, km, cm):

        """
            // very slow

            calibration principle: only generate fft once for each pair of (s, t), calibrate for all expiries
            
            Step One: Data Reconstruction - from {(s,t,k,c)} to {(s,t): {(k,c)}}
            
                original data structure is tuple (s, t, k, c) for every option
                build dictionary d with tuple (s, t) as key
                for each key, the value is a set of pairs of (k, c)
                
            Step Two: Define the Object Function
            
            Step Three: Optimize
                loop through the dictionary to calibrate
            
            n: int / the length of the payoffs (or the strikes)

            tm: the market expiry
            km: np.array / the market strikes
            cm: np.array / the market prices for options
        """
        
        d = {}
        
        for i in range(n):
            d.setdefault(tm[i], set()).add((km[i], cm[i]))

        def obj(x):

            """ x = [ sigma, kappa, theta, xi, rho ] """

            self.sigma, self.kappa, self.theta, self.xi, self.rho = x
            sse = 0

            for key, value in d.items():

                self.t = key
                combo = self.fft()

                for k, c in value:

                    sse += (c - self.payoff(k, combo)) ** 2

            return sse

        x = [self.sigma, self.kappa, self.theta, self.xi, self.rho]
        print(obj(x))
        bd = ((0, 10), (0, 10), (0, 10), (0, 10), (-1, 0))

        sol = minimize(obj, x, method='L-BFGS-B', bounds=bd)
        print(sol.x)
        print(obj(sol.x))
        return sol.x
    
    def simulate(self, n=1000, output='num', grid='Euler', method='cutoff'):

        """
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path'), 'num' in default

                         ds = [r * s] * dt + [sqrt(sigma) * s] * dW_1
                   d(sigma) = [kappa * (theta - sigma)] * dt + [xi * sqrt(sigma)] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        dt = self.t / n
        z1, z2 = np.random.multivariate_normal([0, 0], [[1, self.rho], [self.rho, 1]], n).T
        sigma = self.sigma
        
        if output == 'num':
            s = self.s
            for i in range(n):
                s += _generate(self.r * s, _modify(sigma) * s, dt, z1[i])
                sigma += _generate(self.kappa * (self.theta - sigma), self.xi * np.sqrt(_modify(sigma)), dt, z2[i])
            return s
        else:
            record = np.zeros(n + 1)
            record[0] = self.s
            for i in range(n):
                record[i+1] = record[i] + _generate(self.r * record[i], _modify(sigma) * record[i], dt, z1[i])
                sigma += _generate(self.kappa * (self.theta - sigma), self.xi * np.sqrt(_modify(sigma)), dt, z2[i])
            return record


class SABR(CEV):

    def __init__(self, sigma, alpha, beta, rho, s, r, t=1, q=0.0):

        """
            the price follows CEV
            the volatility is log-normal

            alpha: growth rate of the volatility
            rho: correlation between two Brownian motions
        """

        CEV.__init__(self, s, r, sigma, t, beta, q)
        self.alpha = alpha
        self.rho = rho

    def calibrate(self, n, sigma_m, km):

        """
            calibrate the approximate closed-form implied volatility for every expiry

            n: the number of sample
            sigma_m: the market volatility
            km: the market strike

            Note that self.sigma is the initial volatility
        """

        # the forward price
        f = self.s * np.exp(self.r * self.t)

        def obj(x):

            self.alpha, self.beta, self.rho, self.sigma = x
            sse = 0

            for i in range(n):

                p = (f * km[i]) ** ((1 - self.beta) / 2)
                q = np.log(f / km[i])
                z = self.alpha * p * q / self.sigma

                y = np.log((np.sqrt(1 - 2 * self.rho * z + z * z) + z - self.rho) / (1 - self.rho))
                a = 1 + ((1 - self.beta) ** 2 * self.sigma ** 2 / (
                            24 * p ** 2) + self.rho * self.alpha * self.beta * self.sigma
                         / (4 * p) + self.alpha ** 2 * (2 - 3 * self.rho ** 2) / 24) * self.t
                b = 1 + (1 - self.beta) ** 2 * q ** 2 / 24 + (1 - self.beta) ** 4 * q ** 4 / 1920

                vol = self.alpha * q / y * a / b

                sse += (vol - sigma_m[i]) ** 2

            return sse

        x = np.array([self.alpha, self.beta, self.rho, self.sigma])
        bd = ((0, None), (0.5, 0.5), (-1, 1), (0, None))
        sol = minimize(obj, x, method='L-BFGS-B', bounds=bd)

        return sol.x

    def simulate(self, n=1000, output='num'):

        """
            n: the number of randoms, 1000 in default
            output: the final value ('num') or the entire simulation ('path'), 'num' in default

                         ds = [r * s] * dt + [sigma * s ** beta] * dW_1
                   d(sigma) = [alpha * sigma] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        dt = self.t / n
        z1, z2 = np.random.multivariate_normal([0, 0], [[1, self.rho], [self.rho, 1]], n).T
        sigma = self.sigma
        
        if output == 'num':
            s = self.s
            for i in range(n):
                s += _generate(self.r * s, sigma * s ** self.beta, dt, z1[i])
                sigma += _generate(0, self.alpha * _modify(sigma), dt, z2[i])
            return s
        else:
            record = np.zeros(n + 1)
            record[0] = self.s
            for i in range(n):
                record[i+1] = record[i] + _generate(self.r * record[i], sigma * record[i] ** self.beta, dt, z1[i])
                sigma += _generate(0, self.alpha * _modify(sigma), dt, z2[i])
            return record
