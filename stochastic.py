# Author: Yunfei Luo
# Date: Aug 7, 2019

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.sparse import dia_matrix
from scipy.stats import norm


def _generate(mu, v, dt, z, method='Euler', dv=0.0):

    """
        Generate simulation

        The general stochastic process can be written as

            dx = mu(x, t) * dt + v(x, t) * dW

        where

            x is the object of the process
            mu(x, t) is the drift
            v(x, t) is the volatility

        Parameters
        ----------
        mu: float
            drift at time t

        v: float
            volatility

        dt: float
            time mash

        z: float
            random generated from (possibly correlated) standard normal distribution

        method: str, 'Euler' in default, optional
            simulation method

            Euler:
                delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z
            Milstein:
                delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z
                                                          + 1 / 2 * v(x, t) * delta_v(x, t) * (z ** 2 - 1) * delta_t

        dv: float, required if Milstein method is chosen
            volatility mesh

        Return
        ------
        _mu: float
            drift at time t + 1
    """

    if method == 'Euler':
        return mu * dt + v * np.sqrt(dt) * z
    elif method == 'Milstein':
        return mu * dt + v * np.sqrt(dt) * z + v * dv * dt * (z * z - 1) / 2


def _modify(v, method='truncate'):

    """
        Modify volatility if negative during simulation

        Parameters
        ----------
        vol: float
            volatility

        method: str, 'truncate' in default, optional
            method to adjust volatility

            truncate: assume vol=0 if vol < 0
                cause volatility to be less
            reflect: assume vol=-vol if vol < 0
                cause volatility to be more

        Return
        ------
        _v: float
            modified volatility
    """

    if v >= 0:
        return v

    if method == 'truncate':
        return 0
    elif method == 'reflect':
        return -v


class Stochastic:
    
    """ General stochastic process """
    
    def __init__(self, s, r, sigma, t, q=0.0):

        """
            Parameters
            ----------
            s: float
                initial price of underlying assets

            r: float
                initial (excess) interest rate / rate of return

            sigma: float
                volatility

            t: float
                expiry in years

            q: float, 0.0 in default, optional
                dividend
        """

        self.s = s
        self.r = r-q
        self.sigma = sigma
        self.t = t


class OrnsteinUhlenbeck(Stochastic):

    """ Also known as Vasicek process """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):
        
        """
            Additional Parameters
            ---------------------
            kappa: float
                speed of mean reversion

            theta: float
                level of mean reversion
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=1000, output='num'):

        """
            dr = [kappa * (theta - r)] * dt + [sigma] * dW

            Parameters
            ----------
            n: int, 1000 in default
                number of randoms

            output: str, 'num' in default
                the final value ('num') or the entire simulation ('path')

            Return
            ------
            r or record: float if output is 'num', np.array if output is 'path'
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

    """ Interest rate model """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):

        """
            Additional Parameters
            ---------------------
            kappa: float
                speed of mean reversion

            theta: float
                level of mean reversion
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=1000, output='num'):

        """
            dr = [kappa * (theta - r)] * dt + [sigma * sqrt(r)] * dW

            Parameters
            ----------
            n: int, 1000 in default
                number of randoms

            output: str, 'num' in default
                the final value ('num') or the entire simulation ('path')

            Return
            ------
            r or record: float if output is 'num', np.array if output is 'path'
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

    def __init__(self, s, r, sigma, t, beta, q=0.0):

        """
            Additional Parameter
            --------------------
            beta: skewness of volatility surface
                between 0 and 1, inclusive, for stocks
                greater than 1 for commodity
        """
        
        Stochastic.__init__(self, s, r, sigma, t, q)
        self.beta = beta

    def simulate(self, n=1000, output='num'):

        """
            ds = [r * s] * dt + [sigma * s ** beta] * dW

            Parameters
            ----------
            n: int, 1000 in default
                number of randoms

            output: str, 'num' in default
                the final value ('num') or the entire simulation ('path')

            Return
            ------
            r or record: float if output is 'num', np.array if output is 'path'
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

    def pde(self, ht, hs, smax, k1, k2, method='CN'):

        """ PDE Scheme """

        def lower(x): return (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) - self.r * x) * ht / 2

        def mid(x): return 1 - (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) + self.r) * ht

        def upper(x): return (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) + self.r * x) * ht / 2

        n = int(self.t / ht)
        m = int(smax / hs)

        c = np.zeros((m-1, 1))

        for i in range(m-1):
            if (i+1) * hs <= k1:
                continue
            elif (i+1) * hs >= k2:
                c[i] = 5
            else:
                c[i] = (i+1) * hs - 285

        d = c.copy()

        md = np.array([mid(i) if method == 'EE' else 2 - mid(i) if method == 'EI' else 3 - mid(i) for i in range(1, m)])
        ld = np.array([0 if i == m else lower(i) if method == 'EE' else -lower(i) for i in range(2, m+1)])
        ud = np.array([0 if i == 0 else upper(i) if method == 'EE' else -upper(i) for i in range(0, m-1)])
        diag = np.array([ud, md, ld])

        if method == 'CN':
            diag2 = np.array([-ud, 4 - md, -ld])

        for i in range(n, 0, -1):

            if method == 'EE':
                c = dia_matrix((diag, [1, 0, -1]), shape=(m-1, m-1)).dot(c)
                c[-1] += upper(m-1) * d[-1] * np.exp(-self.r * (self.t - ht * i))
            elif method == 'EI':
                c[-1] += upper(m-1) * d[-1] * np.exp(-self.r * (self.t - ht * i))
                c = solve_banded((1, 1), diag, c, overwrite_b=True)
            elif method == 'CN':
                c = dia_matrix((diag2, [1, 0, -1]), shape=(m-1, m-1)).dot(c)
                c[-1] += upper(m - 1) * d[-1] * (
                            np.exp(-self.r * (self.t - ht * (i - 1))) + np.exp(-self.r * (self.t - ht * i)))
                c = solve_banded((1, 1), diag, c, overwrite_b=True)

            for j in range(m-1):
                if (j + 1) * hs <= k1:
                    continue
                elif (j + 1) * hs >= k2:
                    temp = 5
                else:
                    temp = (j + 1) * hs - 285
                c[j] = max(c[j], temp * np.exp(-self.r * (self.t - ht * i)))

        return c[int(self.s / hs)]


class Bachelier(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Bachelier is CEV process with beta = 0 """

        CEV.__init__(self, s, r, sigma, t, 0, q)


class BlackScholes(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Black-Scholes is CEV process with beta = 1 """

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

            k: float
                strike
            option: str
                type of option, 'call' or 'put'
        """

        if option == 'call':
            return norm.cdf(self._d1(k)) * self.s - norm.cdf(self._d2(k)) * np.exp(-self.r * self.t)
        elif option == 'put':
            return 1

    def euro_delta(self, k, option='call'):

        """
            closed-form solutions for deltas of European options

            k: float
                strike

            option: str
                type of option, 'call' or 'put'
        """

        pass


class Heston(CoxIntergellRoss):

    """ Price is log-normal and volatility follows Cox-Intergell-Ross process """

    def __init__(self, sigma, kappa, theta, xi, rho, s, r, t=1, alpha=2.0, q=0.0):

        """
            Additional Parameters
            ---------------------
            xi: float
                skewness

            rho: float
                -1 <= rho <= 1, correlation between two Brownian motions

            alpha: float
                damping factor
        """

        CoxIntergellRoss.__init__(self, r, sigma, t, kappa, theta, s, q)
        self.xi = xi
        self.rho = rho
        self.alpha = alpha

    def psi(self, x):

        """ Characteristic Function """

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

            Calibration principle: only generate fft once for each pair of (s, t), calibrate for all expiries
            
            Step One: Data Reconstruction - from {(s,t,k,c)} to {(s,t): {(k,c)}}
            
                original data structure is tuple (s, t, k, c) for every option
                build dictionary d with tuple (s, t) as key
                for each key, the value is a set of pairs of (k, c)
                
            Step Two: Define the Object Function
            
            Step Three: Optimize

                loop through the dictionary to calibrate

            Parameters
            ----------
            n: int
                length of the payoffs (or the strikes)

            tm: np.array
                market expiry

            km: np.array
                market strikes

            cm: np.array
                market prices for options

            Return
            ------
            calibrated_parameters: np.array
                instance parameters are also updated
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

        x = np.array([self.sigma, self.kappa, self.theta, self.xi, self.rho])
        bd = ((0, 10), (0, 10), (0, 10), (0, 10), (-1, 0))
        sol = minimize(obj, x, method='L-BFGS-B', bounds=bd)

        return sol.x
    
    def simulate(self, n=1000, output='num', grid='Euler', method='cutoff'):

        """
                         ds = [r * s] * dt + [sqrt(sigma) * s] * dW_1
                   d(sigma) = [kappa * (theta - sigma)] * dt + [xi * sqrt(sigma)] * dW_2
            Cov(dW_1, dW_2) = rho * dt

            Parameters
            ----------
            n: int, 1000 in default
                number of randoms

            output: str, 'num' in default
                the final value ('num') or the entire simulation ('path')

            Return
            ------
            r or record: float if output is 'num', np.array if output is 'path'
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

    """ Price follows CEV process and volatility is log-normal"""

    def __init__(self, sigma, alpha, beta, rho, s, r, t=1, q=0.0):

        """
            Additional Parameters
            ---------------------
            alpha: float
                growth rate of the volatility

            rho: float
                -1 <= rho <= 1, correlation between two Brownian motions
        """

        CEV.__init__(self, s, r, sigma, t, beta, q)
        self.alpha = alpha
        self.rho = rho

    def calibrate(self, n, sigma_m, km):

        """
            calibrate the approximate closed-form implied volatility for every expiry

            Paramters
            ---------
            n: int
                number of sample

            sigma_m: np.array
                market volatility

            km: np.array
                market strike

            Return
            ------
            calibrated_parameters: np.array
                instance parameters are also updated
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
                         ds = [r * s] * dt + [sigma * s ** beta] * dW_1
                   d(sigma) = [alpha * sigma] * dW_2
            Cov(dW_1, dW_2) = rho * dt

            Parameters
            ----------
            n: int, 1000 in default
                number of randoms

            output: str, 'num' in default
                the final value ('num') or the entire simulation ('path')

            Return
            ------
            r or record: float if output is 'num', np.array if output is 'path'
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
