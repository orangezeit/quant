# Author: Yunfei Luo
# Date: Aug 13, 2019
# 0.1.0

from functools import reduce
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

    def _simulate(self, n, path, rate, delta_x, delta_w1, sv=False, delta_sigma=None, delta_w2=None, rho=0.0):

        """
            dx = delta_x * dt + delta_w1 * dW_1
            d(sigma) = delta_sigma * dt + delta_w2 * dW_2
            Cov(dW_1, dW_2) = rho

            Parameters
            ----------
            n: int
                number of simulations

            path: bool
                the final value ('num') or the entire simulation ('path')

            rate: bool

            delta_x: function

            delta_w1: function
                to describe how the first Brownian motion varies

            sv: bool, False in default
                whether
            delta_sigma: function, None in default

            delta_w2:

            rho: float, 0.0 in default

            Return
            ------
            x or record: np.array if path is True, float if path is False
        """

        x = self.r if rate else self.s
        sigma = self.sigma

        if sv:
            z1, z2 = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n).T
        else:
            z1 = np.random.normal(size=n)

        if path:
            record = np.zeros(n)

        def move(c, i):
            # global sigma
            c += _generate(delta_x(c), delta_w1(c, sigma), self.t / n, z1[i])
            if sv:
                sigma += _generate(delta_sigma(sigma), delta_w2(sigma), self.t / n, z2[i])
            if path:
                record[i] = c
            return c

        x = reduce(move, range(n), x)

        """
        for i in range(n):
            x += _generate(delta_x(x), delta_w1(x, sigma), self.t / n, z1[i])
            if sv:
                sigma += _generate(delta_sigma(sigma), delta_w2(sigma), self.t / n, z2[i])
            if path:
                record[i] = x
        """

        return record if path else x


class OrnsteinUhlenbeck(Stochastic):

    """ Interest rate model, also known as Vasicek process """

    def __init__(self, r, sigma, t, kappa, theta, s=0.0, q=0.0):
        
        """
            Additional Parameters
            ---------------------
            kappa: float
                speed of mean reversion

            theta: float
                level of mean reversion
        """
        
        super().__init__(s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=100, path=False):

        """ dr = [kappa * (theta - r)] * dt + [sigma] * dW """

        return super()._simulate(n, path, True, lambda r: self.kappa * (self.theta - r), lambda r, sigma: self.sigma)


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
        
        super().__init__(s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=100, path=False, cutoff='truncate'):

        """ dr = [kappa * (theta - r)] * dt + [sigma * sqrt(r)] * dW """

        return super()._simulate(n, path, True, lambda r: self.kappa * (self.theta - r),
                                 lambda r, sigma: self.sigma * np.sqrt(_modify(r, cutoff)))


class CEV(Stochastic):

    def __init__(self, s, r, sigma, t, beta, q=0.0):

        """
            Additional Parameter
            --------------------
            beta: skewness of volatility surface
                between 0 and 1, inclusive, for stocks
                greater than 1 for commodity
        """
        
        super().__init__(s, r, sigma, t, q)
        self.beta = beta

    def simulate(self, n=100, path=False):

        """ ds = [r * s] * dt + [sigma * s ** beta] * dW """

        return super()._simulate(n, path, False,
                                 lambda s: self.r * s, lambda s, sigma: self.sigma * s ** self.beta)

    def pde(self, ht, hs, s_max, euro, option, k1, k2=0.0, grid='CN'):

        """
            PDE Scheme

            Parameters
            ----------
            ht: positive float
                time mesh

            hs: positive float
                price mesh

            s_max: positive float
                maximum stock price in scheme

            euro: bool
                European option if True, American option if False

            option: str
                option type selected from {call, put, call-spread, put-spread}

            k1: float
                (first) strike

            k2: float, 0.0 in default, required if option is call-spread or put-spread
                (second) strike

            grid: str
                scheme type selected from {'EE', 'EI', 'CN'}
                    EE: Euler Explicit
                    EI: Euler Implicit
                    CN: Crank Nicolson
        """

        if not isinstance(ht, float) or ht <= 0.0:
            raise ValueError('Time mesh must be a positive float.')

        if not isinstance(hs, float) or hs <= 0.0:
            raise ValueError('Price mesh must be a positive float.')

        if not isinstance(s_max, float) or s_max <= 0.0:
            raise ValueError('Maximum stock price in scheme must be a positive float.')

        if not isinstance(euro, bool):
            raise ValueError('')

        if not isinstance(k1, float) or k1 <= 0.0:
            raise ValueError('First strike must be positive float.')

        if option in ('call-spread', 'put-spread') and (not isinstance(k2, float) or k2 < k1):
            raise ValueError('Second strike must be positive float not less than first strike.')

        if grid not in ('EE', 'EI', 'CN'):
            raise ValueError('Not Defined. Scheme type must be selected from {EE, EI, CN}.')

        m = int(s_max / hs)

        if option == 'call':
            c = np.array([max(0.0, (i + 1) * hs - k1) for i in range(m - 1)])
        elif option == 'put':
            c = np.array([max(0.0, k1 - (i + 1) * hs) for i in range(m - 1)])
        elif option == 'call-spread':
            c = np.array([min(max(0.0, (i + 1) * hs - k1), k2 - k1) for i in range(m - 1)])
        elif option == 'put-spread':
            c = np.array([max(min(k2 - k1, k1 - (i + 1) * hs), 0.0) for i in range(m - 1)])
        else:
            raise ValueError('Not defined. Option type must be selected from {}.')

        d = c.copy()

        def lower(x):
            return (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) - self.r * x) * ht / 2

        def mid(x):
            return 1 - (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) + self.r) * ht

        def upper(x):
            return (self.sigma ** 2 * x ** (2 * self.beta) * hs ** (2 * self.beta - 2) + self.r * x) * ht / 2

        md = np.array([mid(i) if grid == 'EE' else 2 - mid(i) if grid == 'EI' else 3 - mid(i) for i in range(1, m)])
        ld = np.array([0 if i == m else lower(i) if grid == 'EE' else -lower(i) for i in range(2, m + 1)])
        ud = np.array([0 if i == 0 else upper(i) if grid == 'EE' else -upper(i) for i in range(0, m - 1)])
        diag = np.array([ud, md, ld])

        if grid == 'CN':
            diag2 = np.array([-ud, 4 - md, -ld])

        def euler_explicit(v, i):
            v = dia_matrix((diag, [1, 0, -1]), shape=(m - 1, m - 1)).dot(v)
            v[0] += lower(1) * d[0] * np.exp(-self.r * (self.t - ht * i))
            v[-1] += upper(m - 1) * d[-1] * np.exp(-self.r * (self.t - ht * i))
            return v if euro else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        def euler_implicit(v, i):
            v[0] += lower(1) * d[0] * np.exp(-self.r * (self.t - ht * i))
            v[-1] += upper(m - 1) * d[-1] * np.exp(-self.r * (self.t - ht * i))
            v = solve_banded((1, 1), diag, v, overwrite_b=True)
            return v if euro else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        def crank_nicolson(v, i):
            v = dia_matrix((diag2, [1, 0, -1]), shape=(m - 1, m - 1)).dot(v)
            v[0] += lower(1) * d[0] * (
                    np.exp(-self.r * (self.t - ht * (i - 1))) + np.exp(-self.r * (self.t - ht * i)))
            v[-1] += upper(m - 1) * d[-1] * (
                    np.exp(-self.r * (self.t - ht * (i - 1))) + np.exp(-self.r * (self.t - ht * i)))
            v = solve_banded((1, 1), diag, v, overwrite_b=True)
            return v if euro else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        n = int(self.t / ht)

        if grid == 'EE':
            c = reduce(euler_explicit, range(n, 0, -1), c)
        elif grid == 'EI':
            c = reduce(euler_implicit, range(n, 0, -1), c)
        elif grid == 'CN':
            c = reduce(crank_nicolson, range(n, 0, -1), c)

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


class Heston(Stochastic):

    """ Price is log-normal and volatility follows Cox-Intergell-Ross process """

    def __init__(self, s, r, sigma, kappa, theta, xi, rho, t=1, alpha=2.0, q=0.0):

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

        Stochastic.__init__(self, s, r, sigma, t, q)
        self.kappa = kappa
        self.theta = theta
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

        parameters = np.array([self.sigma, self.kappa, self.theta, self.xi, self.rho])
        bd = ((0, 10), (0, 10), (0, 10), (0, 10), (-1, 0))
        sol = minimize(obj, parameters, method='L-BFGS-B', bounds=bd)

        return sol.x

    def simulate(self, n=100, path=False, cutoff='truncate'):

        """
            ds = [r * s] * dt + [sqrt(sigma) * s] * dW_1
            d(sigma) = [kappa * (theta - sigma)] * dt + [xi * sqrt(sigma)] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        return super()._simulate(n, path, False, lambda s: self.r * s, lambda s, sigma: _modify(sigma, cutoff) * s,
                                 True, lambda sigma: self.kappa * (self.theta - sigma),
                                 lambda sigma: self.xi * np.sqrt(_modify(sigma, cutoff)), self.rho)


class SABR(Stochastic):

    """ Price follows CEV process and volatility is log-normal"""

    def __init__(self, s, r, sigma, alpha, beta, rho, t=1, q=0.0):

        """
            Additional Parameters
            ---------------------
            alpha: float
                growth rate of the volatility

            rho: float
                -1 <= rho <= 1, correlation between two Brownian motions
        """

        super().__init__(s, r, sigma, t, q)
        self.alpha = alpha
        self.beta = beta
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

        parameters = np.array([self.alpha, self.beta, self.rho, self.sigma])
        bd = ((0, None), (0.5, 0.5), (-1, 1), (0, None))
        sol = minimize(obj, parameters, method='L-BFGS-B', bounds=bd)

        return sol.x

    def simulate(self, n=100, path=False, cutoff='truncate'):

        """
            ds = [r * s] * dt + [sigma * s ** beta] * dW_1
            d(sigma) = [alpha * sigma] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        return super()._simulate(n, path, False, lambda s: self.r * s, lambda s, sigma: sigma * s ** self.beta,
                                 True, lambda sigma: 0, lambda sigma: self.alpha * _modify(sigma, cutoff), self.rho)

