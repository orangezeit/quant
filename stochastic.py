# Author: Yunfei Luo
# Date: Aug 18, 2019
# Version: 0.2.1 (in development)


from functools import reduce

import numpy as np

from scipy.fftpack import fft
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.sparse import dia_matrix
from scipy.stats import norm


options = {'call': lambda st, k: np.maximum(0.0, st - k), 'put': lambda st, k: np.maximum(0.0, k - st),
           'call-spread': lambda st, k1, k2: np.minimum(np.maximum(0.0, st - k1), k2 - k1),
           'put-spread': lambda st, k1, k2: np.maximum(np.minimum(k2 - k1, k1 - st), 0.0),
           'call-binary': lambda st, k: (st > k).astype(float), 'put-binary': lambda st, k: (st < k).astype(float)}


def select_basket(basket, paths, based, pack):

    """
        Select basket option based on multiple underlying assets

        Parameters
        ----------
        basket: str
            basket type selected from {'min', 'max', 'atlas', 'everest', 'himalayan'}

        paths: np.array
            simulation paths of assets

        based: str
            based exercise type selected from {'European', 'American'}

        Return
        ------
        path: np.array
            desired simulation path
    """

    if basket == 'min':
        fun = np.maximum
    elif basket == 'max':
        fun = np.minimum
    elif basket in ('atlas', 'everest', 'himalayan'):
        fun = None
    else:
        raise ValueError('Method does not exist.')

    option, *strikes, times = pack

    if based == 'European':
        return np.array([options[option](reduce(fun, paths[:, i, :])[-1], *strikes) for i in range(times)]).mean()
    else:
        pass


def _generate(mu, v, dt, z, grid='Euler', dv=0.0):

    """
        Generate simulation

        General stochastic process can be written as

            dx = mu(x, t) * dt + v(x, t) * dW

        where

            x is object of process
            mu(x, t) is drift
            v(x, t) is volatility

        Parameters
        ----------
        mu: float
            drift at time t

        v: non-negative float
            volatility

        dt: positive float
            time mesh

        z: float
            random generated from (possibly correlated) standard normal distribution

        grid: str, 'Euler' in default, optional
            simulation method selected from {'Euler', 'Milstein'}

            Euler method:
                delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z

            Milstein method:
                delta_x = mu(x, t) * delta_t + v(x, t) * sqrt(delta_t) * z
                                   + 1 / 2 * v(x, t) * delta_v(x, t) * (z ** 2 - 1) * delta_t

        dv: float, 0.0 in default, required if Milstein method is chosen
            volatility mesh

        Return
        ------
        _mu: float
            drift at time t + 1
    """

    if grid == 'Euler':
        return mu * dt + v * np.sqrt(dt) * z
    elif grid == 'Milstein':
        return mu * dt + v * np.sqrt(dt) * z + v * dv * dt * (z * z - 1) / 2
    else:
        raise ValueError('Method does not exist.')


def _modify(v, cutoff):

    """
        Modify volatility if negative during simulation

        Parameters
        ----------
        v: float
            volatility

        cutoff: str
            method to adjust volatility

            truncate: assume vol=0 if vol < 0
                cause volatility to be less

            reflect: assume vol=-vol if vol < 0
                cause volatility to be more

        Return
        ------
        _v: non-negative float
            modified volatility
    """

    if v >= 0:
        return v
    elif cutoff == 'truncate':
        return 0
    elif cutoff == 'reflect':
        return -v
    else:
        raise ValueError('Method does not exist.')


class Stochastic:
    
    """ General stochastic process """
    
    def __init__(self, s, r, sigma, t, q=0.0):

        """
            Parameters
            ----------
            s: non-negative float
                initial price of underlying assets

            r: float
                initial (excess) interest rate / rate of return

            sigma: non-negative float
                volatility

            t: non-negative float
                expiry in years

            q: non-negative float, 0.0 in default, optional
                dividend rate
        """

        if not isinstance(s, float) or s < 0:
            raise ValueError('Initial price must be non-negative float.')

        if not isinstance(r, float):
            raise ValueError('Initial interest rate must be float.')

        if not isinstance(sigma, float) or sigma < 0:
            raise ValueError('Volatility must be non-negative float.')

        if not isinstance(sigma, float) or t < 0:
            raise ValueError('Expiry must be non-negative float.')

        if not isinstance(q, float) or q < 0:
            raise ValueError('Dividend rate must be non-negative float.')

        self.s = s
        self.r = r-q
        self.sigma = sigma
        self.t = t

    def monte_carlo(self, path, n, rate, delta_x, delta_w1, sv=False, delta_sigma=None, delta_w2=None, rho=0.0):

        """
            Monte Carlo simulation process can be written as

                dx = delta_x * dt + delta_w1 * dW_1
                d(sigma) = delta_sigma * dt + delta_w2 * dW_2
                Cov(dW_1, dW_2) = rho

            Parameters
            ----------
            path: bool
                return entire simulation if True and final value if False

            n: int
                number of time intervals

            rate: bool
                object is rate if True and price if False

            delta_x: function
                to describe how object varies

            delta_w1: function
                to describe how first Brownian motion varies

            sv: bool, False in default
                whether volatility is stochastic

            delta_sigma: function, None in default
                to describe how volatility varies

            delta_w2: function, None in default
                to describe how second Brownian motion varies

            rho: float, 0.0 in default
                correlation of two Brownian motions

            Return
            ------
            x or record: np.array if path is True, float if path is False
                entire simulation or final value
        """

        if not isinstance(n, int) or n <= 0:
            raise ValueError('Number of time intervals must be positive int.')

        x = self.r if rate else self.s
        sigma = self.sigma

        if sv:
            z1, z2 = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n).T
        else:
            z1 = np.random.normal(size=n)

        if path:
            record = np.zeros(n)

        for i in range(n):
            x += _generate(delta_x(x), delta_w1(x, sigma), self.t / n, z1[i])
            if sv:
                sigma += _generate(delta_sigma(sigma), delta_w2(sigma), self.t / n, z2[i])
            if path:
                record[i] = x

        return record if path else x

    def _simulate(self, path, setting, pack):

        """
            Simulation and option pricing based on one underlying asset

            Parameters
            ----------
            path: bool
                return entire simulation if True and final value if False

            setting: tuple(n, rate, delta_x, delta_w1[, sv, delta_sigma, delta_w2, rho])
                parameters and functions for Monte-Carlo simulation

            pack: tuple(str, str, float[, ...], int, None or tuple(str, float, str))
                parameters for option pricing

                option: str
                    option type selected from {'call', 'put', 'call-spread', 'put-spread', 'call-binary', 'put-binary'}

                exercise: str
                    exercise type selected from {'European', 'American', 'Asian-fixed', 'Asian-float'
                                                 'lookback-fixed', 'lookback-float', 'barrier'}

                strikes: float

                times: int
                    number of simulations

                info: None or tuple(str, float, str)
                    extra information, required if 'barrier' is selected

                    act: str
                        activation type selected from {'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'}

                    barrier: float
                        the spot price to activate or deactivate based exercise

                    based: str
                        based exercise type selected from {'European', 'American'}

            Return
            ------
            p: float
                (simulated) option price
        """

        if isinstance(path, bool):
            raise TypeError('Variable path must be bool.')

        if pack is None:
            return self.monte_carlo(path, *setting)

        option, exercise, *strikes, times, info = pack

        if not isinstance(times, int) or times <= 0:
            raise ValueError('Number of simulation must be positive int.')

        if option in options:
            if exercise == 'European':
                return options[option](np.array([self.monte_carlo(False, *setting) for _ in range(times)]),
                                       *strikes).mean()
            elif exercise == 'American':  # not correct

                p_sum = 0.0
                n, _ = setting
                for _ in range(times):
                    mc = np.flip(self.monte_carlo(True, *setting))

                    def backward(p, i):
                        return max(options[option](mc[i], *strikes), p * np.exp(-self.r * self.t / n))

                    p_sum += reduce(backward, range(1, n), options[option](mc[0], *strikes))

                return p_sum / times

            elif exercise == 'Asian-fixed':
                return options[option](np.array([self.monte_carlo(True, *setting).mean() for _ in range(times)]),
                                       *strikes).mean()
            elif exercise == 'Asian-float':

                p_sum = 0.0

                for _ in range(times):
                    mc = self.monte_carlo(True, *setting)
                    p_sum += options[option](mc.mean(), mc[-1])

                return p_sum / times

            elif exercise == 'lookback-fixed':

                if option.split('-')[0] == 'call':
                    mc = np.array([self.monte_carlo(True, *setting).max() for _ in range(times)])
                else:
                    mc = np.array([self.monte_carlo(True, *setting).min() for _ in range(times)])
                return options[option](mc, *strikes).mean()

            elif exercise == 'lookback-float':

                p_sum = 0.0

                for _ in range(times):
                    mc = self.monte_carlo(True, *setting)
                    if option.split('-')[0] == 'call':
                        p_sum += options[option](mc.max(), mc[-1])
                    else:
                        p_sum += options[option](mc.min(), mc[-1])

                return p_sum / times

            elif exercise == 'barrier':

                act, barrier, based = info

                if not isinstance(barrier, float):
                    raise ValueError('Barrier must be float.')

                if based not in ('European', 'American'):
                    raise ValueError('Based exercise type must be selected from {European, American}.')

                acts = {'up-and-out': lambda p: (p < barrier).all(),
                        'down-and-out': lambda p: (p > barrier).all(),
                        'up-and-in': lambda p: (p > barrier).any(),
                        'down-and-in': lambda p: (p < barrier).any()}

                if act not in acts:
                    raise ValueError('Activation type does not exist.')

                p_sum = 0.0

                for _ in range(times):
                    mc = self.monte_carlo(True, *setting)

                    if acts[act](mc):
                        if based == 'European':
                            p_sum += options[option](mc[-1], *strikes)
                        else:
                            pass

                return p_sum / times

            else:
                raise ValueError('Exercise type does not exist.')
        else:
            raise ValueError('Option type does not exist.')


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

        if not isinstance(kappa, float):
            raise ValueError('Speed of mean reversion must be float.')

        if not isinstance(theta, float):
            raise ValueError('Level of mean reversion must be float.')

        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=100, path=False, pack=None):

        """ dr = [kappa * (theta - r)] * dt + [sigma] * dW """

        return super()._simulate(path, (n, True, lambda x: self.kappa * (self.theta - x), lambda x, y: self.sigma),
                                 pack)


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

        if not isinstance(kappa, float):
            raise ValueError('Speed of mean reversion must be float.')

        if not isinstance(theta, float):
            raise ValueError('Level of mean reversion must be float.')

        self.kappa = kappa
        self.theta = theta

    def simulate(self, n=100, path=False, cutoff='truncate', pack=None):

        """ dr = [kappa * (theta - r)] * dt + [sigma * sqrt(r)] * dW """

        return super()._simulate(path, (n, True, lambda r: self.kappa * (self.theta - r),
                                 lambda r, sigma: self.sigma * np.sqrt(_modify(r, cutoff))), pack)


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

        if not isinstance(beta, float):
            raise ValueError('Skewness of volatility surface must be float.')

        self.beta = beta

    def simulate(self, n=10000, path=False, pack=None):

        """ ds = [r * s] * dt + [sigma * s ** beta] * dW """

        return super()._simulate(path, (n, False, lambda x: self.r * x, lambda x, y: self.sigma * x ** self.beta),
                                 pack)

    def pde(self, ht, hs, pack, mul=2.0, grid='CN'):

        """
            PDE scheme for option pricing

            Parameters
            ----------
            ht: positive float
                time mesh

            hs: positive float
                price mesh

            pack: tuple(str, str, float[, ...])

                option: str
                    option type selected from {'call', 'put', 'call-spread', 'put-spread'}
                exercise: str
                    exercise type selected from {'European', 'American'}
                strikes: float

            mul: positive float
                constant multiplier to determine upper bound of price in grid
                mul > 1

            grid: str
                scheme type selected from {'EE', 'EI', 'CN'}
                    Euler Explicit / Euler Implicit / Crank Nicolson

            Return
            ------
            p: float
                option price
        """

        if not isinstance(ht, float) or ht <= 0.0:
            raise ValueError('Time mesh must be a positive float.')

        if not isinstance(hs, float) or hs <= 0.0:
            raise ValueError('Price mesh must be a positive float.')

        if not isinstance(mul, float) or mul <= 1.0:
            raise ValueError('Constant multiplier must be float greater than 1.')

        option, exercise, *strikes = pack
        s_max = max(self.s, *strikes) * mul

        if exercise not in ('European', 'American'):
            raise ValueError('Exercise type must be selected from {European, American}.')

        if grid not in ('EE', 'EI', 'CN'):
            raise ValueError('Scheme type must be selected from {EE, EI, CN}.')

        m = int(s_max / hs)

        if option in options:
            c = options[option](hs * np.array(range(1, m)), *strikes)
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
            return v if exercise == 'European' else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        def euler_implicit(v, i):
            v[0] += lower(1) * d[0] * np.exp(-self.r * (self.t - ht * i))
            v[-1] += upper(m - 1) * d[-1] * np.exp(-self.r * (self.t - ht * i))
            v = solve_banded((1, 1), diag, v, overwrite_b=True)
            return v if exercise == 'European' else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        def crank_nicolson(v, i):
            v = dia_matrix((diag2, [1, 0, -1]), shape=(m - 1, m - 1)).dot(v)
            v[0] += lower(1) * d[0] * (
                    np.exp(-self.r * (self.t - ht * (i - 1))) + np.exp(-self.r * (self.t - ht * i)))
            v[-1] += upper(m - 1) * d[-1] * (
                    np.exp(-self.r * (self.t - ht * (i - 1))) + np.exp(-self.r * (self.t - ht * i)))
            v = solve_banded((1, 1), diag, v, overwrite_b=True)
            return v if exercise == 'European' else np.maximum(v, d * np.exp(-self.r * (self.t - ht * i)))

        n = int(self.t / ht)

        if grid == 'EE':
            c = reduce(euler_explicit, range(n, 0, -1), c)
        elif grid == 'EI':
            c = reduce(euler_implicit, range(n, 0, -1), c)
        elif grid == 'CN':
            c = reduce(crank_nicolson, range(n, 0, -1), c)

        return c[int(self.s / hs)]

    def calibrate(self, tm, km, cm, pm):

        parameters = np.array([self.sigma, self.beta])

        def obj(x):
            self.sigma, self.beta = x
            print(self.sigma, self.beta)
            err = 0.0

            for t, k, c, p in zip(tm, km, cm, pm):

                self.t = t
                err += abs(self.pde(t / 1000, 0.01, ('call', 'European', k)) - c) ** 2
                err += abs(self.pde(t / 1000, 0.01, ('put', 'European', k)) - p) ** 2

            return err

        bd = ((0, None), (0, 1))
        sol = minimize(obj, parameters, bounds=bd)

        return sol.x


class Bachelier(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Bachelier is CEV process with beta = 0 """

        CEV.__init__(self, s, r, sigma, t, 0.0, q)


class BlackScholes(CEV):

    def __init__(self, s, r, sigma, t, q=0.0):

        """ Black-Scholes is CEV with beta = 1 """

        CEV.__init__(self, s, r, sigma, t, 1.0, q)

    def european_option_formula(self, k, option, output):

        """
            Closed-form solutions for price or Greek of European options

            Parameters
            ----------
            k: float
                strike

            option: str
                option 'call' or 'put'

            output: str
                desired output type selected from {'price', 'delta', 'gamma', 'vega', 'theta', 'rho'}

            Return
            ------
            ans: float
                desired price or Greek
        """

        d1 = (np.log(self.s / k) + (self.r + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)

        formula = {'call-value': norm.cdf(d1) * self.s - norm.cdf(d2) * k * np.exp(-self.r * self.t),
                   'put-value': norm.cdf(-d2) * k * np.exp(-self.r * self.t) - norm.cdf(-d1) * self.s,
                   'call-delta': norm.cdf(d1), 'put-delta': norm.cdf(d1) - 1,
                   'call-gamma': norm.pdf(d1) / (self.s * self.sigma * np.sqrt(self.t)),
                   'put-gamma': norm.pdf(d1) / (self.s * self.sigma * np.sqrt(self.t)),
                   'call-vega': self.s * norm.pdf(d1) * np.sqrt(self.t),
                   'put-vega': self.s * norm.pdf(d1) * np.sqrt(self.t),
                   'call-theta': -self.s * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t)) - self.r * k * np.exp(
                       -self.r * self.t) * norm.cdf(d2),
                   'put-theta': -self.s * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t)) + self.r * k * np.exp(
                       -self.r * self.t) * norm.cdf(-d2),
                   'call-rho': k * self.t * np.exp(-self.r * self.t) * norm.cdf(d2),
                   'put-rho': -k * self.t * np.exp(-self.r * self.t) * norm.cdf(-d2)}

        return formula[f'{option}-{output}']


class Heston(Stochastic):

    """ Price is log-normal and volatility follows Cox-Intergell-Ross process """

    def __init__(self, s, r, sigma, kappa, theta, xi, rho, t, alpha=2.0, q=0.0):

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

    def simulate(self, n=100, path=False, cutoff='truncate', pack=None):

        """
            ds = [r * s] * dt + [sqrt(sigma) * s] * dW_1
            d(sigma) = [kappa * (theta - sigma)] * dt + [xi * sqrt(sigma)] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        return super()._simulate(path, (n, False, lambda s: self.r * s, lambda s, sigma: _modify(sigma, cutoff) * s,
                                 True, lambda sigma: self.kappa * (self.theta - sigma),
                                 lambda sigma: self.xi * np.sqrt(_modify(sigma, cutoff)), self.rho), pack)

    def heston_fft(self, n=65536, dv=0.01):

        """ Fast Fourier Transform """

        # characteristic function
        def psi(x):

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

        weights = np.array([dv / 2 if i == 0 else dv for i in range(n)])
        y = fft([np.exp(-1j * i * (dv * np.log(self.s) - np.pi)) * psi(i * dv) for i in range(n)] * weights)
        k = np.exp(np.log(self.s) + 2 * np.pi / (n * dv) * (np.array(range(n)) - n / 2))

        return k ** (-self.alpha) / np.pi * y.real, k
    
    def payoff(self, k, combo=None):
        
        """ 
            Given the payoffs and strikes (combo) from FFT, find the model price for the strike k
            
                Find the desired strikes range by the binary search
                Linearly interpolate in the corresponding payoffs range and output the price
        """
        
        payoffs, strikes = combo if combo else self.heston_fft()
        
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
        
    def calibrate(self, tm, km, cm):

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
                market strike

            cm: np.array
                market option price

            Return
            ------
            None, update instance parameters
        """

        n = len(km)
        d = {}
        
        for i in range(n):
            d.setdefault(tm[i], set()).add((km[i], cm[i]))

        parameters = np.array([self.sigma, self.kappa, self.theta, self.xi, self.rho])

        def obj(x):

            """ x = [ sigma, kappa, theta, xi, rho ] """

            sse = 0
            self.sigma, self.kappa, self.theta, self.xi, self.rho = x

            for key, value in d.items():

                self.t = key
                combo = self.heston_fft()
                sse += np.sum((c - self.payoff(k, combo)) ** 2 for k, c in value)
            print(sse, x)
            return sse

        bd = ((0.001, 10), (0, 10), (0, 10), (0.001, 10), (-1, 0))
        minimize(obj, parameters, bounds=bd)


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

    def simulate(self, n=100, path=False, cutoff='truncate', pack=None):

        """
            ds = [r * s] * dt + [sigma * s ** beta] * dW_1
            d(sigma) = [alpha * sigma] * dW_2
            Cov(dW_1, dW_2) = rho * dt
        """

        return super()._simulate(path, (n, False, lambda s: self.r * s, lambda s, sigma: sigma * s ** self.beta,
                                 True, lambda sigma: 0, lambda sigma: self.alpha * _modify(sigma, cutoff),
                                 self.rho), pack)

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
            None, update instance parameters
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
        minimize(obj, parameters, method='L-BFGS-B', bounds=bd)
