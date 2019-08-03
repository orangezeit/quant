# Author: Yunfei Luo
# Date: Aug 3, 2019
# version: 0.11.1 (in development)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from scipy.cluster.hierarchy import single, leaves_list
from scipy.linalg import solve, inv
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf, oas


def estimate(df, mean_est=0, cov_est=0, alpha=1e-10):

    """
        Estimate mean and covariance given historical data

        Parameters
        ----------
        df: pd.DataFrame (n.sample, n.feature)
            historical data

        mean_est: int from {0, 1}
            method to estimate mean
            respectively from {'equal_weights', 'exponential_weights'}

        cov_est: int from {0, 1, 2, 3}
            method to estimate covariance
            respectively from {'equal_weights', 'exponential_weights', 'ledoit_wolf', 'oas'}

        alpha: float, required if exponential_weights selected
            0 < alpha <= 1, larger alpha means more weights on near
            exponential_weights -> equal_weights if alpha -> 0

        Return
        ------
        mean, cov: np.array
            estimated mean (n.feature) and covariance (n.feature * n.feature)
    """

    if mean_est == 0:
        mean = df.mean().values
    elif mean_est == 1:
        mean = df.ewm(alpha=alpha).mean().iloc[-1].values
    else:
        raise ValueError('Method does not exist.')

    if cov_est == 0:
        cov = df.cov().values
    elif cov_est == 1:
        cov = df.ewm(alpha=alpha).cov().iloc[-df.shape[1]:].values
    elif cov_est == 2:
        cov, _ = ledoit_wolf(df)
    elif cov_est == 3:
        cov, _ = oas(df)
    else:
        raise ValueError('Method does not exist.')

    return mean, cov


class Markowitz:

    """
        Markowitz Portfolio Allocation
        ------------------------------

        Given n risky assets (and possibly one risk-free asset)
        Find the optimal portfolio to maximize return and to minimize variance
    """

    def __init__(self, cov_mat, exp_ret, idx, target=None, rf=None,
                 bounds=None, mkt_neutral=False, gamma=0.0, tol=1e-10):

        """
            Parameters
            ----------
            cov_mat : np.array
                n * n covariance matrix of risky assets

            exp_ret : np.array
                n-length array of expected (excess) returns of risky assets

            idx: int from {-2, -1, 1, 2}
                desired type of portfolio
                    -2: Global Minimum Variance (GMV)
                    -1: Maximum Sharpe Ratio (MSR)
                     1: Optimized given expected (excess) portfolio return
                     2: Optimized given expected portfolio variance

            target: positive float or None[default], required if idx is 1 or 2
                target return if idx == 1, target variance if idx == 2

            rf: positive float or None[default], optional
                expected return for the risk-free asset (if applicable)

            bounds: tuple(tuple(float)) or None[default], optional
                bounds for optimized weights
                    same bounds for each if len(bounds) is 1
                    different bounds for each if len(bounds) is n

            mkt_neutral: bool, False[default], optional
                indicates whether the portfolio is market-neutral
                the sum of weights is 0 if true or 1 if false

            gamma: float, 0.0[default], optional
                parameter for L2 (penalty for over-fitting)

            tol: positive float, 1e-10[default], optional
                tolerance for termination
        """

        if not isinstance(cov_mat, np.ndarray) or len(cov_mat.shape) != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError('Covariance matrix must be a square matrix')

        if not isinstance(exp_ret, np.ndarray) or len(exp_ret.shape) != 1 or len(exp_ret) != cov_mat.shape[0]:
            raise ValueError('Expected return must be an array with same length as size of covariance matrix')

        if idx not in {-2, -1, 1, 2}:
            raise ValueError('Indicator must be -2, -1, 1 or 2.')

        if idx in {1, 2} and not isinstance(target, float):
            raise ValueError('Target return or variance must be a positive float.')

        if rf is not None and not isinstance(rf, float):
            raise ValueError('Risk-free rate must be None or a positive float')

        if not isinstance(mkt_neutral, bool):
            raise ValueError('Market neutral must be true or false.')

        if not isinstance(gamma, float) or gamma < 0:
            raise ValueError('Gamma must be a non-negative float.')

        if not isinstance(tol, float) or tol <= 0:
            raise ValueError('Tolerance must be a positive float.')

        self.n = len(exp_ret)

        self.cov_mat = cov_mat
        self.rf = rf
        self.exp_ret = exp_ret - rf if rf else exp_ret
        self.idx = idx
        self.bounds = (bounds[0],) * self.n if bounds and len(bounds) == 1 else bounds

        if bounds or mkt_neutral:

            self.x0 = np.array([1 / self.n] * self.n)
            self.constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) if mkt_neutral else np.sum(x) - 1}

        self.mkt_neutral = mkt_neutral
        self.tol = tol
        self.gamma = gamma

        if idx == 1:
            self.target = max(target - rf if rf else target, self._gmv() @ exp_ret)
        elif idx == 2:
            self.target = max(target, self._gmv() @ cov_mat @ self._gmv())

        # save weights for performance analysis
        self.weights = None

    def _obj1(self, weights):

        # portfolio volatility
        return weights @ self.cov_mat @ weights + self.gamma * np.sum(weights ** 2)

    def _obj2(self, weights):

        # negative sharpe ratio
        return -weights @ self.exp_ret / np.sqrt(self._obj1(weights))

    def _gmv(self):

        """
            Global Minimum Variance (GMV)

            Return
            ------
            weights: np.array, n-length
        """

        if self.bounds or self.mkt_neutral:

            self.weights = minimize(self._obj1, self.x0, bounds=self.bounds,
                                    constraints=self.constraint, tol=self.tol).x

        else:

            weights = solve(self.cov_mat, np.ones(self.n))  # calculate weights
            self.weights = weights / np.sum(weights)        # scale to sum 1

        return self.weights

    def _msr(self):

        """
            Maximum Sharpe Ratio (MSR)

            Return
            ------
            weights: np.array, n-length
        """

        if self.bounds or self.mkt_neutral:

            self.weights = minimize(self._obj2, self.x0, bounds=self.bounds,
                                    constraints=self.constraint, tol=self.tol).x

        else:

            weights = solve(self.cov_mat, self.exp_ret)  # calculate weights
            self.weights = weights / np.sum(weights)     # scale to sum 1

        return self.weights

    def _optimize_return(self):

        """
            Optimal portfolio given the expected portfolio return

            Return
            ------
            weights: np.array, n-length
        """

        if self.bounds or self.mkt_neutral:

            cons = [{'type': 'eq', 'fun': lambda x: x @ self.exp_ret - self.target}]
            if self.rf is None:
                cons.append(self.constraint)
            self.weights = minimize(self._obj1, self.x0, bounds=self.bounds,
                                    constraints=cons, tol=self.tol).x

        else:

            # one-asset / two-asset theorems
            gmv = np.zeros(self.n) if self.rf else self._gmv()
            msr = self._msr()
            a = (self.target - self.exp_ret @ gmv) / (self.exp_ret @ (msr - gmv))
            self.weights = a * msr + (1 - a) * gmv

        return self.weights

    def _optimize_risk(self):

        """
            Optimal portfolio given the expected portfolio variance

            Return
            ------
            weights: np.array, n-length
        """

        if self.bounds or self.mkt_neutral:

            cons = [{'type': 'eq', 'fun': lambda x: x @ self.cov_mat @ x - self.target}]
            if self.rf is None:
                cons.append(self.constraint)
            self.weights = minimize(self._obj2, self.x0, bounds=self.bounds,
                                    constraints=cons, tol=self.tol).x

        else:

            # one-asset / two-asset theorems
            gmv = np.zeros(self.n) if self.rf else self._gmv()
            msr = self._msr()
            a = (self.target - gmv @ self.cov_mat @ gmv) / (msr @ self.cov_mat @ msr - gmv @ self.cov_mat @ gmv)
            self.weights = a * msr + (1 - a) * gmv

        return self.weights

    def allocate(self):

        """ User API, return desired optimized weights given indicator """

        if self.idx == -2:
            return self._gmv()
        elif self.idx == -1:
            return self._msr()
        elif self.idx == 1:
            return self._optimize_return()
        else:
            return self._optimize_risk()

    def efficient_frontier(self, n_sim=10000):

        """ User API, plot scatter plot of simulated portfolios to visualize efficient frontier """

        pylab.rcParams.update({'legend.fontsize': 'x-large',
                               'axes.labelsize': 'x-large',
                               'axes.titlesize': 'x-large',
                               'xtick.labelsize': 'x-large',
                               'ytick.labelsize': 'x-large'})

        weights = np.empty((n_sim, self.n))

        for i in range(n_sim):
            weights[i] = np.random.random(self.n)
            weights[i] /= weights[i].sum()

        ret_ports = np.array([weights[i] @ self.exp_ret * 252 for i in range(n_sim)])
        vol_ports = np.array([np.sqrt(weights[i] @ self.cov_mat @ weights[i] * 252) for i in range(n_sim)])
        sharpe_ports = ret_ports / vol_ports

        msr_ret, msr_vol = ret_ports[sharpe_ports.argmax()], vol_ports[sharpe_ports.argmax()]
        gmv_ret, gmv_vol = ret_ports[vol_ports.argmin()], vol_ports[vol_ports.argmin()]

        plt.figure(figsize=(12, 8), facecolor='lightgray')
        plt.scatter(vol_ports, ret_ports, c=sharpe_ports, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.scatter(msr_vol, msr_ret, c='red', s=50)
        plt.scatter(gmv_vol, gmv_ret, c='orangered', s=50)
        plt.show()

    def performance(self):

        """ User API, print performance analysis """

        port_ret = self.weights @ self.exp_ret
        port_var = self.weights @ self.cov_mat @ self.weights

        print('Optimized weights:   ', self.weights)
        print('Expected Return:     ', port_ret)
        print('Portfolio Variance:  ', port_var)
        print('Sharpe Ratio:        ', port_ret / port_var)


class BlackLitterman:

    """
        Black-Litterman Portfolio Allocation
        ------------------------------------

        Given n risky assets
        Find the optimal portfolio given the subjective views
    """

    def __init__(self, c, w, p, q, omega, xi=0.42667, tau=0.1):

        """
            Parameters
            ----------

            c: np.array
                n * n covariance matrix of risky assets
            w:
                n-length weights of market cap

            p * r = q + omega

            p:
            q:
            omega:

            xi: historical stock index
                For US investors, it is 0.42667
            tau: subjective constant
                sigma = tau * c
                smaller -> larger confidence in covariance matrix
        """

        self.c = c
        self.pi = c @ w / xi
        self.sigma = c * tau
        self.p = p
        self.q = q
        self.xi = xi

        temp = self.sigma @ p.T
        self.temp = temp @ inv(p @ temp + omega)

    def _post_exp_ret(self):

        """ Return posterior expected return """

        return self.pi + self.temp @ (self.q - self.p @ self.pi)

    def _post_sigma(self):

        """ Return posterior covariance matrix """

        return self.sigma + self.c - self.temp @ self.p @ self.sigma

    def optimal_weights(self):

        """ Return optimal weights based on posterior expected return and covariance matrix """

        return solve(self._post_sigma(), self._post_vr()) * self.xi


class RiskParity:

    """ Focuses on risk allocation """

    def __init__(self, df):

        """
            Parameters
            ----------
            df: np.array (n.sample, n.feature)
                historical data

            cov: np.array
                covariance matrix of historical data (n.feature * n.feature)

        """

        self.df = df
        _, self.cov = estimate(df)

    def ivp(self, cov=None):

        """
            Inverse Variance Portfolio (IVP)
                Weights are proportional to inverses of asset variances

            Parameter
            ---------
            cov: np.array, None[default] which maps to self.cov later
                covariance matrix, optional for IVP, required for cluster variance

            Return
            ------
            weights: np.array, n-length
        """

        diag = np.diag(self.cov if cov is None else cov)

        weights = np.array([0 if d == 0 else 1 / d for d in diag])

        return weights / weights.sum() if weights.sum() > 0 else weights

    def _cluster_var(self, idx):

        """
            Calculate variance of sub-portfolio (cluster variance)

            Parameter
            ---------
            idx: indices of the cluster

            Return
            ------
            var: float
                cluster variance
        """

        cov = self.cov[idx, :][:, idx]
        weights = self.ivp(cov)
        return weights @ cov @ weights

    def hrp(self):

        """
            Hierarchy Risk Parity

            Return
            ------
            weights: np.array, n-length
        """

        # construct distance matrix
        dist_mat = squareform(((1 - self.df.corr().values) / 2) ** 0.5, checks=False)
        if np.isnan(dist_mat).any():
            np.nan_to_num(dist_mat, False)

        # tree clustering / quasi-diagonalization
        idx = leaves_list(single(dist_mat))

        # recursive bisection
        weights = pd.Series([1] * len(idx))
        idx = [idx]

        while len(idx):
            # bisect
            idx = [i[j:k] for i in idx for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            # assign pairwise weights
            for i in range(0, len(idx), 2):

                var1 = self._cluster_var(idx[i])
                var2 = self._cluster_var(idx[i + 1])
                if var1 and var2:
                    weights[idx[i]] *= var2 / (var1 + var2)
                    weights[idx[i + 1]] *= var1 / (var1 + var2)
                else:
                    weights[idx[i]] *= .5
                    weights[idx[i + 1]] *= .5

        return weights.values

    def trp(self, cutoff=0):

        """
            Tail Risk Parity
                Minimize risk of potential loss

            Parameter
            ---------
            cutoff: end point of the integral (risk measure)
                indicates maximum loss of portfolio

            Return
            ------
            weights: np.array, n-length
        """

        _, n = self.df.shape
        x0 = np.array([1 / n] * n)
        bound = ((0, np.inf),) * n
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        return minimize(lambda weights: gaussian_kde((self.df * weights).sum(1)).integrate_box(-np.inf, cutoff),
                        x0, bounds=bound, constraints=constraint).x


import random

if __name__ == '__main__':

    # test code
    """
    r = np.array([0.10, 0.09, 0.16])
    r2 = np.array([-0.001237, 0.004848, -0.003694, 0.007403, -0.000610])
    covs = np.array([[0.09, -0.03, 0.084], [-0.03, 0.04, 0.012], [0.084, 0.012, 0.16]])
    covs2 = np.array([[0.002027, -0.000362,  0.000099, -0.000220, -0.000305],
                      [-0.000362, 0.002421, 0.000297, 0.000090, 0.000151],
                      [0.000099, 0.000297, 0.002420, 0.000020, 0.000113],
                      [-0.000220,  0.000090,  0.000020,  0.002302, 0.000047],
                      [-0.000305,  0.000151,  0.000113,  0.000047,  0.002877]])

    #mka = Markowitz(covs, returns, 0.2, None, None)
    #mk = Markowitz(covs, returns, 0.2, None, (0, 0))
    mk_test = Markowitz(covs, r, 1, bounds=((-np.inf, np.inf),), target=0.00376)
    mk_test2 = Markowitz(covs, r, 1, target=0.00376)

    w = np.array([35/50, 10/50, 5/50])
    c = np.array([[7.344, 2.015, 3.309], [2.015, 4.410, 1.202], [3.309, 1.202, 3.497]]) / 100

    p = np.array([[1, 0, 0], [-1, 1, 0]])
    q = np.array([2.5/100, 2/100])
    omega = np.array([[0.01, 0], [0, 0.015]]) ** 2

    bl = BlackLitterman(c, w, p, q, omega)
    print(bl.optimal_weights())
    """

    # [201.22, 244.88, 16.83, 259.53, 370.59, 369.38, 221.56, 1060, 100.92, 332.47]

    test_set = pd.DataFrame([[random.uniform(-1, 1) for _ in range(10)] for _ in range(30)])

    #cvar(test_set)
    res = RiskParity(test_set).hrp()
    print(res)


