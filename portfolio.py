# Author: Yunfei Luo
# Date: Jul 28, 2019
# version: 0.10.2 (in development)

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage
from scipy.linalg import solve, inv
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.covariance import ledoit_wolf, oas

import random


def estimate(df, mean_est=0, cov_est=0, alpha=0.0):

    """
        estimate mean and covariance given historical data

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
            0 <= alpha < 1, larger alpha means more weights on near
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
        cov = ledoit_wolf(df)[0]
    elif cov_est == 3:
        cov = oas(df)[0]
    else:
        raise ValueError('Method does not exist.')

    return mean, cov


class Markowitz:

    """
        Markowitz Portfolio Allocation Problem
        --------------------------------------

        Given n risky assets (and possibly one risk-free asset)
        Find the optimal portfolio to maximize return and minimize variance
    """

    def __init__(self, cov_mat, exp_ret, idx, target=None, rf=None,
                 bounds=None, mkt_neutral=False, gamma=0.0, tol=1e-10):

        """
            Parameters
            ----------
            cov_mat : np.ndarray
                n * n covariance matrix of risky assets

            exp_ret : np.ndarray
                n-length array of expected (excess) returns of risky assets

            idx: int from {-2, -1, 1, 2}
                desired type of portfolio
                    -2: Global Minimum Variance (GMV)
                    -1: Maximum Sharpe Ratio (MSR)
                     1: Optimized given expected (excess) portfolio return
                     2: Optimized given expected portfolio variance

            target: positive float or None[default], required if idx is 1 or 2, optional otherwise
                target return if idx == 1, target variance if idx == 2

            rf: positive float or None[default], optional
                Expected return for the risk-free asset (if applicable)

            bounds: tuple(tuple(float)) or None[default], optional
                Bounds for weights
                    same bounds for each if len(bounds) is 1
                    different bounds for each if len(bounds) is n

            mkt_neutral: bool, False in default, optional
                indicates whether the portfolio is market-neutral
                the sum of weights is 0 if true or 1 if false

            gamma: float, 0.0 in default
                parameter for L2

            tol: positive float, 1e-10 in default, optional
                Tolerance for termination
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
            weights: np.ndarray, n-length
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
            weights: np.ndarray, n-length
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
            weights: np.ndarray, n-length
        """

        if self.bounds or self.mkt_neutral:

            cons = [{'type': 'eq', 'fun': lambda x: x @ self.exp_ret - self.target}]
            if self.rf is None:
                cons.append(self.constraint)
            self.weights = minimize(self._obj1, self.x0, bounds=self.bounds, constraints=cons, tol=self.tol).x

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
            weights: np.ndarray, n-length
        """

        if self.bounds or self.mkt_neutral:

            cons = [{'type': 'eq', 'fun': lambda x: x @ self.cov_mat @ x - self.target}]
            if self.rf is None:
                cons.append(self.constraint)
            self.weights = minimize(self._obj2, self.x0, bounds=self.bounds, constraints=cons, tol=self.tol).x

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
        Black-Litterman Portfolio Allocation Problem
        --------------------------------------------

        Given n risky assets
        Find the optimal portfolio given the subjective views
    """

    def __init__(self, c, w, p, q, omega, xi=0.42667, tau=0.1):

        """
            Parameters
            ----------

            c: 2d np.array
                n * n covariance matrix of risky assets
            w:
                weight of market cap

            p * r = q + omega

            p:
            q:
            omega:

            xi: historical stock index
                For US investors,
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

    def _post_vr(self):

        return self.pi + self.temp @ (self.q - self.p @ self.pi)

    def _post_sigma(self):

        return self.sigma + self.c - self.temp @ self.p @ self.sigma

    def optimal_weights(self):

        return solve(self._post_sigma(), self._post_vr()) * self.xi


class RiskParity:

    """ Portfolio that focuses on risk allocation """

    def __init__(self, df):

        _, self.cov = estimate(df)
        self.corr = df.corr().values

    def ivp(self, cov=None):

        """ inverse variance portfolio """

        _cov = self.cov if cov is None else cov
        weights = 1 / np.diag(_cov)
        return weights / weights.sum()

    def _cluster_var(self, idx):

        cov = self.cov[idx, :][:, idx]
        weights = self.ivp(cov)
        return weights @ cov @ weights

    def hrp(self):

        """ Hierarchy Risk Parity """

        # construct distance matrix
        dist_mat = ((1 - self.corr) / 2) ** 0.5
        link = linkage(dist_mat).astype(int)

        # sort clustered indices by distance
        idx = pd.Series([link[-1, 0], link[-1, 1]])
        num = link[-1, 3]

        while idx.max() >= num:

            idx.index *= 2
            i, j = idx[idx >= num].index, idx[idx >= num].values - num
            idx[i] = link[j, 0]
            idx = idx.append(pd.Series(link[j, 1], index=i + 1)).sort_index()

        # allocate weights by bisection
        weights = pd.Series(1, index=idx)
        idx = [idx.values]

        while len(idx):

            idx = [i[j:k] for i in idx for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bisect

            for i in range(0, len(idx), 2):  # assign pairwise weights

                var1 = self._cluster_var(idx[i])
                var2 = self._cluster_var(idx[i + 1])
                weights[idx[i]] *= var2 / (var1 + var2)
                weights[idx[i + 1]] *= var1 / (var1 + var2)

        return weights.sort_index().values


def cvar(df, alpha=0.05):

    x0 = np.array([1 / 10] * 10)   # initial guess
    bounds = ((0, 1),) * 10
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def obj(w):

        weighted_df = df * w
        sample = gaussian_kde(weighted_df.values.T).resample(10000)
        cutoff = weighted_df.quantile(alpha).values.reshape(10, 1)
        return -sample[sample < cutoff].mean()

    obj(x0)




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
    res = mk_test.allocate()
    res2 = mk_test2.allocate()

    mk_test.performance()
    mk_test2.performance()

    

    w = np.array([35/50, 10/50, 5/50])
    c = np.array([[7.344, 2.015, 3.309], [2.015, 4.410, 1.202], [3.309, 1.202, 3.497]]) / 100

    p = np.array([[1, 0, 0], [-1, 1, 0]])
    q = np.array([2.5/100, 2/100])
    omega = np.array([[0.01, 0], [0, 0.015]]) ** 2

    bl = BlackLitterman(c, w, p, q, omega)
    print(bl.optimal_weights())
    """


    # [201.22, 244.88, 16.83, 259.53, 370.59, 369.38, 221.56, 1060, 100.92, 332.47]

    test_set = pd.DataFrame([[random.random() for _ in range(10)] for _ in range(30)])
    rp = RiskParity(test_set)
    w = rp.hrp()
    print(w, sum(w))








