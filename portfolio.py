import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve


def obj(delta, mode, sigma, vr=None):

    return - delta @ vr / np.sqrt(delta @ sigma @ delta) if mode == 'msr' else delta @ sigma @ delta


class markovitz:

    def __init__(self, sigma, vr, er, rf=None, bounds=None):

        """
            Markovitz Portfolio Allocation Problem
            --------------------------------------

            Given N risky assets (and possibly one risk-free asset)
            Find the optimal portfolio to maximize return and minimize variance

            Parameters
            ----------

            sigma : np.array, 2-dimensional
                N * N covariance matrix of risky assets

            vr : np.array, 1-dimensional
                N-length array of expected returns of risky assets

            er: -2, -1, or positive float
                Indicates the type of portfolio to return
                    -2: GMV
                    -1: MSR
                    positive float: optimized portfolio given expected portfolio return

            rf: positive float or None, optional
                Expected return for the risk-free asset (if applicable)
                None in default

            bounds: (negative float, negative float) or None, optional
                Limits on shorting
                    First float: limit on one asset
                    Second float: limit on the sum of assets
                None in default
        """

        self.n = len(vr)
        self.sigma = sigma
        self.zcb = rf is not None
        self.vr = vr - rf if self.zcb else vr
        self.er = er - rf if self.zcb and er > 0 else er

        self.bounds = None if bounds is None else [(bounds[0], np.inf)] * self.n

        if bounds is not None:
            self.x0 = np.array([1 / self.n] * self.n)
            self.constraint1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            self.constraint2 = {'type': 'eq', 'fun': lambda x: x @ self.vr - self.er}
            self.constraint3 = {'type': 'ineq', 'fun': lambda x: sum(min(0, i) for i in x) - bounds[1]}

    def gmv(self):

        """ Global Minimum Variance """

        if self.bounds is None:
            m = solve(self.sigma, np.ones(self.n))
            return m / np.sum(m)
        else:
            cons = [self.constraint1, self.constraint3]
            a = minimize(obj, self.x0, ('gmv', self.sigma), 'SLSQP', bounds=self.bounds, constraints=cons)
            return a.x

    def msr(self):

        """ Maximum Sharpe Ratio (with excess return if applicable) """

        if self.bounds is None:

            m = solve(self.sigma, self.vr)
            return m / np.sum(m)

        else:

            cons = [self.constraint1, self.constraint3]
            sol = minimize(obj, self.x0, ('msr', self.sigma, self.vr), 'SLSQP', bounds=self.bounds, constraints=cons)
            return sol.x

    def optimize(self):

        """ Optimal portfolio given the expected portfolio return """

        if self.bounds is None:

            gmv = np.zeros(self.n) if self.zcb else self.gmv()
            msr = self.msr()
            a = (self.er - (self.vr @ gmv)) / (self.vr @ (msr - gmv))
            return a * msr + (1 - a) * gmv

        else:

            cons = [self.constraint2, self.constraint3]
            if not self.zcb:
                cons.append(self.constraint1)
            sol = minimize(obj, self.x0, ('allocate', self.sigma), 'SLSQP', bounds=self.bounds, constraints=cons)
            return sol.x

    def allocate(self):

        return self.gmv() if self.er == -2 else self.msr() if self.er == -1 else self.optimize()


if __name__ == '__main__':

    returns = np.array([0.10, 0.09, 0.16])
    covs = np.array([[0.09, -0.03, 0.084], [-0.03, 0.04, 0.012], [0.084, 0.012, 0.16]])
    mka = markovitz(covs, returns, 0.2, None)
    mk = markovitz(covs, returns, 0.2, None, (-0.5, -1))

    print(mka.allocate())
    print(mk.allocate())



